"""音频提取 + Whisper 转录。

支持两种后端：
- faster-whisper：CTranslate2 格式，首次需从 HuggingFace 下载模型（国内可能被墙）
- openai-whisper：原版 .pt 模型，默认从 ~/.cache/whisper 离线加载

backend="auto"（默认）：先试 faster-whisper，失败则回退到 openai-whisper。
"""

import os
import tempfile

# 国内直连 huggingface.co 常被墙，faster-whisper 默认走镜像 hf-mirror.com。
# 必须在导入 faster_whisper / huggingface_hub 之前设置才生效。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import ffmpeg


def _transcribe_faster(audio_path: str, model_size: str, lang, prompt) -> list[dict]:
    """faster-whisper 后端（CTranslate2 模型，可能需联网下载）。"""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size)
    seg_iter, _info = model.transcribe(
        audio_path, language=lang, word_timestamps=False,
        initial_prompt=prompt or None,
    )
    return [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in seg_iter
    ]


def _transcribe_openai(audio_path: str, model_size: str, lang, prompt) -> list[dict]:
    """openai-whisper 后端（原版 .pt 模型，从 ~/.cache/whisper 离线加载）。"""
    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(
        audio_path, language=lang, verbose=False,
        initial_prompt=prompt or None,
    )
    return [
        {
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": s["text"].strip(),
        }
        for s in result.get("segments", [])
    ]


def _run_backend(
    audio_path: str, model_size: str, lang, backend: str, prompt: str
) -> list[dict]:
    if backend == "faster":
        return _transcribe_faster(audio_path, model_size, lang, prompt)
    if backend == "openai":
        return _transcribe_openai(audio_path, model_size, lang, prompt)

    # auto：先 faster，失败回退 openai
    try:
        return _transcribe_faster(audio_path, model_size, lang, prompt)
    except Exception as e_fast:
        print("  faster-whisper 不可用，改用 openai-whisper 本地模型……")
        try:
            return _transcribe_openai(audio_path, model_size, lang, prompt)
        except Exception as e_openai:
            raise RuntimeError(
                "两种 Whisper 后端均失败：\n"
                f"  - faster-whisper：{e_fast}\n"
                f"  - openai-whisper：{e_openai}\n"
                "  建议：\n"
                "  1) 用本地已有的原版模型：--whisper-backend openai\n"
                "     （需 pip install openai-whisper，模型放在 ~/.cache/whisper）\n"
                "  2) 或换更小的模型：--whisper-model tiny\n"
                "  3) 或挂代理让 faster-whisper 能访问模型仓库"
            ) from e_openai


def transcribe_video(
    video_path: str,
    model_size: str = "base",
    language: str = "zh",
    limit_sec: float = 0.0,
    backend: str = "auto",
    start_sec: float = 0.0,
    prompt: str = "",
) -> list[dict]:
    """提取音频并转录。

    返回 [{"start": float, "end": float, "text": str}, ...]（时间戳为视频内绝对秒）。
    start_sec / limit_sec：只转录 [start_sec, limit_sec] 这段。
    prompt：术语提示词，提升专有名词识别准确率。
    backend：auto / faster / openai。
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在：{video_path}")

    # 1. 提取音频到临时 wav（16kHz 单声道）
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        in_kwargs = {}
        if start_sec > 0:
            in_kwargs["ss"] = start_sec  # 从 start_sec 开始截取
        out_kwargs = dict(ar=16000, ac=1, loglevel="quiet")
        if limit_sec > 0:
            out_kwargs["t"] = max(0.0, limit_sec - start_sec)  # 截到 limit_sec
        (
            ffmpeg
            .input(video_path, **in_kwargs)
            .output(tmp, **out_kwargs)
            .overwrite_output()
            .run()
        )

        # 2. 转录（language=None 表示自动检测）
        lang = None if (language in (None, "", "None", "none")) else language
        segments = _run_backend(tmp, model_size, lang, backend, prompt)
        # 截取的音频时间从 0 起，补回 start_sec 还原为视频内绝对时间
        if start_sec > 0:
            for s in segments:
                s["start"] += start_sec
                s["end"] += start_sec
    finally:
        # 3. 清理临时文件
        if os.path.exists(tmp):
            os.remove(tmp)

    print(f"转录完成，共 {len(segments)} 个片段")
    return segments


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：python transcriber.py VIDEO_PATH")
        sys.exit(1)
    segs = transcribe_video(sys.argv[1])
    print("前 10 个片段：")
    for s in segs[:10]:
        print(f"  [{s['start']:.1f}-{s['end']:.1f}] {s['text']}")
