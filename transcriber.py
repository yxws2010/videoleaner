"""音频提取 + Whisper 转录。"""

import os
import tempfile

import ffmpeg
from faster_whisper import WhisperModel


def transcribe_video(
    video_path: str,
    model_size: str = "base",
    language: str = "zh",
    limit_sec: float = 0.0,
) -> list[dict]:
    """提取音频并转录。

    返回 [{"start": float, "end": float, "text": str}, ...]
    limit_sec > 0：只转录前这么多秒（配合大文件快速测试）。
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在：{video_path}")

    # 1. 提取音频到临时 wav（16kHz 单声道）
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        out_kwargs = dict(ar=16000, ac=1, loglevel="quiet")
        if limit_sec > 0:
            out_kwargs["t"] = limit_sec  # 只截取前 limit_sec 秒音频
        (
            ffmpeg
            .input(video_path)
            .output(tmp, **out_kwargs)
            .overwrite_output()
            .run()
        )

        # 2. 转录
        # language=None 表示自动检测
        lang = None if (language in (None, "", "None", "none")) else language
        model = WhisperModel(model_size)
        seg_iter, _info = model.transcribe(
            tmp,
            language=lang,
            word_timestamps=False,
        )

        segments = [
            {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in seg_iter
        ]
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
