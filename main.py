"""视频网课分析工具 — CLI 入口。"""

import os
import time

import click

from frame_extractor import extract_key_frames
from transcriber import transcribe_video
from aligner import align_frames_with_transcript
from analyzer import analyze_course, MODEL_PRICING, DEFAULT_MODEL
from reporter import save_report


def get_video_duration(video_path: str) -> float:
    """返回视频时长（秒）。"""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps if fps > 0 else 0


def _step(label: str):
    print(f"\n{label}  ({time.strftime('%H:%M:%S')})")
    return time.time()


def _done(start: float):
    print(f"  完成，耗时 {time.time() - start:.1f}s")


_TOKENS_PER_IMAGE = 1600  # 长边 1024px JPEG 的大致上限


def estimate_cost(aligned: list[dict], batch_size: int, model: str) -> dict:
    """粗略预估第 4 步 Claude 分析的 token 与费用（按所选模型定价）。"""
    price = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])
    price_in, price_out = price["in"], price["out"]

    n_frames = len(aligned)
    n_batches = max(1, (n_frames + batch_size - 1) // batch_size)

    image_tokens = n_frames * _TOKENS_PER_IMAGE
    # 中文按 ~1 token/字 粗估转录文字
    text_tokens = sum(len(item.get("transcript", "")) for item in aligned)
    prompt_overhead = n_batches * 150
    input_tokens = image_tokens + text_tokens + prompt_overhead

    # 输出：每批最多 4096，按典型 ~40% 估算
    out_tokens_typical = int(n_batches * 4096 * 0.4)
    out_tokens_max = n_batches * 4096

    cost_typical = (
        input_tokens / 1e6 * price_in
        + out_tokens_typical / 1e6 * price_out
    )
    cost_max = (
        input_tokens / 1e6 * price_in
        + out_tokens_max / 1e6 * price_out
    )
    return {
        "n_frames": n_frames,
        "n_batches": n_batches,
        "input_tokens": input_tokens,
        "out_tokens_typical": out_tokens_typical,
        "cost_typical": cost_typical,
        "cost_max": cost_max,
    }


@click.command()
@click.argument("video_path", type=click.Path())
@click.option("--output-dir", default=".", help="笔记输出目录 [默认: 当前目录]")
@click.option("--subject", default="", help="课程主题提示，帮助 Claude 理解上下文")
@click.option("--threshold", default=0.12, type=float, help="关键帧差异阈值 0~1 [默认: 0.12]")
@click.option("--interval", default=30, type=int, help="强制兜底间隔（秒）[默认: 30]")
@click.option("--batch-size", default=8, type=int, help="每次发给 Claude 的帧数 [默认: 8]")
@click.option("--whisper-model", default="base", help="Whisper 模型大小 [默认: base]")
@click.option("--language", default="zh", help="音频语言，zh/en/None [默认: zh]")
@click.option("--model", default=DEFAULT_MODEL,
              type=click.Choice(list(MODEL_PRICING.keys())),
              help=f"Claude 模型，越贵质量越好 [默认: {DEFAULT_MODEL}]")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="跳过 Claude 分析前的费用确认（用于自动化脚本）")
def main(
    video_path,
    output_dir,
    subject,
    threshold,
    interval,
    batch_size,
    whisper_model,
    language,
    model,
    yes,
):
    """视频网课分析工具：输入视频，输出结构化 Markdown 笔记。"""
    print("=== 视频网课分析工具 ===")

    if not os.path.exists(video_path):
        raise click.ClickException(f"视频文件不存在：{video_path}")

    duration = get_video_duration(video_path)
    print(f"视频文件：{video_path}")
    m, s = divmod(int(duration), 60)
    print(f"视频时长：{m:02d}:{s:02d}")

    # 步骤 1/4：提取关键帧
    t = _step("[步骤 1/4] 提取关键帧")
    frames = extract_key_frames(
        video_path,
        diff_threshold=threshold,
        max_interval_sec=float(interval),
    )
    _done(t)

    # 步骤 2/4：转录音频
    t = _step("[步骤 2/4] 转录音频")
    segments = transcribe_video(
        video_path,
        model_size=whisper_model,
        language=language,
    )
    _done(t)

    # 步骤 3/4：对齐时间轴
    t = _step("[步骤 3/4] 对齐时间轴")
    aligned = align_frames_with_transcript(frames, segments)
    _done(t)

    # === 费用确认（第 4 步开始前，这是唯一花钱的一步）===
    est = estimate_cost(aligned, batch_size, model)
    print("\n" + "=" * 48)
    print("⚠️  下一步将调用 Claude API，会消耗你的 token（扣费）")
    print("=" * 48)
    print(f"  模型          : {model}")
    print(f"  关键帧数      : {est['n_frames']} 帧")
    print(f"  分批数        : {est['n_batches']} 批（每批 {batch_size} 帧）")
    print(f"  预计输入 token: ~{est['input_tokens']:,}")
    print(f"  预计输出 token: ~{est['out_tokens_typical']:,}（每批上限 4096）")
    print(f"  粗略费用      : ~${est['cost_typical']:.2f}"
          f"（最坏约 ${est['cost_max']:.2f}）")
    print(f"  注：费用按 {model} 标准定价估算，实际以 Anthropic 官方账单为准")
    print("=" * 48)

    if not yes:
        if not click.confirm("确认继续分析并扣费？", default=False):
            print("已取消。前 3 步均为本地处理，未产生任何费用。")
            return

    # 步骤 4/4：Claude 分析
    t = _step("[步骤 4/4] Claude 分析")
    raw_notes = analyze_course(
        aligned,
        subject_hint=subject,
        batch_size=batch_size,
        model=model,
    )
    _done(t)

    # 保存笔记
    path = save_report(
        raw_notes,
        video_path,
        output_dir=output_dir,
        duration=duration,
        frame_count=len(frames),
    )
    print(f"\n✅ 完成！笔记已保存：{path}")


if __name__ == "__main__":
    main()
