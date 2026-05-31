"""视频网课分析工具 — CLI 入口。"""

import os
import time

import click

from frame_extractor import extract_key_frames
from transcriber import transcribe_video
from aligner import align_frames_with_transcript
from analyzer import (
    analyze_course, MODEL_PRICING, DEFAULT_MODEL, DEFAULT_MINIMAX_MODEL,
)
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


def _tokens_per_image(image_max_side: int) -> int:
    """估算单张图片的 token 数。

    Anthropic 视觉大致按 (宽×高)/750 计费，这里用正方形长边作上限估算，
    随尺寸平方缩放：长边减半 → token 约降到 1/4。
    """
    return max(1, int(image_max_side * image_max_side / 750))


def estimate_cost(
    aligned: list[dict],
    batch_size: int,
    model: str,
    image_max_side: int = 1024,
) -> dict:
    """粗略预估第 4 步 Claude 分析的 token 与费用（按所选模型定价）。"""
    price = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])
    price_in, price_out = price["in"], price["out"]

    n_frames = len(aligned)
    n_batches = max(1, (n_frames + batch_size - 1) // batch_size)

    image_tokens = n_frames * _tokens_per_image(image_max_side)
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
@click.option("--whisper-backend", default="auto",
              type=click.Choice(["auto", "faster", "openai"]),
              help="转录后端：auto(先faster失败回退openai)/faster/openai [默认: auto]")
@click.option("--language", default="zh", help="音频语言，zh/en/None [默认: zh]")
@click.option("--provider", default="anthropic",
              type=click.Choice(["anthropic", "minimax"]),
              help="大模型来源：anthropic=Claude，minimax=你的 MiniMax [默认: anthropic]")
@click.option("--model", default="",
              help="模型名。anthropic 可选 claude-opus/sonnet/haiku-4-5；"
                   "minimax 需用多模态模型（默认 MiniMax-M2.5，M2.7 不支持图片）")
@click.option("--base-url", default="",
              help="MiniMax OpenAI 兼容接口地址（留空用默认或 MINIMAX_BASE_URL 环境变量）")
@click.option("--image-max-side", default=1024, type=int,
              help="发给 Claude 的图片长边像素，越小越省 token [默认: 1024]")
@click.option("--image-quality", default=85, type=click.IntRange(1, 100),
              help="JPEG 压缩质量 1~100，越低越省 token [默认: 85]")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="跳过 Claude 分析前的费用确认（用于自动化脚本）")
@click.option("--dry-run", is_flag=True, default=False,
              help="只跑前 3 步并显示预估，不调用大模型、不扣额度（验证流程用）")
@click.option("--limit-sec", default=0.0, type=float,
              help="只处理视频前 N 秒，大文件快速测试用 [默认: 0=整段]")
@click.option("--max-frames", default=0, type=int,
              help="最多提取 N 个关键帧后停止，快速测试用 [默认: 0=不限]")
def main(
    video_path,
    output_dir,
    subject,
    threshold,
    interval,
    batch_size,
    whisper_model,
    whisper_backend,
    language,
    provider,
    model,
    base_url,
    image_max_side,
    image_quality,
    yes,
    dry_run,
    limit_sec,
    max_frames,
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
        max_frames=max_frames,
        limit_sec=limit_sec,
    )
    _done(t)

    # 步骤 2/4：转录音频
    t = _step("[步骤 2/4] 转录音频")
    segments = transcribe_video(
        video_path,
        model_size=whisper_model,
        language=language,
        limit_sec=limit_sec,
        backend=whisper_backend,
    )
    _done(t)

    # 步骤 3/4：对齐时间轴
    t = _step("[步骤 3/4] 对齐时间轴")
    aligned = align_frames_with_transcript(frames, segments)
    _done(t)

    # 解析模型默认值（按 provider）
    if not model:
        model = DEFAULT_MODEL if provider == "anthropic" else DEFAULT_MINIMAX_MODEL

    # === 费用确认（第 4 步开始前，这是唯一花钱的一步）===
    est = estimate_cost(aligned, batch_size, model, image_max_side)
    print("\n" + "=" * 48)
    print("⚠️  下一步将调用大模型 API，会消耗你的额度（扣费）")
    print("=" * 48)
    print(f"  来源/模型     : {provider} / {model}")
    print(f"  图片设置      : 长边 {image_max_side}px / 质量 {image_quality}")
    print(f"  关键帧数      : {est['n_frames']} 帧")
    print(f"  调用次数      : {est['n_batches']} 次（每次 {batch_size} 帧）")
    print(f"  预计输入 token: ~{est['input_tokens']:,}")
    print(f"  预计输出 token: ~{est['out_tokens_typical']:,}（每次上限 4096）")
    if provider == "anthropic":
        print(f"  粗略费用      : ~${est['cost_typical']:.2f}"
              f"（最坏约 ${est['cost_max']:.2f}）")
        print(f"  注：费用按 {model} 标准定价估算，实际以 Anthropic 官方账单为准")
    else:
        print(f"  扣费方式      : 从你的 MiniMax 套餐扣 {est['n_batches']} 次模型调用")
        print("  注：MiniMax 按套餐次数/token 计费，具体以 MiniMax 账户为准")
    print("=" * 48)

    if dry_run:
        print("🧪 dry-run：仅验证流程，未调用大模型、未扣任何额度。")
        print(f"   （已就绪 {est['n_frames']} 帧 / {est['n_batches']} 次调用，"
              "去掉 --dry-run 即可真正分析）")
        return

    if not yes:
        if not click.confirm("确认继续分析并扣费？", default=False):
            print("已取消。前 3 步均为本地处理，未产生任何费用。")
            return

    # 步骤 4/4：大模型分析
    t = _step("[步骤 4/4] 大模型分析")
    raw_notes = analyze_course(
        aligned,
        subject_hint=subject,
        batch_size=batch_size,
        model=model,
        image_max_side=image_max_side,
        image_quality=image_quality,
        provider=provider,
        base_url=(base_url or os.environ.get("MINIMAX_BASE_URL") or None),
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
