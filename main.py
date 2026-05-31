"""视频网课分析工具 — CLI 入口。"""

import os
import time

import click

from frame_extractor import extract_key_frames
from transcriber import transcribe_video
from aligner import align_frames_with_transcript
from analyzer import (
    analyze_course, MODEL_PRICING, DEFAULT_MODEL, DEFAULT_MINIMAX_MODEL,
    PROVIDER_MODELS, DEFAULT_MODELS, USD_BILLED_PROVIDERS,
)

# provider 友好名称（向导显示用）
PROVIDER_LABELS = {
    "anthropic": "Claude (Anthropic)",
    "openai": "ChatGPT (OpenAI)",
    "minimax": "MiniMax",
}
# 各 provider 的密钥环境变量名
PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}
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


def _resolve_base_url(provider: str, base_url: str) -> str | None:
    """决定传给客户端的 base_url：显式 --base-url 优先，其次按 provider 取默认。"""
    if base_url:
        return base_url
    if provider == "minimax":
        return os.environ.get("MINIMAX_BASE_URL") or None
    return None  # openai 用官方默认；anthropic 不使用


def _pick(title: str, options: list[str], default_idx: int = 1) -> str:
    """打印带编号的选项并返回所选项（编号从 1 起）。"""
    print(title)
    for idx, opt in enumerate(options, 1):
        print(f"   [{idx}] {opt}")
    choice = click.prompt(
        "   请选择", default=default_idx, type=click.IntRange(1, len(options))
    )
    return options[choice - 1]


def run_wizard(provider, model, max_frames, start_sec, end_sec,
               whisper_backend, whisper_model):
    """交互式向导：逐步询问各项配置（含按需输入密钥），返回更新后的值。"""
    print("\n=== 配置向导 ===（直接回车用默认值）")

    # 1. 关键帧上限
    max_frames = click.prompt(
        "1) 最多提取多少个关键帧？（0=不限）", default=max_frames, type=int
    )
    # 2. 处理时间区间（从第几秒到第几秒）
    start_sec = click.prompt(
        "2a) 从第几秒开始？（0=从头）", default=start_sec, type=float
    )
    end_sec = click.prompt(
        "2b) 到第几秒结束？（0=到结尾）", default=end_sec, type=float
    )

    # 3. 转录后端
    backends = ["auto", "faster", "openai"]
    whisper_backend = _pick(
        "3) 转录后端（国内离线选 openai）：", backends,
        backends.index(whisper_backend) + 1,
    )
    # 4. 转录模型
    wmodels = ["tiny", "base", "small", "medium", "large"]
    default_wm = wmodels.index(whisper_model) + 1 if whisper_model in wmodels else 2
    whisper_model = _pick(
        "4) 转录模型（越大越准越慢）：", wmodels, default_wm
    )

    # 5. 大模型来源
    providers = list(PROVIDER_LABELS.keys())
    provider = _pick(
        "5) 选择大模型来源：", [PROVIDER_LABELS[p] for p in providers],
        providers.index(provider) + 1,
    )
    provider = providers[[PROVIDER_LABELS[p] for p in providers].index(provider)]

    # 6. 具体模型
    model = _pick(
        f"6) 选择 {PROVIDER_LABELS[provider]} 的模型：",
        PROVIDER_MODELS[provider], 1,
    )

    # 7. 密钥：环境变量已有就跳过，否则当场输入（隐藏、仅本次有效）
    env_name = PROVIDER_ENV[provider]
    if os.environ.get(env_name):
        print(f"7) 密钥：已从环境变量 {env_name} 读到，无需重复输入")
    else:
        key = click.prompt(
            f"7) 未检测到 {env_name}，请输入 {PROVIDER_LABELS[provider]} 密钥",
            hide_input=True,
        )
        os.environ[env_name] = key.strip()
        print("   已设置（仅本次运行有效，未保存到任何文件）")

    print("=" * 32)
    return (provider, model, max_frames, start_sec, end_sec,
            whisper_backend, whisper_model)


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
              type=click.Choice(["anthropic", "openai", "minimax"]),
              help="大模型来源：anthropic=Claude，openai=ChatGPT，minimax=MiniMax")
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
@click.option("--start-sec", default=0.0, type=float,
              help="从第 N 秒开始处理（跳过之前内容）[默认: 0]")
@click.option("--end-sec", default=0.0, type=float,
              help="处理到第 N 秒为止 [默认: 0=到结尾]")
@click.option("--limit-sec", default=0.0, type=float,
              help="（兼容旧用法）处理时长 N 秒；等价于 --end-sec=start+N")
@click.option("--max-frames", default=0, type=int,
              help="最多提取 N 个关键帧后停止，快速测试用 [默认: 0=不限]")
@click.option("--whisper-prompt", default="",
              help="转录术语提示词，提升专有名词识别（如 'AI Skill,Agent,Prompt,Cursor'）")
@click.option("--no-images", is_flag=True, default=False,
              help="不在笔记中嵌入关键帧图片")
@click.option("--interactive", "-i", is_flag=True, default=False,
              help="交互式向导：逐步选择帧数/秒数/大模型来源与型号")
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
    start_sec,
    end_sec,
    limit_sec,
    max_frames,
    whisper_prompt,
    no_images,
    interactive,
):
    """视频网课分析工具：输入视频，输出结构化 Markdown 笔记。"""
    print("=== 视频网课分析工具 ===")

    if not os.path.exists(video_path):
        raise click.ClickException(f"视频文件不存在：{video_path}")

    # 交互式向导（覆盖帧数/时间区间/转录后端与模型/来源/模型，并按需输入密钥）
    if interactive:
        (provider, model, max_frames, start_sec, end_sec,
         whisper_backend, whisper_model) = run_wizard(
            provider, model, max_frames, start_sec, end_sec,
            whisper_backend, whisper_model,
        )

    # 解析模型默认值（按 provider）
    if not model:
        model = DEFAULT_MODELS[provider]

    # 处理区间：end_sec 优先；否则用旧的 limit_sec（时长）换算为绝对结束秒
    end_abs = end_sec if end_sec > 0 else (
        start_sec + limit_sec if limit_sec > 0 else 0.0
    )

    duration = get_video_duration(video_path)
    print(f"视频文件：{video_path}")
    m, s = divmod(int(duration), 60)
    print(f"视频时长：{m:02d}:{s:02d}")
    if start_sec > 0 or end_abs > 0:
        print(f"处理区间：{start_sec:.0f}s ~ "
              f"{(end_abs if end_abs > 0 else duration):.0f}s")

    # 步骤 1/4：提取关键帧
    t = _step("[步骤 1/4] 提取关键帧")
    frames = extract_key_frames(
        video_path,
        diff_threshold=threshold,
        max_interval_sec=float(interval),
        max_frames=max_frames,
        start_sec=start_sec,
        limit_sec=end_abs,
    )
    _done(t)

    # 步骤 2/4：转录音频
    t = _step("[步骤 2/4] 转录音频")
    segments = transcribe_video(
        video_path,
        model_size=whisper_model,
        language=language,
        start_sec=start_sec,
        limit_sec=end_abs,
        backend=whisper_backend,
        prompt=whisper_prompt or subject,
    )
    _done(t)

    # 步骤 3/4：对齐时间轴
    t = _step("[步骤 3/4] 对齐时间轴")
    aligned = align_frames_with_transcript(frames, segments)
    _done(t)

    # === 费用确认（第 4 步开始前，这是唯一花钱的一步）===
    est = estimate_cost(aligned, batch_size, model, image_max_side)
    print("\n" + "=" * 48)
    print("⚠️  下一步将调用大模型 API，会消耗你的额度（扣费）")
    print("=" * 48)
    print(f"  来源/模型     : {PROVIDER_LABELS.get(provider, provider)} / {model}")
    print(f"  图片设置      : 长边 {image_max_side}px / 质量 {image_quality}")
    print(f"  关键帧数      : {est['n_frames']} 帧")
    print(f"  调用次数      : {est['n_batches']} 次（每次 {batch_size} 帧）")
    print(f"  预计输入 token: ~{est['input_tokens']:,}")
    print(f"  预计输出 token: ~{est['out_tokens_typical']:,}（每次上限 4096）")
    if provider in USD_BILLED_PROVIDERS:
        print(f"  粗略费用      : ~${est['cost_typical']:.2f}"
              f"（最坏约 ${est['cost_max']:.2f}）")
        print(f"  注：费用按 {model} 标准定价估算，实际以官方账单为准")
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

    # 准备关键帧图片目录（与笔记同名时间戳，便于对应）
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    images_dir = images_rel = None
    if not no_images:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        images_rel = f"{stem}_images_{run_ts}"
        images_dir = os.path.join(output_dir, images_rel)

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
        base_url=_resolve_base_url(provider, base_url),
        images_dir=images_dir,
        images_rel=images_rel or "",
    )
    _done(t)

    # 保存笔记
    path = save_report(
        raw_notes,
        video_path,
        output_dir=output_dir,
        duration=duration,
        frame_count=len(frames),
        timestamp=run_ts,
    )
    print(f"\n✅ 完成！笔记已保存：{path}")
    if images_dir:
        print(f"   关键帧图片：{images_dir}/（共 {len(frames)} 张，已嵌入笔记）")


if __name__ == "__main__":
    main()
