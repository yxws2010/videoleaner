"""视频网课分析工具 — CLI 入口。"""

import os
import time

import click

from frame_extractor import extract_key_frames
from transcriber import transcribe_video
from aligner import align_frames_with_transcript
from analyzer import analyze_course
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


@click.command()
@click.argument("video_path", type=click.Path())
@click.option("--output-dir", default=".", help="笔记输出目录 [默认: 当前目录]")
@click.option("--subject", default="", help="课程主题提示，帮助 Claude 理解上下文")
@click.option("--threshold", default=0.12, type=float, help="关键帧差异阈值 0~1 [默认: 0.12]")
@click.option("--interval", default=30, type=int, help="强制兜底间隔（秒）[默认: 30]")
@click.option("--batch-size", default=8, type=int, help="每次发给 Claude 的帧数 [默认: 8]")
@click.option("--whisper-model", default="base", help="Whisper 模型大小 [默认: base]")
@click.option("--language", default="zh", help="音频语言，zh/en/None [默认: zh]")
def main(
    video_path,
    output_dir,
    subject,
    threshold,
    interval,
    batch_size,
    whisper_model,
    language,
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

    # 步骤 4/4：Claude 分析
    t = _step("[步骤 4/4] Claude 分析")
    raw_notes = analyze_course(
        aligned,
        subject_hint=subject,
        batch_size=batch_size,
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
