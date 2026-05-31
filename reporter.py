"""输出 Markdown 笔记。"""

import os
from datetime import datetime


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def save_report(
    raw_notes: str,
    video_path: str,
    output_dir: str = ".",
    duration: float = 0.0,
    frame_count: int = 0,
    timestamp: str | None = None,
) -> str:
    """保存 Markdown 文件，返回文件路径。

    timestamp 用于文件名（与关键帧图片目录共用同一时间戳，便于对应）。
    """
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.basename(video_path)
    stem = os.path.splitext(video_name)[0]
    now = datetime.now()
    ts = timestamp or now.strftime("%Y%m%d_%H%M%S")
    filename = f"{stem}_notes_{ts}.md"
    path = os.path.join(output_dir, filename)

    header = (
        f"# 课程笔记：{video_name}\n\n"
        f"> 生成时间：{now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"> 视频时长：{_format_duration(duration)}\n"
        f"> 关键帧数：{frame_count}\n\n"
        "---\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + raw_notes + "\n")

    return path
