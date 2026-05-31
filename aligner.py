"""帧与文本时间轴对齐。"""

import numpy as np


def align_frames_with_transcript(
    frames: list[tuple[float, np.ndarray]],
    segments: list[dict],
    context_window_sec: float = 5.0,
) -> list[dict]:
    """将每个关键帧与其附近时间窗口内的文本对齐。

    返回 [{"timestamp": float, "frame": np.ndarray, "transcript": str}, ...]
    """
    aligned: list[dict] = []

    for timestamp, frame in frames:
        lo = timestamp - context_window_sec
        hi = timestamp + context_window_sec

        # 找出与 [lo, hi] 有重叠的所有片段
        texts = [
            seg["text"]
            for seg in segments
            if seg["end"] >= lo and seg["start"] <= hi and seg["text"]
        ]
        transcript = " ".join(texts).strip()

        aligned.append({
            "timestamp": timestamp,
            "frame": frame,
            "transcript": transcript,
        })

    return aligned


if __name__ == "__main__":
    # 简单自测
    fake_frames = [(10.0, np.zeros((2, 2), dtype=np.uint8))]
    fake_segs = [
        {"start": 6.0, "end": 8.0, "text": "你好"},
        {"start": 9.0, "end": 12.0, "text": "世界"},
        {"start": 30.0, "end": 31.0, "text": "无关"},
    ]
    for item in align_frames_with_transcript(fake_frames, fake_segs)[:5]:
        print(f"[{item['timestamp']:.1f}] -> {item['transcript']!r}")
