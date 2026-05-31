"""智能关键帧提取。

通过画面差异检测 + 强制最小/最大间隔策略，在"不漏信息"和"不过度笨重"
之间取得平衡。
"""

import cv2
import numpy as np
from tqdm import tqdm

import cache


def _to_gray_small(frame: np.ndarray, resize_width: int) -> np.ndarray:
    """缩小到指定宽度（保持比例）并转灰度，用于差异计算。"""
    h, w = frame.shape[:2]
    if w > resize_width:
        scale = resize_width / w
        frame = cv2.resize(
            frame, (resize_width, max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def extract_key_frames(
    video_path: str,
    diff_threshold: float = 0.12,
    max_interval_sec: float = 30.0,
    min_interval_sec: float = 2.0,
    resize_width: int = 320,
    max_frames: int = 0,
    limit_sec: float = 0.0,
    start_sec: float = 0.0,
) -> list[tuple[float, np.ndarray]]:
    """提取关键帧。

    返回 [(timestamp_sec, frame_bgr), ...] 列表。帧保留原始分辨率，
    resize 只用于差异计算。

    max_frames > 0：收集到这么多关键帧后立即停止（快速测试用）。
    start_sec > 0：从这一秒开始处理（跳过之前的内容）。
    limit_sec > 0：处理到这一秒为止（视频内的绝对时间，非时长）。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if fps <= 0:
        cap.release()
        raise ValueError(f"无法读取视频帧率：{video_path}")

    # 从 start_sec 跳着开始读（大文件直接 seek，省时）
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    frames: list[tuple[float, np.ndarray]] = []
    prev_small: np.ndarray | None = None
    last_saved_ts: float = -1e9  # 保证第一帧一定被保存

    frame_idx = 0
    pbar = tqdm(total=total if total > 0 else None, unit="帧", desc="提取关键帧")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 用容器实际播放位置作时间戳（seek 之后才准确）
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = pos_msec / 1000.0 if pos_msec > 0 else frame_idx / fps

        # 处理到 limit_sec（视频内绝对时间）为止
        if limit_sec > 0 and timestamp > limit_sec:
            break

        pbar.update(1)

        small = _to_gray_small(frame, resize_width)
        save = False

        if prev_small is None:
            # 第一帧总是保存
            save = True
        elif timestamp - last_saved_ts < min_interval_sec:
            # 1. 强制最小间隔：跳过
            save = False
        else:
            # 2. 画面差异检测
            if small.shape == prev_small.shape:
                diff = float(np.mean(cv2.absdiff(small, prev_small))) / 255.0
            else:
                diff = 1.0  # 尺寸变化视为强差异
            if diff > diff_threshold:
                save = True
            # 3. 强制兜底
            elif timestamp - last_saved_ts >= max_interval_sec:
                save = True

        if save:
            frames.append((timestamp, frame))
            last_saved_ts = timestamp
            # 达到帧数上限即停止（快速测试）
            if max_frames > 0 and len(frames) >= max_frames:
                break

        prev_small = small
        frame_idx += 1

        # 每处理 500 帧释放一次内存
        if frame_idx % 500 == 0:
            import gc
            gc.collect()

    pbar.close()
    cap.release()

    ratio = (len(frames) / total) if total > 0 else 0.0
    print(f"提取关键帧 {len(frames)} 帧，共处理 {total} 帧，压缩比 {ratio:.1%}")
    return frames


def extract_key_frames_cached(
    video_path: str,
    diff_threshold: float = 0.12,
    max_interval_sec: float = 30.0,
    min_interval_sec: float = 2.0,
    resize_width: int = 320,
    max_frames: int = 0,
    limit_sec: float = 0.0,
    start_sec: float = 0.0,
    use_cache: bool = True,
) -> list[tuple[float, np.ndarray]]:
    """带缓存的关键帧提取。

    参数与 extract_key_frames 相同；use_cache=True 时，相同视频+相同参数
    第二次调用直接从 .cache/ 读取（关键帧以 JPEG 编码存盘，体积小），
    跳过逐帧解码，大幅加快重复测试。
    """
    params = {
        "diff_threshold": diff_threshold,
        "max_interval_sec": max_interval_sec,
        "min_interval_sec": min_interval_sec,
        "resize_width": resize_width,
        "max_frames": max_frames,
        "limit_sec": limit_sec,
        "start_sec": start_sec,
    }
    key = cache.make_key("frames", video_path, params)

    if use_cache:
        cached = cache.load(key)
        if cached is not None:
            frames = [
                (ts, cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR))
                for ts, buf in cached
            ]
            print(f"✓ 命中关键帧缓存，跳过提取（{len(frames)} 帧）")
            return frames

    frames = extract_key_frames(
        video_path,
        diff_threshold=diff_threshold,
        max_interval_sec=max_interval_sec,
        min_interval_sec=min_interval_sec,
        resize_width=resize_width,
        max_frames=max_frames,
        limit_sec=limit_sec,
        start_sec=start_sec,
    )

    if use_cache:
        encoded = []
        for ts, frame in frames:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                encoded.append((ts, buf.tobytes()))
        cache.save(key, encoded)
        print(f"  关键帧已缓存（下次相同参数将秒级复用）")

    return frames


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：python frame_extractor.py VIDEO_PATH")
        sys.exit(1)
    result = extract_key_frames(sys.argv[1])
    print(f"共 {len(result)} 个关键帧，时间戳示例：")
    for ts, _ in result[:10]:
        m, s = divmod(int(ts), 60)
        print(f"  [{m:02d}:{s:02d}]")
