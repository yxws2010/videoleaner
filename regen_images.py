"""补出笔记里缺失的关键帧图片（不重新调用大模型）。

两种来源，自动按优先级选择：

1) 【推荐·免费】解析已生成的 .md，按其中引用的时间戳直接从视频里截帧。
   适合：笔记已生成、已付费，但图片没存进文件夹的情况。
       python regen_images.py --md 你的视频_notes_xxx.md --video 你的视频.mp4

2) 从关键帧缓存(.cache/)还原（需先跑过带缓存的流程）。
       python regen_images.py 目标文件夹
       python regen_images.py 目标文件夹 --cache frames_xxxx.pkl
"""

import os
import pickle
import re
import sys

import cv2
import numpy as np

import cache
from analyzer import resize_keep_aspect

# Windows 控制台默认 GBK，重配为 UTF-8 避免打印中文/符号报错
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 匹配 markdown 图片引用里的关键帧：![..](任意路径/frame_003_0804.jpg)
_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]*frame_\d+_(\d{2})(\d{2})\.jpg)\)")


def _arg(flag: str):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


def regen_from_markdown(md_path: str, video_path: str) -> None:
    """解析 md 的图片引用，从视频按时间戳截帧写入对应路径。"""
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    refs = _IMG_RE.findall(text)  # [(rel_path, mm, ss), ...]
    if not refs:
        print("❌ 笔记里没找到关键帧图片引用（frame_xxx_MMSS.jpg）。")
        sys.exit(1)

    base = os.path.dirname(os.path.abspath(md_path))  # 图片相对 md 所在目录
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频：{video_path}")
        sys.exit(1)

    n = 0
    for rel, mm, ss in refs:
        ts = int(mm) * 60 + int(ss)
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"  ⚠️ 跳过 {rel}（{mm}:{ss} 处取帧失败）")
            continue
        out_path = os.path.join(base, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        thumb = resize_keep_aspect(frame, max_side=800)
        good, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if good:
            with open(out_path, "wb") as fp:
                fp.write(buf.tobytes())
            n += 1
    cap.release()
    print(f"✅ 已从视频按笔记时间戳补出 {n}/{len(refs)} 张图片。")


def _latest_frame_cache():
    if not os.path.isdir(cache.CACHE_DIR):
        return None
    files = [
        os.path.join(cache.CACHE_DIR, x)
        for x in os.listdir(cache.CACHE_DIR)
        if x.startswith("frames_") and x.endswith(".pkl")
    ]
    return max(files, key=os.path.getmtime) if files else None


def regen_from_cache(out_dir: str, cache_file: str | None) -> None:
    if cache_file and not os.path.isabs(cache_file):
        cache_file = os.path.join(cache.CACHE_DIR, cache_file)
    cache_file = cache_file or _latest_frame_cache()
    if not cache_file or not os.path.exists(cache_file):
        print("❌ 找不到关键帧缓存。改用 --md + --video 从笔记+视频补图。")
        sys.exit(1)

    with open(cache_file, "rb") as f:
        encoded = pickle.load(f)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for idx, (ts, buf) in enumerate(encoded):
        frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        mm, ssec = divmod(int(ts), 60)
        fname = f"frame_{idx:03d}_{mm:02d}{ssec:02d}.jpg"
        thumb = resize_keep_aspect(frame, max_side=800)
        good, out = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if good:
            with open(os.path.join(out_dir, fname), "wb") as fp:
                fp.write(out.tobytes())
            n += 1
    print(f"✅ 已从缓存 {os.path.basename(cache_file)} 补出 {n} 张图片到：{out_dir}")


def main():
    md = _arg("--md")
    video = _arg("--video")
    if md and video:
        regen_from_markdown(md, video)
        return

    # 否则走缓存模式：第一个位置参数为目标文件夹
    positional = [a for a in sys.argv[1:] if not a.startswith("--")
                  and a not in (md or "", video or "", _arg("--cache") or "")]
    if not positional:
        print(__doc__)
        sys.exit(1)
    regen_from_cache(positional[0], _arg("--cache"))


if __name__ == "__main__":
    main()
