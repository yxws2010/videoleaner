"""本地缓存：把耗时的关键帧提取 / 音频转录结果存盘，重复测试时直接复用。

缓存键 = 视频文件指纹（路径+大小+修改时间）+ 处理参数 的哈希。
任一参数或视频变了，键就变，自动重新计算，不会用到过期结果。
"""

import hashlib
import os
import pickle

# 缓存目录（放在项目下的 .cache/，已在 .gitignore 忽略）
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def _file_sig(path: str) -> str:
    """视频文件指纹：绝对路径 + 字节数 + 修改时间。"""
    st = os.stat(path)
    return f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"


def make_key(kind: str, video_path: str, params: dict) -> str:
    """根据视频指纹与处理参数生成缓存键。"""
    raw = _file_sig(video_path) + "|" + repr(sorted(params.items()))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{kind}_{digest}"


def _path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, key + ".pkl")


def load(key: str):
    """读缓存；不存在或损坏时返回 None。"""
    p = _path(key)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None  # 缓存损坏就当未命中，重新计算


def save(key: str, obj) -> None:
    """写缓存（失败不影响主流程）。"""
    try:
        with open(_path(key), "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        print(f"  ⚠️ 缓存写入失败（忽略）：{e}")


def clear() -> int:
    """清空所有缓存，返回删除的文件数。"""
    if not os.path.isdir(CACHE_DIR):
        return 0
    n = 0
    for name in os.listdir(CACHE_DIR):
        if name.endswith(".pkl"):
            try:
                os.remove(os.path.join(CACHE_DIR, name))
                n += 1
            except OSError:
                pass
    return n
