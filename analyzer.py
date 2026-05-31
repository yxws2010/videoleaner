"""大模型分析：分批把"帧图 + 时间戳文字"发给模型，生成 Markdown 笔记。

支持两种 provider：
- anthropic：调用 Claude（原生 Messages API，按 token 美元计费）
- minimax：通过 OpenAI 兼容接口调用 MiniMax（按你的套餐次数/token 计费）
"""

import base64
import os

import anthropic
import cv2
import numpy as np

DEFAULT_MODEL = "claude-opus-4-5"
MAX_TOKENS = 4096

# MiniMax（OpenAI 兼容接口）默认配置，可被 CLI / 环境变量覆盖
DEFAULT_MINIMAX_MODEL = "MiniMax-VL-01"
DEFAULT_MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"

# Claude 各模型标准定价（美元 / 百万 token），已核对 Anthropic 官方价目（2026-05）。
# 不含缓存/批处理折扣。新增模型或官方调价后在此同步。
MODEL_PRICING = {
    "claude-opus-4-5": {"in": 5.0, "out": 25.0},
    "claude-sonnet-4-5": {"in": 3.0, "out": 15.0},
    "claude-haiku-4-5": {"in": 1.0, "out": 5.0},
}


def resize_keep_aspect(frame: np.ndarray, max_side: int) -> np.ndarray:
    """将帧缩放至长边不超过 max_side，保持宽高比。"""
    h, w = frame.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale == 1.0:
        return frame
    return cv2.resize(
        frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def _encode_jpeg_b64(
    frame: np.ndarray, image_max_side: int, image_quality: int
) -> str | None:
    """把帧压缩为 JPEG 并返回 base64 字符串；失败返回 None。"""
    frame_resized = resize_keep_aspect(frame, max_side=image_max_side)
    ok, buf = cv2.imencode(
        ".jpg", frame_resized, [cv2.IMWRITE_JPEG_QUALITY, image_quality]
    )
    if not ok:
        return None
    return base64.b64encode(buf).decode()


def _prompt_text(subject_hint: str) -> str:
    subject_line = f"课程主题：{subject_hint}\n\n" if subject_hint else ""
    return (
        f"{subject_line}"
        "请分析以上视频帧画面和对应时间戳的音频文字，完成：\n"
        "1. 提取本段的核心知识点（用 ## 小节标题区分不同主题）\n"
        "2. 如果画面中有公式、代码、图表，请完整提取\n"
        "3. 标注重要概念的时间戳，格式：`[MM:SS]`\n"
        "4. 如有小结或总结，单独列出\n"
        "输出格式：Markdown，不要加多余的前言"
    )


def _ts_label(item: dict) -> str:
    minutes, seconds = divmod(int(item["timestamp"]), 60)
    return f"[{minutes:02d}:{seconds:02d}] {item['transcript']}"


def _build_anthropic_content(
    batch: list[dict], subject_hint: str, image_max_side: int, image_quality: int
) -> list[dict]:
    content: list[dict] = []
    for item in batch:
        b64 = _encode_jpeg_b64(item["frame"], image_max_side, image_quality)
        if b64 is None:
            continue
        content.append({
            "type": "image",
            "source": {
                "type": "base64", "media_type": "image/jpeg", "data": b64,
            },
        })
        content.append({"type": "text", "text": _ts_label(item)})
    content.append({"type": "text", "text": _prompt_text(subject_hint)})
    return content


def _build_openai_content(
    batch: list[dict], subject_hint: str, image_max_side: int, image_quality: int
) -> list[dict]:
    """OpenAI 兼容格式（MiniMax 用）：图片走 image_url 的 base64 data URL。"""
    content: list[dict] = []
    for item in batch:
        b64 = _encode_jpeg_b64(item["frame"], image_max_side, image_quality)
        if b64 is None:
            continue
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
        content.append({"type": "text", "text": _ts_label(item)})
    content.append({"type": "text", "text": _prompt_text(subject_hint)})
    return content


def analyze_course(
    aligned_data: list[dict],
    subject_hint: str = "",
    batch_size: int = 8,
    model: str = DEFAULT_MODEL,
    image_max_side: int = 1024,
    image_quality: int = 85,
    provider: str = "anthropic",
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """分批调用大模型，返回完整 Markdown 笔记字符串。"""
    batches = [
        aligned_data[i:i + batch_size]
        for i in range(0, len(aligned_data), batch_size)
    ]
    total_batches = len(batches)
    results: list[str] = []

    if provider == "anthropic":
        client = anthropic.Anthropic()
    elif provider == "minimax":
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "使用 MiniMax 需要安装 openai 库：pip install openai"
            ) from e
        key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not key:
            raise RuntimeError(
                "未找到 MiniMax 密钥，请设置环境变量 MINIMAX_API_KEY"
            )
        client = OpenAI(
            base_url=base_url or DEFAULT_MINIMAX_BASE_URL, api_key=key
        )
    else:
        raise ValueError(f"未知 provider：{provider}")

    for i, batch in enumerate(batches):
        try:
            if provider == "anthropic":
                content = _build_anthropic_content(
                    batch, subject_hint, image_max_side, image_quality
                )
                response = client.messages.create(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": content}],
                )
                text = "".join(
                    block.text for block in response.content
                    if getattr(block, "type", None) == "text"
                )
            else:  # minimax / OpenAI 兼容
                content = _build_openai_content(
                    batch, subject_hint, image_max_side, image_quality
                )
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": content}],
                )
                text = response.choices[0].message.content or ""
            results.append(text.strip())
        except Exception as e:  # 单批失败不中断整体流程
            print(f"⚠️  第 {i + 1}/{total_batches} 批分析失败，已跳过：{e}")

        print(f"已分析 {i + 1}/{total_batches} 批")

    return "\n\n---\n\n".join(results)


if __name__ == "__main__":
    print("analyzer 模块：请通过 main.py 调用，或导入 analyze_course() 测试。")
