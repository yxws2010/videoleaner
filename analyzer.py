"""大模型分析：分批把"帧图 + 时间戳文字"发给模型，生成 Markdown 笔记。

支持两种 provider：
- anthropic：调用 Claude（原生 Messages API，按 token 美元计费）
- minimax：通过 OpenAI 兼容接口调用 MiniMax（按你的套餐次数/token 计费）
"""

import base64
import os
import re

import anthropic
import cv2
import numpy as np

# 去除模型思考过程标签（<think>...</think> / <thinking>...</thinking>）
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """剥掉模型输出里的思考过程标签，返回干净正文。"""
    text = _THINK_RE.sub("", text)
    # 清理可能残留的未闭合开标签
    text = re.sub(r"<think(?:ing)?>", "", text, flags=re.IGNORECASE)
    return text.strip()


def save_batch_images(
    batch: list[dict], images_dir: str, images_rel: str, start_index: int
) -> tuple[str, int]:
    """把一批关键帧存成图片，返回 (markdown 图片块, 下一个序号)。"""
    os.makedirs(images_dir, exist_ok=True)
    lines: list[str] = []
    idx = start_index
    for item in batch:
        minutes, seconds = divmod(int(item["timestamp"]), 60)
        mmss = f"{minutes:02d}{seconds:02d}"
        fname = f"frame_{idx:03d}_{mmss}.jpg"
        thumb = resize_keep_aspect(item["frame"], max_side=800)
        cv2.imwrite(os.path.join(images_dir, fname), thumb)
        rel = f"{images_rel}/{fname}" if images_rel else fname
        lines.append(f"**[{minutes:02d}:{seconds:02d}]**\n\n![{minutes:02d}:{seconds:02d}]({rel})")
        idx += 1
    return ("\n\n".join(lines), idx)

DEFAULT_MODEL = "claude-opus-4-5"
MAX_TOKENS = 4096

# MiniMax（OpenAI 兼容接口）默认配置，可被 CLI / 环境变量覆盖
# 注意：必须用支持图像理解的多模态模型。MiniMax-M2.5 原生多模态可用；
# M2.7 / M2.7-highspeed 等为纯文本模型，无法处理截图。
DEFAULT_MINIMAX_MODEL = "MiniMax-M2.5"
DEFAULT_MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"  # 国内站

# 各模型标准定价（美元 / 百万 token）。已核对官方价目（2026-05）。
# Claude: Anthropic；GPT: OpenAI。不含缓存/批处理折扣。
MODEL_PRICING = {
    "claude-opus-4-5": {"in": 5.0, "out": 25.0},
    "claude-sonnet-4-5": {"in": 3.0, "out": 15.0},
    "claude-haiku-4-5": {"in": 1.0, "out": 5.0},
    "gpt-4o": {"in": 2.5, "out": 10.0},
    "gpt-4o-mini": {"in": 0.15, "out": 0.6},
}

# 各 provider 可选模型（均需支持图像理解/视觉）。
PROVIDER_MODELS = {
    "anthropic": ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"],
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "minimax": ["MiniMax-M2.5"],
}
DEFAULT_MODELS = {
    "anthropic": DEFAULT_MODEL,
    "openai": "gpt-4o",
    "minimax": DEFAULT_MINIMAX_MODEL,
}
# 按 token 美元计费的 provider（minimax 按套餐次数计费，单列）。
USD_BILLED_PROVIDERS = {"anthropic", "openai"}


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
        "重要：音频文字是语音识别结果，专业术语可能有错（例如把 'Skill' 识别成"
        "'四个有/技能败'、把 'Prompt' 识别成 '提是此'）。请优先结合画面内容"
        "纠正这些术语，以画面为准。\n"
        "输出格式：直接输出 Markdown 正文，不要写思考过程，不要加多余前言。"
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
    images_dir: str | None = None,
    images_rel: str = "",
) -> str:
    """分批调用大模型，返回完整 Markdown 笔记字符串。

    images_dir 非空时，把每批关键帧存图并嵌入对应小节上方。
    """
    batches = [
        aligned_data[i:i + batch_size]
        for i in range(0, len(aligned_data), batch_size)
    ]
    total_batches = len(batches)
    results: list[str] = []
    img_index = 0  # 关键帧图片全局序号

    if provider == "anthropic":
        client = anthropic.Anthropic()
    elif provider in ("minimax", "openai"):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                f"使用 {provider} 需要安装 openai 库：pip install openai"
            ) from e
        if provider == "minimax":
            key = api_key or os.environ.get("MINIMAX_API_KEY")
            base = base_url or DEFAULT_MINIMAX_BASE_URL
            env_name = "MINIMAX_API_KEY"
        else:  # openai / ChatGPT
            key = api_key or os.environ.get("OPENAI_API_KEY")
            base = base_url or None  # None = 用 OpenAI 官方默认地址
            env_name = "OPENAI_API_KEY"
        if not key:
            raise RuntimeError(
                f"未找到 {provider} 密钥，请设置环境变量 {env_name}"
            )
        client = OpenAI(base_url=base, api_key=key)
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
            text = strip_think(text)  # 剥掉模型思考过程
        except Exception as e:  # 单批失败不中断整体流程
            print(f"⚠️  第 {i + 1}/{total_batches} 批分析失败，已跳过：{e}")
            text = ""

        # 关键帧配图（放在该批笔记上方）
        if images_dir:
            img_md, img_index = save_batch_images(
                batch, images_dir, images_rel, img_index
            )
            section = f"{img_md}\n\n{text}" if text else img_md
        else:
            section = text
        if section.strip():
            results.append(section.strip())

        print(f"已分析 {i + 1}/{total_batches} 批")

    return "\n\n---\n\n".join(results)


if __name__ == "__main__":
    print("analyzer 模块：请通过 main.py 调用，或导入 analyze_course() 测试。")
