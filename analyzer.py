"""Claude API 调用，分批分析对齐后的帧+文本。"""

import base64

import anthropic
import cv2
import numpy as np

DEFAULT_MODEL = "claude-opus-4-5"
MAX_TOKENS = 4096

# 各模型标准定价（美元 / 百万 token），已核对 Anthropic 官方价目（2026-05）。
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


def _build_content(
    batch: list[dict],
    subject_hint: str,
    image_max_side: int = 1024,
    image_quality: int = 85,
) -> list[dict]:
    content: list[dict] = []
    for item in batch:
        frame_resized = resize_keep_aspect(item["frame"], max_side=image_max_side)
        ok, buf = cv2.imencode(
            ".jpg", frame_resized, [cv2.IMWRITE_JPEG_QUALITY, image_quality]
        )
        if not ok:
            continue
        b64 = base64.b64encode(buf).decode()
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        })
        ts = item["timestamp"]
        minutes, seconds = divmod(int(ts), 60)
        content.append({
            "type": "text",
            "text": f"[{minutes:02d}:{seconds:02d}] {item['transcript']}",
        })

    subject_line = f"课程主题：{subject_hint}\n\n" if subject_hint else ""
    content.append({
        "type": "text",
        "text": (
            f"{subject_line}"
            "请分析以上视频帧画面和对应时间戳的音频文字，完成：\n"
            "1. 提取本段的核心知识点（用 ## 小节标题区分不同主题）\n"
            "2. 如果画面中有公式、代码、图表，请完整提取\n"
            "3. 标注重要概念的时间戳，格式：`[MM:SS]`\n"
            "4. 如有小结或总结，单独列出\n"
            "输出格式：Markdown，不要加多余的前言"
        ),
    })
    return content


def analyze_course(
    aligned_data: list[dict],
    subject_hint: str = "",
    batch_size: int = 8,
    model: str = DEFAULT_MODEL,
    image_max_side: int = 1024,
    image_quality: int = 85,
) -> str:
    """分批调用 Claude，返回完整 Markdown 笔记字符串。"""
    client = anthropic.Anthropic()

    batches = [
        aligned_data[i:i + batch_size]
        for i in range(0, len(aligned_data), batch_size)
    ]
    total_batches = len(batches)
    results: list[str] = []

    for i, batch in enumerate(batches):
        content = _build_content(
            batch, subject_hint, image_max_side, image_quality
        )
        try:
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": content}],
            )
            text = "".join(
                block.text for block in response.content
                if getattr(block, "type", None) == "text"
            )
            results.append(text.strip())
        except anthropic.APIError as e:
            print(f"⚠️  第 {i + 1}/{total_batches} 批分析失败，已跳过：{e}")

        print(f"已分析 {i + 1}/{total_batches} 批")

    return "\n\n---\n\n".join(results)


if __name__ == "__main__":
    print("analyzer 模块：请通过 main.py 调用，或导入 analyze_course() 测试。")
