# 视频网课分析工具 — 完整实现规格

## 目标

构建一个命令行工具，输入一段网课视频，输出结构化 Markdown 笔记。
核心挑战：在"不漏信息"和"不过度笨重"之间取得平衡，通过**智能关键帧提取**解决。

---

## 项目结构

```
video-course-summarizer/
├── main.py                # 入口，CLI 参数解析 + 流程编排
├── frame_extractor.py     # 智能关键帧提取
├── transcriber.py         # 音频提取 + Whisper 转录
├── aligner.py             # 帧与文本时间轴对齐
├── analyzer.py            # Claude API 调用
├── reporter.py            # 输出 Markdown 笔记
├── requirements.txt
└── README.md
```

---

## 依赖

```
# requirements.txt
opencv-python>=4.8.0
faster-whisper>=1.0.0
anthropic>=0.40.0
numpy>=1.24.0
Pillow>=10.0.0
tqdm>=4.66.0
click>=8.1.0
ffmpeg-python>=0.2.0
```

系统依赖：需要安装 ffmpeg（用于音频提取）。在 README 里注明：`brew install ffmpeg` 或 `apt install ffmpeg`。

---

## 模块详细规格

### 1. `frame_extractor.py` — 智能关键帧提取

**函数签名**：
```python
def extract_key_frames(
    video_path: str,
    diff_threshold: float = 0.12,
    max_interval_sec: float = 30.0,
    min_interval_sec: float = 2.0,
    resize_width: int = 320,
) -> list[tuple[float, np.ndarray]]:
    """
    返回 [(timestamp_sec, frame_bgr), ...] 列表
    """
```

**关键帧判定逻辑**（按优先级）：

1. **强制最小间隔**：两帧时间差 < `min_interval_sec` 时，跳过（防止短时间内重复截帧）。
2. **画面差异检测**：
   - 将当前帧和上一帧都缩小到 `resize_width` 宽度（保持比例），转灰度
   - 计算像素绝对差的均值，除以 255 归一化为 [0,1]
   - 差异 > `diff_threshold` → 判定为关键帧
3. **强制兜底**：距上次保存超过 `max_interval_sec` 秒，无论差异大小都保存一帧

**实现注意**：
- 用 `tqdm` 显示进度条，单位为帧数
- 每处理 500 帧释放一次内存（`cap.grab()` 跳帧比 `cap.read()` 快，在不需要解码的帧用 grab）
- 返回的帧保留**原始分辨率**（resize 只用于差异计算，不影响发给 Claude 的图片质量）
- 在函数末尾打印统计信息：`f"提取关键帧 {len(frames)} 帧，共处理 {total} 帧，压缩比 {ratio:.1%}"`

---

### 2. `transcriber.py` — 音频转录

**函数签名**：
```python
def transcribe_video(
    video_path: str,
    model_size: str = "base",
    language: str = "zh",
) -> list[dict]:
    """
    返回 [{"start": float, "end": float, "text": str}, ...]
    """
```

**实现步骤**：

1. 用 `ffmpeg-python` 从视频中提取音频到临时 `.wav` 文件（16kHz 单声道）：
   ```python
   import ffmpeg, tempfile, os
   tmp = tempfile.mktemp(suffix=".wav")
   ffmpeg.input(video_path).output(tmp, ar=16000, ac=1, loglevel="quiet").run()
   ```
2. 用 `faster-whisper` 转录，开启 `word_timestamps=False`，获取 segment 级别时间戳
3. 返回前清理临时文件
4. 转录完成后打印：`f"转录完成，共 {len(segments)} 个片段"`

**模型选择建议**（写进 README）：
- `tiny`：最快，准确率低，适合测试
- `base`：均衡，推荐首选
- `small`/`medium`：质量更好，适合重要课程
- 中文课程设 `language="zh"`；英文课程设 `language="en"`；不确定用 `language=None`（自动检测）

---

### 3. `aligner.py` — 时间轴对齐

**函数签名**：
```python
def align_frames_with_transcript(
    frames: list[tuple[float, np.ndarray]],
    segments: list[dict],
    context_window_sec: float = 5.0,
) -> list[dict]:
    """
    返回 [{"timestamp": float, "frame": np.ndarray, "transcript": str}, ...]
    """
```

**对齐逻辑**：
- 对每个关键帧，找出时间范围 `[timestamp - context_window_sec, timestamp + context_window_sec]` 内的所有文本片段
- 将这些片段的文本拼接为一个字符串，作为该帧的 `transcript`
- 如果该帧附近没有任何文本（纯沉默段），`transcript` 设为空字符串

---

### 4. `analyzer.py` — Claude 分析

**函数签名**：
```python
def analyze_course(
    aligned_data: list[dict],
    subject_hint: str = "",
    batch_size: int = 8,
) -> str:
    """
    分批调用 Claude，返回完整 Markdown 笔记字符串
    """
```

**实现细节**：

批次划分：每 `batch_size` 个帧为一批，依次发给 Claude。

每批请求的 content 构建方式：
```python
content = []
for item in batch:
    # 将帧压缩为 JPEG，限制长边 1024px（减少 token 用量）
    frame_resized = resize_keep_aspect(item["frame"], max_side=1024)
    _, buf = cv2.imencode(".jpg", frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()
    content.append({
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}
    })
    ts = item["timestamp"]
    minutes, seconds = divmod(int(ts), 60)
    content.append({
        "type": "text",
        "text": f"[{minutes:02d}:{seconds:02d}] {item['transcript']}"
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
    )
})
```

调用参数：
```python
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=4096,
    messages=[{"role": "user", "content": content}]
)
```

**批次间的拼接**：在每批结果之间插入分隔行 `\n\n---\n\n`，最终合并为一个完整字符串。

**错误处理**：
- 用 `try/except anthropic.APIError` 捕获 API 错误，失败的批次打印警告并跳过，不中断整体流程
- 每批调用后打印进度：`f"已分析 {i+1}/{total_batches} 批"`

---

### 5. `reporter.py` — 输出笔记

**函数签名**：
```python
def save_report(
    raw_notes: str,
    video_path: str,
    output_dir: str = ".",
) -> str:
    """
    保存 Markdown 文件，返回文件路径
    """
```

**输出文件名规则**：
- 基于视频文件名生成：`{视频名}_notes_{YYYYMMDD_HHMMSS}.md`
- 例如：`lecture01_notes_20250601_143022.md`

**文件头部自动附加**：
```markdown
# 课程笔记：{视频文件名}

> 生成时间：{datetime}
> 视频时长：{duration}
> 关键帧数：{frame_count}

---

{raw_notes}
```

---

### 6. `main.py` — CLI 入口

用 `click` 实现命令行接口：

```
用法：python main.py [OPTIONS] VIDEO_PATH

参数：
  VIDEO_PATH           视频文件路径（支持 mp4/mkv/avi/mov）

选项：
  --output-dir TEXT    笔记输出目录 [默认: 当前目录]
  --subject TEXT       课程主题提示，帮助 Claude 理解上下文（如"机器学习基础"）
  --threshold FLOAT    关键帧差异阈值 0~1 [默认: 0.12]
  --interval INT       强制兜底间隔（秒）[默认: 30]
  --batch-size INT     每次发给 Claude 的帧数 [默认: 8]
  --whisper-model TEXT Whisper 模型大小 [默认: base]
  --language TEXT      音频语言，zh/en/None [默认: zh]
  --help               显示帮助
```

**运行流程**：
```
1. 打印 "=== 视频网课分析工具 ===" 和视频信息
2. [步骤 1/4] 提取关键帧 → frame_extractor.py
3. [步骤 2/4] 转录音频   → transcriber.py
4. [步骤 3/4] 对齐时间轴  → aligner.py
5. [步骤 4/4] Claude 分析 → analyzer.py（显示分批进度）
6. 保存笔记              → reporter.py
7. 打印完成信息和文件路径
```

每步前打印当前步骤名和时间，完成后打印耗时。

---

## README.md 要求

包含以下内容：
1. **安装步骤**：`pip install -r requirements.txt` + ffmpeg 安装命令（macOS/Linux/Windows 分别列出）
2. **环境变量**：需要设置 `ANTHROPIC_API_KEY`
3. **快速开始**：3 个典型用法示例
4. **关键帧阈值调整建议**：
   - PPT 课程（切换明显）：`--threshold 0.08`
   - 板书课程（渐进变化）：`--threshold 0.06`
   - 实验演示（背景复杂）：`--threshold 0.18`
5. **费用估算**：1 小时课程约消耗多少 Claude API token（粗略估算）
6. **已知限制**：不支持实时流，不支持字幕轨道直接提取（未来可扩展）

---

## 辅助函数（可放在各模块内）

```python
def resize_keep_aspect(frame: np.ndarray, max_side: int) -> np.ndarray:
    """将帧缩放至长边不超过 max_side，保持宽高比"""
    h, w = frame.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def get_video_duration(video_path: str) -> float:
    """返回视频时长（秒）"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps if fps > 0 else 0
```

---

## 测试要求

实现完成后，用一段 **3~5 分钟的测试视频**验证：
1. 关键帧数量合理（不应超过 50 帧/每5分钟）
2. 转录结果有时间戳且文字基本正确
3. 最终 Markdown 文件可正常打开，包含知识点和时间戳
4. 对不存在的视频路径给出友好错误提示

---

## 实现顺序建议

1. `frame_extractor.py`（可独立测试，运行后展示截帧数量）
2. `transcriber.py`（可独立测试，打印前10个片段）
3. `aligner.py`（打印前5个对齐结果确认逻辑正确）
4. `analyzer.py`（先用1个帧测试 API 调用是否成功）
5. `reporter.py` + `main.py`（串联完整流程）
6. `README.md`
