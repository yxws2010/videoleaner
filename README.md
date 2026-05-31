# 视频网课分析工具

输入一段网课视频，自动输出结构化 Markdown 笔记。

核心思路：通过**智能关键帧提取**（画面差异检测 + 强制最小/最大间隔），在
"不漏信息"和"不过度笨重"之间取得平衡；再结合 Whisper 语音转录与 Claude
多模态分析，生成带时间戳的知识点笔记。

---

## 安装

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 ffmpeg（用于音频提取）

- **macOS**：`brew install ffmpeg`
- **Linux**：`apt install ffmpeg`
- **Windows**：`winget install Gyan.FFmpeg`（或从 https://ffmpeg.org/download.html 下载并加入 PATH）

### 3. 设置环境变量

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

---

## 快速开始

```bash
# 1. 最简用法（中文课程，默认参数）
python main.py lecture01.mp4

# 2. 指定课程主题与输出目录，提升分析质量
python main.py lecture01.mp4 --subject "机器学习基础" --output-dir ./notes

# 3. PPT 课程 + 更高质量的转录模型
python main.py slides.mp4 --threshold 0.08 --whisper-model small
```

---

## 关键帧阈值调整建议

不同类型课程画面变化幅度不同，可通过 `--threshold` 调整灵敏度（值越小越敏感）：

| 课程类型 | 特征 | 推荐阈值 |
| --- | --- | --- |
| PPT 课程 | 切换明显 | `--threshold 0.08` |
| 板书课程 | 渐进变化 | `--threshold 0.06` |
| 实验演示 | 背景复杂 | `--threshold 0.18` |

---

## Whisper 模型选择

| 模型 | 速度 | 准确率 | 适用场景 |
| --- | --- | --- | --- |
| `tiny` | 最快 | 低 | 快速测试 |
| `base` | 均衡 | 中 | **推荐首选** |
| `small` / `medium` | 较慢 | 高 | 重要课程 |

语言设置：中文课程 `--language zh`；英文课程 `--language en`；不确定用
`--language None`（自动检测）。

---

## Claude 模型选择（影响质量与费用）

通过 `--model` 切换,越贵质量越好。第 4 步开始前会按所选模型预估费用并等你确认:

| 模型 | 输入 / 输出价（每百万 token） | 适用场景 |
| --- | --- | --- |
| `claude-opus-4-5`（默认） | $5 / $25 | 质量最佳,重要课程 |
| `claude-sonnet-4-5` | $3 / $15 | 均衡,日常推荐 |
| `claude-haiku-4-5` | $1 / $5 | 最省钱,快速预览 |

```bash
# 用更便宜的 Sonnet
python main.py lecture01.mp4 --model claude-sonnet-4-5

# 自动化脚本：跳过费用确认
python main.py lecture01.mp4 --model claude-haiku-4-5 -y
```

> 价格已核对 Anthropic 官方价目（2026-05），不含缓存/批处理折扣，实际以官方账单为准。

---

## 费用估算

每个关键帧会以 JPEG（长边 ≤1024px）形式发送给 Claude，约消耗 1000~1600 token；
加上转录文本与输出，**1 小时课程**通常产生 40~120 个关键帧，粗略估算约消耗
**十几万到二十几万输入 token + 数万输出 token**。实际费用取决于课程画面变化频率
和所选 Claude 模型定价，请以官方价目为准。可通过调大 `--threshold` 或
`--interval` 减少帧数以降低费用。

---

## 已知限制

- 不支持实时流（仅支持本地视频文件：mp4 / mkv / avi / mov）
- 不支持直接提取字幕轨道（始终走语音转录）
- 上述功能均为未来可扩展方向

---

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `main.py` | CLI 参数解析 + 流程编排 |
| `frame_extractor.py` | 智能关键帧提取 |
| `transcriber.py` | 音频提取 + Whisper 转录 |
| `aligner.py` | 帧与文本时间轴对齐 |
| `analyzer.py` | Claude API 调用 |
| `reporter.py` | 输出 Markdown 笔记 |
