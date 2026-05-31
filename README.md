# 视频网课分析工具

输入一段网课视频，自动输出结构化 Markdown 笔记。

核心思路：通过**智能关键帧提取**（画面差异检测 + 强制最小/最大间隔），在
"不漏信息"和"不过度笨重"之间取得平衡；再结合 Whisper 语音转录与 Claude
多模态分析，生成带时间戳的知识点笔记。

---

## 💰 省钱速查表

费用 = 模型单价 × token，而 token 主要由**图片**决定。下表为 40 个关键帧的
粗略预估，按需组合 `--model` 和 `--image-max-side`：

| 档位 | 命令组合 | 40 帧预估 | 适合 |
| --- | --- | --- | --- |
| 🏆 最高质量 | `--model claude-opus-4-5 --image-max-side 1280` | ~$0.55 | 公式/板书密集 |
| ⚖️ 均衡（默认） | `--model claude-opus-4-5`（1024px） | ~$0.49 | 一般课程 |
| 💵 经济 | `--model claude-sonnet-4-5 --image-max-side 768` | ~$0.22 | 日常学习 |
| 🪙 极省 | `--model claude-haiku-4-5 --image-max-side 512` | ~$0.057 | 快速预览/纯讲解 |

> 越往下越省钱，但图片变小会让公式、代码、板书小字变模糊。第 4 步确认框会
> 显示当前设置与预估费用，可据此调整。详见下方
> [模型选择](#claude-模型选择影响质量与费用)与[压小图片](#进一步省-token压小图片)。

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

## 用自己的 MiniMax 模型（OpenAI 兼容接口）

如果你有 MiniMax 套餐，可以用 `--provider minimax` 替代 Claude。
本工具会把课程截图发给模型，**必须选支持图像理解的多模态模型**：

| MiniMax 模型 | 能否用 | 说明 |
| --- | --- | --- |
| `MiniMax-M2.5`（默认） | ✅ | 原生多模态，支持图像理解 |
| `MiniMax-M2.7` / `-highspeed` | ❌ | 纯文本模型，无法处理截图 |
| `MiniMax-M2` / `M2.1` 系列 | ❌ | 纯文本 |

### 配置环境变量

```bash
# Windows (PowerShell)
$env:MINIMAX_API_KEY="你的-minimax-key"

# macOS / Linux
export MINIMAX_API_KEY="你的-minimax-key"
# 可选：覆盖接口地址（默认国内站 https://api.minimaxi.com/v1）
export MINIMAX_BASE_URL="https://api.minimaxi.com/v1"
```

> 🔒 **安全**：API Key 只放环境变量，**切勿**写进代码或提交到 Git。一旦泄露请
> 立刻到 MiniMax 控制台重置。

需要先装 openai 库（已在 requirements.txt）：`pip install openai`

### 用法

```bash
# 用 MiniMax-M2.5（默认，国内站地址也是默认）
python main.py lecture01.mp4 --provider minimax

# 显式指定模型 / 接口地址
python main.py lecture01.mp4 --provider minimax \
  --model MiniMax-M2.5 --base-url https://api.minimaxi.com/v1
```

> MiniMax 按**套餐次数/token**计费，不是美元。第 4 步确认框会显示
> **「本次约 N 次模型调用」**（N = 关键帧数 ÷ batch-size），方便你对照套餐余额。
> 调大 `--batch-size` 可减少调用次数（但每次内容更长）。

### 先用 `--dry-run` 零成本验证

加 `--dry-run` 会跑完前 3 步（提取关键帧 / 转录 / 对齐）并显示预估，但
**不调用大模型、不扣任何额度**。用来确认 ffmpeg、Whisper、关键帧都正常：

```bash
python main.py 测试视频.mp4 --provider minimax --dry-run
```

确认无误后，去掉 `--dry-run` 即可真正分析。

### 视频太大？只测前一小段

大文件不必整段跑。用下面两个开关快速测试（可与 `--dry-run` 叠加）：

| 选项 | 作用 |
| --- | --- |
| `--limit-sec N` | 只处理**前 N 秒**（关键帧 + 转录都只跑前 N 秒，不读完大文件） |
| `--max-frames N` | 最多提取 **N 个关键帧**就停 |

```bash
# 只跑前 60 秒、最多 5 帧，且不调用模型 —— 秒级验证整条流程
python main.py 大视频.mp4 --provider minimax --limit-sec 60 --max-frames 5 --dry-run

# 确认 OK 后，去掉 --dry-run 真正分析前 60 秒（只扣 1 次左右调用）
python main.py 大视频.mp4 --provider minimax --limit-sec 60 --max-frames 5
```

---

## 进一步省 token：压小图片

图片是输入 token 的大头，可调小尺寸/质量大幅降费（按 `(宽×高)/750` 计费，
长边减半 → 图片 token 约降到 1/4）：

| 选项 | 默认 | 作用 |
| --- | --- | --- |
| `--image-max-side` | 1024 | 图片长边像素，越小越省（建议不低于 512，否则公式/小字可能看不清） |
| `--image-quality` | 85 | JPEG 质量 1~100，越低越省（建议不低于 60） |

```bash
# 省钱档：小图 + Haiku（适合快速预览/纯文字课程）
python main.py lecture01.mp4 --model claude-haiku-4-5 --image-max-side 640 --image-quality 70

# 高清档：保留细节（适合公式密集/板书课程）
python main.py lecture01.mp4 --image-max-side 1280
```

> ⚠️ 图片太小会让 Claude 看不清公式、代码、板书。省钱和可读性要权衡，
> 第 4 步的确认框会显示当前图片设置和预估费用，可据此调整。

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
