# MiniGPT Lab

**从零实现、训练并观察字符级自回归 Transformer。** 适用于 PyTorch 入门、注意力机制教学和可复现的小规模对照实验。

[模型实现](model.py) · [实验方法与结果](docs/EXPERIMENTS.md) · [验证记录](docs/VALIDATION.md) · [原始指标](results/)

本项目使用 Python、PyTorch 和 Gradio，约 **11.4 万参数**，默认配置适合 CPU；环境支持时自动选择 CUDA 或 Apple MPS。它是字符语言模型，不是通用聊天助手，与 MiniGPT-4 无关。

## 项目背景

将复杂语言模型拆成可读、可测试的基本组件，可以更清楚地理解下一字符预测、因果遮罩和自回归生成。本项目以公开英文小说为语料，保留训练配置、随机种子、loss、参数量和环境信息，并在固定参数量下比较单头与多头注意力。

代码优先服务教学和实验透明性，没有引入分布式训练、模型编译或预训练模型下载。

## 核心功能

- 独立实现 Token/Position Embedding、因果单头/多头注意力、FFN 和 Transformer Block。
- 字符级编码，训练词表与验证数据隔离，未知字符使用 UNK。
- CPU 低资源默认配置，支持 `--device auto/cpu/cuda/mps`。
- 保存训练/验证 loss、最佳/最终 checkpoint、完整配置、种子、环境和数据/源码校验值。
- 三个配对种子的单头/四头对照实验，参数量与初始化可核查。
- 命令行与 Gradio 文本生成，支持 temperature、top-k 和可复现采样种子。
- 中文关键注释和自动测试，包括注意力数值对照、因果性、梯度及端到端训练生成。

## 模型架构

```mermaid
flowchart LR
    A[字符编号] --> B[Token Embedding + Position Embedding]
    B --> C[Transformer Blocks × 2]
    C --> D[LayerNorm]
    D --> E[词表投影]
    E --> F[下一字符概率与采样]
```

每个 Block 采用 Pre-LayerNorm 和两条残差连接：

```text
x = x + CausalMultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
FeedForward: Linear(d, 4d) → GELU → Linear(4d, d)
```

注意力为 `softmax(QKᵀ / sqrt(d_head) + causal_mask)V`。遮罩使每个位置只能观察自身及前文。单头使用同一实现的 `H=1` 特例；总宽度保持不变，所以改变 head 数不改变投影参数量。

默认配置：2 层、宽度 64、4 个头、上下文 64、batch 8、AdamW 学习率 0.0003、dropout 0、训练 200 步、seed 42、CPU 两线程。当前语料词表为 76（含 UNK），总参数量 **113,920**。

## 安装

要求 **Python >= 3.11**。本次实际验证环境为 Windows 10 / Python 3.14.2 / PyTorch 2.14.1+cpu。其他操作系统和 Python 版本尚未实测。

```bash
git clone https://github.com/qinzhong631-ux/minigpt-lab.git
cd minigpt-lab
python -m venv .venv
```

激活环境：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

安装依赖：

```bash
python -m pip install --upgrade pip
# Windows/Linux 仅需 CPU 时，可先从官方 CPU 索引安装对应构建
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

macOS 或 CUDA 环境请按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装适配设备的 PyTorch，再安装其余依赖。原始实验使用 NumPy 2.4.0，该版本后来被上游撤回。发布依赖改为未撤回的 2.4.6，原始实验元数据不改写。仓库固定直接依赖版本；`torch==2.14.1` 允许对应的 `+cpu` / CUDA 构建。完整环境见 [展示运行元数据](results/default/metadata.json)。没有锁定全部传递依赖，因此不承诺跨版本或硬件位级一致。

## 从准备数据到生成文本

**仓库不包含训练语料全文或模型权重。首次克隆后需要先准备数据并训练。** 已公开的 `results/` 是实验记录，不能直接加载为模型。

```bash
# 1. 下载公开文本并保留原件、来源和校验值
python prepare_data.py

# 2. 默认低资源训练：200 步；自动选择可用设备
python train.py

# 3. 读取训练产生的 runs/default/best.pt
python generate.py --prompt "Alice" --max-new-tokens 200 --seed 42

# 4. 启动本地交互界面
python app.py
```

打开终端显示的地址，默认是 `http://127.0.0.1:7860`。可调整起始文本、生成长度、temperature、top-k 和随机种子，并查看 loss 曲线。服务仅绑定本机，默认不公开分享；端口被占用时加 `--port 7861`。

更多运行方式：

```bash
# CPU 快速流程检查
python train.py --device cpu --steps 10 --out runs/smoke_new

# 重现本项目 1000 步展示训练
python train.py --device cpu --steps 1000 --seed 42 --out runs/showcase_new
python generate.py --checkpoint runs/showcase_new/best.pt --prompt "Alice"
python app.py --checkpoint runs/showcase_new/best.pt

# 六组对照：1/4 heads × 三个种子，各 200 步
python compare.py --device cpu --steps 200 --seeds 42 43 44 --out runs/comparison_new

# 运行全部测试（无须下载语料或预训练权重）
python -m pytest -q

# 完成默认训练后，实际启动并调用 Gradio 生成接口
python scripts/verify_app.py
```

每次训练请选择空目录；非空 `--out` 会报错以保护既有实验。使用 `python train.py --help` 查看完整参数。默认训练和1000步展示训练的预算不同，不要直接混用结果。

## 数据、日志与复现

语料为 Lewis Carroll 的 *Alice’s Adventures in Wonderland*，[Project Gutenberg #11](https://www.gutenberg.org/ebooks/11)。本次下载提取 144,600 字符：前 130,140 字符训练，后 14,460 字符验证。词表只从训练区间建立，批次窗口不跨越边界；本次验证集 UNK 计数为 0。

- `metadata.json`：环境、配置、随机种子、参数量、数据及源码 SHA256、初始权重校验值。
- `metrics.csv`：每步训练批次 loss，以及评估点的训练/验证 loss。
- `summary.json`：最终指标与最佳 checkpoint 步数。
- `vocab.json`、`best.pt`、`last.pt`：词表和推理权重。

训练批次与评估使用独立随机生成器。`train_step_loss` 是更新前当前批次的损失；`train_loss` / `val_loss` 是更新后 eval 模式下固定抽样窗口的均值，单位为 nats/字符。评估不是全验证集的无重叠遍历。验证集用于选 best checkpoint，**没有独立测试集**。

数据来源及原始文件 SHA256 见 [data/source.json](data/source.json)。上游文本可能更新，重现实验时应核对校验值。生成超过上下文时使用最近字符窗口并重新编号位置；尚无 KV cache 或精确断点续训。

## 实际实验结果

以下均来自 2026-10-12 的真实 CPU 运行。完整 CSV、配置与原始生成文本保存在 [results/](results/)。

| 配置 | 步数 | 参数量 | 最终训练 loss | 最终验证 loss |
|---|---:|---:|---:|---:|
| 展示模型：4 heads，seed 42 | 1000 | 113,920 | 2.331909 | 2.368591 |

固定总宽度、层数、训练预算、数据与优化器，比较单头和四头（每组三种子、每次200步）：

| 注意力头数 | 参数量 | 验证 loss 均值 | 样本标准差 |
|---|---:|---:|---:|
| 1 | 113,920 | 2.674768 | 0.019984 |
| 4 | 113,920 | 2.677110 | 0.019667 |

同一种子的两组初始权重哈希和样本序列一致。报告固定最终步指标，避免对不同组挑选不同的最优时刻。单头在这次小实验中数值略低，差异很小，**不能推导单头或多头普遍更优**。

等参数不等于等计算：每头维度从64变为16，注意力矩阵数量、softmax 行为及中间激活内存不同。仅一个语料、三个种子、短训练，也没有统计显著性检验；详见 [实验报告](docs/EXPERIMENTS.md)。模型生成仍有拼写错误和不连贯句子，[真实生成样本](results/default/sample.txt) 未人工润色。

## 项目结构

```text
minigpt-lab/
├── model.py              # Transformer 与各基础组件
├── data.py               # 字符词表、切分与批次
├── runtime.py            # 设备、随机种子和环境记录
├── prepare_data.py       # 语料下载与来源追踪
├── train.py              # 训练、评估和 checkpoint
├── generate.py           # 自回归采样
├── compare.py            # 单头/四头实验
├── app.py                # Gradio 界面
├── tests/                # 单元与集成测试
├── scripts/verify_app.py  # 真实 HTTP 生成验收
├── data/source.json      # 实验所用语料来源及校验值
├── results/              # 去除本地绝对路径的实验记录
├── docs/                 # 实验与验证说明
├── licenses/             # 第三方许可与数据说明
├── requirements.txt
└── LICENSE
```

`.gitignore` 排除虚拟环境、缓存、下载语料、权重、密钥文件以及本地机器日志。训练结果仍保存在本地 `runs/`；公开的 `results/` 是已审核、保留真实指标的导出副本。

## 测试与已知局限

项目测试覆盖字符编码、数据隔离、因果性、注意力数值、梯度、小批量过拟合、真实训练/权重加载/采样复现、参数公平性与 Gradio 接口。发布前复测为 **17 passed，1 skipped**；跳过项为本机无 CUDA 的真实 GPU 测试。发布验收的具体证据与尚未验证事项见 [验证报告](docs/VALIDATION.md)。

未实测 CUDA/MPS、全部支持的 Python 版本或长期训练稳定性；不能以当前测试替代所有平台兼容性验证。本项目用于教学和小规模研究演示，不提供通用语言理解、聊天或高质量写作能力。

## 参考资料、第三方代码和 AI 辅助开发

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：缩放点积与多头注意力。
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)：参考简洁 GPT 结构、字符建模及训练/采样组织方式。未直接复制其 Python 源文件或加载其权重；保留 [nanoGPT MIT 许可](licenses/nanoGPT-MIT.txt)。
- [PyTorch 官方文档](https://docs.pytorch.org/docs/stable/index.html)与 [Gradio 官方文档](https://www.gradio.app/docs)：第三方库 API。checkpoint 使用 `weights_only=True`，仍仅应加载可信文件。
- 本项目由用户提出需求，**OpenAI Codex 辅助编写代码、中文注释、测试与文档，并执行测试和修订**。AI 辅助不构成正确性保证；实际指标与未验证部分分别披露。

原创代码按 [MIT License](LICENSE) 提供。第三方依赖按各自许可证使用，许可信息和随包取得的声明见 [licenses/](licenses/)。数据不因代码采用 MIT 而成为 MIT 数据；Gutenberg 来源页标示作品在美国属于公有领域，下载脚本保留原始电子书与条款，使用和再分发请遵守适用要求，见 [数据说明](licenses/DATA_NOTICE.md)。
