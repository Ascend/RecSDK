# OxygenREC v2 Full

在 RecSDK 的 `benchmark/` 目录运行下面两个测试。GPU 和 NPU 共用配置，都使用 Eager。数据下载地址和目录结构见 [Benchmark 运行说明](../../README.md#oxygenrec-v2-full-模型)。若设备环境使用单独的 Python，请设置 `PYTHON_BIN` 为该解释器的绝对路径。

| 测试 | 命令 | 成功标志 |
| --- | --- | --- |
| Full 训练 | `"${PYTHON_BIN:-python3}" run.py OXYGENREC_V2_TRAIN.json --eager` | `OK stage=v2_full_epoch_training`，默认训练 1 个完整 epoch，打印新 checkpoint 的路径 |
| 推理验证 | `"${PYTHON_BIN:-python3}" run.py OXYGENREC_V2_VALIDATION.json --eager` | `OK stage=v2_fixed_validation`，默认对 32 个固定列表执行 greedy 和 beam@5 解码 |

推理默认读取资产目录中的初始权重。测试刚训练出的模型时，把训练日志中的 `checkpoint=` 完整路径赋给 `V2_FULL_CHECKPOINT` 后运行验证。训练、验证均支持 `OXYGENREC_PRECISION=bf16`（默认）或 `fp32`，验证还支持 `all`。对照训练摘要的 `checkpoint.sha256` 与推理摘要的 `inputs.checkpoint_sha256` 可确认使用了哪份权重。

## 从原始 CSV 生成起始权重

Full 训练入口在缺少默认的 SID 注册表和初始权重时自动完成相应步骤。预处理和预训练使用 CPU，生成的文件可供 GPU 和 NPU 共用。显式指定的 SID 注册表或权重路径须指向已有文件。

需要单独生成文件时，在 `benchmark/` 目录的同一 Bash 终端依次执行以下两步。

**1. 数据集预处理。** 从训练时段的事件中选择商品，用训练截止前的商品属性为它们生成语义编码，保存为 `sid_registry.json`。已有配套注册表时可跳过第一条 Python 命令。

```bash
assets="${OXYGENREC_ASSET_ROOT:-$PWD/datasets/oxygenrec}"
raw="$assets/data/raw/retailrocket"
sid="$assets/data/processed/rq_comparison/w256_kmeanspp"
prep=models/oxygenrec/original/prepare
"${PYTHON_BIN:-python3}" "$prep/fit_retailrocket_sid.py" --device cpu --events "$raw/events.csv" \
  --properties "$raw/item_properties_part1.csv" "$raw/item_properties_part2.csv" \
  --width 256 --initialization kmeans++ --output-dir "$sid"
```

**2. 预训练。** 用事件表和注册表训练一轮，将初始权重保存为 `checkpoints/gpu_pretraining/full-epoch-1.pt`（相对于数据目录）。已有这份配套权重时可直接运行 Benchmark 的训练或推理配置。

```bash
"${PYTHON_BIN:-python3}" "$prep/train_v2_pretraining_ablation_retailrocket.py" --device cpu \
  --events "$raw/events.csv" --sid-registry "$sid/sid_registry.json" --epochs 1 \
  --max-train-samples 5000 --max-validation-samples 32 --output-dir "$assets/checkpoints/gpu_pretraining"
```

## 每个文件的用途

以下只列 OxygenREC 接入涉及的文件。正式运行时，公共运行器 [run.py](../../run.py) 按 JSON 配置检出固定源码提交 `fc711c66f784e256da68e8713f58d18964039e45`，安装模型启动脚本，再调用训练或验证。运行器生成的 `RecSDK/` 是独立源码缓存。`original/` 在本分支中保留模型代码、两项测试的实现以及从原始数据准备资产所需的工具。

### Benchmark 配置与输出

| 文件 | 作用 |
| --- | --- |
| [benchmark/README.md](../../README.md#oxygenrec-v2-full-模型) | 提供数据下载、GPU/NPU 环境、精度选择和测试步骤。 |
| [OXYGENREC_V2_TRAIN.json](../../configs/OXYGENREC_V2_TRAIN.json) | 固定模型版本，启动 Full 训练，设置默认 BF16 和 1 个 epoch。 |
| [OXYGENREC_V2_VALIDATION.json](../../configs/OXYGENREC_V2_VALIDATION.json) | 固定同一模型版本，选择推理验证模式。 |
| [oxygenrec.patch](../../patches/oxygenrec.patch) | 在缓存中添加 `run_oxygenrec.sh`，解析平台、精度及路径并调用相应入口。验证完成后调用结果导出工具。 |
| [train_full.sh](train_full.sh) | 准备缺少的默认训练资产，再调用 Full 整轮训练。 |
| [export_results.py](export_results.py) | 读取验证摘要，生成 `precision_result.txt`。启用逐样本导出时还保存目标、greedy 和 beam 预测张量，供框架比对。 |
| [README.md](README.md) | 本文件，说明两个测试入口和文件分工。 |
| [.gitignore](.gitignore) | 忽略运行器下载的模型缓存、测试输出和框架结果目录。 |

### 测试实现与首次数据准备

以下路径相对于 `original/`。两个测试的 Python 实现放在 `scripts/`；`prepare/` 仅在首次从原始 CSV 构建资产时使用。

| 文件 | 作用 |
| --- | --- |
| [run_v2_full_training.sh](original/run_v2_full_training.sh) | 检查训练输入、选择设备，将参数传给 epoch 训练脚本并记录日志。 |
| [scripts/train_v2_full_epochs.py](original/scripts/train_v2_full_epochs.py) | 按训练时间切分构造列表，执行反向传播和优化器更新，每轮保存权重及摘要。 |
| [run_v2_validation.sh](original/run_v2_validation.sh) | 检查推理输入，按所选精度调用固定集验证并写日志。 |
| [scripts/evaluate_v2_fixed_validation.py](original/scripts/evaluate_v2_fixed_validation.py) | 加载指定权重，执行 greedy 和受约束的 beam 解码，记录 loss、SID 召回、预测和输入指纹。 |
| [prepare/fit_retailrocket_sid.py](original/prepare/fit_retailrocket_sid.py) | 从训练截止前的商品属性构造向量，拟合残差 K-means，生成 SID 注册表。 |
| [prepare/train_v2_pretraining_ablation_retailrocket.py](original/prepare/train_v2_pretraining_ablation_retailrocket.py) | 使用事件与注册表运行配对预训练，生成正式续训所需的 Full 初始权重。 |
| [.gitignore](original/.gitignore) | 排除本地原始数据、预处理数据、权重和 Python 缓存。 |

### 模型、数据与方法模块

下表路径相对于 `original/src/oxygenrec/`。其中多模态、Slow LLM、SFT 和后训练模块保留原模型的方法实现，日常运行两项测试时无需单独执行这些文件。

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 导出包的公共接口，包括 SID、量化、评测和指令工具。 |
| `model.py` | 定义模型配置、编码器与解码器、行为条件、损失及受约束生成。 |
| `device.py` | 初始化 CPU、CUDA 或 NPU，处理种子设置、设备同步和显存统计。 |
| `sid.py` | 定义 Semantic ID、商品与 SID 的映射，以及约束解码用的前缀树。 |
| `quantization.py` | 提供确定性的残差 K-means 参考实现。 |
| `quantization_torch.py` | 用 PyTorch 分块拟合残差码本并编码商品向量。 |
| `sid_metrics.py` | 计算码本利用、前缀覆盖、负载分布和 SID 碰撞指标。 |
| `evaluation.py` | 计算 SID token 匹配、排序命中、召回和合法率。 |
| `migration_alignment.py` | 从 checkpoint 恢复模型、重建固定输入，并计算源码和输入指纹。 |
| `migration_validation.py` | 聚合列表预测指标，计算固定验证的目标与预测指纹。 |
| `data/__init__.py` | 导出事件、时间切分、样本和模型批次接口。 |
| `data/events.py` | 定义交互事件与行为类型，逐行读取 RetailRocket CSV。 |
| `data/temporal.py` | 按时间切分事件，构造下一商品和每日目标列表，限制历史只包含目标之前的交互。 |
| `data/model_inputs.py` | 将历史和目标转换成补齐的 SID、行为 ID 与掩码。 |
| `data/property_embeddings.py` | 用训练截止前的属性快照生成商品哈希向量，供 SID 拟合。 |
| `data/bootstrap.py` | 按训练期商品频次构造确定性 SID，供数据管道自检。 |
| `multimodal.py` | 用残差 Q-Former 融合预计算的文本和图像特征。 |
| `instructions.py` | 根据历史行为构建可审查的指令文本，提供哈希编码和 JSONL 存储。 |
| `instruction_cache.py` | 按稳定样本键保存、加载离线指令特征。 |
| `llm_features.py` | 用冻结的本地语言模型提取指令特征。 |
| `llm_reasoning.py` | 构建基于历史证据的提示，生成并校验结构化检索推理结果。 |
| `retrieval_planning.py` | 将结构化检索计划转换成 IGR 使用的数值控制参数。 |
| `retrieval_selection.py` | 按固定规则选择检索命中、漏召和行为案例，供审阅。 |
| `sft_data.py` | 校验推理审阅记录并构造 SFT 对话样本。 |
| `sft_sampling.py` | 按历史行为证据分层抽取 SFT 样本。 |
| `alignment.py` | 实现 SA-GCPO 的组内优势、阈值与训练目标。 |
| `rewards.py` | 汇总格式、相对得分、排序和多样性奖励。 |
| `sa_review.py` | 按目标覆盖、奖励差异等规则选择 SA-GCPO 代表轨迹。 |
| `ea_tosd.py` | 计算轨迹奖励并选择 best-of-G 候选，提供 EA-TOSD 蒸馏所需的计算。 |
