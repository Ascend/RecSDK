# TokenMixer-Large

TokenMixer-Large 是 RankMixer2.0 在 RecSDK 中的独立实现，面向 Amazon Books 序列召回任务。当前工程同时保留 GPU 和 NPU 运行分支，并支持 dense/sparse MoE、Token Parallel、NPU 融合算子、训练指标记录和性能 profiling。

论文参考：

- [RankMixer2.0 / TokenMixer-Large](https://arxiv.org/pdf/2602.06563)

## 一、代码入口

模型目录：

    rec_model_zoo_pytorch/tokenmixer-large/ScalingLaw-Rank-Generic-recall/

主要入口：

    sample_configs/train_amazon_books_4B.ini
    sample_configs/train_amazon_books_7B.ini
    sample_configs/train_amazon_books_15B.ini
    ScalingLaw/run.sh
    ScalingLaw/main.py
    ScalingLaw/modeling/generic/sequential_v2/tokenmixer_large.py
    ScalingLaw/modeling/generic/sequential_v2/DLRM.py
    ScalingLaw/modeling/generic/sequential_v2/GR_model.py

推荐通过 benchmark 统一入口运行：

    cd /path/to/RecSDK/benchmark
    python run.py TOKENMIXER_LARGE.json

对应配置和补丁：

    benchmark/configs/TOKENMIXER_LARGE.json
    benchmark/patches/tokenmixer-large.patch

直接运行模型时，需要进入 ScalingLaw 目录：

    cd rec_model_zoo_pytorch/tokenmixer-large/ScalingLaw-Rank-Generic-recall/ScalingLaw

## 二、数据准备

### 2.1 数据格式

数据采用 Meta GR/SASRec 风格的序列召回输入。需要先将原始 Amazon Books 数据转换为 CSV 文件，至少保证数据目录下存在 CSV 文件，常用文件名为 sasrec_format.csv。

数据目录可以是：

    rec_model_zoo_pytorch/tokenmixer-large/amzn_books.ori/

也可以使用任意绝对路径，通过 --data_dir 传入。

### 2.2 默认数据路径

run.sh 默认从 ScalingLaw 目录计算数据路径：

    ../../amzn_books.ori

因此默认目录实际对应：

    rec_model_zoo_pytorch/tokenmixer-large/amzn_books.ori/

启动前可以检查：

    ls ../../amzn_books.ori/*.csv

如果目录不存在或没有 CSV 文件，启动脚本会直接退出，并提示使用 --data_dir 指定数据目录。

## 三、运行方式

### 3.1 NPU 单卡功能验证

单卡验证时需要关闭 Token Parallel，否则 4B/7B 的 TP4 或 15B 的 TP8 无法在单卡上组成完整并行组。

将配置中的以下参数改为：

    "use_token_parallel": false,
    "token_parallel_size": 1

然后运行：

    VISIBLE_DEVICES=0 bash run.sh npu \
      --config_file=../sample_configs/train_amazon_books_4B.ini \
      --data_dir=/path/to/amzn_books.ori \
      --save_dir=/path/to/tokenmixer_runs/4B_single_card

### 3.2 NPU 多卡训练

4B 和 7B 默认使用 TP4：

    VISIBLE_DEVICES=0,1,2,3 bash run.sh npu \
      --config_file=../sample_configs/train_amazon_books_4B.ini \
      --data_dir=/path/to/amzn_books.ori

15B 默认使用 TP8：

    VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh npu \
      --config_file=../sample_configs/train_amazon_books_15B.ini \
      --data_dir=/path/to/amzn_books.ori

VISIBLE_DEVICES 会同时设置 ASCEND_RT_VISIBLE_DEVICES 和 CUDA_VISIBLE_DEVICES。启动脚本使用 torchrun，默认每个可见设备启动一个进程。

Token Parallel 的约束：

    进程数 >= token_parallel_size
    进程数必须能够被 token_parallel_size 整除

如果出现以下提示：

    Token Parallel disabled: world_size=1 is not divisible by token_parallel_size=4

说明当前实际只启动了单进程，需要增加 VISIBLE_DEVICES，或者关闭 Token Parallel。

### 3.3 GPU 运行

GPU 使用同一个模型入口和配置，启动脚本会自动切换到 GPU 分支：

    VISIBLE_DEVICES=0 bash run.sh gpu \
      --config_file=../sample_configs/train_amazon_books_4B.ini \
      --data_dir=/path/to/amzn_books.ori

GPU 分支不会调用 NPU 专用算子。模型会回退到 PyTorch 的 bmm、普通 SwiGLU、dense expert 或 Python dispatch 路径，因此 GPU 和 NPU 性能对比时应保持模型配置、batch、数据、step 数和精度一致。

### 3.4 训练、评估和推理模式

默认运行训练：

    bash run.sh npu

通过 MODEL_MODE 切换运行模式：

    MODEL_MODE=eval bash run.sh npu
    MODEL_MODE=infer bash run.sh npu
    MODEL_MODE=save_user_emb bash run.sh npu

eval 和 infer 会关闭训练状态，save_user_emb 用于保存用户侧 embedding。是否执行完整评估还受配置中的 is_recall、eval_recall 和 eval_after_each_epoch 控制。

## 四、4B、7B、15B 配置

当前不需要新增独立启动脚本，只需要通过 --config_file 选择配置。

| 配置 | item embedding | local batch | TokenMixer dim | 层数 | heads | TP size | experts |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4B | 1024 | 32 | 64 | 8 | 4 | 4 | 3 |
| 7B | 1536 | 16 | 128 | 10 | 8 | 4 | 3 |
| 15B | 2048 | 8 | 128 | 20 | 16 | 8 | 7 |

对应文件：

    sample_configs/train_amazon_books_4B.ini
    sample_configs/train_amazon_books_7B.ini
    sample_configs/train_amazon_books_15B.ini

这些配置的 batch 是单卡 batch。多卡全局 batch 约等于：

    global_batch_size = local_batch_size * world_size

做 GPU/NPU 性能对比时，建议同时记录单卡 batch、卡数、全局 batch 和每步耗时，不能只比较单卡 QPS。

性能测试或快速功能验证可以调整：

    "batch_limit": 40,
    "e2e_average_nums": 20,
    "epochs": 2,
    "num_epochs": 2

正式收敛验证时，应去掉较小的 batch_limit，并使用相同 epoch、数据顺序和随机种子。

## 五、模型结构

TokenMixer-Large 的主干路径：

    Embedding / Tokenization
      -> Pre-Norm
      -> Mixing
      -> Per-token SwiGLU 或 Sparse-Pertoken MoE
      -> Reverting
      -> Pre-Norm
      -> Per-token SwiGLU 或 Sparse-Pertoken MoE
      -> Interval Residual
      -> Pooling / Recall Head

核心实现位于 tokenmixer_large.py：

- PerTokenSwiGLU：每个 token 独立的 gate/up/down 参数。
- SparsePerTokenMoE：支持 top-k router、shared expert、gate scaling 和稀疏 dispatch。
- TokenMixerLargeBlock：实现 Pre-Norm、Mixing/Reverting、残差和 block 级 aux loss。
- TokenParallelRuntime：管理 token 维度切分、AllGather/Reduce 和通信重叠。
- TokenMixerLarge：模型入口，负责构造 block、连接 DLRM 和输出结果。

DLRM.py 和 GR_model.py 负责把 TokenMixer-Large 接入 RecSDK 的统一模型接口，并向训练链路传递 deep_outputs、deep_loss、deep_sparsity 和 aux_loss。

## 六、NPU 性能优化

### 6.1 SwiGLU 和 GEMM 优化

NPU 路径优先使用以下优化：

    einsum -> bmm
    gate/up 两次投影 -> fused gate_up bmm
    split + SiLU + multiply -> npu_swiglu

相关开关：

    "npu_bmm": true,
    "fused_gate_up": true,
    "use_npu_swiglu": true

### 6.2 Grouped MoE

开启 use_npu_grouped_moe 后，稀疏 MoE 优先使用 NPU grouped expert 路径：

    MoE token permute
      -> grouped matmul
      -> npu_swiglu
      -> grouped matmul
      -> MoE token unpermute

这条路径用于减少 Python expert loop、nonzero、index_select 和大量小 GEMM。相关开关：

    "use_npu_grouped_moe": true,
    "use_npu_grouped_moe_bias_fusion": true,
    "fallback_on_npu_grouped_moe_failure": true

如果当前 torch_npu 或 CANN 没有对应算子，模型会回退到 PyTorch 路径。回退机制用于保证 GPU 和不完整 NPU 环境仍可运行，但性能可能下降。

### 6.3 RMSNorm 和残差融合

NPU 侧支持融合残差加法和 RMSNorm：

    "use_rmsnorm": true,
    "use_fused_add_rmsnorm": true,
    "use_fused_interval_rmsnorm": true

如果遇到算子不支持、精度异常或编译失败，可以先将两个 fused 开关关闭，确认基础路径正确后再逐个打开。

### 6.4 Token Parallel 和通信重叠

Token Parallel 相关配置：

    "use_token_parallel": true,
    "token_parallel_size": 4,
    "tp_comm_overlap": true,
    "tp_comm_dtype": "bfloat16",
    "tp_reduce_aux_loss": false

15B 配置的 token_parallel_size 为 8。训练脚本默认设置：

    TP_ALL_GATHER_INTO_TENSOR=1
    TP_COMM_OVERLAP=1
    HCCL_OVERLAPPING=1
    HCCL_FUSION=1
    HCCL_BUFFSIZE=128

通信性能对路由分布、卡数、全局 batch 和 HCCL 拓扑都敏感。对比优化前后版本时，需要固定随机种子和数据顺序，并至少统计稳定区间内的平均 step time、P95 step time、通信耗时和 Free time。

## 七、编译、图模式和 profiling

首次功能验证建议关闭 compile 和 graph：

    ENABLE_COMPILE=0 ENABLE_GRAPH=0 bash run.sh npu

确认 loss、输出 shape 和评估指标正确后，再逐步开启：

    ENABLE_COMPILE=1 ENABLE_GRAPH=0 bash run.sh npu
    ENABLE_COMPILE=1 ENABLE_GRAPH=1 bash run.sh npu

精度对比可以开启：

    CHECK_PRECISION=1 ENABLE_COMPILE=1 bash run.sh npu

### 7.1 NPU profiling

直接运行时使用：

    MODEL_PROFILING_FLAG=1 bash run.sh npu

NPU profiling 默认输出到：

    ScalingLaw/profiling/1B_gemm_tokenmixer_large_graph/

日志中重点关注：

    Computing
    Comm
    Free
    Overlapped
    MoE token permute/unpermute
    GroupedMatmul
    SwiGlu
    RmsNorm / AddRmsNorm
    ZerosLike

### 7.2 GPU profiling

GPU profiling 使用：

    MODEL_PROFILING_FLAG=1 bash run.sh gpu

输出目录为：

    ScalingLaw/gpu_profiling/profiling/

profiling 会改变运行开销，因此性能对比需要分别进行无 profiling 的正式压测和带 profiling 的算子归因测试。

## 八、训练指标和结果文件

配置中开启：

    "record_training_metrics": true

训练结束后，指标默认写入：

    <save_dir>/modelfile/run_metrics/

主要文件：

| 文件 | 内容 |
|---|---|
| loss_curve.csv | step、loss、ms/step、samples/s |
| epoch_metrics.csv | epoch 级评估指标，包括 AUC 或召回指标 |
| metrics_summary.json | 最终 loss、平均 QPS、最后一次评估结果和 AUC 是否可用 |
| loss_auc_curve.png | loss 曲线和 AUC 曲线 |
