# 半异步训练框架 Demo

本demo提供了一个通用的半异步训练框架，旨在解决分布式训练中计算与通信的耦合问题。

本demo使用 DLRM 模型作为示例载体，但核心交付物是底层的 TrainPipelineSparseDist 流水线引擎。该引擎通过精细的 pytorch Stream 管理和梯度解耦，实现了高效的“计算 - 通信”重叠，特别适用于推荐系统、大语言模型（LLM）等包含大规模稀疏嵌入查找的场景。

## 实验结果

核心实验结果：dlrm模型反向100% 通信掩盖

结论： 在本 Demo 的配置下，我们实现了 Sparse Backward的 100% 通信掩盖。

在推荐系统的分布式训练中，瓶颈通常在于 Embedding 参数的通信（AllGather/AllReduce）。我们的目标是让计算“填满”通信的等待时间。

## 现象分析

### Dense Forward (计算瓶颈)

由于 DLRM 的 Dense 部分算子较轻量，且高度优化，它成为了流水线的“短板”。
在当前的 Trace 中，Dense Forward 的时间片较短，无法完全覆盖 Data Distribution 的耗时。

### Sparse Backward (完美掩盖)

通过半异步策略，Sparse Backward 被调度到了非关键路径上。

实验数据显示：Sparse Backward 的耗时完全落在了 Dense Forward 或 Data Dist 的时间窗口内。

这意味着：在进行反向传播计算的同时，数据传输也在并行进行，反向传播没有增加额外的端到端训练时间。

## 核心功能描述

稀疏计算异步消除了batch i+1的稀疏前向计算对batch i的稀疏反向传播的依赖，使得batch i+1的稀疏前向计算可以再batch i的稀疏反向计算之前执行。这种半异步训练的能够在保留稠密计算batch同步计算依赖关系的同时，将稀疏前向计算和通信提前N个step。

训练步骤被划分为6个阶段：CPU数据加载->特征all2all通信->稀疏前向计算->稠密前向计算->稠密反向计算->稀疏反向计算。

## 快速开始

本 Demo 使用 DLRM 模型演示框架效果。

参考[DLRM模型迁移样例说明](../dlrm/README.md)完成 DLRM 模型的迁移和数据准备。

### 半异步训练适配

修改 dlrm_main.py 中的训练脚本，替换原有的训练循环为 TrainPipelineSparseDist 流水线引擎的调用。

export PYTHONPATH=$PYTHONPATH:/path/to/RecSDK/examples_benchmark/develop/semi_async

line 532-547
替换成

```python
    from semi_async import TrainPipelineSparseDistAsyncEmbedding
    # 初始化半异步训练流水线
    pipeline = TrainPipelineSparseDistAsyncEmbedding(
        model, optimizer, device, execute_all_batches=True
    )
```

run.sh 中55行替换成
MODES=("semi_async")

调整负载

为了观察半异步的效果，需要确保dense模型有足够的计算负载

在 run_dlrm_model.sh 中

降低GLOBALSIZE到1024以降低sparse forward的计算负载

增加dcn_num_layers到40以增加dense forward的计算负载

代码执行

```bash
bash run.sh
```
