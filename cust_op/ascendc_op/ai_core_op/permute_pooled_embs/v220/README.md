# PermutePooledEmbs 算子说明

## 概述

`permute_pooled_embs` 是用于对池化后的嵌入（pooled embeddings）输出进行特征维度重排列的核心算子。该算子主要用于推荐系统和深度学习场景中，当需要对多个特征的嵌入输出进行重新排序时使用。

### 主要功能
- 对嵌入输出张量沿特征维度进行重排列
- 支持 float32、float16、bfloat16 数据类型
- 支持 CPU 和 NPU 两种计算后端
- 针对 NPU 进行了 SIMD 并行优化

### 应用场景
- **分布式训练**：在不同设备间重新组织嵌入特征
- **特征重组**：根据业务需求调整特征顺序
- **模型并行**：在模型并行训练中重新分配特征

## 接口定义

### 算子输入与输出

| 名称                | 输入/输出 | 参数类型 | 数据类型                       | 数据格式                    | 范围/说明 |
|---------------------|----------|----------|-------------------------------|----------------------------|-----------|
| pooled_embs         | 输入     | Tensor   | float32/float16/bfloat16      | [B_local, total_global_D]  | 池化后的嵌入输出张量。B_local为batch size，total_global_D为所有特征的embedding维度之和 |
| offset_dim_list     | 输入     | Tensor   | int64                         | [T+1]                      | 每个特征embedding维度的累积和。offset_dim_list[0]=0, offset_dim_list[T]=total_global_D |
| permute_list        | 输入     | Tensor   | int64                         | [T]                        | 输出特征顺序，值范围[0, T-1]，不可重复 |
| inv_offset_dim_list | 输入| Tensor  | int64                         | [T+1]                      | 重排后特征embedding维度的累积和|
| inv_permute_list    | 输入| Tensor  | int64                         | [T]                        | permute_list的逆索引/逆排列 |
| permuted_pooled_embs | 输出     | Tensor   | float32/float16/bfloat16      | [B_local, total_global_D]  | 重排列后的嵌入张量。与输入pooled_embs形状一致，特征列顺序发生改变 |


## 算法原理

### 核心思想
对每个特征块独立执行重排列，通过预计算的偏移列表确定每个特征在输入和输出张量中的位置范围。

### 并行策略
- **核心级并行**: 按total_global_D维度进行多核均匀拆分
- **SIMD并行**: 使用DataCopy进行列数据的非连续搬运，根据输入shape分为对齐情况和非对齐情况。

### 性能优化
- **DataCopy优化**: 使用AscendC的DataCopy进行高效的数据搬运
- **列数据处理**: 直接操作列数据，避免转置开销
- **内存对齐**: 非对齐场景下通过padding确保数据对齐以提高访问效率

## 文件结构
```
permute_pooled_embs/
├── c310/
│   └── run.sh                         # 编译脚本
├── v220/
│   ├── op_host/                       # Host侧实现
│   │   ├── permute_pooled_embs_tiling.h
│   │   └── permute_pooled_embs.cpp
│   ├── op_kernel/                     # Kernel侧实现
│   │   ├── permute_pooled_embs_kernel.h
│   │   └── permute_pooled_embs.cpp
│   ├── permute_pooled_embs.json       # 算子配置
│   ├── README.md                      # 说明文档
│   └── run.sh                         # 编译脚本
```


## 算子编译部署

算子编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"1.算子编译"章节。

注：详细算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/permute_pooled_embs/README.md)

## 单算子测试
算子编译与部署、算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/permute_pooled_embs/README.md)。