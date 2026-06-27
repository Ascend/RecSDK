# hybrid_torchrec测试套件

本目录包含hybrid_torchrec组件的多个测试用例，用于验证**NPU纯显存模式下**EmbeddingBagCollection，HashEmbeddingBagCollection的功能和精度正确性。

> 说明
>
> EmbeddingBagCollection为开源TorchRec原生稀疏表实现。
>
> HashEmbeddingBagCollection为hybrid_torchrec组件新增的稀疏表实现，对应TorchRec原生的EmbeddingBagCollection，支持特征ID Hash映射、ID去重等功能。

## 目录结构

```shell
├── README.md                           # 测试套件说明文档
├── dataset.py                          # 数据集定义和随机数据生成器
├── model.py                            # 测试模型定义
├── util.py                             # 工具函数（日志设置等）
├── test_all.sh                         # 执行全部测试用例的脚本
├── test_hybrid_embeddingbag.py         # 测试TorchRec原生的EmbeddingBagCollection前反向查表
├── test_hybrid_hash_embeddingbag.py    # 测试支持特征ID Hash映射的HashEmbeddingBagCollection前反向查表
├── test_hybrid_hash_embeddingbag_dp.py # 测试DP分表模式下的HashEmbeddingBagCollection
├── test_hybrid_pipeline_hash_embeddingbag.py  # 测试pipeline场景下的HashEmbeddingBagCollection前反向查表
├── test_ids_process.py                 # 测试C++侧实现的特征ID去重分桶相关功能
├── test_module.py                      # 测试HashEmbeddingBagCollection的在CPU/NPU设备上功能/精度正确性
└── test_train_and_eval.py              # 测试训练和评估功能
```

## 运行测试

### 环境准备

请参见[hybrid_torchrec文档](../../README.md)安装hybrid_torchrec及其依赖项。

### 运行单个测试

```bash
# 运行单个测试文件
pytest test_hybrid_embeddingbag.py
```

### 运行所有测试

```bash
# 使用提供的脚本运行所有测试
bash test_all.sh
```

## 注意事项

1. 确保在NPU环境下运行测试
2. 分布式测试需要足够的NPU资源（2张NPU卡）
3. 测试数据是随机生成的，结果可能因种子而异
