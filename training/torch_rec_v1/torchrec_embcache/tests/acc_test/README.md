# torchrec_embcache测试套件

本目录包含torchrec_embcache组件的多个测试用例，用于验证**多级缓存模式下**EmbcacheEmbeddingCollection，EmbcacheEmbeddingBagCollection的功能和精度正确性。

> 说明
>
> EmbcacheEmbeddingCollection为torchrec_embcache组件新增的稀疏表实现，对应TorchRec原生的EmbeddingCollection。
>
> EmbcacheEmbeddingBagCollection为torchrec_embcache组件新增的稀疏表实现，对应TorchRec原生的EmbeddingBagCollection。

## 目录结构

```text
├── README.md                              # 测试用例说明文档
├── dataset.py                             # 数据集定义和随机数据生成器
├── model.py                               # 测试模型定义
├── run_test.sh                            # 执行所有测试用例的脚本
├── test_embedding_cache_pipeline.py       # 测试多级缓存pipeline+EmbCacheEmbeddingBagCollection的功能和精度
├── test_embedding_ec_cache_aggregation.py # 测试多级缓存pipeline+EmbcacheEmbeddingCollection+梯度累积优化器的功能和精度
├── test_embedding_ec_cache_pipeline.py    # 测试多级缓存pipeline+EmbcacheEmbeddingCollection的功能和精度
├── test_feature_filter.py                 # 测试（基于特征时间戳和计数）特征准入淘汰功能
├── test_kjt_with_time.py                  # 测试带时间戳的KeyedJaggedTensor功能
├── test_save_and_load.py                  # 测试保存和加载功能
├── test_show_click.py                     # 测试（基于特征展示次数和点击次数的加权分数）特征准入淘汰功能
└── util.py                                # 工具函数（日志设置等）
```

## 运行测试

### 环境准备

请参见[README文档](../../README.md)安装torchrec_embcache及其依赖项。

### 运行单个测试

```bash
# 运行单个测试文件示例
pytest test_embedding_cache_pipeline.py
```

### 运行所有测试

```bash
# 使用提供的脚本运行测试
bash run_test.sh
```

## 注意事项

1. 确保在NPU环境下运行测试
2. 分布式测试需要足够的NPU资源（大部分用例需2张NPU卡，保存和加载测试用例需3张NPU卡）
3. 测试数据是随机生成的，结果可能因种子而异
