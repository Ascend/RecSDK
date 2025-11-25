# SIM (Search-based User Interest Modeling) 模型实现说明

## 1. 概述

SIM (Search-based User Interest Model) 是一种用于处理用户长期行为序列的点击率(CTR)预估模型。该模型采用两阶段搜索策略，有效处理长用户行为序列：

- **通用搜索单元 (General Search Unit, GSU)**: 从长期行为序列中搜索出与目标项目相关的子序列
- **精确搜索单元 (Exact Search Unit, ESU)**: 对候选项目与子序列之间的精确关系进行建模

## 2. 模型架构

### 2.1 通用搜索单元 (GSU)

支持两种搜索方式：

#### 硬搜索 (Hard Search)
- 无参数方法，根据类别匹配选择行为
- 计算效率高，适合线上部署
- 公式: r_i = {1 if C_i = C_a, 0 otherwise}

#### 软搜索 (Soft Search)
- 参数化方法，通过计算用户行为和目标物品的相似度
- 使用嵌入向量进行内积计算
- 公式: r_i = MLP(W_a * e_a + W_b * e_i)

### 2.2 精确搜索单元 (ESU)

- 使用多头注意力机制捕捉用户的多样化兴趣
- 引入时间间隔信息，建模行为的时间状态属性
- 使用Transformer架构进行序列建模

## 3. 实现细节

### 3.1 模型组件
- `GeneralSearchUnit`: 实现GSU功能
- `ExactSearchUnit`: 实现ESU功能
- `SIMModel`: 整合GSU和ESU的完整模型
- `SIMLoss`: 模型损失函数

### 3.2 输入输出
- **输入**:
  - 用户行为序列: [batch_size, seq_len, embedding_dim]
  - 目标物品嵌入: [batch_size, embedding_dim]
  - 用户特征: [batch_size, user_feature_dim]（可选）
  - 行为类别: [batch_size, seq_len]（硬搜索需要）
  - 时间间隔: [batch_size, k]（离散化时间间隔）

- **输出**:
  - 预测CTR: [batch_size, 1]
  - 注意力权重: [batch_size, k]
  - 选中行为索引: [batch_size, k]

## 4. 使用方法

### 4.1 模型初始化
```python
model = SIMModel(
    item_embedding_dim=64,       # 物品嵌入维度
    user_feature_dim=32,         # 用户特征维度
    hidden_dim=128,              # 隐藏层维度
    num_heads=8,                 # 注意力头数
    dropout=0.1,                 # dropout比率
    search_type='hard',          # 搜索类型 ('hard' 或 'soft')
    num_categories=100           # 类别数量（硬搜索需要）
)
```

### 4.2 前向传播
```python
# 前向传播
pred_ctr, attention_weights, selected_indices = model(
    user_behavior_seq=user_behavior_seq,      # 用户行为序列
    target_item_emb=target_item_emb,          # 目标物品嵌入
    user_features=user_features,              # 用户特征（可选）
    behavior_categories=behavior_categories,  # 行为类别（硬搜索需要）
    time_intervals=time_intervals             # 时间间隔
)
```

### 4.3 可选配置项
```
export PROFILING_FLAG=1  #开启profiling性能采集
export E2E_FLAG=1        #开启端到端耗时计算
export INDUCTOR_FLAG=1   #开启Inductor模式
```

### 4.4 运行命令
```
python SIM_demo.py  #运行demo
python SIM_train.py #完整运行
```

## 5. 特性

- **高效性**: 通过两阶段搜索策略，将长序列缩减为固定长度子序列
- **可扩展性**: 支持软搜索和硬搜索两种方式
- **时间建模**: 引入时间间隔信息，更好地建模长期兴趣
- **多头注意力**: 捕捉用户兴趣的多样性
- **工业友好**: 考虑了线上部署的性能要求

## 6. 参考文献

- CIKM 2020: "Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction"
- 阿里巴巴SIM模型论文及实践