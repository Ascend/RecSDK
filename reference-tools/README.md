# Rec SDK Reference Tools 工具集说明

本目录包含RecSDK的辅助工具集合，用于模型性能分析、精度验证、数据解析等功能。工具仅限于离线开发验证测试使用，禁止在生产环境使用。

## 目录结构概览

```
├── atomic          # 原子操作相关工具
├── graph_partition # 图分割工具
├── model_convert   # 模型转换工具
├── parse_data      # 数据解析工具
├── perf            # 性能分析工具
├── perfrec-python  # 性能记录工具
└── precrec-python  # 精度验证工具
├── mx_rec_perf.sh  # 性能分析脚本，主要作用是对推荐系统模型训练过程中的性能瓶颈进行诊断和分析
```

## 子目录详细说明

### atomic（原子操作）
- **功能**：提供稀疏操作支持，包含生成多热点数据、模型信息展示等功能
- **主要文件**：
  - `sparse_ops/`：稀疏操作实现模块
  - `gen_mt_data_0to1e.py`：生成多热点数据
  - `model_info.md`：模型信息模板
  - `sparse.sh`：稀疏操作执行脚本
  - `sparse_lookup.py`：稀疏查找示例
  - `sparse_lookup_with_grad.py`：带梯度的稀疏查找示例

### graph_partition（图分割）
- **功能**：图分割工具，用于分布式训练中的计算图划分
- **主要文件**：
  - `gen_config.py`：生成配置文件
  - `graph_partition.py`：核心图分割逻辑
  - `template.cfg`：配置模板

### model_convert（模型转换）
- **功能**：模型格式转换工具，支持多任务模型版本
- **主要文件**：
  - `model_convert.py`：主转换脚本
  - `model_convert_mt_v2.py`：多任务模型转换脚本
  - `common.py`：公共函数
  - `README.md`：使用说明

### parse_data（数据解析）
- **功能**：TensorFlow数据解析性能分析工具
- **主要文件**：
  - `data_parser.py`：数据解析性能测试主程序
  - `run.sh`：多worker执行脚本

### perf（性能分析）
- **功能**：性能分析工具集合
- **主要文件**：
  - `fast.sh`：快速性能测试
  - `host_set.sh`：主机设置
  - `msprof.sh`：Ascend Profiler调用脚本
  - `perf_flame_graph.sh`：火焰图生成脚本

### perfrec-python（性能记录）
- **功能**：性能追踪与融合分析工具
- **主要文件**：
  - `fusion_tracing.py`：融合操作追踪
  - `perf.py`：性能统计
  - `config.toml`：配置文件
  - `README.md`：使用说明

### precrec-python（精度验证）
- **功能**：精度验证工具，支持稠密和稀疏检查点比较
- **主要文件**：
  - `precision_check.py`：精度对比工具
  - `sparse_ckpt.py`：稀疏检查点处理
  - `dense_ckpt.py`：稠密检查点处理
  - `loss.py`：损失值计算
  - `ops.py`：操作验证

## 安全规范

1. **文件读取安全**
   - 执行文件操作前应验证文件是否存在及可读性
   - 对网络或其他不可信来源的数据进行合法性校验
   - 使用`pickle`模块加载时确保文件可信

2. **参数合法性检查**
   - 所有输入参数必须进行有效性验证（类型、范围、格式）
   - 特别是在处理外部配置值时，应确保它们不会超过物理限制

3. **异常处理规范**
   - 避免捕获通用异常(Exception)，应明确捕获特定异常类型(ValueError, IndexError等)
   - 文件操作需增加权限检查和异常处理逻辑

## 免责条款

1. **技术限制**
   - 本工具集专为Rec SDK设计，依赖特定版本的Python、TensorFlow及其他组件
   - 部分工具可能仅适用于昇腾平台

2. **已知风险**
   - 开发分支可能包含未完成或不稳定的功能
   - 部分工具在aarm64环境下可能存在问题

3. **使用建议**
   - 不建议在生产环境中直接使用未经充分测试的工具版本
   - 所有运行命令应在容器或镜像中执行，确保环境一致性