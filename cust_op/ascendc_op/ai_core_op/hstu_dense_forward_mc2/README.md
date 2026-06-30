# HSTU Dense Forward MC2 —— 通算融合算子

## 概述

`hstu_dense_forward_mc2` 是一个 **MC2（Model Communication + Computation）通算融合算子**，将多卡间的 **All-to-All 集合通信** 与 HSTU Attention 计算融合在 AI Core Kernel 内部执行，显著减少通信与计算之间的串行等待开销。

### 核心特性

| 特性 | 说明 |
|------|------|
| **通算融合** | All-to-All 通信与 Attention 计算在 Kernel 内重叠执行 |
| **8 卡分布式** | K/V 序列分布在各卡 HBM 上，K/V Seq = Q Seq × 8 |
| **GQA 支持** | K/V 头数可小于 Q 头数，支持 4 组及以上分组 |
| **dim 不等** | qk head_dim 与 v_dim 可不等 |
| **数据格式** | Normal（4D Dense） |
| **多精度** | fp32 / fp16 / bf16 |

## 产品支持情况

| 硬件型号 | 是否支持 |
|----------|----------|
| Atlas A2 训练系列 | ✅ |
| Atlas A3 训练系列 | ✅ |
| Atlas 推理系列 | ❌ 不支持 |

## 文件结构

```text
hstu_dense_forward_mc2/
├── v220/
│   ├── op_host/                    # Host 侧 Tiling 实现
│   │   ├── hstu_dense_forward_def.cpp        # 算子 Check/Tiling 入口
│   │   ├── hstu_dense_forward_proto.cpp      # 算子原型注册
│   │   ├── hstu_dense_forward_tiling.cpp     # Tiling 数据计算
│   │   ├── hstu_dense_forward_tiling.h       # Tiling 数据结构
│   │   ├── hcom_topo_info.h                  # HCCL 拓扑信息
│   │   └── tiling_policy_define.h            # Tiling 策略常量
│   ├── op_kernel/                  # Device 侧 Kernel 实现（含 MC2 通信）
│   │   ├── collectives.h                     # All-to-All 集合通信原语
│   │   ├── comm_args.h                       # 通信参数结构体
│   │   ├── sync_collectives.h                # 同步通信原语
│   │   ├── datacopy_gm2gm.h                  # GM 间数据拷贝
│   │   ├── datacopy_gm2gm_delay.h            # 延迟数据拷贝
│   │   ├── hstu_dense_forward_normal_kernel.h  # Dense 模式核心 Kernel
│   │   ├── hstu_dense_kernel_patten_bsnd.h   # 基础 Kernel 模板
│   │   ├── hstu_dense_base.h                 # 基础定义
│   │   ├── hstu_dense_causal_mask.h          # Causal Mask 生成
│   │   ├── hstu_dense_forward.cpp            # Kernel 入口
│   │   └── tiling_policy_define.h            # Tiling 策略常量
│   ├── test/                      # 8 卡测试
│   │   ├── build.sh                          # 一键构建 & 运行
│   │   ├── test_hstu_dense_forward.cpp       # 8 卡 C++ 测试
│   │   ├── generate_golden.py                # Golden 数据生成（torchrun 8 进程）
│   │   ├── CMakeLists.txt                    # CMake 配置
│   │   ├── README.md                         # 测试说明
│   │   └── 测试计划_融合vs非融合对比.md        # 测试计划
│   ├── hstu_dense_forward.json    # 算子原型配置
│   └── run.sh                     # 编译安装脚本
└── README.md                      # 本文档
```

## 通算融合原理

### 分布式计算模型

MC2 算子运行在 **8 张 NPU 卡**上，Q 的数据按卡切分，K/V 的数据全量分布在 8 张卡上：

```text
卡 i：Output[i] = Attention( Q[i], K[0..7], V[0..7] )
```

其中 K/V 序列长度 = Q 序列长度 × 8（每张卡持有 1/8 的 K/V 序列）。

### 计算流程

```text
┌─────────────────────────────────────────────────┐
│  Step 1: 本卡 Q 与 本地 K/V 计算 QK^T            │
│  Step 2: SiLU 激活 × silu_scale × mask           │
│  Step 3: 跨卡 All-to-All 交换中间结果             │
│  Step 4: 与远程 K/V 计算局部 Attention            │
│  Step 5: 汇总得到完整 Output                     │
└─────────────────────────────────────────────────┘
```

### 关键通信文件

| 文件 | 功能 |
|------|------|
| `collectives.h` | `Collectives` 类，封装 All-to-All 数据拷贝（GM → GM）|
| `comm_args.h` | `CommArgs` 结构体，通信缓冲区相关参数 |
| `sync_collectives.h` | `SyncCollectives` 类，跨核同步原语 |
| `datacopy_gm2gm.h` | GM 到 GM 的直接数据拷贝（Ping-Pong 缓冲区）|

## 计算公式

$$
\text{Output} = \big(\text{SiLU}(QK^T + \text{bias}) \times \text{silu\_scale} \times \text{mask}\big) \cdot V
$$

其中 SiLU 激活函数替代了传统 Softmax，提供了更好的梯度特性。

## 算子输入与输出

### Dense（Normal）模式

| 名称 | 输入/输出 | 数据类型 | 形状 | 说明 |
|------|-----------|---------|------|------|
| q | 输入 | fp32/fp16/bf16 | `[B, S, N_q, D_q]` | B∈[1,2048], S∈[1,20480], N_q∈[1,16], D_q∈[1,512] |
| k | 输入 | fp32/fp16/bf16 | `[B, S×rank_size, N_k, D_q]` | K 序列长 = Q × rank_size；N_k∈[1,N_q] 且 N_q % N_k == 0，见 GQA 配置 |
| v | 输入 | fp32/fp16/bf16 | `[B, S×rank_size, N_k, D_v]` | D_v∈[16,512] 且 16 的倍数；N_k 为 K/V 头数，见 GQA 配置 |
| mask | 输入(可选) | fp32/fp16/bf16 | `[B, N, S, S]` | None 或全 1 表示不使用；传入时为自定义 mask（需配合 mask_type=3）；支持下三角 causal mask（mask_type=0）由算子内部自动生成 |
| attn_bias | 输入(可选) | fp32/fp16/bf16 | `[B, N, S, S]` | None 表示不使用 |
| rank_id | 属性 | int | 0 ~ 7 | 当前卡的 rank ID |
| rank_size | 属性 | int | 8 | 总卡数 |
| mask_type | 属性 | int | 0/2/3 | 0=下三角mask, 2=无mask, 3=自定义mask |
| max_seq_len | 属性 | int | [1, 20480] | Q 序列最大长度 |
| silu_scale | 属性 | float | 默认 1/max_seq_len | SiLU 缩放系数 |
| group | 属性 | string | HCCL 通信域名称 | HcclGetCommName 获取 |
| layout | 属性(可选) | string | "normal" | 数据格式 |
| seq_offsets | 属性(可选) | list_int | [] | jagged 模式时传入 |
| attn_output | 输出 | fp32/fp16/bf16 | `[B, S, N_q, D_v]` | 注意力输出 |

## GQA 支持

| 配置 | N_q | N_k | 说明 |
|------|-----|-----|------|
| MHA | 8 | 8 | 标准多头注意力 |
| GQA-4 | 8 | 2 | 每 4 个 Q 头共享 1 个 K/V 头 |
| GQA-2 | 8 | 4 | 每 2 个 Q 头共享 1 个 K/V 头 |
| MQA | 8 | 1 | 所有 Q 头共享 1 个 K/V 头 |

约束：`N_q % N_k == 0`

## 编译部署

### 算子编译（run.sh）

```bash
cd v220
bash run.sh
```

编译产物安装到 `$ASCEND_TOOLKIT_HOME/opp/vendors/hstu_dense_forward/`

### 测试编译与运行（test/build.sh）

```bash
cd v220/test

# 一键：8 卡、bs=31、seq=1024
bash build.sh

# 自定义参数
bash build.sh 8 31 2048

# 跳过 golden 数据生成（已有 golden 数据时）
bash build.sh 8 31 1024 skip

# 分步执行
python generate_golden.py --bs 31 --seq 1024    # Step 1: 生成 golden
torchrun --nproc_per_node=8 generate_golden.py --bs 31 --seq 1024  # 8 卡 golden
cmake . && make                                   # Step 2: 编译
./test_hstu_dense_forward 8 ./bin_file 31 1024   # Step 3: 运行
```

测试会在 80 次迭代后输出每卡的平均耗时和正确性比对结果。

## 注意事项

1. **必须 8 卡运行**：MC2 融合依赖 8 卡 All-to-All 通信拓扑，不支持少于 8 卡
2. **K/V 序列长度**：K/V 的 seq 维度必须是 Q 的 rank_size（8）倍
3. **先编译算子再编译测试**：test 依赖 run.sh 安装的 API 头文件，需先执行 `bash run.sh`
4. **Golden 数据**：由 `torchrun --nproc_per_node=8 generate_golden.py` 生成，每卡独立一份
5. **内存管理**：Golden 生成对大 seq_len（>8192）会分 chunk 计算以避免 OOM
