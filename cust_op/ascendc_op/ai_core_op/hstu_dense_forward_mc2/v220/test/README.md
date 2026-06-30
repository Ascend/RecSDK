# HSTU Dense Forward MC2 Test

MC2 通算融合算子（All-to-All + Attention）的 8 卡测试。

## 文件说明

| 文件 | 用途 |
|------|------|
| `build.sh` | 一键：生成 golden → 编译 → 运行 |
| `generate_golden.py` | `torchrun` 多进程生成 8 卡 golden 期望输出 |
| `test_hstu_dense_forward.cpp` | C++ 测试：HCCL 初始化 → All-to-All → 算子调用 → 精度对比 |
| `CMakeLists.txt` | 独立 CMake 构建配置 |

## 前置条件

- **8 卡 NPU 环境**，P2P 已启用
- 算子已通过 `v220/run.sh` 编译安装
- PyTorch + torchrun（用于 golden 数据生成）

```bash
conda activate fbgemm_npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 一键运行

```bash
cd /path/to/hstu_dense_forward_mc2/v220/test
bash build.sh [num_cards] [batch_size] [seq_len] [skip_golden]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_cards` | 8 | 卡数（1~8） |
| `batch_size` | 31 | batch size |
| `seq_len` | 1024 | sequence length（K/V seq = Q seq × `rankSize`） |
| `skip_golden` | 0 | 设为 `skip` 或 `1` 跳过 golden 生成（复用已有数据） |

执行流程：

1. **Golden 生成** — `torchrun --nproc_per_node=N generate_golden.py` 生成各卡期望输出
2. **编译** — `cmake . && make`
3. **运行** — 多线程启动 N 卡，每卡独立 HCCL 通信域 + 算子调用，输出精度和耗时

## 分步执行

```bash
# Step 1: 生成 golden
source ${ASCEND_TOOLKIT_HOME}/bin/setenv.bash
torchrun --nproc_per_node=8 generate_golden.py --bs 31 --seq 1024

# Step 2: 编译
cmake . && make

# Step 3: 运行
export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/opp/vendors/hstu_dense_forward/op_api/lib/:${LD_LIBRARY_PATH}
./test_hstu_dense_forward 8 ./bin_file 31 1024
```

## 性能数据

测试程序内部通过 `aclrtCreateEvent` / `aclrtEventElapsedTime` 在设备侧计时，warmup 20 次后计时 60 次取平均，运行后直接打印各卡平均耗时（单位 us）：

```text
[SUMMARY] Device 0: sync_ret=0, avg=10271.23 us
[SUMMARY] Device 1: sync_ret=0, avg=10271.45 us
...
avg=10271.77 us, min=10271.23 us, max=10272.22 us
```

## MC2 测试要点

- 测试通过 `HcclCommInitAll` 建立 8 卡 HCCL 通信域，每卡独立线程调用算子
- 算子入参包含 `rankId`、`rankSize`、`group`（HCCL 通信域名），由框架层在 Kernel 外部编排 All-to-All 交换 K/V
- K/V 的 seq_len = Q 的 seq_len × `rankSize`（All-to-All 后每卡持有全局 K/V）
- 输出与 `generate_golden.py` 生成的参考值对比，验证计算正确性
