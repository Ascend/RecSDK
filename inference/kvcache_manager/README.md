# recsys_kvcache_manager_npu

Ascend 950 KVCache Manager —— 基于 CANN ACL 的 NPU + Host 双层分页 KV cache 管理器。

把 HSTU 等推理框架在 GPU 上的 paged KV cache 流程移植到 Ascend 950 上，实现
NPU 设备侧页分配/换入换出与 Host 侧存储/异步搬运的完整生命周期。

## 架构概览

```text
┌─────────────────────────────────────────────────────────────┐
│                      KVCacheManager                         │
│  顶层编排：lookup → allocate → onboard → offload → evict    │
├──────────────────────┬──────────────────────────────────────┤
│  NPUKVCacheManager   │  HostKVStorageManagerBase (ABC)      │
│  NPU 侧页分配器      │  ├─ NativeHostKVCacheManager        │
│  - paged cache table │  │   (CANN ACL 双 stream 异步搬运)  │
│  - LRU 页回收        │  └─ FlexKV... (后续)               │
│  - offload 页锁定    │                                    │
└──────────────────────┴──────────────────────────────────────┘
         │                           │
         │  C++ (pybind11)           │  C++ (pybind11)
         ▼                           ▼
  NPUKVCacheManagerImpl       HostKVStorageImpl
  (LRU / page table /         (pinned host mem /
   metadata buffer)            ACL async D2H/H2D /
                               gather/scatter kernel)
```

## 功能模块

### KVCacheManager（顶层编排）

协调 NPU 页分配器与 Host 存储后端，提供完整推理生命周期：

- `lookup_kvcache` — 同时查询 NPU 和 Host 侧缓存状态
- `allocate_kvcache` — 为新 token 分配 NPU 页
- `onboard_launch` / `onboard_wait` — 异步将 Host 侧 KV 搬入 NPU
- `offload_launch` / `offload_try_wait` — 异步将 NPU 侧 KV 搬至 Host
- `evict` / `evict_all` — 按 user_id 或全量驱逐
- `from_config` — 从 `KVCacheConfig` 一次性构建整个管理器

### NPUKVCacheManager（NPU 侧页分配器）

管理 NPU 上的 paged KV cache tensor 和页表：

- `lookup(uids)` — 返回各 user 在 NPU 上的缓存起止位置
- `allocate(uids, seq_lengths, lookup_results)` — 分配页、填充 metadata buffer
- `check_for_offload` / `acquire_offload_pages` / `release_offload_pages` — offload 流的页锁定/释放
- `revoke_onboard_pages` — onboard 失败时回退已分配页
- `put` / `get` — 调试用，直接写入/读出 cache tensor（内部调用 `append_kvcache` kernel）

### NativeHostKVCacheManager（Host 侧存储后端）

基于 CANN ACL RAII 工具的实现，双 stream 异步搬运：

- `lookup_kvcache` — 查询 Host 侧已缓存长度
- `onboard_kvcache_launch` / `onboard_kvcache_wait_by_layer` — 按 layer 粒度同步的 H2D 搬入
- `offload_kvcache_launch` / `offload_kvcache_wait` — D2H 搬出，支持超时检测
- `finish_task` / `cancel_task` — 完成/取消 offload 任务
- 内部使用 4 条 ACL stream（onload / offload / scatter / gather）

### C++ Kernels

| Kernel | 功能 |
|--------|------|
| `append_paged_kvcache` | 将新 token KV 写入 paged cache（AscendC 算子） |
| `gather_paged_kvcache` | 从 paged cache 按页索引 gather 到连续 buffer |
| `scatter_paged_kvcache` | 从连续 buffer scatter 到 paged cache |
| `get_paged_batch_indices_positions` | 计算批索引和位置 |

### KVCacheConfig

统一配置数据类，涵盖模型维度、页/块大小、容量、超时、策略等全部参数，
支持 `get_kvcache_config()` 工厂函数构造。

## 公开 API

```python
from recsys_kvcache_manager_npu import (
    KVCacheManager,          # 顶层编排
    NPUKVCacheManager,       # NPU 侧页分配器
    NativeHostKVCacheManager,# Host 侧存储后端
    HostKVStorageManagerBase,# Host 侧抽象基类
    KVCacheConfig,           # 配置数据类
    KVCacheOffloadMode,      # offload 模式枚举 (lazy / eager)
)
```

版本：`0.3.0`

## 目录结构

```text
recsys_kvcache_manager_npu/
├── README.md
├── CMakeLists.txt
├── setup.py
├── .gitattributes
├── recsys_kvcache_manager_npu/       # Python 包
│   ├── __init__.py
│   ├── kvcache_manager.py            # 顶层 KVCacheManager
│   ├── npu_kvcache_manager.py        # NPU 侧页分配器
│   ├── host_kvstorage_manager.py     # 抽象基类与公共类型
│   ├── native_host_kvcache_manager.py# 昇腾 native host 实现
│   ├── kvcache_config.py             # KVCacheConfig
│   ├── kvcache_metadata.py           # KVCacheMetadata
│   └── kvcache_utils.py              # KVIndexMeta / KVLookupResult
├── src/                               # C++ 扩展源码
│   ├── pybind.cpp                     # pybind11 绑定
│   ├── npu_kvcache_manager_impl.{h,cpp}
│   ├── native_host_kvcache_manager_impl.{h,cpp}
│   ├── kvcache_npu_utils.h            # ACL RAII 工具 (Stream/Event/HostMem)
│   ├── gather_scatter.cpp             # gather/scatter 入口
│   └── kernels/                       # AscendC 算子
│       ├── common/                    # 共享头文件 (dispatch_head_dim, fastdiv, vec_dtypes)
│       ├── append_paged_kvcache/
│       ├── gather_paged_kvcache/
│       ├── scatter_paged_kvcache/
│       └── get_paged_batch_indices_positions/
└── test/
    ├── test_native_host_unit.py       # NativeHostKVCacheManager 单元测试
    ├── test_npu_kvcache_manager.py    # NPUKVCacheManager 单元测试
    ├── test_kvcache_manager.py        # KVCacheManager 集成测试
    └── test_native.py                 # C++ 扩展直测
```

## 环境要求

- Linux（Ubuntu 22.04+ 推荐；Windows 仅做代码编辑）
- Python 3.9+
- `torch` 与 `torch_npu` 版本匹配（**必须先装好**，见下文）
- CANN Toolkit（`ASCEND_HOME` 指向安装路径）
- CMake ≥ 3.18，g++ ≥ 9，ninja-build，patchelf
- pybind11

## 编译前置：装好 torch + torch_npu

`setup.py` 依赖 `torch.utils.cpp_extension`，所以**当前 Python 环境必须已能
`import torch` 和 `import torch_npu`**，否则 `setup.py` 会立即抛
`ModuleNotFoundError`。

```bash
# 激活已装好 torch + torch_npu 的 conda 环境
conda activate <env-name>

# 验证
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
```

## 编译（在 Ascend 950 上）

WSL 没有 CANN/torch_npu，**只能做 Python 语法检查，不能编译 C++ 扩展**。

```bash
conda activate <env-name>

export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=$ASCEND_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$ASCEND_HOME/include:$CPATH

# 可选：指定 SoC 版本和编译线程数
export SOC_VERSION=Ascend950PR_9579   # 默认值
export MAX_COMPILE_THREADS=8          # 默认值

# 方式 A：pip 安装（推荐；--no-build-isolation 复用当前环境的 torch）
pip3 install --no-build-isolation .

# 方式 B：就地编译 .so（开发期增量编译更快）
python3 setup.py build_ext --inplace
```

构建产物：`kvcache_npu_cpp.cpython-3xx-*.so`（在包根目录）。

## 验证编译产物

```bash
python3 -c "import kvcache_npu_cpp; print('cpp ext ok')"
python3 -c "from recsys_kvcache_manager_npu import KVCacheManager; print('py wrapper ok')"
```

## 运行测试（仅在 Ascend 950 上）

```bash
# NativeHostKVCacheManager 单元测试
python test/test_native_host_unit.py

# NPUKVCacheManager 单元测试
pytest test/test_npu_kvcache_manager.py

# KVCacheManager 集成测试
pytest test/test_kvcache_manager.py

# C++ 扩展直测
python test/test_native.py
```

## 快速上手

```python
import torch
from recsys_kvcache_manager_npu import KVCacheManager, KVCacheConfig

# 方式 1：从配置构建（推荐）
config = KVCacheConfig(
    num_layers=3, num_heads=4, head_dim=128,
    page_size=32, offload_chunksize=128,
    num_primary_cache_pages=1024, num_buffer_pages=64,
    host_capacity_per_layer=1024 * 128 * 4 * 128 * 2,  # bytes
    max_batch_size=32, max_seq_len=4096,
    dtype=torch.bfloat16, device=0,
    offload_mode="lazy",
    host_kvstorage_fail_policy="fail_open",
)
mgr = KVCacheManager.from_config(config)

# 方式 2：手动组装
from recsys_kvcache_manager_npu import NPUKVCacheManager, NativeHostKVCacheManager
npu_mgr = NPUKVCacheManager(
    num_layers=3, num_heads=4, head_dim=128,
    num_tokens_per_page=32, num_tokens_per_chunk=128,
    num_primary_cache_pages=1024, num_buffer_pages=64,
    max_batch_size=32, max_sequence_length=4096,
    dtype=torch.bfloat16, device_idx=0,
)
host_mgr = NativeHostKVCacheManager(
    num_layers=3, num_heads=4, head_dim=128,
    num_tokens_per_page=32, num_tokens_per_chunk=128,
    bytes_capacity_per_layer=1024 * 128 * 4 * 128 * 2,
    max_batch_size=32, max_sequence_length=4096,
    dtype=torch.bfloat16, device_idx=0,
)
mgr = KVCacheManager(npu_mgr, host_mgr, offload_mode="lazy")
```

## 后续规划

- FlexKV 后端（后续）
- HSTU 端到端集成
