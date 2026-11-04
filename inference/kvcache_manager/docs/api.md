# recsys_kvcache_manager API

> 版本：26.06

## 概述

recsys_kvcache_manager 是面向生成式推荐模型（如 HSTU）推理场景的 KV cache 管理器，以 user ID 为粒度管理 KV cache，将 KV 数据组织在 NPU 显存与主机锁页内存两级存储中，兼容 LLM 的 paged KV cache 格式。

业务通过 `KVCacheManager` 完成 KV cache 的完整生命周期：

1. `lookup_kvcache`：查询 user 在 NPU 与 Host 两侧已有的缓存。
2. `allocate_kvcache`：在 NPU 上分配本次推理所需的页，并生成页表与元数据。
3. `onboard_launch`：把 Host 上的历史 KV 异步搬入 NPU，按层等待后就绪。
4. 推理读写 NPU 上的 KV cache；新 token 的 K/V 通过 `append_kvcache` 算子写入。
5. `offload_launch` / `offload_try_wait`：把 NPU 数据异步搬回 Host。

本章仅列出业务可直接调用的对外接口与算子接口。C++ 扩展内部的实现类、页管理流程与 Host 搬运内部方法不属于对外接口，不在本章说明。

## API 分类概览

| 类别 | 接口 | 说明 |
| --- | --- | --- |
| 顶层编排 | `KVCacheManager` | 协调 `lookup → allocate → onboard → offload → evict` 全流程，业务入口。 |
| NPU 页分配器 | `NPUKVCacheManager` | 管理 NPU 侧 cache 表与页资源。 |
| Host 存储后端 | `NativeHostKVCacheManager` | 基于 CANN ACL 的 Host 存储与异步搬运。 |
| 配置与枚举 | `KVCacheConfig`、`KVCacheOffloadMode`、`HostKVStorageBackend`、`HostKVStorageFailPolicy` | 统一配置数据类与配置相关的枚举。 |
| 返回值结构 | `KVLookupResult`、`KVCacheMetadata`、`HostKVTaskHandle` | 对外接口的返回值类型，字段供业务读取。 |
| 自定义算子 | `append_kvcache`、`gather_paged_kvcache`、`scatter_paged_kvcache`、`get_paged_batch_indices_positions` | AscendC 实现的 paged KV 读写算子。 |

## 接口调用约定

### 导入路径

```python
import torch  # 必须先于 kvcache_npu_cpp 导入
from recsys_kvcache_manager import (
    KVCacheManager,
    NPUKVCacheManager,
    NativeHostKVCacheManager,
    KVCacheConfig,
    KVCacheOffloadMode,
)
```

`kvcache_npu_cpp` 链接了 torch 的符号，导入前必须先 `import torch`，否则会因符号未解析导致导入失败。

`HostKVStorageBackend` 与 `HostKVStorageFailPolicy` 不在包根 `__all__` 中，需从 `recsys_kvcache_manager.kvcache_config` 导入；`KVCacheConfig` 中对应字段也接受其字符串取值（如 `"native_host"`、`"fail_open"`）。

### 张量与设备约定

| 数据 | 约定 |
| --- | --- |
| `user_ids` / `uids` | CPU 上的 `torch.int64` 一维连续张量。 |
| `sequence_lengths` 等长度参数 | CPU 上的 `torch.int32` 或 `torch.int64` 一维连续张量；dtype 在 `allocate_kvcache` 中校验。 |
| 元数据张量（`kv_indices`、`kv_indptr`、`batch_indices` 等） | NPU 上的 `torch.int32` 一维连续张量。 |
| KV 数据张量 | NPU 上的 2 字节浮点类型。算子按 2 字节元素宽度访问。 |

## KVCacheManager

### \_\_init\_\_

**功能描述**

构造顶层管理器，并把 NPU cache 表注册到 Host 后端。一般通过 `KVCacheManager.from_config()` 创建，无需直接调用本接口。

**函数原型**

```python
def __init__(
    self,
    npu_kvcache_manager: NPUKVCacheManager,
    host_kvstorage_manager: HostKVStorageManagerBase,
    offload_mode: str = KVCacheOffloadMode.LAZY.value,
    host_kvstorage_fail_policy: str = HostKVStorageFailPolicy.FAIL_OPEN.value,
):
```

**参数说明**

| 参数名 | 类型 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| npu_kvcache_manager | NPUKVCacheManager | 必选 | NPU 侧页分配器实例。 |
| host_kvstorage_manager | HostKVStorageManagerBase | 必选 | Host 侧存储后端实例。 |
| offload_mode | str | 可选 | offload 触发模式，取值见 [KVCacheOffloadMode](#kvcacheoffloadmode)，默认 `"lazy"`。 |
| host_kvstorage_fail_policy | str | 可选 | 搬运失败策略，取值为 `"fail_open"` 或 `"fail_close"`，默认 `"fail_open"`。 |

**返回值说明**

- 成功：返回 `KVCacheManager` 实例。
- 失败：`offload_mode` 或 `host_kvstorage_fail_policy` 非法时抛出 `ValueError`。

### from_config

**功能描述**

根据 `KVCacheConfig` 一次性构建 NPU 页分配器与 Host 存储后端，并返回顶层管理器。构建过程中会校验配置并计算对齐后的最大序列长度。

**函数原型**

```python
@classmethod
def from_config(cls, kvcache_config: KVCacheConfig) -> "KVCacheManager":
```

**参数说明**

| 参数名 | 类型 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| kvcache_config | KVCacheConfig | 必选 | 组件配置，字段见 [KVCacheConfig](#kvcacheconfig)。 |

**返回值说明**

- 成功：返回 `KVCacheManager` 实例。
- 失败：`offload_chunksize` 不是 `page_size` 的整数倍、`offload_mode` 或 `host_kvstorage_fail_policy` 非法时抛出 `ValueError`；`host_kvstorage_backend` 不受支持时抛出 `NotImplementedError`。

**使用示例**

```python
import torch
from recsys_kvcache_manager import KVCacheConfig, KVCacheManager

config = KVCacheConfig(
    num_layers=3, num_heads=4, head_dim=128,
    page_size=32, offload_chunksize=128,
    num_primary_cache_pages=1024, num_buffer_pages=64,
    host_capacity_per_layer=1024 * 128 * 4 * 128 * 2,  # bytes
    max_batch_size=32, max_seq_len=4096,
    dtype=torch.bfloat16, device=0,
)
mgr = KVCacheManager.from_config(config)
```

> [!NOTE]
>
> `max_seq_len` 在内部按 `page_size` 向上取整；NPU 与 Host 使用同一个对齐后的长度，因此配置值不必是 `page_size` 的整数倍。

### lookup_kvcache

**功能描述**

查询当前 batch 中各 user 已有的 KV cache。同时查询 NPU 与 Host，并合并两侧结果。

**函数原型**

```python
def lookup_kvcache(
    self,
    user_ids: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> Tuple[KVIndexMeta, KVLookupResult]:
```

**参数说明**

| 参数名 | 类型 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| user_ids | torch.Tensor | 必选 | `int64` 一维张量，元素个数不超过 `max_batch_size`。 |
| sequence_lengths | torch.Tensor | 必选 | `int32`/`int64` 一维张量，与 `user_ids` 等长，表示各 user 当前的历史序列长度（含本次待写入的 token）。 |

**返回值说明**

- 成功：返回二元组 `(index_meta, lookup_result)`。`index_meta` 为 `KVIndexMeta`，包含本次的 `user_ids` 与 `sequence_lengths`，供后续接口原样传递；`lookup_result` 为合并后的查询结果，字段见 [KVLookupResult](#kvlookupresult)。
- 失败：缓存状态不满足合并规则时抛出 `ValueError`；`user_ids` 不是 `int64` 一维连续张量时，NPU 侧校验抛出 `RuntimeError`。

**使用示例**

```python
user_ids = torch.tensor([0, 1], dtype=torch.int64)
seq_lengths = torch.tensor([512, 256], dtype=torch.int32)
index_meta, lookup = mgr.lookup_kvcache(user_ids, seq_lengths)
```

### allocate_kvcache

**功能描述**

为当前 batch 在 NPU 上分配页，并填充本次推理所需的页表与元数据。新分配 token 数由 `sequence_lengths - cached_lengths` 得出，命中缓存的部分不会再分配新页。

**函数原型**

```python
def allocate_kvcache(
    self,
    index_meta: KVIndexMeta,
    lookup_results: KVLookupResult,
    output_kvcache_metadata: Optional[KVCacheMetadata] = None,
) -> KVCacheMetadata:
```

**参数说明**

| 参数名 | 类型 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| index_meta | KVIndexMeta | 必选 | `lookup_kvcache` 返回的索引元数据。 |
| lookup_results | KVLookupResult | 必选 | `lookup_kvcache` 返回的查询结果。 |
| output_kvcache_metadata | KVCacheMetadata | 可选 | 复用的元数据缓冲区。为 None 时按 batch 内实际 token/page 数新建。 |

**返回值说明**

- 成功：返回 `KVCacheMetadata`，字段见 [KVCacheMetadata](#kvcachemetadata)。
- 失败：`sequence_lengths` 不是 `int32`/`int64` 一维连续张量、序列长度超限、缓冲区容量不足、缓存状态不一致或页资源无法回收时抛出 `RuntimeError`。

**使用示例**

```python
metadata = mgr.allocate_kvcache(index_meta, lookup)

# 模型侧按层取用 cache 表：[num_pages, 2, page_size, num_heads, head_dim]
for layer_idx in range(3):
    k_cache, v_cache = metadata.kv_cache_table[layer_idx].unbind(dim=1)
```

### onboard_launch

**功能描述**

启动异步 onboard（Host → NPU）：计算每个 user 需要从 Host 搬入的 token 区间，下发搬运与写入，并把任务句柄写入 `kvcache_metadata.kv_onload_handle`。

**函数原型**

```python
def onboard_launch(
    self,
    index_meta: KVIndexMeta,
    lookup_result: KVLookupResult,
    kvcache_metadata: KVCacheMetadata,
) -> HostKVTaskHandle:
```

**参数说明**

| 参数名 | 类型 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| index_meta | KVIndexMeta | 必选 | 本 batch 的索引元数据。 |
| lookup_result | KVLookupResult | 必选 | 本 batch 的查询结果，用于计算 onboard 区间。 |
| kvcache_metadata | KVCacheMetadata | 必选 | `allocate_kvcache` 返回的元数据，提供本次分配的页索引。 |

**返回值说明**

- 成功：返回 `HostKVTaskHandle`。本 batch 无需 onboard 时，句柄的 `status` 为 `SKIPPED`。
- 失败：参数不满足约束时抛出异常。

**使用示例**

```python
index_meta, lookup = mgr.lookup_kvcache(user_ids, seq_lengths)
metadata = mgr.allocate_kvcache(index_meta, lookup)
handle = mgr.onboard_launch(index_meta, lookup, metadata)

# 推理逐层使用时按层等待 onboard 完成
for layer_idx in range(3):
    metadata.kv_onload_handle.stream_wait_layer(layer_idx)
    # ... 本层注意力计算 ...
```

### offload_launch

**功能描述**

启动异步 offload（NPU → Host）：筛选出 NPU 上存在未 offload 数据的 user，锁定其待搬运的页后下发搬运。返回的句柄同时记录在 `ongoing_offload_tasks` 中，供 `offload_try_wait` 轮询。

当前 offload 由调用方按需触发（`lazy` 语义），组件不会在推理后自动 offload。

**函数原型**

```python
def offload_launch(
    self,
    index_meta: KVIndexMeta,
    kvcache_metadata: Optional[KVCacheMetadata] = None,
) -> Optional[HostKVTaskHandle]:
```

**参数说明**

| 参数名 | 类型 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| index_meta | KVIndexMeta | 必选 | 本 batch 的索引元数据。 |
| kvcache_metadata | KVCacheMetadata | 可选 | 本 batch 的元数据。 |

**返回值说明**

- 成功：返回 `HostKVTaskHandle`，`status` 为 `LAUNCHED`。
- 无需 offload 或 Host 侧拒绝搬运（如空间不足）：返回 `None`。
- 失败：抛出异常。

**使用示例**

```python
task_handle = mgr.offload_launch(index_meta)
if task_handle is not None:
    while mgr.ongoing_offload_tasks:
        mgr.offload_try_wait()
```

### offload_try_wait

**功能描述**

轮询所有进行中的 offload 任务，不阻塞。已完成的任务落盘并释放页锁；被 Host 侧跳过或失败的任务释放页锁，失败时按策略处理；未完成的任务保留到下次轮询。

**函数原型**

```python
def offload_try_wait(self) -> None:
```

**返回值说明**

- 成功：无返回值。
- 失败：`fail_close` 策略下搬运失败时抛出 `RuntimeError`；出现未知状态时抛出 `RuntimeError`。

### evict

**功能描述**

按 user ID 回收 KV cache，可分别指定回收 NPU 侧或 Host 侧。

**函数原型**

```python
def evict(
    self,
    user_ids: torch.Tensor,
    for_npu: bool = False,
    for_host: bool = False,
) -> None:
```

**参数说明**

| 参数名 | 类型 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| user_ids | torch.Tensor | 必选 | CPU `int64` 一维张量。 |
| for_npu | bool | 可选 | 是否回收 NPU 侧缓存，默认 False。 |
| for_host | bool | 可选 | 是否回收 Host 侧缓存，默认 False。 |

**返回值说明**

- 成功：无返回值。
- 失败：`for_npu` 与 `for_host` 同时为 False 时抛出 `ValueError`。

**使用示例**

```python
mgr.evict(torch.tensor([0, 1], dtype=torch.int64), for_npu=True, for_host=True)
```

### evict_all

**功能描述**

回收全部 user 的 KV cache，可分别指定 NPU 侧或 Host 侧。

**函数原型**

```python
def evict_all(self, for_npu: bool = False, for_host: bool = False) -> None:
```

**参数说明**

| 参数名 | 类型 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| for_npu | bool | 可选 | 是否清空 NPU 侧缓存，默认 False。 |
| for_host | bool | 可选 | 是否清空 Host 侧缓存，默认 False。 |

**返回值说明**

- 成功：无返回值。
- 失败：`for_npu` 与 `for_host` 同时为 False 时抛出 `ValueError`。

## NPUKVCacheManager

NPU 侧页分配器。构造时申请 `[num_layers, num_primary_cache_pages, 2, page_size, num_heads, head_dim]` 的 NPU 显存并切成逐层视图。业务一般通过 `KVCacheManager.from_config()` 创建，仅在需要自行组装管理器时直接实例化。

### \_\_init\_\_

**功能描述**

申请 NPU 侧 KV cache 显存。

**函数原型**

```python
def __init__(
    self,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    num_tokens_per_page: int,
    num_tokens_per_chunk: int,
    num_primary_cache_pages: int,
    num_buffer_pages: int,
    max_batch_size: int,
    max_sequence_length: int,
    dtype: torch.dtype,
    device_idx: int,
):
```

**参数说明**

| 参数名 | 类型 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| num_layers | int | 必选 | 模型层数。 |
| num_heads | int | 必选 | KV head 数。 |
| head_dim | int | 必选 | 每个 head 的维度。算子支持 64 / 128 / 256 / 512。 |
| num_tokens_per_page | int | 必选 | 每页 token 数（`page_size`）。 |
| num_tokens_per_chunk | int | 必选 | 一次 onboard / offload 搬运的最小 token 数（`offload_chunksize`），须为 `page_size` 的整数倍。 |
| num_primary_cache_pages | int | 必选 | NPU 主缓存页数。 |
| num_buffer_pages | int | 必选 | 用于 offload 的缓冲页数。 |
| max_batch_size | int | 必选 | 单次可处理的 user 数上限。 |
| max_sequence_length | int | 必选 | 单 user 序列长度上限，已按 `page_size` 对齐。 |
| dtype | torch.dtype | 必选 | KV 数据类型，算子要求为 2 字节类型（推荐 `torch.bfloat16`）。 |
| device_idx | int | 必选 | NPU 设备号。 |

**返回值说明**

- 成功：返回 `NPUKVCacheManager` 实例。
- 失败：显存申请或初始化失败时抛出异常。

**使用示例**

```python
from recsys_kvcache_manager import NPUKVCacheManager

npu_mgr = NPUKVCacheManager(
    num_layers=3, num_heads=4, head_dim=128,
    num_tokens_per_page=32, num_tokens_per_chunk=128,
    num_primary_cache_pages=1024, num_buffer_pages=64,
    max_batch_size=32, max_sequence_length=4096,
    dtype=torch.bfloat16, device_idx=0,
)
```

页回收不在此类上单独调用，统一由 `KVCacheManager.evict()` / `evict_all()` 调度。

## NativeHostKVCacheManager

基于 CANN ACL 的 Host 存储后端：使用主机锁页内存保存历史 KV，异步完成 Host 与 NPU 之间的搬运。业务一般通过 `KVCacheManager.from_config()` 创建。

### \_\_init\_\_

**功能描述**

创建 Host 侧 pinned 内存存储与异步搬运资源。

**函数原型**

```python
def __init__(
    self,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    num_tokens_per_page: int,
    num_tokens_per_chunk: int,
    bytes_capacity_per_layer: int,
    max_batch_size: int,
    max_sequence_length: int,
    onload_timeout_ms: float = 0.0,
    offload_timeout_ms: float = 0.0,
    dtype: torch.dtype = torch.bfloat16,
    device_idx: int = 0,
):
```

**参数说明**

| 参数名 | 类型 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| num_layers | int | 必选 | 模型层数。 |
| num_heads | int | 必选 | KV head 数。 |
| head_dim | int | 必选 | 每个 head 的维度。 |
| num_tokens_per_page | int | 必选 | 每页 token 数。 |
| num_tokens_per_chunk | int | 必选 | 存储与搬运的最小 token 块（`offload_chunksize`）。 |
| bytes_capacity_per_layer | int | 必选 | 单层 Host 存储容量，单位字节。内部按 chunk 大小（`num_tokens_per_chunk * 2 * num_heads * head_dim * 元素字节数`）向下取整；建议配置为 chunk 大小的整数倍以用满容量。 |
| max_batch_size | int | 必选 | 单次搬运的 user 数上限。 |
| max_sequence_length | int | 必选 | 单 user 序列长度上限，已按 `page_size` 对齐。 |
| onload_timeout_ms | float | 可选 | onboard 超时阈值。**当前未实现**：批级 onboard 等待不存在，该参数不参与任何逻辑。默认 0.0。 |
| offload_timeout_ms | float | 可选 | offload 超时阈值，单位毫秒；大于 0 时，轮询超过该时长返回超时状态。默认 0.0。 |
| dtype | torch.dtype | 可选 | KV 数据类型，默认 `torch.bfloat16`。 |
| device_idx | int | 可选 | NPU 设备号，默认 0。 |

**返回值说明**

- 成功：返回 `NativeHostKVCacheManager` 实例。
- 失败：pinned 内存申请或 ACL 资源创建失败时抛出异常。

**使用示例**

```python
from recsys_kvcache_manager import NativeHostKVCacheManager

host_mgr = NativeHostKVCacheManager(
    num_layers=3, num_heads=4, head_dim=128,
    num_tokens_per_page=32, num_tokens_per_chunk=128,
    bytes_capacity_per_layer=1024 * 2 * 32 * 4 * 128 * 2,
    max_batch_size=32, max_sequence_length=4096,
    offload_timeout_ms=100.0, dtype=torch.bfloat16, device_idx=0,
)
```

存储回收不在此类上单独调用，统一由 `KVCacheManager.evict()` / `evict_all()` 调度。

## KVCacheConfig

组件统一配置数据类。

**函数原型**

```python
@dataclass
class KVCacheConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    page_size: int
    offload_chunksize: int
    num_primary_cache_pages: int
    num_buffer_pages: int
    host_capacity_per_layer: int
    max_batch_size: int
    max_seq_len: int
    dtype: torch.dtype
    device: int
    host_kvstorage_backend: str = "native_host"
    onload_timeout_ms: float = 0.0
    offload_timeout_ms: float = 1000.0
    offload_mode: str = "lazy"
    host_kvstorage_fail_policy: str = "fail_open"
    extra_configs: Dict[str, Any] = field(default_factory=dict)
```

**参数说明**

| 参数名 | 类型 | 可选/必选 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| num_layers | int | 必选 | - | 模型层数。 |
| num_heads | int | 必选 | - | KV head 数。 |
| head_dim | int | 必选 | - | 每个 head 的维度，算子支持 64 / 128 / 256 / 512。 |
| page_size | int | 必选 | - | 每页 token 数。 |
| offload_chunksize | int | 必选 | - | 一次搬运的最小 token 数，须为 `page_size` 的整数倍。 |
| num_primary_cache_pages | int | 必选 | - | NPU 主缓存页数。 |
| num_buffer_pages | int | 必选 | - | NPU offload 缓冲页数。 |
| host_capacity_per_layer | int | 必选 | - | 单层 Host 存储容量，单位字节。 |
| max_batch_size | int | 必选 | - | 单次处理的 user 数上限。 |
| max_seq_len | int | 必选 | - | 单 user 序列长度上限，内部按 `page_size` 向上取整。 |
| dtype | torch.dtype | 必选 | - | KV 数据类型，算子要求为 2 字节类型（如 `torch.bfloat16`）。 |
| device | int | 必选 | - | NPU 设备号。 |
| host_kvstorage_backend | str | 可选 | `"native_host"` | Host 后端类型，当前仅支持 `"native_host"`。 |
| onload_timeout_ms | float | 可选 | 0.0 | onboard 超时阈值。**当前未实现**：批级 onboard 等待不存在，该字段不参与任何逻辑。 |
| offload_timeout_ms | float | 可选 | 1000.0 | offload 超时阈值，单位毫秒。 |
| offload_mode | str | 可选 | `"lazy"` | offload 触发模式，见 [KVCacheOffloadMode](#kvcacheoffloadmode)。 |
| host_kvstorage_fail_policy | str | 可选 | `"fail_open"` | 搬运失败策略，见 [HostKVStorageFailPolicy](#hostkvstoragefailpolicy)。 |
| extra_configs | Dict[str, Any] | 可选 | `{}` | 预留扩展字段。**当前未实现**：不参与任何逻辑。 |

**使用示例**

```python
config = KVCacheConfig(
    num_layers=3, num_heads=4, head_dim=128,
    page_size=32, offload_chunksize=128,
    num_primary_cache_pages=1024, num_buffer_pages=64,
    host_capacity_per_layer=1024 * 128 * 4 * 128 * 2,
    max_batch_size=32, max_seq_len=4096,
    dtype=torch.bfloat16, device=0,
    offload_mode="lazy", host_kvstorage_fail_policy="fail_open",
)
```

## HostKVStorageBackend

Host 后端类型枚举（`str, Enum`）。

| 取值 | 说明 |
| --- | --- |
| `HostKVStorageBackend.NATIVE_HOST` = `"native_host"` | 基于 CANN ACL 的原生 Host 存储后端。当前唯一支持的后端。 |

## HostKVStorageFailPolicy

搬运失败策略枚举（`str, Enum`）。

| 取值 | 说明 |
| --- | --- |
| `HostKVStorageFailPolicy.FAIL_OPEN` = `"fail_open"` | 搬运失败时回退已分配的页（取消任务并释放页锁），推理继续。 |
| `HostKVStorageFailPolicy.FAIL_CLOSE` = `"fail_close"` | 搬运失败时回退页并抛出 `RuntimeError`。 |

## KVCacheOffloadMode

offload 触发模式枚举。

**函数原型**

```python
class KVCacheOffloadMode(Enum):
    LAZY = "lazy"
    EAGER = "eager"  # 当前未实现
```

| 取值 | 说明 |
| --- | --- |
| `KVCacheOffloadMode.LAZY` = `"lazy"` | 按需 offload：由调用方在合适时机显式调用 `offload_launch()`。 |
| `KVCacheOffloadMode.EAGER` = `"eager"` | 推理后立即 offload。**当前未实现**：仅做合法性校验，不会自动触发 offload，实际行为与 `lazy` 一致。 |

> [!NOTE]
>
> 传入未定义的字符串取值时，`KVCacheManager.__init__()` 抛出 `ValueError`。

**使用示例**

```python
from recsys_kvcache_manager import KVCacheOffloadMode

config.offload_mode = KVCacheOffloadMode.LAZY.value
```

## 返回值结构

### KVLookupResult

缓存查询结果，包含合并结果与 NPU / Host 两侧的原始结果。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| user_ids | torch.Tensor | 本 batch 的 user ID。 |
| cached_start_indices | torch.Tensor | 合并后可复用的起始位置，`int32`。 |
| cached_lengths | torch.Tensor | 合并后可复用的 token 数，`int32`。 |
| npu_cached_start_indices | torch.Tensor | NPU 侧缓存起点，`int32`。 |
| npu_cached_lengths | torch.Tensor | NPU 侧缓存长度，`int32`。 |
| host_cached_start_indices | torch.Tensor | Host 侧缓存起点，`int32`。 |
| host_cached_lengths | torch.Tensor | Host 侧缓存长度，`int32`。 |
| extra | Dict[str, Any] | 预留扩展字段。**当前未实现**：不参与任何逻辑。 |

### KVCacheMetadata

单 batch 的元数据，`allocate_kvcache` 的返回值，供 onboard 与写入算子使用。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| kv_indices | torch.Tensor | NPU `int32`，本 batch 的页索引表。 |
| kv_indptr | torch.Tensor | NPU `int32`，各序列页索引的偏移，长度 `batch_size + 1`。 |
| kv_last_page_len | torch.Tensor | NPU `int32`，各序列最后一页的有效 token 数。 |
| total_history_lengths | torch.Tensor | NPU `int32`，各序列历史总长度。 |
| new_history_offsets | torch.Tensor | NPU `int32`，新 token 前缀和，长度 `batch_size + 1`。 |
| batch_indices | torch.Tensor | NPU `int32`，每个新 token 所属的序列序号，长度 `new_history_nnz`。 |
| position | torch.Tensor | NPU `int32`，每个新 token 在序列中的位置，长度 `new_history_nnz`。 |
| new_history_nnz | int | 新 token 总数（host 侧）。 |
| new_history_nnz_npu | torch.Tensor | NPU `int32`，新 token 总数（device 侧），写入算子读取该值。 |
| kv_cache_table | List[torch.Tensor] | NPU 各层 cache 表视图，形状 `[num_pages, 2, page_size, num_heads, head_dim]`。 |
| kv_onload_handle | HostKVTaskHandle | onboard 句柄，`onboard_launch` 时写入。 |
| max_seqlen | int | 本 batch 最大序列长度。 |

### HostKVTaskHandle

搬运任务句柄。

| 字段/方法 | 类型 | 说明 |
| --- | --- | --- |
| user_ids | torch.Tensor | 本任务涉及的 user；`SKIPPED` 句柄为 `None`。 |
| status | HostKVTaskStatus | 任务状态，取值包括 `UNINITIALIZED`、`SKIPPED`、`LAUNCHED`、`READY`、`EVENT_READY`、`TIMEOUT`、`FAILED`、`CANCELLED`。 |
| metadata | Dict[str, Any] | 任务附加信息。onboard 任务为 dict，包含 `onboard_start_indices`、`onboard_lengths`；offload 任务为 `None`。 |
| is_layerwise | bool | 是否按层同步：onboard 任务为 `True`，offload 任务为 `False`。 |
| stream_wait_layer(layer_idx) | 方法 | 按层等待。仅在 `is_layerwise` 为 True 时生效，否则为空操作。 |

## 自定义算子

组件包含 4 个 AscendC 算子，用于完成 paged KV cache 的写入、按页聚集与散布，以及 token 位置推导。下文的 C++ 启动封装均声明在 `kvcache` 命名空间（`src/gather_scatter.h`）。

**表 1** 算子列表

| 算子 | 接口形式 | 功能介绍 |
| --- | --- | --- |
| `append_kvcache` | Python（`kvcache_npu_cpp` 扩展） | 把新计算的 K/V 按 token 位置写入 paged cache 表。 |
| `gather_paged_kvcache` | C++（`src/gather_scatter.h`） | 按页索引把 paged cache 中的页聚集到连续缓冲区。 |
| `scatter_paged_kvcache` | C++（`src/gather_scatter.h`） | 把连续缓冲区中的页散布写回 paged cache。 |
| `get_paged_batch_indices_positions` | C++（`src/gather_scatter.h`） | 由 `append_indptr` 与序列长度推导每个新 token 的 batch 序号与序列内位置。 |

> [!NOTICE]
>
> 1. 算子为高性能计算接口，调用方需自行保证参数类型、形状、取值满足约束；算子内部不对页索引做边界检查，越界会导致数据损坏或执行失败。
> 2. `gather_paged_kvcache` / `scatter_paged_kvcache` / `get_paged_batch_indices_positions` 未在 Python 侧绑定，仅提供 C++ 启动封装，由组件内部调用。
> 3. 所有算子均为异步下发，调用返回不代表执行完成。

### append_kvcache

**功能描述**

把新计算的 K/V 写入 paged cache 表。对每个新 token `i`：

1. 由 `positions[i]` 与 `page_size` 计算目标页序号 `page_iter` 与页内偏移 `entry_idx`。
2. 目标页由 `kv_indices[kv_indptr[batch_indices[i]] + page_iter]` 给出。
3. 写入位置为 `page * stride_page + head_idx * stride_h + entry_idx * stride_n + lane`。
4. 源数据位于 `append_key[(i + seqlen_offsets[batch_indices[i]]) * append_k_stride_n + head_idx * append_k_stride_h + lane]`，V 同理。

**函数原型**

```python
kvcache_npu_cpp.append_kvcache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    seqlen_offsets: torch.Tensor,
    nnz_npu: torch.Tensor,
    nnz: int,
    paged_k_cache: torch.Tensor,
    paged_v_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    kv_layout: int,
) -> None
```

**参数说明**

| 参数名 | 类型 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| append_key | torch.Tensor | 必选 | NPU 2 字节类型，3 维，形状 `[nnz, num_heads, head_dim]`，第 1、2 维须与 cache 表一致。不要求整体连续，按 `stride(0)` / `stride(1)` 访问；最后一维须连续。 |
| append_value | torch.Tensor | 必选 | NPU 2 字节类型，形状须与 `append_key` 完全一致。 |
| batch_indices | torch.Tensor | 必选 | NPU `int32` 一维连续张量，长度 `nnz`，每个新 token 所属的序列序号（取值小于 batch 大小）。 |
| positions | torch.Tensor | 必选 | NPU `int32` 一维连续张量，长度 `nnz`，每个新 token 在序列中的位置。 |
| seqlen_offsets | torch.Tensor | 必选 | NPU `int32` 一维连续张量，长度等于 batch 大小，各序列在 `append_key` 中的起始偏移。 |
| nnz_npu | torch.Tensor | 必选 | NPU `int32` 一维连续张量，第 0 个元素为新 token 总数；算子以该值为准。 |
| nnz | int | 必选 | host 侧新 token 总数。为兼容既有调用签名保留，当前实现不读取。 |
| paged_k_cache | torch.Tensor | 必选 | NPU 2 字节类型，4 维。`kv_layout=0` 时形状 `[num_pages, page_size, num_heads, head_dim]`；`kv_layout=1` 时 `[num_pages, num_heads, page_size, head_dim]`。`num_heads`、`head_dim`、`page_size` 由该张量推导；可以是 `unbind()` 产生的非连续视图，但页内须为对应布局的稠密存储。 |
| paged_v_cache | torch.Tensor | 必选 | NPU 2 字节类型，形状约束同上。 |
| kv_indices | torch.Tensor | 必选 | NPU `int32` 一维连续张量，页索引表，元素为合法页号。 |
| kv_indptr | torch.Tensor | 必选 | NPU `int32` 一维连续张量，各序列在 `kv_indices` 中的偏移，长度 `batch_size + 1`。 |
| kv_last_page_len | torch.Tensor | 必选 | NPU `int32` 一维连续张量，各序列最后一页的有效 token 数。为接口一致性保留，当前实现不读取。 |
| kv_layout | int | 必选 | cache 表的 K/V 排布：0 表示 NHD，1 表示 HND。其他取值抛异常。 |

**返回值说明**

- 成功：返回 None。算子异步执行，完成前数据不可见。
- 失败：设备、dtype、维度、`kv_layout` 不满足约束时抛出 `RuntimeError`；`head_dim` 不在 {64, 128, 256, 512} 或启动线程数超过 1024 时，主机侧校验抛出 `RuntimeError`。

**使用示例**

```python
import torch
import kvcache_npu_cpp

k = torch.randn(8, 4, 128, dtype=torch.bfloat16, device="npu:0")   # [nnz, heads, dim]
v = torch.randn_like(k)
paged_k, paged_v = metadata.kv_cache_table[0].unbind(dim=1)       # [num_pages, page_size, heads, dim]
kvcache_npu_cpp.append_kvcache(
    k, v,
    metadata.batch_indices,
    metadata.position,
    torch.zeros(2, dtype=torch.int32, device="npu:0"),  # seqlen_offsets
    metadata.new_history_nnz_npu,
    metadata.new_history_nnz,
    paged_k, paged_v,
    metadata.kv_indices,
    metadata.kv_indptr,
    metadata.kv_last_page_len,
    0,  # NHD
)
```

### gather_paged_kvcache

**功能描述**

按页索引把 paged cache 中的页聚集到连续缓冲区，K 与 V 分别写到连续缓冲区的两个平面。对每个 token `i`：

- `page_idx = i / page_size`，`entry_idx = i % page_size`。
- 源偏移：`page_ids[page_idx] * stride_page + head_idx * stride_h + entry_idx * stride_n + lane`。
- 目的偏移：`page_idx * stride_page + head_idx * stride_h + entry_idx * stride_n + lane`。

V 平面位于 `kv_cache + stride_k2v` / `continuous_kv + stride_k2v`。

**函数原型**

```cpp
void GatherPagedKVCache(uint16_t* kv_cache,           // 输入：paged cache
                        int* page_ids,                // 输入：页索引
                        uint32_t num_heads,
                        uint32_t head_dim,
                        uint32_t page_size,
                        uint32_t stride_page,
                        uint32_t stride_k2v,
                        uint32_t stride_n,
                        uint32_t stride_h,
                        uint32_t num_pages,
                        uint16_t* continuous_kv,      // 输出：连续缓冲区
                        uint32_t num_ctas,
                        aclrtStream stream);
```

**参数说明**

| 参数名 | 类型 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| kv_cache | uint16_t* | 必选 | 设备侧 paged cache 基址。`uint16_t*` 仅表示 2 字节存储视图，实际为 bf16/fp16 数据。 |
| page_ids | int* | 必选 | 设备侧 `int32` 页索引数组，长度 `num_pages`。页号必须合法，算子不做边界检查。 |
| num_heads | uint32_t | 必选 | KV head 数，参与启动几何校验；kernel 本身不读取。 |
| head_dim | uint32_t | 必选 | head 维度，取值须为 64 / 128 / 256 / 512。 |
| page_size | uint32_t | 必选 | 每页 token 数。 |
| stride_page | uint32_t | 必选 | 相邻页之间的元素步长。 |
| stride_k2v | uint32_t | 必选 | 同页内 K 平面到 V 平面的元素步长（通常为 `page_size * num_heads * head_dim`）。 |
| stride_n | uint32_t | 必选 | 相邻 token 之间的元素步长。 |
| stride_h | uint32_t | 必选 | 相邻 head 之间的元素步长。 |
| num_pages | uint32_t | 必选 | 本次聚集的页数，处理的 token 总数为 `num_pages * page_size`。 |
| continuous_kv | uint16_t* | 必选 | 设备侧连续输出缓冲区基址，布局与 paged cache 的页内布局一致。 |
| num_ctas | uint32_t | 必选 | 启动的 AIV 核数，通常取 `PlatformAscendCManager::GetCoreNumAiv()`。 |
| stream | aclrtStream | 必选 | 下发流的句柄。 |

**返回值说明**

- 成功：无返回值。算子异步执行，需在 `stream` 上同步或插入事件后再读取结果。
- 失败：`head_dim` 非法或启动线程数超过 1024 时抛出 `std::runtime_error`。

### scatter_paged_kvcache

**功能描述**

把连续缓冲区中的页散布写回 paged cache，是 `gather_paged_kvcache` 的逆操作。页内 token 映射与偏移公式与 gather 相同，方向相反。

**函数原型**

```cpp
void ScatterPagedKVCache(uint16_t* continuous_kv,     // 输入：连续缓冲区
                         int* page_ids,                // 输入：页索引
                         uint32_t num_heads,
                         uint32_t head_dim,
                         uint32_t page_size,
                         uint32_t stride_page,
                         uint32_t stride_k2v,
                         uint32_t stride_n,
                         uint32_t stride_h,
                         uint32_t num_pages,
                         uint16_t* kv_cache,           // 输出：paged cache
                         uint32_t num_ctas,
                         aclrtStream stream);
```

**参数说明**

参数含义与 [gather_paged_kvcache](#gather_paged_kvcache) 一致，仅输入、输出方向相反：`continuous_kv` 为输入，`kv_cache` 为输出。

**返回值说明**

- 成功：无返回值。异步执行。
- 失败：`head_dim` 非法或启动线程数超过 1024 时抛出 `std::runtime_error`。

**使用示例**

```cpp
#include "gather_scatter.h"

// 把连续缓冲区中的页写回第 layer_idx 层 cache
kvcache::ScatterPagedKVCache(
    reinterpret_cast<uint16_t*>(continuous_buffer),
    page_ids.data_ptr<int>(),
    num_heads, head_dim, page_size,
    page_stride, k2v_stride, inner_token_stride, head_dim,
    num_pages,
    reinterpret_cast<uint16_t*>(npu_cache_table[layer_idx]),
    num_ctas, stream);
```

### get_paged_batch_indices_positions

**功能描述**

由 `append_indptr` 与序列长度推导每个新 token 的 batch 序号与序列内位置。对每个序列 `s`（一个 block 负责一个序列）：

- `seq_start = append_indptr[s]`，`append_per_seq = append_indptr[s + 1] - seq_start`。
- `pos_start = seq_lens[s] - append_per_seq`。
- 对 `i ∈ [0, append_per_seq)`：`batch_indices[seq_start + i] = s`，`positions[seq_start + i] = pos_start + i`。

**函数原型**

```cpp
void GetPagedBatchIndicesPositions(int32_t batch_size,
                                   int32_t* append_indptr,
                                   int32_t* seq_lens_ptr,
                                   int32_t* batch_indices_ptr,
                                   int32_t* positions_ptr,
                                   aclrtStream stream);
```

**参数说明**

| 参数名 | 类型 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| batch_size | int32_t | 必选 | 序列数，同时作为启动的 block 数。 |
| append_indptr | int32_t* | 必选 | 设备侧 `int32`，长度至少 `batch_size + 1`，各序列新 token 的偏移。 |
| seq_lens_ptr | int32_t* | 必选 | 设备侧 `int32`，长度至少 `batch_size`，各序列历史总长度。 |
| batch_indices_ptr | int32_t* | 必选 | 设备侧 `int32` 输出，长度至少 `append_indptr[batch_size]`。 |
| positions_ptr | int32_t* | 必选 | 设备侧 `int32` 输出，长度与 `batch_indices_ptr` 相同。 |
| stream | aclrtStream | 必选 | 下发流的句柄。 |

**返回值说明**

- 成功：无返回值。异步执行。
- 失败：缓冲区长度不足时行为未定义，算子内部不做 `batch_size` 与缓冲区长度的交叉校验。
