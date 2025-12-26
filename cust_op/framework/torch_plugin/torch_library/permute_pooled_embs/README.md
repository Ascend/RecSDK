**使用pytorch框架调用方式调用permute_pooled_embs算子**

该样例基于Pytorch2.6.0、python3.11.0运行

### Pytorch框架对外接口原型

```python
torch.ops.fbgemm.permute_pooled_embs(Tensor pooled_embs, 
                                     Tensor offset_dim_list, 
                                     Tensor permute_list,
                                     Tensor inv_offset_dim_list,
                                     Tensor inv_permute_list) -> (Tensor)


torch.ops.mxrec.permute_pooled_embs(Tensor pooled_embs, 
                                    Tensor offset_dim_list, 
                                    Tensor permute_list,
                                    Tensor inv_offset_dim_list,
                                    Tensor inv_permute_list) -> (Tensor)
```

### 输入参数说明

#### `pooled_embs` (Tensor)
- **形状**: `[B_local, total_global_D]`
- **类型**: `float32`, `float16`, `bfloat16`
- **描述**: 池化后的嵌入输出张量
  - `B_local`: 本地批次大小（batch size）
  - `total_global_D`: 所有特征的嵌入维度总和 = `sum(embs_dims)`
- **内存布局**: 行优先（row-major），每行代表一个样本的所有特征嵌入拼接

#### `offset_dim_list` (Tensor)
- **形状**: `[T+1]`，其中 `T` 是特征数量
- **类型**: `int64`
- **描述**: 嵌入维度的累积和（cumulative sum），包含起始偏移
  - `offset_dim_list[0] = 0`
  - `offset_dim_list[i] = sum(embs_dims[0:i])`
  - `offset_dim_list[T] = total_global_D`
- **示例**: 如果 `embs_dims = [4, 4, 8]`，则 `offset_dim_list = [0, 4, 8, 16]`

#### `permute_list` (Tensor)
- **形状**: `[T]`
- **类型**: `int64`
- **描述**: 排列顺序列表，`permute_list[i]` 表示输出位置 `i` 应该放置原始特征 `permute_list[i]`
  - 每个元素范围: `[0, T-1]`
  - 可以是任意排列（permutation），也支持重复（duplicate）
- **示例**: `permute_list = [2, 0, 1]` 表示：
  - 输出位置 0 放置原始特征 2
  - 输出位置 1 放置原始特征 0
  - 输出位置 2 放置原始特征 1

#### `inv_offset_dim_list` (Tensor)
- **形状**: `[T+1]`
- **类型**: `int64`
- **描述**: 排列后嵌入维度的累积和
  - `inv_offset_dim_list[i] = sum(embs_dims[permute_list[0:i]])`
  - 用于确定输出张量中每个特征段的起始位置
- **示例**: 如果 `embs_dims = [4, 4, 8]` 且 `permute_list = [2, 0, 1]`，则：
  - `inv_embs_dims = [8, 4, 4]`（按 permute 顺序）
  - `inv_offset_dim_list = [0, 8, 12, 16]`

#### `inv_permute_list` (Tensor)
- **形状**: `[T]`
- **类型**: `int64`
- **描述**: 逆排列列表，用于反向传播
  - `inv_permute_list[i]` 表示原始特征 `i` 在输出中的位置
  - 满足: `inv_permute_list[permute_list[i]] = i`
- **示例**: 如果 `permute_list = [2, 0, 1]`，则 `inv_permute_list = [1, 2, 0]`

### 输出参数说明

#### 返回值 (Tensor)
- **形状**: `[B_local, total_global_D]`（与输入相同，除非允许重复）
- **类型**: 与 `pooled_embs` 相同
- **描述**: 重排列后的嵌入输出
  - 输出张量的特征顺序按照 `permute_list` 指定的顺序排列
  - 每个样本的特征嵌入被重新组织

### 示例

假设：
- `B_local = 3`
- `embs_dims = [4, 4, 8]`（3个特征，维度分别为4, 4, 8）
- `permute_list = [2, 0, 1]`

**输入 `pooled_embs`**:
```
样本0: [f0_0, f0_1, f0_2, f0_3, f1_0, f1_1, f1_2, f1_3, f2_0, f2_1, ..., f2_7]
样本1: [f0_0, f0_1, f0_2, f0_3, f1_0, f1_1, f1_2, f1_3, f2_0, f2_1, ..., f2_7]
样本2: [f0_0, f0_1, f0_2, f0_3, f1_0, f1_1, f1_2, f1_3, f2_0, f2_1, ..., f2_7]
```

**输出**:
```
样本0: [f2_0, f2_1, ..., f2_7, f0_0, f0_1, f0_2, f0_3, f1_0, f1_1, f1_2, f1_3]
样本1: [f2_0, f2_1, ..., f2_7, f0_0, f0_1, f0_2, f0_3, f1_0, f1_1, f1_2, f1_3]
样本2: [f2_0, f2_1, ..., f2_7, f0_0, f0_1, f0_2, f0_3, f1_0, f1_1, f1_2, f1_3]
```

## 约束条件

### 输入约束
1. **张量维度**: `pooled_embs` 必须是至少 2 维张量
2. **数据类型**: 
   - `pooled_embs`: 必须是浮点类型（float32/float16/bfloat16）
   - `offset_dim_list`, `permute_list`, `inv_offset_dim_list`, `inv_permute_list`: 必须是 `int64` 类型
3. **设备一致性**: 所有输入张量必须在同一设备上（CPU 或 NPU）
4. **形状一致性**:
   - `offset_dim_list.numel() == T + 1`
   - `permute_list.numel() == T`
   - `inv_offset_dim_list.numel() == T + 1`
   - `inv_permute_list.numel() == T`
   - `offset_dim_list[T] == pooled_embs.size(1)`（除非允许重复）
5. **排列有效性**: `permute_list` 中的值必须在 `[0, T-1]` 范围内


### 运行算子样例

#### 算子编译与部署

permute_pooled_embs算子的运行依赖index_select算子，需先安装index_select算子。

算子编译部署请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。

#### Pytorch编译

Pytorch框架适配层编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子适配层编译"。

#### 算子调用示例,以下以pytest方式调用为例
```python
import itertools
import random
import sysconfig

import pytest
import torch
import torch_npu
import fbgemm_gpu
import numpy as np

DEVICE = "npu:0"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

def get_result(tensors: dict, device: str = 'cpu', is_mxrec: bool = False):
    tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in tensors.items()}

    if device and device.startswith('npu'):
        torch.npu.set_device(device)
        tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}

    if is_mxrec:
        results = torch.ops.mxrec.permute_pooled_embs(**tensors)
    else:
        results = torch.ops.fbgemm.permute_pooled_embs(**tensors)

    if device and device.startswith('npu'):
        torch_npu.npu.synchronize()
    return [x.cpu() if isinstance(x, torch.Tensor) else x for x in results]


T = np.random.randint(2, 20, 6)
B = [32, 128, 1024, 2048, 10240, 102400]
SHAPE_LIST = list(itertools.product(T, B))

@pytest.mark.parametrize("types", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shapes", SHAPE_LIST)
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_permute_pooled_embs_aligned(types, shapes, is_mxrec):
    """
    Params:
        pooled_embs: (B, sum(embs_dims)) dtype=etype
        offset_dim: (T + 1) dtype=int64
        permute: (T) dtype=int64
        inv_permute: (T) dtype=int64
        inv_offset_dim: (T + 1) dtype=int64
    """
    etype = types[0] if isinstance(types, tuple) else types
    t, b = shapes

    # 每个特征的维度随机选择[32, 64, 128]中的一个
    choices = torch.tensor([32, 64, 128], dtype=torch.int64)
    embs_dims = choices[torch.randint(0, len(choices), (t,), dtype=torch.int64)]
    offset_dim_list = torch.cat([torch.tensor([0], dtype=torch.int64), torch.cumsum(embs_dims, dim=0)])
    permute = torch.randperm(t, dtype=torch.int64)
    inv_permute = torch.empty_like(permute)
    for i, p in enumerate(permute):
        inv_permute[p] = i
    inv_embs_dims = embs_dims[permute]
    inv_offset_dim_list = torch.cat([torch.tensor([0], dtype=torch.int64), torch.cumsum(inv_embs_dims, dim=0)])
    pooled_embs = torch.arange(0, embs_dims.sum().item() * b, dtype=etype).reshape(b, -1)

    params = {
        'pooled_embs': pooled_embs,
        'offset_dim_list': offset_dim_list,
        'permute_list': permute,
        'inv_offset_dim_list': inv_offset_dim_list,
        'inv_permute_list': inv_permute,
    }

    golden = get_result(params)
    result = get_result(params, DEVICE, is_mxrec)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-4)
```

注：上述用例为通用场景执行，更详细精度、多场景测试用例请参考用例[`RecSDK/cust_op/test/permute2d_sparse_data_test/torch/test_permute_pooled_embs.py`](../../../../test/permute2d_sparse_data_test/torch/test_permute_pooled_embs.py)。