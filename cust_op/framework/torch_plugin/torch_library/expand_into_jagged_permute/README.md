# 使用 PyTorch 调用 expand_into_jagged_permute

本示例在 PyTorch 2.6.0 / Python 3.11.0 环境验证。

## 示例
```python
import sysconfig
import torch
import torch_npu

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
DEVICE = "npu:0"

def demo():
    # 示例：3个表，长度分别为 [10, 20, 15]
    # permute = [2, 0, 1] 表示将表顺序置换为 [表2, 表0, 表1]
    permute = torch.tensor([2, 0, 1], dtype=torch.int32, device=DEVICE)
    
    # input_offsets: [0, 10, 30, 45] 表示原始表的累积偏移量
    input_offsets = torch.tensor([0, 10, 30, 45], dtype=torch.int32, device=DEVICE)
    
    # output_offsets: [0, 15, 25, 45] 表示置换后表的累积偏移量
    # 置换后的长度顺序为 [15, 10, 20]
    output_offsets = torch.tensor([0, 15, 25, 45], dtype=torch.int32, device=DEVICE)
    
    # output_size = 45 (所有表的总长度)
    output_size = 45
    
    # 使用 mxrec 命名空间调用
    output_permute = torch.ops.mxrec.expand_into_jagged_permute(
        permute, input_offsets, output_offsets, output_size
    )
    print("output_permute shape:", output_permute.shape)
    print("output_permute:", output_permute.cpu())
    
    # 也可以使用 fbgemm 命名空间调用（与CUDA实现保持一致）
    output_permute_fbgemm = torch.ops.fbgemm.expand_into_jagged_permute(
        permute, input_offsets, output_offsets, output_size
    )
    print("output_permute_fbgemm:", output_permute_fbgemm.cpu())

if __name__ == "__main__":
    demo()
```

## 接口说明

### 函数签名
```python
torch.ops.mxrec.expand_into_jagged_permute(
    permute: Tensor,
    input_offsets: Tensor,
    output_offsets: Tensor,
    output_size: SymInt
) -> Tensor

torch.ops.fbgemm.expand_into_jagged_permute(
    permute: Tensor,
    input_offsets: Tensor,
    output_offsets: Tensor,
    output_size: SymInt
) -> Tensor
```

### 参数说明
- **permute** (Tensor): 表级别的置换索引，1D张量，数据类型为 INT32 或 INT64
- **input_offsets** (Tensor): 输入表的累积偏移量，1D张量，长度为 `permute.numel() + 1`，数据类型与 permute 相同
- **output_offsets** (Tensor): 输出表的累积偏移量，1D张量，长度为 `permute.numel() + 1`，数据类型与 permute 相同
- **output_size** (SymInt): 输出结果的长度，通常等于 `output_offsets[-1]`

### 返回值
- **output_permute** (Tensor): 扩展后的置换索引，1D张量，形状为 `[output_size]`，数据类型与输入相同

### 约束条件
- `permute.numel() == input_offsets.numel() - 1`
- `permute.numel() == output_offsets.numel() - 1`
- `permute`、`input_offsets`、`output_offsets` 必须具有相同的数据类型
- 所有输入张量必须位于相同的 NPU 设备上
- `output_size` 必须等于 `output_offsets[-1]`

## 编译与部署
- 参考 [RecSDK/cust_op/README.md](../../../../README.md) 中"单算子使用说明"的算子编译与适配层编译章节。
- 完整精度/功能测试可查看 `cust_op/test/expand_into_jagged_permute/torch/test_expand_into_jagged_permute.py`。

