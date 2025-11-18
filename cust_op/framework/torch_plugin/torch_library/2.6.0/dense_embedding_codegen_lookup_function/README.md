# 使用PyTorch框架调用dense_embedding_codegen_lookup_function算子

该样例基于 PyTorch 2.6.0 和 Python 3.11.0 运行。

### 算子调用示例

```python
import sysconfig
import torch

# 加载NPU算子库
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

def test_dense_embedding_lookup():
    # 设备ID
    DEVICE_ID = "npu:0"
    
    # 创建测试数据
    dev_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32).to(DEVICE_ID)
    weights_offsets = torch.tensor([0, 3, 6], dtype=torch.int64).to(DEVICE_ID)
    d_offsets = torch.tensor([0, 2, 4], dtype=torch.int32).to(DEVICE_ID)
    hash_size_cumsum = torch.tensor([0, 2, 4], dtype=torch.int64).to(DEVICE_ID)
    indices = torch.tensor([0, 1, 1, 0], dtype=torch.int64).to(DEVICE_ID)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int64).to(DEVICE_ID)
    
    # 调用算子
    result = torch.ops.mxrec.dense_embedding_codegen_lookup_function(
        devWeights=dev_weights,
        weightsOffsets=weights_offsets,
        dOffsets=d_offsets,
        totalD=4,
        maxD=2,
        hashSizeCumsum=hash_size_cumsum,
        totalHashSizeBits=2,
        indices=indices,
        offsets=offsets,
        poolingMode=0,
        indiceWeightsOptional=None,
        featureRequiresGrad=None,
        outputDtypeOptional=0,
        bOffsetOptional=None,
        vbeOutputOffsetsFeatureRankOptional=None,
        vbeBOffsetsRankPerFeatureOptional=None,
        maxB=0,
        maxBFeatureRank=0,
        vbeOutputSize=0
    )
    
    print("Result:", result)

if __name__ == "__main__":
    test_dense_embedding_lookup()
```

### 反向传播算子调用示例

```python
import sysconfig
import torch

# 加载NPU算子库
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

def test_dense_embedding_lookup_grad():
    # 设备ID
    DEVICE_ID = "npu:0"
    
    # 创建测试数据
    dev_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32).to(DEVICE_ID)
    weights_grad = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=torch.float32).to(DEVICE_ID)
    weights_offsets = torch.tensor([0, 3, 6], dtype=torch.int64).to(DEVICE_ID)
    d_offsets = torch.tensor([0, 2, 4], dtype=torch.int32).to(DEVICE_ID)
    hash_size_cumsum = torch.tensor([0, 2, 4], dtype=torch.int64).to(DEVICE_ID)
    indices = torch.tensor([0, 1, 1, 0], dtype=torch.int64).to(DEVICE_ID)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int64).to(DEVICE_ID)
    
    # 调用反向传播算子
    result = torch.ops.mxrec.dense_embedding_codegen_lookup_function_grad(
        devWeights=dev_weights,
        grad=weights_grad,
        weightsOffsets=weights_offsets,
        dOffsets=d_offsets,
        totalD=4,
        maxD=2,
        hashSizeCumsum=hash_size_cumsum,
        totalHashSizeBits=2,
        indices=indices,
        offsets=offsets,
        poolingMode=0,
        indiceWeightsOptional=None,
        featureRequiresGrad=None,
        outputDtypeOptional=0,
        bOffsetOptional=None,
        vbeOutputOffsetsFeatureRankOptional=None,
        vbeBOffsetsRankPerFeatureOptional=None,
        maxB=0,
        maxBFeatureRank=0,
        vbeOutputSize=0
    )
    
    print("Grad Result:", result[0])

if __name__ == "__main__":
    test_dense_embedding_lookup_grad()
```

## 编译与部署

算子编译与部署请参考 [RecSDK\cust_op\README.md](../../../../../README.md) 中 "单算子使用说明" 章节：
- [算子编译](../../../../../README.md#1算子编译)
- [算子适配层编译](../../../../../README.md#2算子适配层编译)

> **提示**
> 以上示例仅展示基本用法，如需更全面的精度测试与边界用例，请参考完整测试文件：
> - [`RecSDK/cust_op/test/dense_embedding_codegen_lookup_function_test/torch/test_dense_embedding_codegen_lookup_function_forward.py`](../../../../../test/dense_embedding_codegen_lookup_function_test/torch/test_dense_embedding_codegen_lookup_function_forward.py)
> - [`RecSDK/cust_op/test/dense_embedding_codegen_lookup_function_test/torch/test_dense_embedding_codegen_lookup_function_backward.py`](../../../../../test/dense_embedding_codegen_lookup_function_test/torch/test_dense_embedding_codegen_lookup_function_backward.py)