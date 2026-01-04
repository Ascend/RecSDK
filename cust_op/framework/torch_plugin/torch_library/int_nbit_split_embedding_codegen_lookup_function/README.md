# 使用 PyTorch 调用 int_nbit_split_embedding_codegen_lookup_function

本示例在 PyTorch 2.6.0 / Python 3.11.0 环境验证。
**提示：该接口用于内部查表，推荐通过 hybrid_torchrec/torchrec_embcache 框架调用。**

## 示例
```python
import sysconfig
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import IntNBitTableBatchedEmbeddingBagsCodegen

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
DEVICE = "npu:0"

def demo():
    specs = [
        ("table0", 128, 64, SparseType.FP8, PoolingMode.SUM, SparseType.FP32),
        ("table1", 256, 32, SparseType.FP8, PoolingMode.SUM, SparseType.FP32),
    ]
    op = IntNBitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=specs,
        device=DEVICE,
        pooling_mode=PoolingMode.SUM,
        output_dtype=SparseType.BF16,
    )
    indices = torch.randint(0, 256, (1024,), dtype=torch.int32, device=DEVICE)
    offsets = torch.arange(0, 1024 + len(specs), len(specs), dtype=torch.int32, device=DEVICE)
    offsets = torch.cat([offsets, offsets[-1:]])
    out = torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(
        op.weights_dev,
        op.weights_uvm,
        op.weights_placements,
        op.weights_offsets,
        op.weights_tys,
        op.D_offsets,
        op.total_D,
        op.max_int2_D,
        op.max_int4_D,
        op.max_int8_D,
        op.max_float16_D,
        op.max_float32_D,
        indices,
        offsets,
        int(PoolingMode.SUM),
        None,
        int(SparseType.BF16),
        None,
        None,
        op.row_alignment,
        op.max_float8_D,
        op.fp8_exponent_bits,
        op.fp8_exponent_bias,
    )
    print("out shape:", out.shape)

if __name__ == "__main__":
    demo()
```

## 编译与部署
- 参考 [RecSDK/cust_op/README.md](../../../../README.md) 中“单算子使用说明”的算子编译与适配层编译章节。
- 完整精度/功能测试可查看 `cust_op/test/int_nbit_split_embedding_codegen_lookup_function_test/torch/test_int_nbit_split_embedding_codegen_lookup_function.py`。
