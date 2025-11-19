# 使用PyTorch框架调用dense_embedding_codegen_lookup_function和dense_embedding_codegen_lookup_function_grad算子

该样例基于 PyTorch 2.6.0 和 Python 3.11.0 运行。

**注意：该算子为内部查表接口，不建议直接调用，推荐通过hybrid_torchrec框架进行调用**

### 算子调用示例

```python
import sysconfig
from collections import defaultdict

import torch
import fbgemm_gpu
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode
from torchrec.distributed.batched_embedding_kernel import DenseTableBatchedEmbeddingBagsCodegen

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
DEVICEID = "npu:0"

def test_lookup_single_table():
    # 定义嵌入表参数
    num_embeddings = 100
    embedding_dim = 16
    batch_size = 32
    indices_num = 200
    
    # 创建测试数据
    indices = torch.randint(0, num_embeddings, (indices_num,)).to(DEVICEID)
    offsets = torch.tensor([0, indices_num]).to(DEVICEID)
    weights = torch.randn(num_embeddings, embedding_dim)

    # 通过torchrec提供的接口调用算子进行查表
    tbe = DenseTableBatchedEmbeddingBagsCodegen(
        [(num_embeddings, embedding_dim)],
        use_cpu=False,
        pooling_mode=PoolingMode.NONE,
    )
    tbe.weights.data.copy_(weights.reshape(-1))
    tbe.weights.requires_grad = True
    tbe.weights.retain_grad()
    
    # 执行前向传播
    output = tbe(indices, offsets)
    loss = torch.sum(output ** 2 / 2)
    
    # 执行反向传播
    loss.backward()

if __name__ == "__main__":
    test_lookup_single_table()
```

## 编译与部署

算子编译与部署请参考 [RecSDK/cust_op/README.md](../../../../README.md) 中 "单算子使用说明" 章节：
- [算子编译](../../../../README.md#1算子编译)
- [算子适配层编译](../../../../README.md#2算子适配层编译)

> **提示**
> 以上示例仅展示基本用法，如需更全面的精度测试与边界用例，请参考完整测试文件：  
> - [`RecSDK/cust_op/test/dense_embedding_codegen_lookup_function_test/torch/test_dense_embedding_codegen_lookup_function_auto.py`](../../../../test/dense_embedding_codegen_lookup_function_test/torch/test_dense_embedding_codegen_lookup_function_auto.py)
