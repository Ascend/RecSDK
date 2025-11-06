# 使用PyTorch框架调用split_embedding_codegen_forward_unweighted和backward_codegen_adagrad_unweighted_exact算子

该样例基于 PyTorch 2.6.0 和 Python 3.11.0 运行。

**注意：该算子为内部查表接口，不建议直接调用，推荐通过hybrid_torchrec/torchrec_embcache框架进行调用**

### 算子调用示例


```python
import sysconfig
from collections import defaultdict

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from hybrid_torchrec.distributed.batched_embedding_kernel import HybridSplitTableBatchedEmbeddingBagsCodegen
from torch.optim import SGD

import torchrec
from torchrec import JaggedTensor, KeyedJaggedTensor, PoolingType, ComputeDevice

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
DEVICEID = "npu:0"

def test_lookup_multi_tables():
    tables = [(98, 16), (14, 16), (20, 16)]
    optim = EmbOptimType.EXACT_SGD
    pooling_mode = PoolingMode.SUM
    embedding_specs = [
        (num_embeddings, embedding_dim, EmbeddingLocation.DEVICE, ComputeDevice.NPU)
        for (num_embeddings, embedding_dim) in tables
    ]

    batch_size = 64
    max_len = 20 # 最大的查表索引偏移
    lengths = torch.randint(0, max_len, (batch_size * len(tables),))
    offsets = torch.zeros((lengths.shape[0] + 1,), dtype=lengths.dtype)
    torch.cumsum(lengths, 0, out=offsets[1:])
    # 随机生成查表输入数据
    indices_test = []
    for ind, table in enumerate(tables):
        num = torch.sum(lengths[batch_size * ind: batch_size * (ind + 1)])
        indices = torch.randint(0, table[0], (num.item(),)).to(torch.int64)
        indices_test.append(indices)
    indices = torch.cat(indices_test).to(DEVICEID).to(torch.int64)
    offsets = offsets.to(DEVICEID).to(torch.int64)

    # 通过hybrid_torchrec 提供的接口调用算子进行查表
    ebc_class = HybridSplitTableBatchedEmbeddingBagsCodegen

    tbe = ebc_class(
        embedding_specs,
        optimizer=optim,
        device=torch.device(DEVICEID),
        pooling_mode=pooling_mode
    )
    output = tbe(indices, offsets, **kwargs)
    loss = torch.sum(output ** 2 / 2)
    loss.backward()

if __name__ == "__main__":
    test_lookup_multi_tables()
```

## 编译与部署

算子编译与部署请参考 [RecSDK\cust_op\README.md](../../../../../README.md) 中 "单算子使用说明" 章节：
- [算子编译](../../../../../README.md#1算子编译)
- [算子适配层编译](../../../../../README.md#2算子适配层编译)

> **提示**
> 以上示例仅展示基本用法，如需更全面的精度测试与边界用例，请参考完整测试文件：  
> - [`RecSDK/cust_op\test\split_embedding_codegen_lookup_adagrad_function_test\torch\test_split_embedding_codegen_lookup_function.py`](../../../../../test/split_embedding_codegen_lookup_adagrad_function_test/torch/test_split_embedding_codegen_lookup_function.py)
