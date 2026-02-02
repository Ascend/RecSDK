# 使用PyTorch框架调用reverse_sequence算子

该样例基于 PyTorch 2.6.0 和 Python 3.11.0 运行。

### 算子调用示例

```python
import sysconfig

import fbgemm_gpu
import numpy as np
import torch_npu
import torch

DEVICE = "npu:0"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
np.random.seed(42)

bs = 10
max_seq_len = 10
data_dim = 16
input_data = np.random.randn(bs, max_seq_len, data_dim).astype(np.float32)
seq_lengths = np.random.randint(0, max_seq_len, bs)

input_dt = torch.float32
seq_lengths_dt = torch.int64
input_data = torch.from_numpy(input_data).to(input_dt).to(DEVICE)
seq_lengths = torch.from_numpy(seq_lengths).to(seq_lengths_dt).to(DEVICE)
output = torch.ops.mxrec.reverse_sequence(input_data, seq_lengths)
```

## 编译与部署

算子编译与部署请参考 [RecSDK/cust_op/README.md](../../../../README.md) 中 "单算子使用说明" 章节：
- [算子编译](../../../../README.md#算子编译)
- [算子适配层编译](../../../../README.md#算子适配层编译)

> **提示**
> 以上示例仅展示基本用法，如需更全面的精度测试与边界用例，请参考完整测试文件：
> - [test](../../../../test/reverse_sequence_test/torch/test_reverse_sequence.py)