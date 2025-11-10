**使用pytorch框架调用方式调用hstu_dense_backward_fuxi算子**

该样例基于Pytorch2.6.0、python3.11.0运行

### Pytorch框架对外接口原型

```python
torch.ops.mxrec.hstu_dense_backward_fuxi(Tensor grad, Tensor q, Tensor k, Tensor v, Tensor? mask=None,
    Tensor? biasPosition=None, Tensor? biasTimestamp=None,
    str layout="jagged", int maskType=0, int maxSeqLen=0, float siluScale=0.0,
    int[]? seqOffset=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

### 参数说明

| 名称 | 输入/输出 | 数据类型 | 数据格式 | 备注 |
|----|----|----|----|----|
| grad | 输入| float32/float16/bfloat16 | [s_b, N, D] |
| q | 输入| float32/float16/bfloat16 | [s_b, N, D] |
| k | 输入| float32/float16/bfloat16 | [s_b, N, D] |
| v | 输入| float32/float16/bfloat16 | [s_b, N, D] |
| mask | 可选输入 | float32/float16/bfloat16 | B,N,S,S | S为模型最大的序列长度max_seq_len |
| bias_position | 可选输入 | float32/float16/bfloat16 | 1,S,S | S为模型最大的序列长度max_seq_len |
| bias_timestamp | 可选输入 | float32/float16/bfloat16 | B,S,S | S为模型最大的序列长度max_seq_len |
| grad_bias_position | 输入| float32/float16/bfloat16 | [s_b, N, D] |
| grad_bias_timestamp | 输入| float32/float16/bfloat16 | [s_b, N, D] |
| layout | 属性 | string | N/A | 当前仅支持"jagged"，“jagged”代表Q,K,V数据格式为s_b,N,D格式 |
| mask_type | 属性 | int | N/A | 0:使用内置下三角掩码 1:使用内置上三角掩码(未支持) 2:不使用mask(即使mask传值) 3:使用自定义mask(需要输入mask) |
| max_seq_len | 属性 | int | N/A | 表示模型最大序列长度 |
| silu_scale | 属性 | float | N/A | 支持用户传入自定义silu_scale, 不传入时默认值为1/max_seq_len|
| seq_offsets | 可选属性 | list[int64] | N/A | 表示每个序列的偏移，其中第一个序列的偏移一定是0，此选项只对jagged格式下生效，normal格式不生效。|
| q_grad | 输出 | float32/float16/bfloat16 | [s_b, N, D] |
| k_grad | 输出 | float32/float16/bfloat16 | [s_b, N, D] |
| v_grad | 输出 | float32/float16/bfloat16 | [s_b, N, D] |
| position_bias_grad | 输出 | float32/float16/bfloat16 | 1,S,S | S为变长序列中最大的序列长度 |
| timestamp_bias_grad | 输出 | float32/float16/bfloat16 | B,S,S | S为变长序列中最大的序列长度 |

参数范围说明：
* s_b：为jagged格式下各batch的实际序列长度之和
* B: batch_size 表征批处理的大小，当前取值范围[1, 512]。
* S: seq_lens 表征序列长度，当前取值范围[1, 20480]。
* N：head_num 表征头个数，当前取值为[2,4,6,8]。
* D: head_dim 表征维度，当前取值范围范围[16, 512]，并且需要满足是16的倍数。
* 以上四个维度数值均不能为0，为0时算子输入为空数据，不会执行算子计算;并且其中B、N、S参数影响bias、mask占用显存大小，请根据实际内存合理设置参数大小。
* jagged模式下 S为所有序列中最大的序列长度，比如此时有两个序列，一个序列长度为256，另一个序列长度为512，则S为512。


### 运行算子样例

#### 算子编译与部署

算子编译部署请参考[RecSDK\cust_op\README.md](../../../../../README.md)中"单算子使用说明"-"算子编译"章节。

#### Pytorch编译

Pytorch框架适配层编译请参考[RecSDK\cust_op\README.md](../../../../../README.md)中"单算子使用说明"-"算子适配层编译"。

#### 算子调用示例,以下以pytest方式调用为例

##### hstu_dense_backward_fuxi 接口

```python
import os
import sys
import sysconfig
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch_npu

current_dir = os.path.dirname(os.path.abspath(__file__))
common_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "common"))
sys.path.append(common_dir)
from utils import allclose

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

device_id: int = 0
mask_tril: int = 0
mask_triu: int = 1
mask_none: int = 2
mask_custom: int = 3

bfloat16_pre: float = 5e-3
float16_pre: float = 1e-3
float32_pre: float = 1e-4

torch.manual_seed(3)


def jagged_data_gen(batch_size, max_seq_len, num_heads, attention_dim, mask_type, data_type):
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int64)
    seq_offset = torch.concat((torch.zeros((1,), dtype=torch.int64), torch.cumsum(seq_lens, axis=0))).numpy()
    
    total_seqs = torch.sum(seq_lens)

    start = -1
    end = 1
    grad = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(start, end)
    q = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(start, end)
    k = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(start, end)
    v = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(start, end)
    bpos = torch.empty(1, max_seq_len, max_seq_len, dtype=data_type).uniform_(start, end)
    bts = torch.empty(batch_size, max_seq_len, max_seq_len, dtype=data_type).uniform_(start, end)
    grad_pos = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(start, end)
    grad_ts = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(start, end)

    if mask_type == 0:
        mask = torch.tril(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type))
    elif mask_type == 1:
        mask = torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type))
    elif mask_type == 2:
        mask = None
    else:
        mask = torch.empty(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type).uniform_(start, end)

    return grad, q, k, v, bpos, bts, grad_pos, grad_ts, mask, max_seq_len, seq_offset


torch.npu.set_device(device_id)


class TestHstuJaggedDemo:
    def jagged_to_dense(self, jagged_tensor, seq_lens, max_seq_len, head_num, head_dim):

        batch_size = len(seq_lens)
        dense_tensor = torch.zeros(batch_size, max_seq_len, head_num, head_dim, dtype=jagged_tensor.dtype)

        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            dense_tensor[batch_id, :seq_len, :, :] = jagged_tensor[offset: offset + seq_len, :, :]
            offset = offset + seq_len

        return dense_tensor

    def dense_to_jagged(self, jagged_tensor, dense_tensor, seq_lens):
        tensor = torch.zeros_like(jagged_tensor)

        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            tensor[offset: offset + seq_len, :, :] = dense_tensor[batch_id, 0: seq_len, :, :]
            offset = offset + seq_len

        return tensor

    
    def custom_op_exec(self, grad, q, k, v, bpos, bts, grad_pos, grad_ts, mask, seq_offset, 
                       mask_type, max_seq_len, silu_scale, enable_bias, data_type):
        grad_npu = grad.to(f"npu:{device_id}")
        q_npu = q.to(f"npu:{device_id}")
        k_npu = k.to(f"npu:{device_id}")
        v_npu = v.to(f"npu:{device_id}")
        bpos_npu = bpos.to(f"npu:{device_id}")
        bts_npu = bts.to(f"npu:{device_id}")

        mask_npu = None
        if mask_type == 3:
            mask_npu = mask.to(f"npu:{device_id}")

        if enable_bias:
            q_grad, k_grad, v_grad, bpos_grad, bts_grad = torch.ops.mxrec.hstu_dense_backward_fuxi(
                grad_npu, q_npu, k_npu, v_npu, mask_npu, bpos_npu, bts_npu, "jagged", 
                mask_type, max_seq_len, silu_scale, seq_offset
            )
        else:
            q_grad, k_grad, v_grad, bpos_grad, bts_grad = torch.ops.mxrec.hstu_dense_backward_fuxi(
                grad_npu, q_npu, k_npu, v_npu, mask_npu, None, None, "jagged",
                mask_type, max_seq_len, silu_scale, seq_offset
            )

        torch.npu.synchronize()
        return (q_grad.cpu(), k_grad.cpu(), v_grad.cpu(),
                (enable_bias and bpos_grad.cpu()), (enable_bias and bts_grad.cpu()))
    
    def execute(self, batch_size, max_seq_len, head_num, head_dim, mask_type, silu_scale, enable_bias, data_type):
        grad, q, k, v, bpos, bts, grad_pos, grad_ts, mask, max_seq_len, seq_offset = \
            jagged_data_gen(batch_size, max_seq_len, head_num, head_dim, mask_type, data_type)

        grads = torch.cat((grad, grad_ts, grad_pos), -1) if enable_bias else grad
        q_grad, k_grad, v_grad, bpos_grad, bts_grad = self.custom_op_exec(
            grads, q, k, v, bpos, bts, None, None, mask, seq_offset,
            mask_type, max_seq_len, silu_scale, enable_bias, data_type
        )

    @pytest.mark.parametrize("batch_size", [2, 32, 64])
    @pytest.mark.parametrize("max_seq_len", [256, 1024, 1234, 2048])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("head_dim", [32, 128])
    @pytest.mark.parametrize("mask_type", [0, 2, 3])
    @pytest.mark.parametrize("silu_scale", [1.0 / 256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
    def test_hstu_dens_jagged(self, batch_size, max_seq_len, head_num, head_dim, mask_type, silu_scale,
                              enable_bias, data_type):
        self.execute(batch_size, max_seq_len, head_num, head_dim, mask_type, silu_scale, enable_bias, data_type)

```

注：上述用例为简易调用场景，更详细精度、多场景测试请参考用例[RecSDK/cust_op/test/hstu_dense_backward_fuxi/torch/test_hstu_dense_backward_fuxi.py](../../../../../test/hstu_dense_backward_fuxi/torch/test_hstu_dense_backward_fuxi.py)

