**使用pytorch框架调用方式调用hstu_dense_forward_fuxi算子**

该样例基于Pytorch2.6.0、python3.11.0运行

### Pytorch框架对外接口原型

```python
torch.ops.mxrec.hstu_fuxi(Tensor q, Tensor k, Tensor v, Tensor? timestampBias=None, Tensor? positionBias=None,
    Tensor? mask=None, int maskType=0, int maxSeqLen=0, float siluScale=0.0, str layout="normal",
    int[]? seqOffset=None) -> Tensor
```

### 参数说明

### torch.ops.mxrec.hstu_fuxi接口

#### Atlas A2/A3训练产品

|  名称  |  输入/输出  | 参数类型 |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |  ----  |
|  q | 输入 | Tensor | float32/float16/bfloat16 | [s_b, N, D] | B∈[1, 2048]<br>S∈[1, 20480]<br>N∈[2, 8]且是2的倍数<br>D∈[16, 512]且是16的倍数 | 只支持三维 |
|  k | 输入 | Tensor | float32/float16/bfloat16 | [s_b, N, D] | 同q | 同q |
|  v | 输入 | Tensor | float32/float16/bfloat16 | [s_b, N, D] | 同q | 同q |
|  timestampBias | 输入(可选) | Tensor | float32/float16/bfloat16 | [B, S, S] | NA | S为模型最大序列长度max_seq_len<br>不使用timestampBias时传入None，类型需与q一致 |
|  positionBias | 输入(可选) | Tensor | float32/float16/bfloat16 | [1, S, S] | NA | S为模型最大序列长度max_seq_len<br>不使用positionBias时传入None，类型需与q一致 |
|  mask | 输入(可选) | Tensor | float32/float16/bfloat16 | [B, N, S, S] | NA | S为模型最大序列长度max_seq_len<br>不使用mask时传入None，类型需与q一致，默认为None |
|  maskType | 输入(可选) | int | int | NA | 0：使用内置下三角mask，不需要传入mask<br>1：使用内置上三角mask，不需要传入mask(当前暂不支持)<br>2：不使用mask<br>3：使用自定义mask，此时mask需要用户定义并传入 | 默认值为0 |
|  maxSeqLen | 输入(可选) | int | int | NA | [1, 20480] | 表示模型最大序列长度,默认值为0 |
|  siluScale | 输入(可选) | float | float | NA | NA | 支持用户传入自定义，不传入时默认为0 |
|  layout | 输入(可选) | str | string | NA | 仅支持"jagged":代表q,k,v数据格式为[s_b, N, D] | 默认值为"normal" |
|  seqOffset | 输入(可选) | int[] | int[] | NA | NA | 表示每个batch的实际序列长度偏移，从0开始递增，需用户自行保证合法性，仅在jagged格式下生效，默认为None |
|  output | 输出 | Tensor | float32/float16/bfloat16 | [s_b, N, x * D] | 同q | 当输入RAB为空时，x=1，此时结果为qkv计算结果<br>当输入RAB不为空时，x=3，此时结果为qkv结果与两个rab结果cat，将最后一维整合所得 |

#### Atlas 推理系列产品
|  名称  |  输入/输出  | 参数类型 |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |  ----  |
|  q | 输入 | Tensor | float16 | [B, S, N, D] | B∈[1, 10]<br>S∈[64, 20480]且是64的倍数<br>N∈[1, 8]<br>D∈[16, 128]且是16的倍数 | 只支持四维 |
|  k | 输入 | Tensor | float16 | [B, S, N, D] | 同q | 同q |
|  v | 输入 | Tensor | float16 | [B, S, N, D] | 同q | 同q |
|  timestampBias | 输入(可选) | Tensor | float32/float16/bfloat16 | [B, S, S] | NA | S为模型最大序列长度max_seq_len<br>不使用timestampBias时传入None，类型需与q一致 |
|  positionBias | 输入(可选) | Tensor | float32/float16/bfloat16 | [B, S, S] | NA | S为模型最大序列长度max_seq_len<br>不使用positionBias时传入None，类型需与q一致 |
|  mask | 输入 | Tensor | float16 | [B, N, S, S] | NA | S为模型最大序列长度max_seq_len<br>不使用mask时传入None，类型需与q一致，必须传入 |
|  maskType | 输入(可选) | int | int | NA | 3：使用自定义mask，此时mask需要用户定义并传入 | 默认值为0 |
|  maxSeqLen | 输入(可选) | int | int | NA | [128, 4096]且是128的倍数 | 表示模型最大序列长度,默认值为0 |
|  siluScale | 输入(可选) | float | float | NA | NA | 支持用户传入自定义，不传入时默认为0 |
|  layout | 输入(可选) | str | string | NA | 仅支持"normal":代表q,k,v数据格式为[B, S, N, D] | 默认为"normal" |
|  seqOffset | 输入(可选) | int[] | int[] | NA | NA | 推理产品不用传入，默认为None |
|  output | 输出 | Tensor | float16 | [B, S, N, x * D] | 同q | 当输入RAB为空时，x=1，此时结果为qkv计算结果<br>当输入RAB不为空时，x=3，此时结果为qkv结果与两个rab结果cat，将最后一维整合所得 |

### 运行算子样例

#### 算子编译与部署

算子编译部署请参考[RecSDK\cust_op\README.md](../../../../../README.md)中"单算子使用说明"-"算子编译"章节。

#### Pytorch编译

Pytorch框架适配层编译请参考[RecSDK\cust_op\README.md](../../../../../README.md)中"单算子使用说明"-"算子适配层编译"。

#### 算子调用示例,以下以pytest方式调用为例

##### hstu_fuxi接口

```python
import os
import sys
import sysconfig
import pytest
import torch
import torch_npu
import torch.nn.functional as F
import numpy as np

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

device_id: int = 0

mask_tril: int = 0
mask_triu: int = 1
mask_none: int = 2
mask_custom: int = 3

torch.npu.set_device(device_id)


def jagged_data_gen(batch_size, max_seq_len, num_heads, attention_dim, mask_type):
    seq_lens = np.random.randint(1, max_seq_len + 1, (batch_size))

    seq_offset = torch.concat((torch.zeros((1, ), dtype=torch.int64), \
        torch.cumsum(torch.from_numpy(seq_lens), axis=0))).to(torch.int64).numpy()
    
    total_seqs = np.sum(seq_lens)

    q = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32).uniform_(-1, 1)
    k = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32).uniform_(-1, 1)
    v = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32).uniform_(-1, 1)

    ts_bias = torch.zeros(batch_size, max_seq_len, max_seq_len).to(torch.float32).uniform_(-1, 1)
    pos_bias = torch.zeros(1, max_seq_len, max_seq_len).to(torch.float32).uniform_(-1, 1)
    for batch_id in range(batch_size):
        seq_len = seq_lens[batch_id]
        ts_bias[batch_id, 0:seq_len, 0:seq_len] = torch.rand(seq_len, seq_len).to(torch.float32)
        pos_bias[0, 0:seq_len, 0:seq_len] = torch.rand(seq_len, seq_len).to(torch.float32)

    if mask_type == mask_tril:
        mask = 1 - torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len), diagonal=1)
    else:
        mask = torch.randint(0, 2, size=(batch_size, num_heads, max_seq_len, max_seq_len))
    mask = mask.cpu().to(torch.float32)

    return q, k, v, seq_offset, ts_bias, pos_bias, mask


class TestHstuJaggedFuxi:
    @staticmethod
    def custom_op_exec(q, k, v, seq_offset, ts_bias, pos_bias, mask, mask_type, max_seq_len, silu_scale, \
        enable_bias, data_type):
        q_npu = q.to(f"npu:{device_id}").to(data_type)
        k_npu = k.to(f"npu:{device_id}").to(data_type)
        v_npu = v.to(f"npu:{device_id}").to(data_type)
        ts_bias_npu = ts_bias.to(f"npu:{device_id}").to(data_type)
        pos_bias_npu = pos_bias.to(f"npu:{device_id}").to(data_type)
        mask_npu = mask.to(f"npu:{device_id}").to(data_type)

        if enable_bias:
            output = torch.ops.mxrec.hstu_fuxi(
                q_npu, k_npu, v_npu, ts_bias_npu, pos_bias_npu, mask_npu, mask_type, max_seq_len, silu_scale, \
                    "jagged", seq_offset
            )
        else:
            output = torch.ops.mxrec.hstu_fuxi(
                q_npu, k_npu, v_npu, None, None, mask_npu, mask_type, max_seq_len, silu_scale, "jagged", seq_offset
            )
        torch.npu.synchronize()
        return output.cpu().to(data_type).reshape(-1)


    def execute(self, batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale, data_type):
        q, k, v, seq_offset, ts_bias, pos_bias, mask = jagged_data_gen(batch_size, max_seq_len, head_num, head_dim, \
            mask_type)


        output = self.custom_op_exec(q, k, v, seq_offset, ts_bias, pos_bias, mask, mask_type, max_seq_len, silu_scale, \
            enable_bias, data_type)


    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("max_seq_len", [15, 1024])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("head_dim", [16, 128])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [mask_tril, mask_none, mask_custom])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    def test_hstu_forward_jagged_fuxi(self, batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, \
        silu_scale, data_type):
        self.execute(batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale, data_type)

```

注：上述用例为normal格式简易调用场景，更详细精度、多场景测试请参考用例[RecSDK/cust_op/test/hstu_dense_forward_fuxi/torch/test_hstu_forward_jagged_fuxi.py](../../../../../test/hstu_dense_forward_fuxi/torch/test_hstu_forward_jagged_fuxi.py)