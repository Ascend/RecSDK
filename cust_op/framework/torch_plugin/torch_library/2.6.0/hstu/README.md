**使用pytorch框架调用方式调用hstu_dense_forward算子**

该样例基于Pytorch2.6.0、python3.11.0运行

### Pytorch框架对外接口原型

```python
torch.ops.mxrec.hstu_dense(Tensor q, Tensor k, Tensor v, Tensor? mask=None, Tensor? attnBias=None, int maskType=0,
    int maxSeqLen=0, float siluScale=0.0, str layout="normal", int[]? seqOffset=None) -> Tensor
torch.ops.mxrec.hstu_dense_backward(Tensor grad, Tensor q, Tensor k, Tensor v, Tensor? mask, Tensor? attnBias,
    str layout, int maskType, int maxSeqLen, float siluScale=0.0, int[]? seqOffset=None) -> (Tensor, Tensor, Tensor, Tensor)
```

### 参数说明

### torch.ops.mxrec.hstu_dense接口

#### Atlas A2/A3训练产品

|  名称  |  输入/输出  | 参数类型 |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |  ----  |
|  q | 输入 | Tensor | float32/float16/bfloat16 | [B, S, N, D]/<br>[s_b, N, D] | B∈[1, 2048]<br>S∈[1, 20480]<br>N∈[2, 8]且是2的倍数<br>D∈[16, 512]且是16的倍数 | "normal"格式下只支持四维<br>"jagged"格式下只支持三维 |
|  k | 输入 | Tensor | float32/float16/bfloat16 | [B, S, N, D]/<br>[s_b, N, D] | 同q | 同q |
|  v | 输入 | Tensor | float32/float16/bfloat16 | [B, S, N, D]/<br>[s_b, N, D] | 同q | 同q |
|  mask | 输入(可选) | Tensor | float32/float16/bfloat16 | [B, N, S, S] | NA | S为模型最大序列长度max_seq_len<br>不使用mask时传入None，类型需与q一致，默认为None |
|  attnBias | 输入(可选) | Tensor | float32/float16/bfloat16 | [B, N, S, S] | NA | S为模型最大序列长度max_seq_len<br>不使用attn_bias时传入None，类型需与q一致 |
|  maskType | 输入(可选) | int | int | NA | 0：使用内置下三角mask，不需要传入mask<br>1：使用内置上三角mask，不需要传入mask(当前暂不支持)<br>2：不使用mask<br>3：使用自定义mask，此时mask需要用户定义并传入 | 默认值为0 |
|  maxSeqLen | 输入(可选) | int | int | NA | [1, 20480] | 表示模型最大序列长度,默认值为0 |
|  siluScale | 输入(可选) | float | float | NA | NA | 支持用户传入自定义，不传入时默认为0 |
|  layout | 输入(可选) | str | string | NA | "normal":代表q,k,v数据格式为[B, S, N, D]<br>"jagged":代表q,k,v数据格式为[s_b, N, D] | 默认值为"normal" |
|  seqOffset | 输入(可选) | int[] | int[] | NA | NA | 表示每个batch的实际序列长度偏移，从0开始递增，需用户自行保证合法性，仅在jagged格式下生效，默认为None |
|  output | 输出 | Tensor | float32/float16/bfloat16 | [B, S, N, D]/<br>[s_b, N, D] | 同q | 同q |

#### Atlas 推理系列产品
|  名称  |  输入/输出  | 参数类型 |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |  ----  |
|  q | 输入 | Tensor | float16 | [B, S, N, D] | B∈[1, 2048]<br>S∈[128, 4096]且是128的倍数<br>N∈[1, 8]<br>D∈[16, 128]且是16的倍数 | 只支持四维 |
|  k | 输入 | Tensor | float16 | [B, S, N, D] | 同q | 同q |
|  v | 输入 | Tensor | float16 | [B, S, N, D] | 同q | 同q |
|  mask | 输入 | Tensor | float16 | [B, N, S, S] | NA | S为模型最大序列长度max_seq_len<br>不使用mask时传入None，类型需与q一致，必须传入 |
|  attnBias | 输入(可选) | Tensor | float16 | [B, N, S, S] | NA | S为模型最大序列长度max_seq_len<br>不使用attn_bias时传入None，类型需与q一致，默认为None |
|  maskType | 输入(可选) | int | int | NA | 3：使用自定义mask，此时mask需要用户定义并传入 | 默认值为0 |
|  maxSeqLen | 输入(可选) | int | int | NA | [128, 4096]且是128的倍数 | 表示模型最大序列长度,默认值为0 |
|  siluScale | 输入(可选) | float | float | NA | NA | 支持用户传入自定义，不传入时默认为0 |
|  layout | 输入(可选) | str | string | NA | 仅支持"normal":代表q,k,v数据格式为[B, S, N, D] | 默认为"normal" |
|  seqOffset | 输入(可选) | int[] | int[] | NA | NA | 推理产品不用传入，默认为None |
|  output | 输出 | Tensor | float16 | [B, S, N, D] | 同q | 同q |

### torch.ops.mxrec.hstu_dense_backward接口

|  名称  |  输入/输出  | 参数类型 |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |  ----  |
|  grad | 输入 | Tensor | float32/float16/bfloat16 | [B, S, N, D]/<br>[s_b, N, D] | B∈[1, 2048]<br>S∈[1, 20480]<br>N∈[2, 8]且是2的倍数<br>D∈[16, 512]且是16的倍数 | "normal"格式下只支持四维<br>"jagged"格式下只支持三维 |
|  q | 输入 | Tensor | float32/float16/bfloat16 | [B, S, N, D]/<br>[s_b, N, D] | 同grad | 同grad |
|  k | 输入 | Tensor | float32/float16/bfloat16 | [B, S, N, D]/<br>[s_b, N, D] | 同grad | 同grad |
|  v | 输入 | Tensor | float32/float16/bfloat16 | [B, S, N, D]/<br>[s_b, N, D] | 同grad | 同grad |
|  mask | 输入(可选) | Tensor | float32/float16/bfloat16 | [B, N, S, S] | NA | S为模型最大序列长度max_seq_len<br>不使用mask时传入None，类型需与q一致，默认为None |
|  attnBias | 输入(可选) | Tensor | float32/float16/bfloat16 | [B, N, S, S] | NA | S为模型最大序列长度max_seq_len<br>不使用attn_bias时传入None，类型需与q一致 |
|  layout | 输入 | str | string | NA | "normal":代表q,k,v数据格式为[B, S, N, D]<br>"jagged":代表q,k,v数据格式为[s_b, N, D] | NA |
|  maskType | 输入 | int | int | NA | 0：使用内置下三角mask，不需要传入mask<br>1：使用内置上三角mask，不需要传入mask(当前暂不支持)<br>2：不使用mask<br>3：使用自定义mask，此时mask需要用户定义并传入 | NA |
|  maxSeqLen | 输入 | int | int | NA | [1, 20480] | 表示模型最大序列长度 |
|  siluScale | 输入(可选) | float | float | NA | NA | 支持用户传入自定义，不传入时默认为0 |
|  seqOffset | 输入(可选) | int[] | int[] | NA | NA | 表示每个batch的实际序列长度偏移，从0开始递增，需用户自行保证合法性，仅在jagged格式下生效，默认为None |
|  q_grad | 输出 | Tensor | float32/float16/bfloat16 | [B, S, N, D]/<br>[s_b, N, D] | 同q | 同q |
|  k_grad | 输出 | Tensor | float32/float16/bfloat16 | [B, S, N, D]/<br>[s_b, N, D] | 同k | 同k |
|  v_grad | 输出 | Tensor | float32/float16/bfloat16 | [B, S, N, D]/<br>[s_b, N, D] | 同v | 同v |
|  attn_bias_grad | 输出 | Tensor | float32/float16/bfloat16 | [B, N, S, S] | 同attn_bias | 同attn_bias |

### 运行算子样例

#### 算子编译与部署

算子编译部署请参考[RecSDK\cust_op\README.md](../../../../../README.md)中"单算子使用说明"-"算子编译"章节。

#### Pytorch编译

Pytorch框架适配层编译请参考[RecSDK\cust_op\README.md](../../../../../README.md)中"单算子使用说明"-"算子适配层编译"。

#### 算子调用示例,以下以pytest方式调用为例

##### hstu_dense接口

```python
import os
import sys
import subprocess
import sysconfig
import numpy as np
import pytest
import torch
import torch.nn.functional as F

torch.npu.config.allow_internal_format = False

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

device_id: int = 0
mask_tril: int = 0
mask_triu: int = 1
mask_none: int = 2
mask_custom: int = 3


def generate_tensor(batch_size, max_seq_len, num_heads, attention_dim, data_type, mask_type):
    total_num = batch_size * max_seq_len * num_heads * attention_dim

    q = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim).uniform_(-1, 1)
    k = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim).uniform_(-1, 1)
    v = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim).uniform_(-1, 1)
    rel_attn_bias = torch.rand(batch_size, num_heads, max_seq_len, max_seq_len).uniform_(-1, 1)
    if mask_type == mask_tril:
        invalid_attn_mask = 1 - torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len), diagonal=1)
    else:
        invalid_attn_mask = torch.randint(0, 2, size=(batch_size, num_heads, max_seq_len, max_seq_len))
    return q.to(data_type).to(f"npu:{device_id}"), k.to(data_type).to(f"npu:{device_id}"), v.to(data_type).to(
        f"npu:{device_id}"), rel_attn_bias.to(data_type).to(f"npu:{device_id}"), invalid_attn_mask.to(data_type).to(
        f"npu:{device_id}")


torch.npu.set_device(device_id)


class TestHstuNormalDemo:
    @staticmethod
    def custom_op_exec(q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type):
        if enable_bias:
            output = torch.ops.mxrec.hstu_dense(
                q, k, v, mask, bias, mask_type, max_seq_len, silu_scale, "normal"
            )
        else:
            output = torch.ops.mxrec.hstu_dense(
                q, k, v, mask, None, mask_type, max_seq_len, silu_scale, "normal"
            )

        torch.npu.synchronize()
        return output.cpu().to(data_type).reshape(-1)

    def execute(self, batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale, data_type):
        q, k, v, bias, mask = generate_tensor(batch_size, max_seq_len, head_num, head_dim, data_type, mask_type)

        torch.npu.synchronize()
        output = self.custom_op_exec(q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type)
        torch.npu.synchronize()

    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("max_seq_len", [1, 15, 31, 256, 768, 1023, 4095])
    @pytest.mark.parametrize("head_dim", [32, 64])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [mask_tril, mask_none, mask_custom])
    @pytest.mark.parametrize("silu_scale", [1 / 256])
    @pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
    def test_hstu_dens_normal(self, batch_size, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                              data_type):
        self.execute(batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale, data_type)

```

注：上述用例为normal格式简易调用场景，更详细精度、多场景测试请参考用例[RecSDK/cust_op/test/hstu_dense/torch/test_hstu_dense_forward_demo.py](../../../../../test/hstu_dense/torch/test_hstu_dense_forward_demo.py)


##### hstu_dense_backward接口

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


def generate_tensor(batch_size, max_seq_len, num_heads, attention_dim, mask_type, data_type):
    grad = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)
    q = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)
    k = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)
    v = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)

    bias = torch.empty(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type).uniform_(-1, 1)

    if mask_type == 0:
        mask = torch.tril(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type))
    elif mask_type == 1:
        mask = torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type))
    elif mask_type == 2:
        mask = None
    else:
        mask = torch.randint(0, 2, size=(batch_size, num_heads, max_seq_len, max_seq_len)).to(data_type)

    return grad, q, k, v, bias, mask


torch.npu.set_device(device_id)


class TestHstuNormalDemo:
    @staticmethod
    def custom_op_exec(grad, q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type):
        if enable_bias:
            q_grad, k_grad, v_grad, attn_bias_grad = torch.ops.mxrec.hstu_dense_backward(
                grad, q, k, v, mask, bias, "normal", mask_type, max_seq_len, silu_scale)
        else:
            q_grad, k_grad, v_grad, attn_bias_grad = torch.ops.mxrec.hstu_dense_backward(
                grad, q, k, v, mask, None, "normal", mask_type, max_seq_len, silu_scale)

        torch.npu.synchronize()
        return q_grad.cpu(), k_grad.cpu(), v_grad.cpu(), attn_bias_grad.cpu()

    def execute(self, batch_size, max_seq_len, head_num, head_dim, mask_type, silu_scale, enable_bias, data_type):
        grad, q, k, v, bias, mask = generate_tensor(batch_size, max_seq_len, head_num, head_dim, mask_type, data_type)

        grad_npu = grad.to(f"npu:{device_id}")
        q_npu = q.to(f"npu:{device_id}")
        k_npu = k.to(f"npu:{device_id}")
        v_npu = v.to(f"npu:{device_id}")
        bias_npu = None
        if enable_bias:
            bias_npu = bias.to(f"npu:{device_id}")
        mask_npu = None
        if mask_type == 0 or mask_type == 3:
            mask_npu = mask.to(f"npu:{device_id}")

        q_grad, k_grad, v_grad, attn_bias_grad = self.custom_op_exec(
            grad_npu, q_npu, k_npu, v_npu, bias_npu, mask_npu, mask_type, max_seq_len, silu_scale, enable_bias,
            data_type)

        torch.npu.synchronize()


    @pytest.mark.parametrize("batch_size", [2, 4])
    @pytest.mark.parametrize("max_seq_len", [256, 257, 512, 1234])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("head_dim", [32, 64])
    @pytest.mark.parametrize("mask_type", [0, 2, 3])
    @pytest.mark.parametrize("silu_scale", [0.0, 1.0 / 256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
    def test_hstu_dens_normal(self, batch_size, max_seq_len, head_num, head_dim, mask_type, silu_scale, enable_bias,
                              data_type):
        self.execute(batch_size, max_seq_len, head_num, head_dim, mask_type, silu_scale, enable_bias, data_type)

```

注：注：上述用例为normal格式简易调用场景，更详细精度、多场景测试请参考用例[RecSDK/cust_op/test/hstu_dense/torch/test_hstu_dense_backward_demo.py](../../../../../test/hstu_dense/torch/test_hstu_dense_backward_demo.py)