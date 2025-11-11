**说明**

本算子仅支持NPU调用

# 产品支持情况
| 硬件型号              | 是否支持                  |
| -------------------- | ------------------------ |
| Atlas A2训练系列产品  | 是  |
| Atlas A3训练系列产品  | 是  |
| Atlas 推理系列产品    | 是  |

# hstu_dense_forward算子文件结构

```shell
-- hstu_dense_forward
   |-- onnx_plugin   # hstu_dense_forward支持onnx模型转换
   |-- v220
      |-- op_host    # hstu_dense_forward算子Host侧实现
      |-- op_kernel  # hstu_dense_forward算子Kernel侧实现
      |-- pic        # 算子实现原理图
      |-- hstu_dense_forward.json    # 算子原型配置
      |-- README.md  # hstu_dense_forward算子说明文档
      |-- run.sh     # hstu_dense_forward算子安装脚本
```

# 功能

推荐场景下，使用Hstu融合算子实现推荐场景中注意力机制。


# 算子实现原理

1. 计算公式

$$
HSTU(q, k, v, mask, bias, siluScale) = (Silu(qk_{}^{T} + bias) \times siluScale \times mask)v
$$

2. 数据格式

输入参数q, k, v数据格式为normal或者jagged。
* normal格式：shape为[B, S, N, D]的4维数据格式，排布如下图所示：

![alt text](pic/hstu_normal.png)

* jagged格式：shape为[s_b, N, D]的3维数据格式，排布如下图所示：

![alt text](pic/hstu_jagged.png)

3. 计算原理

![alt text](pic/hstu_image.png)

4. 计算逻辑

```python
def hstu_dense_forward(q_np, k_np, v_np, rel_attn_bias_np, invalid_attn_mask_np):
    q = torch.nn.Parameter(torch.Tensor(q_np).reshape(batch_size, max_seq_len, num_heads, attention_dim).to(dataType).npu(), 
        requires_grad=True)
    k = torch.nn.Parameter(torch.Tensor(k_np).reshape(batch_size, max_seq_len, num_heads, attention_dim).to(dataType).npu(), 
        requires_grad=True)
    v = torch.nn.Parameter(torch.Tensor(v_np).reshape(batch_size, max_seq_len, num_heads, attention_dim).to(dataType).npu(), 
        requires_grad=True)
    real_attn_bias = torch.nn.Parameter(torch.Tensor(rel_attn_bias_np).to(dataType).npu(), requires_grad=True)   
    invalid_attn_mask = torch.Tensor(invalid_attn_mask_np).to(dataType).npu()
     
    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        q,
        k,
    )
    qk_attn = qk_attn + rel_attn_bias

    qk_attn = F.silu(qk_attn) / max_seq_len
    qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
    attn_output = torch.einsum(
            "bhnm,bmhd->bnhd",
            qk_attn,
            v
        ).reshape(batch_size, seq_len, num_heads * attention_dim)

    return npu2cpu(attn_output)

```

# 算子输入与输出

## Atlas A2/A3训练产品

|  名称  |  输入/输出  |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |
|  q | 输入 | float32/float16/bfloat16 | [B, S, N, D]/<br>[s_b, N, D] | B∈[1, 2048]<br>S∈[1, 20480]<br>N∈[1, 16]<br>D∈[16, 512]且是16的倍数 | B:batch_size,表征批处理大小<br>S:seq_len,表征序列长度<br>N:head_num,表征头个数<br>D:head_dim,表征维度<br>s_b为jagged格式下各batch的实际序列长度之和 |
|  k | 输入 | float32/float16/bfloat16 | [B, S, N, D]/<br>[s_b, N, D] | 同q | 同q |
|  v | 输入 | float32/float16/bfloat16 | [B, S, N, D]/<br>[s_b, N, D] | 同q | 同q |
|  mask | 输入(可选) | float32/float16/bfloat16 | [B, N, S, S] | NA | S为模型最大序列长度max_seq_len<br>不使用mask时传入None，类型需与q一致 |
|  attn_bias | 输入(可选) | float32/float16/bfloat16 | [B, N, S, S] | NA | S为模型最大序列长度max_seq_len<br>不使用attn_bias时传入None，类型需与q一致 |
|  maskType | 输入(属性) | int | NA | 0：使用内置下三角mask，不需要传入mask<br>1：使用内置上三角mask，不需要传入mask(当前暂不支持)<br>2：不使用mask<br>3：使用自定义mask，此时mask需要用户定义并传入 | NA |
|  max_seq_len | 输入(属性) | int | NA | [1, 20480] | 表示模型最大序列长度 |
|  silu_scale | 输入(属性) | float | NA | NA | 支持用户传入自定义，不传入时默认为1/max_seq_len |
|  layout | 输入(属性) | string | NA | "normal":代表q,k,v数据格式为[B, S, N, D]<br>"jagged":代表q,k,v数据格式为[s_b, N, D] | NA |
|  seq_offsets | 输入(属性, 可选) | list[int64] | NA | NA | 表示每个batch的实际序列长度偏移，从0开始递增，需用户自行保证合法性，仅在jagged格式下生效 |
|  attn_output | 输出 | float32/float16/bfloat16 | [B, S, N, D]/<br>[s_b, N, D] | 同q | 同q |

## Atlas 推理系列产品
|  名称  |  输入/输出  |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |
|  q | 输入 | float16 | [B, S, N, D] | B∈[1, 2048]<br>S∈[128, 4096]且是128的倍数<br>N∈[1, 8]<br>D∈[16, 128]且是16的倍数 | B:batch_size,表征批处理大小<br>S:seq_len,表征序列长度<br>N:head_num,表征头个数<br>D:head_dim,表征维度 |
|  k | 输入 | float16 | [B, S, N, D] | 同q | 同q |
|  v | 输入 | float16 | [B, S, N, D] | 同q | 同q |
|  mask | 输入 | float16 | [B, 1, S, S] | NA | S为模型最大序列长度max_seq_len，mask为基于下三角的自定义，需要用户基于下三角自定义传入 |
|  attn_bias | 输入(可选) | float16 | [B, N, S, S] | NA | S为模型最大序列长度max_seq_len<br>不使用attn_bias时传入None |
|  maskType | 输入(属性) | int | NA | 3：使用自定义mask，此时mask需要用户定义并传入 | NA |
|  max_seq_len | 输入(属性) | int | NA | [128, 4096]且为128的倍数 | 表示模型最大序列长度 |
|  silu_scale | 输入(属性) | float | NA | NA | 支持用户传入自定义，不传入时默认为1/max_seq_len |
|  layout | 输入(属性) | string | NA | "normal":代表q,k,v数据格式为[B, S, N, D] | NA |
|  seq_offsets | 输入(属性, 可选) | list[int64] | NA | NA | 不支持使用 |
|  attn_output | 输出 | float16 | [B, S, N, D] | 同q | 同q |

注：
* B,S,N,D四个维度数据均不能为0，为0时算子输入为空数据，不会执行算子计算。
* 其中B,S,N参数影响bias、mask占用显存大小，请根据实际内存合理设置参数大小。

# 算子编译部署

算子编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"1.算子编译"章节。

注：详细算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/2.6.0/hstu/README.md)