**使用pytorch框架调用方式调用in_linear_silu算子**

该样例基于Pytorch2.6.0、python3.11.0运行

### Pytorch框架对外接口原型

#### in_linear_silu 接口
```python
torch.ops.mxrec.distance_in_linear_silu(Tensor x, Tensor weight, Tensor bias, int[] attr_dict) -> Tensor[]
```

#### in_linear_silu_backward 接口
```python
torch.ops.mxrec.in_linear_silu_backward(
    Tensor x, Tensor weight, Tensor? bias,
    Tensor user_grad, Tensor value_grad, Tensor query_grad, Tensor key_grad,
    Tensor linear_output,
    int[] attr_dict) -> Tensor[]
```

#### in_linear_silu 自动微分接口
```python
torch.ops.mxrec.in_linear_silu(Tensor x, Tensor weight, Tensor bias, int[] attr_dict) -> Tensor[]
```

### 参数说明

### torch.ops.mxrec.distance_in_linear_silu接口

|  名称  |  输入/输出  | 参数类型 |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |  ----  |
|  x | 输入 | Tensor | float16/bfloat16 | [m, k] | m∈[1, +∞)<br>k∈[16, 8192]且是16的倍数 | 输入张量，不可为空 |
|  weight | 输入 | Tensor | float16/bfloat16 | [n, k] | n∈[64, 32768]且是16的倍数<br>n必须是4*k的倍数 | 权重张量，不可为空 |
|  bias | 输入 | Tensor | float32 | [n] | n∈[64, 32768]且是16的倍数 | 偏置张量，不可为空 |
|  split_arg_list | 输入(属性) | int[] | int[] | [H_u, H_v, H_q, H_k] | 每个元素∈[16, 8192]且是16的倍数<br>sum(split_arg_list)=n | 分割参数，长度为4，不可为空 |
|  user_out | 输出 | Tensor | float16/bfloat16 | [m, H_u] | 同x的m<br>H_u∈[16, 8192]且是16的倍数 | 分割后的user输出张量 |
|  value_out | 输出 | Tensor | float16/bfloat16 | [m, H_v] | 同x的m<br>H_v∈[16, 8192]且是16的倍数 | 分割后的value输出张量 |
|  query_out | 输出 | Tensor | float16/bfloat16 | [m, H_q] | 同x的m<br>H_q∈[16, 8192]且是16的倍数 | 分割后的query输出张量 |
|  key_out | 输出 | Tensor | float16/bfloat16 | [m, H_k] | 同x的m<br>H_k∈[16, 8192]且是16的倍数 | 分割后的key输出张量 |
|  linear_output_out | 输出 | Tensor | float16/bfloat16 | [m, n] | 同x的m<br>n∈[64, 32768]且是16的倍数 | 线性变换后的中间结果张量 |

### torch.ops.mxrec.in_linear_silu_backward接口

|  名称  |  输入/输出  | 参数类型 |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |  ----  |
|  x | 输入 | Tensor | float16/bfloat16 | [m, k] | m∈[1, +∞)<br>k∈[16, 8192]且是16的倍数 | 输入张量，不可为空 |
|  weight | 输入 | Tensor | float16/bfloat16 | [n, k] | n∈[64, 32768]且是16的倍数<br>n必须是4*k的倍数 | 权重张量，不可为空 |
|  bias | 输入(可选) | Tensor | float32 | [n] | n∈[64, 32768]且是16的倍数 | 偏置张量，可选 |
|  user_grad | 输入 | Tensor | float16/bfloat16 | [m, H_u] | 同x的m<br>H_u∈[16, 8192]且是16的倍数 | user输出的梯度张量 |
|  value_grad | 输入 | Tensor | float16/bfloat16 | [m, H_v] | 同x的m<br>H_v∈[16, 8192]且是16的倍数 | value输出的梯度张量 |
|  query_grad | 输入 | Tensor | float16/bfloat16 | [m, H_q] | 同x的m<br>H_q∈[16, 8192]且是16的倍数 | query输出的梯度张量 |
|  key_grad | 输入 | Tensor | float16/bfloat16 | [m, H_k] | 同x的m<br>H_k∈[16, 8192]且是16的倍数 | key输出的梯度张量 |
|  linear_output | 输入 | Tensor | float16/bfloat16 | [m, n] | 同x的m<br>n∈[64, 32768]且是16的倍数 | 前向传播的中间结果张量 |
|  split_arg_list | 输入(属性) | int[] | int[] | [H_u, H_v, H_q, H_k] | 每个元素∈[16, 8192]且是16的倍数<br>sum(split_arg_list)=n | 分割参数，长度为4，不可为空 |
|  isTrans | 输入(属性) | bool | bool | - | 默认为false | 权重是否需要转置，当前版本未使用 |
|  x_grad | 输出 | Tensor | float16/bfloat16 | [m, k] | 同x | 输入x的梯度张量 |
|  weight_grad | 输出 | Tensor | float16/bfloat16 | [n, k] | 同weight | 权重weight的梯度张量 |
|  bias_grad | 输出 | Tensor | float32 | [n] | 同bias | 偏置bias的梯度张量，仅当输入包含bias时输出 |

## 接口范围限制说明

本文档基于代码实现中的实际限制，详细说明各接口的参数范围限制和约束条件。

### in_linear_silu 接口范围限制

#### 输入张量维度要求
- **x, weight**: 必须是 **2D** 张量
  - `x`: 格式为 `[m, k]`
    - `m`: 序列长度或批次大小
    - `k`: 嵌入维度
  - `weight`: 格式为 `[n, k]`
    - `n`: 输出维度，必须等于 `sum(split_arg_list)`
    - `k`: 嵌入维度，必须与x的k一致
- **bias**: 必须是 **1D** 张量，格式为 `[n]`

#### 形状参数范围限制

| 参数 | 范围 | 倍数要求 | 说明 |
| ---- | ---- | -------- | ---- |
| **k (嵌入维度)** | [16, 8192] | 必须是 **16** 的倍数 | x的第2维，weight的第2维 |
| **n (输出维度)** | [64, 32768] | 必须是 **16** 的倍数 | weight的第1维，bias的第1维 |
| **split_arg_list元素** | [16, 8192] | 必须是 **16** 的倍数 | 每个分割维度的大小 |

#### 其他参数限制

| 参数 | 类型 | 范围/取值 | 说明 |
| ---- | ---- | --------- | ---- |
| **split_arg_list** | int[] | 长度为4 | 必须包含4个元素，分别对应user、value、query、key的维度 |
| **sum(split_arg_list)** | int | 等于n | 分割维度之和必须等于输出维度n |
| **n与k的关系** | int | n必须是4*k的倍数 | 输出维度必须是嵌入维度的4倍的整数倍 |

### 错误检查

接口会在以下情况抛出错误：

1. **维度检查失败**: 输入张量维度不符合要求
2. **范围检查失败**: 参数超出允许范围
3. **倍数检查失败**: 各维度不是16的倍数
4. **形状一致性检查失败**: 
   - weight的第2维与x的第2维不一致
   - bias的第1维与weight的第1维不一致
   - sum(split_arg_list)与weight的第1维不一致
5. **n与k的关系检查失败**: n不是4*k的倍数

### 运行算子样例

#### 算子编译与部署

算子编译部署请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。

#### Pytorch编译

Pytorch框架适配层编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子适配层编译"。

#### 算子调用示例,以下以pytest方式调用为例

##### in_linear_silu接口

```python
import os
import sys
import sysconfig
import numpy as np
import pytest
import torch
import torch.nn.functional as F

torch.npu.config.allow_internal_format = False

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

def generate_input(m, k, n, split_list, data_type=torch.float16, device_id=0):
    """
    生成in_linear_silu算子的输入数据
    
    参数:
    m: 序列长度
    k: 嵌入维度
    n: 输出维度
    split_list: 分割参数列表
    data_type: 数据类型
    device_id: NPU设备ID
    
    返回:
    x, weight, bias: 输入张量
    """
    # 生成x [m, k]
    x = torch.rand((m, k), dtype=data_type).uniform_(-1, 1)
    
    # 生成weight [n, k]
    weight = torch.rand((n, k), dtype=data_type).uniform_(-1, 1)
    
    # 生成bias [n]
    bias = torch.rand((n,), dtype=torch.float32).uniform_(-1, 1)
    
    return x.to(f"npu:{device_id}"), weight.to(f"npu:{device_id}"), bias.to(f"npu:{device_id}")

def test_in_linear_silu_forward():
    """
    测试in_linear_silu前向传播接口
    """
    # 设置参数
    device_id = 0
    m = 1024  # 序列长度
    k = 128   # 嵌入维度
    n = 512   # 输出维度 (必须是4*k的倍数)
    split_list = [128, 128, 128, 128]  # 分割参数列表
    data_type = torch.float16
    # 生成输入数据
    x, weight, bias = generate_input(m, k, n, split_list, data_type, device_id)
    
    # 调用前向传播接口
    outputs = torch.ops.mxrec.distance_in_linear_silu(x, weight, bias, split_list)

def test_in_linear_silu_autograd():
    """
    测试in_linear_silu自动微分接口
    """
    # 设置参数
    device_id = 0
    m = 1024  # 序列长度
    k = 128   # 嵌入维度
    n = 512   # 输出维度 (必须是4*k的倍数)
    split_list = [128, 128, 128, 128]  # 分割参数列表
    data_type = torch.float16
    
    # 生成输入数据
    x, weight, bias = generate_input(m, k, n, split_list, data_type, device_id)
    
    # 设置需要计算梯度
    x.requires_grad = True
    weight.requires_grad = True
    bias.requires_grad = True
    
    # 调用自动微分接口
    outputs = torch.ops.mxrec.in_linear_silu(x, weight, bias, split_list)
    user_out, value_out, query_out, key_out, linear_output_out = outputs
    
    # 计算损失并反向传播
    loss = user_out.sum() + value_out.sum() + query_out.sum() + key_out.sum()
    loss.backward()

if __name__ == "__main__":
    test_in_linear_silu_forward()
    test_in_linear_silu_autograd()
```

### 注意事项

1. **数据类型一致性**: 输入张量x和weight的数据类型必须一致，支持float16、bfloat16
2. **形状限制**: 所有维度参数必须是16的倍数，并且满足n = sum(split_arg_list)和n = 4*k*倍数的要求
3. **设备要求**: 所有输入张量必须在NPU设备上
4. **梯度计算**: 只有调用`in_linear_silu`接口时才会自动计算梯度
5. **bias参数**: 前向传播时bias是必填参数，反向传播时bias是可选参数