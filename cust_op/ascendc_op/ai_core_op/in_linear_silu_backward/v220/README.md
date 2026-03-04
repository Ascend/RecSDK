**说明**

本算子仅支持NPU调用。

# 产品支持情况
| 硬件型号              | 是否支持                  |
| -------------------- | ------------------------ |
| Atlas A2训练系列产品  | 是  |
| Atlas A5训练系列产品    | 是  |

# in_linear_silu_backward算子目录层级
```shell
-- in_linear_silu_backward
   |-- v220
      |-- op_host                 # 算子host侧实现
      |-- op_kernel               # 算子kernel侧实现
      |-- in_linear_silu_backward.json   # 算子原型配置
      |-- README.md               # 算子说明文档
      |-- run.sh                  # 算子编译部署脚本
```

# 功能

in_linear_silu_backward是in_linear_silu算子的反向传播实现，用于计算输入x、权重weight和偏置bias的梯度。

# 算子实现原理

算子工作原理说明：
1. 输入张量x ND格式，支持FLOAT16、FLOAT、BFLOAT16类型，shape = (m, k) 不可为空
2. 输入张量weight ND格式，支持FLOAT16、FLOAT、BFLOAT16类型，shape = (n, k), 如果为(k, n)需要转置后传入 不可为空
3. 输入张量bias ND格式，支持FLOAT类型，shape = (n,) 可选
4. 输入张量user_grad、value_grad、query_grad、key_grad ND格式，支持FLOAT16、FLOAT、BFLOAT16类型，分别对应前向传播输出user、value、query、key的梯度
5. 输入张量linear_output ND格式，支持FLOAT16、FLOAT、BFLOAT16类型，来自前向传播的中间结果
6. attr属性 split_arg_list(List[int], 计算输入)： User、Value、Query、Key的长度，4个int类型的List, 成员必须为16的倍数，4个数总和为n，不可为空
7. attr属性 isTrans(Bool, 计算输入)： 权重是否需要转置，可选，默认为true
8. 输出张量x_grad、weights_grad、bias_grad，分别对应输入x、权重weight和偏置bias的梯度
9. 请注意算子输入shape受显存大小限制。


例如：

输入:
```python
m, k = 1024, 128
n = 512
seg_len = int(n / 4)
split_arg_list = [seg_len, seg_len, seg_len, seg_len]

x = torch.rand((m, k), dtype=torch.float16)
weight = torch.rand((n, k), dtype=torch.float16)
bias = torch.rand((n,), dtype=torch.float32)

# 前向传播结果
linear_output = torch.rand((m, n), dtype=torch.float16)

# 梯度输入
user_grad = torch.rand((m, seg_len), dtype=torch.float16)
value_grad = torch.rand((m, seg_len), dtype=torch.float16)
query_grad = torch.rand((m, seg_len), dtype=torch.float16)
key_grad = torch.rand((m, seg_len), dtype=torch.float16)
```

输出：
```python
# x_grad: shape [m, k], 类型与x相同
# weights_grad: shape [n, k], 类型与weight相同
# bias_grad: shape [n,], 类型与bias相同
x_grad, weights_grad, bias_grad = in_linear_silu_backward(
    x, weight, bias, user_grad, value_grad, query_grad, key_grad, linear_output, 
    split_arg_list=split_arg_list, isTrans=True
)
```

# 算子输入与输出
| 名称      | 输入/输出 | 参数类型 | 数据类型         | 数据格式       | 范围         | 说明                                  |
|---------|------------|------|--------------|------------|------------|----------------------------------------|
| x       | 输入       | Tensor | float32/float16/bfloat16 | [seq_len, embed_dim] | embed_dim取值范围[16, 8192] |   embed_dim为16的整数倍              |
| weight   | 输入       | Tensor | float32/float16/bfloat16 | [hidden_size, embed_dim] | hidden_size取值范围[64, 20480] |   hidden_size为16的整数倍，hidden_size = sum(split_arg_list), 数据类型与x保持一致            |
| bias    | 输入       | Tensor | float32 | [hidden_size] | hidden_size取值范围[64, 20480] |   hidden_size为16的整数倍，可选参数            |
| user_grad | 输入       | Tensor | float32/float16/bfloat16 | [seq_len, H_u] | H_u取值范围[16, 8192] |   H_u为16的整数倍, 数据类型与x保持一致          |
| value_grad | 输入       | Tensor | float32/float16/bfloat16 | [seq_len, H_v] | H_v取值范围[16, 8192] |   H_v为16的整数倍, 数据类型与x保持一致          |
| query_grad | 输入       | Tensor | float32/float16/bfloat16 | [seq_len, H_q] | H_q取值范围[16, 8192] |   H_q为16的整数倍 , 数据类型与x保持一致         |
| key_grad | 输入       | Tensor | float32/float16/bfloat16 | [seq_len, H_k] | H_k取值范围[16, 8192] |   H_k为16的整数倍 , 数据类型与x保持一致         |
| linear_output | 输入       | Tensor | float32/float16/bfloat16 | [seq_len, hidden_size] | hidden_size取值范围[64, 20480] |   hidden_size为16的整数倍，来自前向传播的中间结果，数据类型与x保持一致            |
| split_arg_list | 输入(属性)  | ListInt | int   | [H_u, H_v, H_q, H_k]          |   sum(split_arg_list)=hidden_size       |长度为4，不可为空，每个元素为16的整数倍     |
| isTrans | 输入(属性)  | Bool | bool   | -          |   可选，默认为true       |权重是否需要转置     |
| x_grad | 输出     | Tensor | float32/float16/bfloat16 | [seq_len, embed_dim] | embed_dim取值范围[16, 8192] |   embed_dim为16的整数倍, 数据类型与x保持一致          |
| weights_grad | 输出     | Tensor | float32/float16/bfloat16 | [hidden_size, embed_dim] | hidden_size取值范围[64, 20480] |   hidden_size为16的整数倍, 数据类型与weight保持一致          |
| bias_grad | 输出     | Tensor | float32 | [hidden_size] | hidden_size取值范围[64, 20480] |   hidden_size为16的整数倍, 仅当输入包含bias时输出         |

# 算子编译部署

算子编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。

注：详细算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/in_linear_silu_backward/README.md)