**说明**

本算子仅支持NPU调用。

# 产品支持情况

| 硬件型号           | 是否支持 |
|----------------|------|
| Atlas A2训练系列产品 | 是    |
| Atlas A3训练系列产品 | 是    |

## hstu_dense_backward算子目录层级

```shell
├── v220
    ├── hstu_dense_backward.json    # 算子原型配置
    ├── op_host    # hstu_dense_backward算子Host侧实现
    ├── op_kernel  # hstu_dense_backward算子Kernel侧实现
    ├── README.md  # hstu_dense_backward算子说明文档
    └── run.sh     # hstu_dense_backward算子安装脚本
```


# 功能
算子的主要功能是实现HSTU融合算子的反向hstu_dense_backward

# 算子实现原理
```python
q_trans = q.permute(0, 2, 1, 3)
k_trans = k.permute(0, 2, 1, 3)
v_trans = v.permute(0, 2, 1, 3)
g_trans = grad.permute(0, 2, 1, 3)
qk_result = torch.matmul(q_trans, k_trans.permute(0, 1, 3, 2)) 
gv_result = torch.matmul(g_trans, v_trans.permute(0, 1, 3, 2))

qk_add_atten_bias = qk_result + attn_bias
attn_score_forward = F.silu(qk_add_atten_bias) / max_seq_len
attn_score_forward = attn_score_forward * invalid_attn_mask.unsqueeze(0).unsqueeze(0)

score_grad = gv_result / max_seq_len * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
qk_attn_grad = (F.sigmoid(qk_add_atten_bias)*(1+qk_add_atten_bias*(1-F.sigmoid(qk_add_atten_bias)))) * score_grad
rel_attn_bias_grad = qk_attn_grad.sum(1, keepdim=True)

# # V grad
v_grad = torch.matmul(attn_score_forward.permute(0, 1, 3, 2), g_trans)

# # Q K grad
q_grad = torch.matmul(qk_attn_grad, k_trans)
k_grad = torch.matmul(qk_attn_grad.permute(0, 1, 3, 2), q_trans)
```


b) 算子参数说明：
# 算子输入与输出
| 名称            | 输入/输出   | 参数类型      | 数据类型          | 数据格式                                  | 范围           | 说明                                                                     |
|---------------|---------|-----------|---------------|---------------------------------------|--------------|------------------------------------------------------------------------|
| grad        | 输入      | Tensor    | float32/bf16/fp16 | [B_S, Head, Dim] 或者 [B, S, Head, Dim]                          |         参数范围参考前向hstu_dense_forward算子      |                                                           前向输出out反向求导            |
| q       | 输入      | Tensor  | float32/bf16/fp16   | [B_S, Head, Dim] 或者 [B, S, Head, Dim]                                      | 参数范围参考前向hstu_dense_forward算子 | Q，Jagged模式下传入第一种shape，Normal模式先传入第二种 |
| k       | 输入      | Tensor  | float32/bf16/fp16   | [B_S, Head, Dim]   或者 [B, S, Head, Dim]                                    | 参数范围参考前向hstu_dense_forward算子 | K，Shape和类型与Q一致 |
| v       | 输入      | Tensor  | float32/bf16/fp16   | [B_S, Head, Dim]  或者 [B, S, Head, Dim]                                     | 参数范围参考前向hstu_dense_forward算子 | V，Shape和类型与Q一致 |
| mask       | 输入      | Tensor  | float32/bf16/fp16   | [B, Head , S, S]                                      | 参数范围参考前向hstu_dense_forward算子 | mask 使用内置mask或者不需要mask则为None，类型与Q一致 |
| attn_bias       | 输入      | Tensor  | float32/bf16/fp16   | [B, Head , S, S]   | 参数范围参考前向hstu_dense_forward算子 | attn_bias，类型与Q一致 |
| layout       | 输入      | str  | str   | -                                      | ["jagged", "normal"] | QKV内存布局即HSTU的两种模式 |
| mask_type       | 输入      | int  | int   | -                                      | [0, 2, 3] | 使用的mask方式，0表示上三角 2表示不用mask 3表示使用传入的自定义mask |
| max_seq_len       | 输入      | int  | int   | -                                      | 参数范围参考前向hstu_dense_forward算子 | 最大长度 |
| silu_scale       | 输入      | float  | float   | -| - | silu前的scale化 |  
| seq_offsets       | 输入      | List  | int   | [BatchSize+1 ]   | 	参数范围参考前向hstu_dense_forward算子 | Batch中的每一个样本的实际长度 |
| q_grad       | 输入      | Tensor  | float32/bf16/fp16   | [B_S, Head, Dim] 或者 [B, S, Head, Dim]  |与Q保持一致 | Q的梯度 |
| k_grad       | 输入      | Tensor  | float32/bf16/fp16   | [B_S, Head, Dim] 或者 [B, S, Head, Dim]  |与K一致 | k的梯度 |
| v_grad       | 输入      | Tensor  | float32/bf16/fp16   | [B_S, Head, Dim] 或者 [B, S, Head, Dim]  |与attn_bias保持一致 | v的梯度 |
| attn_bias_grad       | 输入      | Tensor  | float32/bf16/fp16   | [B, Head , S, S] | - | attn_bias的反向梯度|

备注： B表示Batch的大小 B_S表示样本的实际长度之和，Head表示Head Num，S表示样本序列长度，Dim表示Attention Dim。参数范围参考前向hstu_dense_forward算子。

# 算子编译部署

算子编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"1.算子编译"章节。

注：详细算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/hstu/README.md)
