**说明**

本算子仅支持NPU调用。

# 产品支持情况

| 硬件型号             | 是否支持 |
| -------------------- | -------- |
| Atlas A2训练系列产品 | 是       |
| Atlas A3训练系列产品 | 是       |

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

## Normal Layout 实现原理

```python
# 1. 计算 QK 和 GV 矩阵乘法
qk = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))
gv = torch.matmul(grad.permute(0, 2, 1, 3), v.permute(0, 2, 3, 1))

# 2. 转换为 float 类型进行计算（提高精度）
qk = qk.float()
gv = gv.float()

# 3. 处理 mask（如果需要）
if mask_type == 0 or mask_type == 3:
    mask = mask.float()

# 4. 添加 attention bias（如果提供）
if enable_bias:
    bias = bias.float()
    qkb = qk + bias
else:
    qkb = qk

# 5. 计算 silu_scale
real_silu_scale = 1 / max_seq_len if silu_scale == 0.0 else silu_scale

# 6. 计算 attention score（前向传播中的 score）
if mask_type == 0 or mask_type == 3:
    score = F.silu(qkb) * real_silu_scale * mask
else:
    score = F.silu(qkb) * real_silu_scale

# 7. 计算 V 的梯度
score = score.to(data_type)  # 转换回原始数据类型
v_grad = torch.matmul(score.permute(0, 1, 3, 2), grad.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

# 8. 计算 attention bias 的梯度（SiLU 的导数）
if mask_type == 0 or mask_type == 3:
    attn_bias_grad = gv * real_silu_scale * mask * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
else:
    attn_bias_grad = gv * real_silu_scale * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))

attn_bias_grad = attn_bias_grad.to(data_type)  # 转换回原始数据类型

# 9. 计算 K 和 Q 的梯度
k_grad = torch.matmul(attn_bias_grad.permute(0, 1, 3, 2), q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
q_grad = torch.matmul(attn_bias_grad, k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
```

## Jagged Layout 实现原理

Jagged Layout 的实现原理与 Normal Layout 类似，主要区别在于：

1. **数据格式转换**：需要将 jagged 格式（`[s_b, N, D]`）转换为 dense 格式（`[B, S, N, D]`）进行计算
2. **Alpha 参数**：在计算 `qkb` 和 `attn_bias_grad` 时需要乘以 `alpha` 参数
3. **结果转换**：计算完成后需要将结果从 dense 格式转换回 jagged 格式

```python
# 1. 将 jagged 格式转换为 dense 格式
grad_dens = jagged_to_dense(grad, seq_lens, max_seq_len, head_nums, head_dim)
q_dens = jagged_to_dense(q, seq_lens, max_seq_len, head_nums, head_dim)
k_dens = jagged_to_dense(k, seq_lens, max_seq_len, head_nums, head_dim)
v_dens = jagged_to_dense(v, seq_lens, max_seq_len, head_nums, head_dim)

# 2. 计算 QK 和 GV 矩阵乘法（与 Normal Layout 相同）
qk = torch.matmul(q_dens.permute(0, 2, 1, 3), k_dens.permute(0, 2, 3, 1))
gv = torch.matmul(grad_dens.permute(0, 2, 1, 3), v_dens.permute(0, 2, 3, 1))

# 3. 转换为 float 类型进行计算
qk = qk.float()
gv = gv.float()
bias = bias.float()

# 4. 处理 mask（如果需要）
if mask_type == 0 or mask_type == 3:
    mask = mask.float()

# 5. 添加 attention bias 并应用 alpha
if enable_bias:
    qkb = qk + bias
else:
    qkb = qk
qkb = qkb * alpha  # Jagged Layout 特有的 alpha 参数

# 6. 计算 silu_scale
real_silu_scale = 1 / max_seq_len if silu_scale == 0.0 else silu_scale

# 7. 计算 attention score
if mask_type == 0 or mask_type == 3:
    score = F.silu(qkb) * real_silu_scale * mask
else:
    score = F.silu(qkb) * real_silu_scale

# 8. 计算 V 的梯度
score = score.to(data_type)
v_grad_dens = torch.matmul(score.permute(0, 1, 3, 2), grad_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

# 9. 计算 attention bias 的梯度（应用 alpha）
if mask_type == 0 or mask_type == 3:
    bias_grad = gv * real_silu_scale * mask * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
else:
    bias_grad = gv * real_silu_scale * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
bias_grad = bias_grad * alpha  # Jagged Layout 特有的 alpha 参数
bias_grad = bias_grad.to(data_type)

# 10. 计算 K 和 Q 的梯度
k_grad_dens = torch.matmul(bias_grad.permute(0, 1, 3, 2), q_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
q_grad_dens = torch.matmul(bias_grad, k_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

# 11. 将结果从 dense 格式转换回 jagged 格式
q_grad = dense_to_jagged(q, q_grad_dens, seq_lens)
k_grad = dense_to_jagged(k, k_grad_dens, seq_lens)
v_grad = dense_to_jagged(v, v_grad_dens, seq_lens)
```


b) 算子参数说明：

# 算子输入与输出

## 输入参数

| 名称         | 输入/输出 | 参数类型          | 数据类型                               | 数据格式                                    | 范围                                                         | 说明                                                         |
| ------------ | --------- | ----------------- | -------------------------------------- | ------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| grad         | 输入      | Tensor (REQUIRED) | float32/float16/bf16                   | Normal: [B, S, N, D]<br>Jagged: [s_b, N, D] | B∈[1, 2048]<br>S∈[1, 20480]<br>N∈[1, 16]<br>D∈[16, 512]且是16的倍数 | 前向输出out的反向梯度，Jagged模式下s_b为总序列长度           |
| q            | 输入      | Tensor (REQUIRED) | float32/float16/bf16<br>与grad类型一致 | Normal: [B, S, N, D]<br>Jagged: [s_b, N, D] | 同grad                                                       | Q张量，Jagged模式下传入[s_b, N, D]，Normal模式下传入[B, S, N, D] |
| k            | 输入      | Tensor (REQUIRED) | float32/float16/bf16<br>与grad类型一致 | Normal: [B, S, N, D]<br>Jagged: [s_b, N, D] | 同grad                                                       | K张量，Shape和类型与Q一致                                    |
| v            | 输入      | Tensor (REQUIRED) | float32/float16/bf16<br>与grad类型一致 | Normal: [B, S, N, D]<br>Jagged: [s_b, N, D] | 同grad                                                       | V张量，Shape和类型与Q一致                                    |
| mask         | 输入      | Tensor (OPTIONAL) | float32/float16/bf16<br>与grad类型一致 | [B, N, S, S]                                | 同grad                                                       | mask张量，当mask_type=3时必须提供，类型与grad一致            |
| attn_bias    | 输入      | Tensor (OPTIONAL) | float32/float16/bf16<br>与grad类型一致 | [B, N, S, S]                                | 同grad                                                       | attn_bias张量，类型与grad一致                                |
| seq_offset_q | 输入      | Tensor (OPTIONAL) | int64                                  | [BatchSize+1]                               | BatchSize∈[1, 2048]                                          | Jagged模式下必须提供，表示每个batch的序列长度偏移，至少包含2个元素 |
| num_context  | 输入      | Tensor (OPTIONAL) | int32/int64                            | [B]                                         | B∈[1, 2048]                                                  | Context掩码长度，当提供时必须与num_target和target_group_size一起提供，且其中的值范围[1, 128] |
| num_target   | 输入      | Tensor (OPTIONAL) | int32/int64                            | [B]                                         | B∈[1, 2048]                                                  | Target掩码长度，当提供时必须与num_context和target_group_size一起提供, 且其中的值范围[1, 512] |

## 属性参数

| 名称              | 参数类型        | 数据类型 | 默认值   | 范围/取值            | 说明                                                         |
| ----------------- | --------------- | -------- | -------- | -------------------- | ------------------------------------------------------------ |
| layout            | Attr            | string   | "normal" | ["normal", "jagged"] | QKV内存布局，normal表示[B, S, N, D]，jagged表示[s_b, N, D]   |
| mask_type         | Attr            | int      | -        | [0, 2, 3]            | mask类型：0表示使用内置下三角mask（TRIL），2表示不使用mask，3表示使用自定义mask（CUSTOM）<br>注意：1（TRIU）当前不支持 |
| max_seq_len       | Attr            | int      | -        | [1, 20480]           | 模型最大序列长度<br>**反向传播特殊约束**: Normal模式下seqLen必须等于max_seq_len，Jagged模式下必须与mask和attn_bias的shape中S相等 |
| silu_scale        | Attr            | float    | 0.0      | 任意值               | SiLU激活函数前的缩放因子，如果为0.0则自动计算为1.0/max_seq_len |
| target_group_size | Attr (OPTIONAL) | int      | 0        | {1, 3}               | Target分组大小，当提供时值必须在{1, 3}中，且必须与num_context和num_target一起提供 |
| alpha             | Attr (OPTIONAL) | float    | 1.0      | 任意值               | 缩放因子，默认值为1.0                                        |

## 输出参数

| 名称           | 输入/输出 | 参数类型          | 数据类型                               | 数据格式                                    | 范围            | 说明                                                 |
| -------------- | --------- | ----------------- | -------------------------------------- | ------------------------------------------- | --------------- | ---------------------------------------------------- |
| q_grad         | 输出      | Tensor (OPTIONAL) | float32/float16/bf16<br>与grad类型一致 | Normal: [B, S, N, D]<br>Jagged: [s_b, N, D] | 与q保持一致     | Q的梯度                                              |
| k_grad         | 输出      | Tensor (OPTIONAL) | float32/float16/bf16<br>与grad类型一致 | Normal: [B, S, N, D]<br>Jagged: [s_b, N, D] | 与k一致         | K的梯度                                              |
| v_grad         | 输出      | Tensor (OPTIONAL) | float32/float16/bf16<br>与grad类型一致 | Normal: [B, S, N, D]<br>Jagged: [s_b, N, D] | 与v一致         | V的梯度                                              |
| attn_bias_grad | 输出      | Tensor (OPTIONAL) | float32/float16/bf16<br>与grad类型一致 | [B, N, S, S]                                | 与attn_bias一致 | attn_bias的反向梯度，如果未提供attn_bias则返回空张量 |

## 接口范围限制说明

由于反向算子通过PTA层进行调用不能直调，参数限制和范围晴参考PTA侧(../../../../framework/torch_plugin/torch_library/hstu/README.md)


# 算子编译部署

算子编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"1.算子编译"章节。

注：详细算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/hstu/README.md)
