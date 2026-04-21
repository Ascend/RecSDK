# torch.ops.mxrec.hstu_backward_v2

```python3
torch.ops.mxrec.hstu_backward_v2(
    grad,
    q,
    k,
    v,
    max_seqlen_q,
    max_seqlen_k,
    seq_offset_q,
    seq_offset_k,
    rab=None,
    num_context=None,
    num_target=None,
    scale=0.0,
    target_group_size=0,
    alpha=1.0
) → (Tensor, Tensor, Tensor, Tensor)
```

HSTU (Hierarchical Sparse Transformer Unit) 算子的反向传播实现，用于计算 Query、Key、Value 以及 RAB注意力分数的梯度。

该算子实现 Transformer 注意力机制的反向传播，通过 CATLASS 模板库在昇腾 NPU 上高效执行矩阵乘法运算。

## 参数说明

- **grad** (*Tensor*) – 输出梯度张量。形状: `[totalSeqLenQ, heads, dimGV]`，数据类型: `float16, bfloat16`

- **q** (*Tensor*) – Query 张量。形状: `[totalSeqLenQ, heads, dimQK]`，数据类型: `float16, bfloat16`

- **k** (*Tensor*) – Key 张量。形状: `[totalSeqLenK, heads, dimQK]`，数据类型: `float16, bfloat16`

- **v** (*Tensor*) – Value 张量。形状: `[totalSeqLenK, heads, dimGV]`，数据类型: `float16, bfloat16`

- **max_seqlen_q** (*int*) – Query 的最大序列长度

- **max_seqlen_k** (*int*) – Key/Value 的最大序列长度

- **seq_offset_q** (*Tensor*) – Query 序列偏移量。形状: `[batchSize + 1]`，数据类型: `int32`

- **seq_offset_k** (*Tensor*) – Key/Value 序列偏移量。形状: `[batchSize + 1]`，数据类型: `int32`

- **rab** (*Tensor, optional*) – RAB。形状: `[batchSize, heads, maxSeqLenQ, maxSeqLenK]`，数据类型: `float16, bfloat16`。默认值: `None`

- **num_context** (*Tensor, optional*) – 每个 batch 的 Context 长度。形状: `[batchSize]`。默认值: `None`

- **num_target** (*Tensor, optional*) – 每个 batch 的 Target 长度。形状: `[batchSize]`。默认值: `None`

- **scale** (*float, optional*) – 缩放因子。默认值: `0.0`（实际使用 `1.0 / maxSeqLenQ`）

- **target_group_size** (*int, optional*) – RAG 目标组大小。必须为 `1` 或 `3`。默认值: `0`

- **alpha** (*float, optional*) – Alpha 系数，用于 RAG 计算。默认值: `1.0`

## 支持的数据类型

`torch.float16, torch.bfloat16`

## Shape

- **输入**:
  - `grad`: `(totalSeqLenQ, heads, dimGV)`
  - `q`: `(totalSeqLenQ, heads, dimQK)`
  - `k`: `(totalSeqLenK, heads, dimQK)`
  - `v`: `(totalSeqLenK, heads, dimGV)`
  - `seqOffsetQ`: `(batchSize + 1)`
  - `seqOffsetK`: `(batchSize + 1)`
  - `rab` (optional): `(batchSize, heads, maxSeqLenQ, maxSeqLenK)`

- **输出**: 4 个 Tensor 的元组
  - `qGrad`: `(totalSeqLenQ, heads, dimQK)`，dtype: `float16, bfloat16`
  - `kGrad`: `(totalSeqLenK, heads, dimQK)`，dtype: `float16, bfloat16`
  - `vGrad`: `(totalSeqLenK, heads, dimGV)`，dtype: `float16, bfloat16`
  - `rabGrad`: `(batchSize, heads, maxSeqLenQ, maxSeqLenK)`，dtype: `float16, bfloat16`（仅当 `rab` 不为 None 时返回）

## 约束条件

- 仅支持 `float16/bfloat16` 数据类型
- `heads` 必须满足: `1 ≤ heads ≤ 16`
- `dimQK` 和 `dimGV` 必须为 16 的倍数
- 当 `numContext` 或 `numTarget` 不为 None 时，`targetGroupSize` 必须为 `1` 或 `3`
- `headQ` 必须等于 `headK`（当前仅支持 MHA，不支持 GQA）

## 使用示例

```python
>>> import torch
>>> import torch_npu

>>> # 参数设置
>>> batch_size = 2
>>> heads = 4
>>> dim_qk = 64
>>> dim_gv = 64
>>> max_seqlen_q = 128
>>> max_seqlen_k = 128
>>> total_seq_len_q = 200
>>> total_seq_len_k = 200

>>> # 构造输入数据
>>> grad = torch.randn(total_seq_len_q, heads, dim_gv, dtype=torch.float16, device="npu:0")
>>> q = torch.randn(total_seq_len_q, heads, dim_qk, dtype=torch.float16, device="npu:0")
>>> k = torch.randn(total_seq_len_k, heads, dim_qk, dtype=torch.float16, device="npu:0")
>>> v = torch.randn(total_seq_len_k, heads, dim_gv, dtype=torch.float16, device="npu:0")

>>> # 序列偏移
>>> seq_lens_q = torch.tensor([100, 100], dtype=torch.int32)
>>> seq_lens_k = torch.tensor([100, 100], dtype=torch.int32)
>>> seq_offset_q = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(seq_lens_q, dim=0)])
>>> seq_offset_k = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(seq_lens_k, dim=0)])

>>> # 调用算子
>>> q_grad, k_grad, v_grad, rab_grad = torch.ops.mxrec.hstu_backward_v2(
...     grad, q, k, v,
...     max_seqlen_q, max_seqlen_k,
...     seq_offset_q.to("npu"),
...     seq_offset_k.to("npu"),
...     None, None, None,
...     0.0, 0, 1.0
... )

>>> # 输出 shape
>>> q_grad.shape
torch.Size([200, 4, 64])
>>> k_grad.shape
torch.Size([200, 4, 64])
>>> v_grad.shape
torch.Size([200, 4, 64])
```

## 返回值

*tuple[Tensor, Tensor, Tensor, Tensor]* – 包含 4 个梯度张量的元组:

- **qGrad** (*Tensor*) – Query 的梯度，形状: `(totalSeqLenQ, heads, dimQK)`
- **kGrad** (*Tensor*) – Key 的梯度，形状: `(totalSeqLenK, heads, dimQK)`
- **vGrad** (*Tensor*) – Value 的梯度，形状: `(totalSeqLenK, heads, dimGV)`
- **rabGrad** (*Tensor*) – RAG 注意力分数的梯度，形状: `(batchSize, heads, maxSeqLenQ, maxSeqLenK)`。如果输入 `rab` 为 `None`，则返回空 Tensor

## 算子原理

HSTU Backward 算子实现以下反向传播计算:

1. **V 梯度计算**: `grad_V = grad_O^T @ P`，其中 P 是注意力概率矩阵
2. **Q 梯度计算**: `grad_Q = grad_S @ K`，其中 `grad_S = grad_P * d(softmax)/d(S)`
3. **K 梯度计算**: `grad_K = grad_S^T @ Q`
4. **RAG 梯度计算**: 考虑 RAG 稀疏掩码的梯度传播

算子使用 CATLASS 模板库实现，包含以下优化:

- 双缓冲 (Double Buffer) 隐藏内存访问延迟
- 跨核同步机制
- L2 Cache 优化
- Swizzle 优化减少 Bank 冲突

## 依赖

算子依赖CATLASS源码, 编译前需要初始化submodule：
```shell
git submodule update --init --recursive
```
