# torch.ops.mxrec.hstu_forward_v2

```python3
torch.ops.mxrec.hstu_forward_v2(
    q,
    k,
    v,
    mask=None,
    rab=None,
    mask_type=0,
    max_seq_len,
    max_seq_len_k=None,
    silu_scale=0.0,
    seq_offset,
    seq_offset_k=None,
    num_context=None,
    num_target=None,
    target_group_size=0,
    alpha=1.0
) → Tensor
```

HSTU (Hierarchical Sparse Transformer Unit) 算子的前向传播实现，用于计算基于 SiLU 激活的稀疏注意力输出。

该算子实现 Transformer 注意力机制的前向传播，通过 CATLASS 模板库在昇腾 NPU 上高效执行矩阵乘法运算，将 QK 矩阵乘法、SiLU 激活、缩放、掩码应用以及与 V 的矩阵乘法融合为单一算子。

## 参数说明

- **q** (*Tensor*) – Query 张量。形状: `[totalSeqLenQ, heads, dimQK]`，数据类型: `float16, bfloat16`

- **k** (*Tensor*) – Key 张量。形状: `[totalSeqLenK, heads, dimQK]`，数据类型: `float16, bfloat16`

- **v** (*Tensor*) – Value 张量。形状: `[totalSeqLenK, heads, dimGV]`，数据类型: `float16, bfloat16`

- **mask** (*Tensor, optional*) – 注意力掩码张量。形状: `[batchSize, heads, maxSeqLenQ, maxSeqLenK]`，数据类型: `float16, bfloat16`。默认值: `None`

- **rab** (*Tensor, optional*) – 相对注意力偏置 (Relative Attention Bias)。形状: `[batchSize, heads, maxSeqLenQ, maxSeqLenK]`，数据类型: `float16, bfloat16`。默认值: `None`

- **mask_type** (*int*) – 掩码类型。`0`: 使用内置下三角掩码（causal mask），无需传入 mask；`2`: 不使用掩码；`3`: 使用自定义 mask。默认值: `0`

- **max_seq_len** (*int*) – Query 的最大序列长度

- **max_seq_len_k** (*int, optional*) – Key/Value 的最大序列长度。默认值: `None`（与 `max_seq_len` 相同）

- **silu_scale** (*float, optional*) – SiLU 激活后的缩放因子。默认值: `0.0`（实际使用 `1.0 / max_seq_len`）

- **seq_offset** (*Tensor*) – Query 序列偏移量。形状: `[batchSize + 1]`，数据类型: `int32, int64`

- **seq_offset_k** (*Tensor, optional*) – Key/Value 序列偏移量。形状: `[batchSize + 1]`，数据类型: `int32, int64`。默认值: `None`（与 `seq_offset` 相同）

- **num_context** (*Tensor, optional*) – 每个 batch 的 Context 长度。形状: `[batchSize]`，数据类型: `int32, int64`。默认值: `None`

- **num_target** (*Tensor, optional*) – 每个 batch 的 Target 长度。形状: `[batchSize]`，数据类型: `int32, int64`。默认值: `None`

- **target_group_size** (*int, optional*) – RAG 目标组大小。必须为 `1` 或 `3`。默认值: `0`

- **alpha** (*float, optional*) – Alpha 系数，用于 RAG 计算。默认值: `1.0`

## 支持的数据类型

`torch.float16, torch.bfloat16`

## Shape

- **输入**:
  - `q`: `(totalSeqLenQ, heads, dimQK)`
  - `k`: `(totalSeqLenK, heads, dimQK)`
  - `v`: `(totalSeqLenK, heads, dimGV)`
  - `mask` (optional): `(batchSize, heads, maxSeqLenQ, maxSeqLenK)`
  - `rab` (optional): `(batchSize, heads, maxSeqLenQ, maxSeqLenK)`
  - `seq_offset`: `(batchSize + 1)`
  - `seq_offset_k` (optional): `(batchSize + 1)`

- **输出**: 1 个 Tensor
  - `attn_output`: `(totalSeqLenQ, heads, dimGV)`，dtype: `float16, bfloat16`

## 约束条件

- 仅支持 `float16/bfloat16` 数据类型
- `heads` 必须满足: `1 ≤ heads ≤ 16`
- `dimQK` 和 `dimGV` 必须为 16 的倍数且 `16 ≤ dimQK, dimGV ≤ 256`
- 当前仅支持 MHA（`headQ` 必须等于 `headK`），不支持 GQA
- 当 `num_context` 或 `num_target` 不为 None 时，`target_group_size` 必须为 `1` 或 `3`
- 仅支持 Jagged 格式输入（3D 张量: `[totalSeqLen, heads, dim]`）

## 使用示例

```python
>>> import torch
>>> import torch_npu

>>> # 参数设置
>>> batch_size = 2
>>> heads = 4
>>> dim_qk = 64
>>> dim_gv = 64
>>> max_seqlen = 128
>>> total_seq_len_q = 200
>>> total_seq_len_k = 200

>>> # 构造输入数据
>>> q = torch.randn(total_seq_len_q, heads, dim_qk, dtype=torch.float16, device="npu:0")
>>> k = torch.randn(total_seq_len_k, heads, dim_qk, dtype=torch.float16, device="npu:0")
>>> v = torch.randn(total_seq_len_k, heads, dim_gv, dtype=torch.float16, device="npu:0")

>>> # 序列偏移
>>> seq_lens_q = torch.tensor([100, 100], dtype=torch.int32)
>>> seq_lens_k = torch.tensor([100, 100], dtype=torch.int32)
>>> seq_offset = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(seq_lens_q, dim=0)])
>>> seq_offset_k = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(seq_lens_k, dim=0)])

>>> # 调用算子
>>> output = torch.ops.mxrec.hstu_forward_v2(
...     q, k, v,
...     None, None,
...     0, max_seqlen, None,
...     0.0, seq_offset.to("npu"), seq_offset_k.to("npu"),
...     None, None, 0, 1.0
... )

>>> # 输出 shape
>>> output.shape
torch.Size([200, 4, 64])
```

## 返回值

*Tensor* – 注意力输出张量:

- **attn_output** (*Tensor*) – 注意力输出，形状: `(totalSeqLenQ, heads, dimGV)`

## 算子原理

HSTU Forward 算子实现以下前向传播计算:

1. **QK 矩阵乘法**: `S = Q @ K^T`，计算 Query 和 Key 的注意力分数矩阵
2. **SiLU 激活与缩放**: `P = SiLU(S) * scale`，对注意力分数应用 SiLU 激活函数并进行缩放
3. **掩码与偏置**: 在注意力分数上应用可选的因果掩码 (causal mask) 和相对注意力偏置 (RAB)
4. **PV 矩阵乘法**: `O = P @ V`，将注意力概率矩阵与 Value 相乘得到最终输出

算子使用 CATLASS 模板库实现，包含以下优化:

- 双缓冲 (Double Buffer) 隐藏内存访问延迟
- 跨核同步机制
- L2 Cache 优化
- Swizzle 优化减少 Bank 冲突
- 两级 Tiling 策略 (L1/L0)，最大化数据复用

## 依赖

算子依赖CATLASS源码, 编译前需要初始化submodule：

```shell
git submodule update --init --recursive
```
