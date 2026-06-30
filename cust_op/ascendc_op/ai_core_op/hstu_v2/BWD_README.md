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
    alpha=1.0,
    window_size_left=-1,
    window_size_right=-1
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

- **window_size_left** (*int*) – 注意力窗口左侧宽度，默认 `-1`。`-1` 表示向左无限延伸。与 `window_size_right` 共同决定 mask 类型

- **window_size_right** (*int*) – 注意力窗口右侧宽度，默认 `-1`。`-1` 表示向右无限延伸，`0` 表示因果掩码。与 `window_size_left` 共同决定 mask 类型

## 支持的数据类型

`torch.float16, torch.bfloat16`

## Shape

- **输入**:
  - `grad`: `(totalSeqLenQ, heads, dimGV)`
  - `q`: `(totalSeqLenQ, heads, dimQK)`
  - `k`: `(totalSeqLenK, heads, dimQK)`
  - `v`: `(totalSeqLenK, heads, dimGV)`
  - `seq_offset_q`: `(batchSize + 1)`
  - `seq_offset_k`: `(batchSize + 1)`
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
- 当 `num_context` 或 `num_target` 不为 None 时，`target_group_size` 必须为 `1` 或 `3`
- `headQ` 必须等于 `headK`（当前仅支持 MHA，不支持 GQA）
- `window_size_left` 和 `window_size_right` 必须 ≥ -1
- 当前仅支持 `(-1, 0)`（causal mask）和 `(-1, -1)`（no mask）两种 `(window_size_left, window_size_right)` 组合
- 当提供 `num_context` 或 `num_target` 时，必须使用 causal mask（`window_size_left=-1, window_size_right=0`）, 同时需满足num_context + num_target < max_seqlen_q

## mask 行为说明

通过 `window_size_left` 和 `window_size_right` 控制注意力 mask 类型：

| window_size_left | window_size_right | mask 类型 | 说明 |
|-----------------|-------------------|----------|------|
| -1 | 0 | causal mask（因果掩码） | 下三角掩码，每个 token 只能关注自身及之前的 token |
| -1 | -1 | no mask（无掩码） | 不做任何 mask，每个 token 可关注所有 token |

当使用 causal mask 时，可通过 `num_context` 和 `num_target` 进一步细分 mask 区域：

- **context 区域**（`num_context` 指定长度）：双向注意力，无 causal 限制
- **target 区域**（`num_target` 指定长度）：分组注意力，group_size 由 `target_group_size` 控制
- **其余区域**：标准 causal mask（下三角）

注意：当前仅支持 causal mask 和 no mask 两种模式，不支持自定义 mask 张量输入。

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

>>> # 调用算子（仅causal mask,未配置context mask及target mask）
>>> q_grad, k_grad, v_grad, rab_grad = torch.ops.mxrec.hstu_backward_v2(
...     grad, q, k, v,
...     max_seqlen_q, max_seqlen_k,
...     seq_offset_q.to("npu"),
...     seq_offset_k.to("npu"),
...     None, None, None,
...     0.0, 0, 1.0,
...     -1, 0
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
