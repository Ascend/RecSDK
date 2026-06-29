
# RecOps Operator List

RecOps is a custom operator set developed by the Rec SDK based on Ascend C for recommendation scenarios. It provides basic operator capabilities for framework components (`tf_rec_v1`, `tf_rec_v2`, `torch_rec_v1`, and `torch_rec_v2`).

## Introduction

| Operator Name | Description | Supported Hardware |
|---------|---------|---------|
| [concat_jagged_tensor](../../../cust_op/ascendc_op/ai_core_op/concat_jagged_tensor/v220/README.md) | Concatenates two jagged tensors along the `dim1` dimension according to the offset and merges them into one tensor. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [concat_jagged_tensor_grad](../../../cust_op/ascendc_op/ai_core_op/concat_jagged_tensor_grad/v220/README.md) | Splits one tensor into two tensors of potentially different lengths according to the offset. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [cust_op_by_addr](../../../cust_op/ascendc_op/ai_core_op/cust_op_by_addr/c310/README.md) | Uses `addr` as input to query the embedding table, replaces the `tf.gather` operator, and supports dynamic expansion. | ascend910, ascend910b, ascend910_93, ascend950 |
| [disentangle_attention](../../../cust_op/ascendc_op/ai_core_op/disentangle_attention/v220/README.md) | Implements the disentangled attention mechanism in the DeBERTa model. | ascend910b, ascend910_93, ascend950 |
| [fused_lazy_adam](../../../cust_op/ascendc_op/ai_core_op/fused_lazy_adam/v220/README.md) | Calculates and updates `m`, `v`, and `variable` during the backward update of the LazyAdam optimizer. | ascend910b, ascend910_93, ascend950 |
| [fused_sgd](../../../cust_op/ascendc_op/ai_core_op/fused_sgd/v220/README.md) | Calculates and updates parameters during the backward update of the SGD optimizer. | ascend910b, ascend910_93, ascend950 |
| [gather_for_rank1](../../../cust_op/ascendc_op/ai_core_op/gather_for_rank1/v220/README.md) | Implements `index_select` for a rank-1 tensor and selects elements from a 1D tensor by index. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [hstu_dense_backward](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_backward/v220/README.md) | Implements backpropagation for the HSTU fusion operator and calculates the gradients of `Q`, `K`, `V`, and `attn_bias`. | ascend910b, ascend910_93, ascend950 |
| [hstu_dense_backward_fuxi](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_backward_fuxi/v220/README.md) | Implements backpropagation for the HSTU-Fuxi fusion operator and calculates gradients in the attention mechanism. | ascend910b, ascend910_93, ascend950 |
| [hstu_dense_forward](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_forward/README.md) | Uses the HSTU fusion operator to implement the attention mechanism in recommendation scenarios. It supports GQA and unequal dimensions. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [hstu_dense_forward_fuxi](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_forward_fuxi/v220/README.md) | Based on the HSTU fusion operator, implements the attention mechanism in the Alpha-Fuxi model for recommendation scenarios. It supports `timestamp_bias` and `position_bias`, as well as the `normal`, `jagged`, and `paged` layouts. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [hstu_backward_v2](../../../cust_op/ascendc_op/ai_core_op/hstu_v2/BWD_README.md) | Implements backpropagation for the HSTU V2 fusion operator and calculates the gradients of `Query`, `Key`, `Value`, and RAB attention scores. | ascend950 |
| [index_select_for_rank1_backward](../../../cust_op/ascendc_op/ai_core_op/index_select_for_rank1_backward/v220/README.md) | Implements backpropagation for `index_select` and calculates gradients. | ascend910b, ascend910_93, ascend950 |
| [in_linear_silu](../../../cust_op/ascendc_op/ai_core_op/in_linear_silu/v220/README.md) | Before HSTU attention, applies `Linear` and `Silu` to the merged and normalized UVQK, then splits it into four tensors: `User`, `Value`, `Query`, and `Key`. | ascend910b, ascend910_93, ascend950 |
| [in_linear_silu_backward](../../../cust_op/ascendc_op/ai_core_op/in_linear_silu_backward/v220/README.md) | Implements backpropagation for the `in_linear_silu` operator and computes the gradients of input `x`, `weight`, and `bias`. | ascend910b, ascend910_93, ascend950 |
| [lccl](../../../cust_op/ascendc_op/ai_core_op/lccl/v220/README.md) | Uses the ability of AI Core to directly access the on-chip memory of the peer and uses memory semantics for collective communication (`AllToAll`, `AllUss`, and `GatherAll`). | ascend910b, ascend910_93 |
| [ln_mul](../../../cust_op/ascendc_op/ai_core_op/ln_mul/v220/README.md) | Applies `LayerNorm` to input `X`, then performs the `gamma` and `beta` computation, and finally multiplies it by input `U`. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [multislice_concat](../../../cust_op/ascendc_op/ai_core_op/multislice_concat/v220/README.md) | Slices the input 2D tensor along the second dimension at the specified positions and lengths, then outputs a tensor composed of several slices. | ascend910b, ascend950 |
| [norm_multiply_dropout](../../../cust_op/ascendc_op/ai_core_op/norm_multiply_dropout/v220/README.md) | Implements the fusion operator for the `layer_norm + multiply + dropout` computation logic. | ascend910b, ascend910_93, ascend950 |
| [norm_multiply_dropout_backward](../../../cust_op/ascendc_op/ai_core_op/norm_multiply_dropout_backward/v220/README.md) | Implements the backpropagation logic for the `layer_norm + multiply + dropout` computation. | ascend910b, ascend910_93, ascend950 |
| [pcie_through](../../../cust_op/ascendc_op/ai_core_op/pcie_through/v220/README.md) | When large amounts of data are exchanged between the host and device, `pcie_through` improves swap-in and swap-out performance. | ascend910b |
| [relative_attn_bias_backward](../../../cust_op/ascendc_op/ai_core_op/relative_attn_bias_backward/v220/README.md) | For the time part of the HSTU model RAB, calculates gradient values during backpropagation for the timestamp parameter. | ascend910b, ascend910_93, ascend950 |
| [relative_attn_bias_pos](../../../cust_op/ascendc_op/ai_core_op/relative_attn_bias_pos/v220/README.md) | Computes the `pos` part of the HSTU model RAB. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [relative_attn_bias_time](../../../cust_op/ascendc_op/ai_core_op/relative_attn_bias_time/v220/README.md) | Computes the `time` part of the HSTU model RAB. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [reverse_sequence](../../../cust_op/ascendc_op/ai_core_op/reverse_sequence/v220/README.md) | Reverses the second dimension of the input data according to the specified length parameter (`seq_lengths`). | ascend910b, ascend910_93, ascend950 |
| [token_mixing](../../../cust_op/ascendc_op/ai_core_op/token_mixing/v220/README.md) | Normalizes the sum of `x` and its transpose `x_t`. | ascend910b, ascend950 |

## Operator Directory Structure

```text
cust_op/ascendc_op/ai_core_op/
├── concat_jagged_tensor/                              # Jagged tensor concatenation
├── concat_jagged_tensor_grad/                         # Jagged tensor concatenation backward
├── cust_op_by_addr/                                   # Address lookup
├── disentangle_attention/                             # Disentangled attention
├── fused_lazy_adam/                                   # LazyAdam optimizer
├── fused_sgd/                                         # SGD optimizer
├── gather_for_rank1/                                  # Rank-1 gather
├── hstu_dense_backward/                               # HSTU backward
├── hstu_dense_backward_fuxi/                          # HSTU-Fuxi backward
├── hstu_dense_forward/                                # HSTU forward
├── hstu_dense_forward_fuxi/                           # HSTU-Fuxi forward
├── hstu_v2/                                           # HSTU V2 backward
├── index_select_for_rank1_backward/                   # index_select backward
├── in_linear_silu/                                    # Linear+Silu fusion
├── in_linear_silu_backward/                           # Linear+Silu backward
├── lccl/                                              # Collective communication
├── ln_mul/                                            # LayerNorm+Multiply fusion
├── multislice_concat/                                 # Multi-slice concatenation
├── norm_multiply_dropout/                             # Norm+Multiply+Dropout fusion
├── norm_multiply_dropout_backward/                    # Norm+Multiply+Dropout backward
├── pcie_through/                                      # PCIe data transfer
├── relative_attn_bias_backward/                       # Relative position bias backward
├── relative_attn_bias_pos/                            # Relative position bias pos
├── relative_attn_bias_time/                           # Relative position bias time
├── reverse_sequence/                                  # Sequence reversal
└── token_mixing/                                      # Token mixing
```
