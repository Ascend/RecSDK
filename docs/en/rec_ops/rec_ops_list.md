
# RecOps Operator List

RecOps is a custom operator set developed by Rec SDK based on Ascend C for recommendation scenarios. It provides basic operator capabilities for framework components (`tf_rec_v1`, `tf_rec_v2`, `torch_rec_v1`, and `torch_rec_v2`).

## Basic Concepts

| Term | Description |
|------|------|
| **Jagged Tensor** | An irregular tensor whose rows can have different lengths. It is commonly used in variable-length sequence scenarios. |
| **dim** | Short for dimension, referring to the dimension of a tensor. |
| **LayerNorm** | Layer normalization that normalizes hidden layers by mean and variance. |
| **Dropout** | A regularization technique that randomly drops the output of some neurons during training. |
| **Silu** | The Sigmoid Linear Unit activation function, defined as `x * sigmoid(x)`.|
| **GQA** | Grouped Query Attention, a grouped query attention mechanism. |
| **HSTU** | Hierarchical Sparse Transformer Unit, a hierarchical sparse Transformer computing unit used in the attention mechanism for recommendation scenarios. |
| **Alpha-Fuxi** | A recommendation model name. The HSTU-Fuxi operator implements the attention mechanism based on this model. |
| **Paged Attention** | A paged attention mechanism that manages the key-value pairs of attention computation in pages, used to efficiently handle variable-length sequences and reduce memory access overhead. |

## Operator List

| Operator Name | Type | Description | Supported Hardware |
|---------|------|---------|---------|
| [concat_jagged_tensor](../../../cust_op/ascendc_op/ai_core_op/concat_jagged_tensor/v220/README.md) | Fusion operator | Concatenates two variable-length tensors into one complete tensor by position offset along the second dimension. Inputs: the two variable-length tensors to concatenate and the list of starting positions of each row. Output: the complete tensor after concatenation. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [concat_jagged_tensor_grad](../../../cust_op/ascendc_op/ai_core_op/concat_jagged_tensor_grad/v220/README.md) | Fusion operator | Splits one complete tensor by position offset into two variable-length tensors (the reverse operation of `concat_jagged_tensor`). Inputs: the tensor to split and the split position information. Output: the two variable-length tensors after splitting. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [in_linear_silu](../../../cust_op/ascendc_op/ai_core_op/in_linear_silu/v220/README.md) | Fusion operator | Applies a linear transformation and Silu activation to the fused features, and then splits the result into four vectors: `User`, `Value`, `Query`, and `Key` (used as preprocessing for the HSTU attention mechanism). Inputs: fused features, weights, bias, and group length. Output: the `User`, `Value`, `Query`, and `Key` vectors. | ascend910b, ascend910_93, ascend950 |
| [in_linear_silu_backward](../../../cust_op/ascendc_op/ai_core_op/in_linear_silu_backward/v220/README.md) | Fusion operator | Backpropagation for `in_linear_silu`: computes the gradients of the fused features, weights, and bias. Inputs: fused features, weights, bias, and the gradients of `User`/`Value`/`Query`/`Key`. Output: the gradients of the fused features, weights, and bias. | ascend910b, ascend910_93, ascend950 |
| [ln_mul](../../../cust_op/ascendc_op/ai_core_op/ln_mul/v220/README.md) | Fusion operator | LayerNorm + Multiply fusion: normalizes the input tensor, computes with the scale factor and bias, and finally multiplies the result by another tensor. Inputs: the tensor to normalize, the tensor to multiply, the scale factor, and the bias. Output: the product of the normalization result and the tensor to multiply. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [norm_multiply_dropout](../../../cust_op/ascendc_op/ai_core_op/norm_multiply_dropout/v220/README.md) | Fusion operator | LayerNorm + Multiply + Dropout fusion: normalizes the input, multiplies it by another tensor, and then applies Dropout. Inputs: the tensor to normalize, the tensor to multiply, the normalization parameters, and the Dropout ratio. Output: the result after normalization, multiplication, and Dropout. | ascend910b, ascend910_93, ascend950 |
| [norm_multiply_dropout_backward](../../../cust_op/ascendc_op/ai_core_op/norm_multiply_dropout_backward/v220/README.md) | Fusion operator | Backpropagation for `norm_multiply_dropout`: computes the gradients of each input tensor. Inputs: output gradient, the tensor to normalize, the tensor to multiply, the normalization parameters, and the Dropout ratio. Output: the gradients of the tensor to multiply, the tensor to normalize, and the normalization parameters. | ascend910b, ascend910_93, ascend950 |
| [token_mixing](../../../cust_op/ascendc_op/ai_core_op/token_mixing/v220/README.md) | Fusion operator | Token mixing normalization: adds the tensor to its transpose and then applies LayerNorm. Inputs: a 3D tensor, the scale factor, and the bias. Output: the normalization result. | ascend910b, ascend950 |
| [disentangle_attention](../../../cust_op/ascendc_op/ai_core_op/disentangle_attention/v220/README.md) | Attention operator | Implements the disentangled attention mechanism in the DeBERTa model and computes content attention and position attention separately. Inputs: `Query`/`Key`/`Value` vectors, position-encoded `Key`/`Query`, relative position indexes, and the attention mask. Output: attention output, attention probability matrix, and attention weights. | ascend910b, ascend910_93, ascend950 |
| [hstu_v2](../../../cust_op/ascendc_op/ai_core_op/hstu_v2/BWD_README.md) | Attention operator | HSTU V2 backpropagation: computes the gradients of `Query`, `Key`, `Value`, and the relative position attention. Inputs: output gradient, `Query`, `Key`, `Value`, sequence offsets, and relative position attention. Output: the gradients of `Query`, `Key`, `Value`, and the position bias. | ascend950 |
| [hstu_dense_backward](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_backward/v220/README.md) | Attention operator | HSTU backpropagation: computes the gradients of `Query`, `Key`, `Value`, and the attention bias. Inputs: output gradient, `Query`, `Key`, `Value`, `mask`, and `attn_bias`. Output: the gradients of `Query`, `Key`, `Value`, and `attn_bias`. | ascend910b, ascend910_93, ascend950 |
| [hstu_dense_backward_fuxi](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_backward_fuxi/v220/README.md) | Attention operator | HSTU-Fuxi backpropagation: computes the gradients of `Query`, `Key`, `Value`, the timestamp bias, and the position bias. Inputs: output gradient, `Query`, `Key`, `Value`, timestamp bias gradient, and position bias gradient. Output: the gradients of `Query`, `Key`, `Value`, the timestamp bias, and the position bias. | ascend910b, ascend910_93, ascend950 |
| [hstu_dense_forward](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_forward/README.md) | Attention operator | HSTU forward attention: implements the attention mechanism for recommendation scenarios and supports GQA and unequal dims. Inputs: `Query`, `Key`, `Value`, `mask`, and `attn_bias`. Output: attention output. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [hstu_dense_forward_fuxi](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_forward_fuxi/v220/README.md) | Attention operator | HSTU-Fuxi forward attention: implements the attention mechanism of the Alpha-Fuxi model and supports the timestamp bias and position bias in three layouts. Inputs: `Query`, `Key`, `Value`, timestamp bias, position bias, and `mask`. Output: attention output. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [relative_attn_bias_backward](../../../cust_op/ascendc_op/ai_core_op/relative_attn_bias_backward/v220/README.md) | Attention operator | Relative position bias backpropagation: computes the gradient of the timestamp bias parameter in the HSTU model. Inputs: timestamp bias gradient, timestamp bucket indexes, and the number of buckets. Output: the timestamp weight gradient. | ascend910b, ascend910_93, ascend950 |
| [relative_attn_bias_pos](../../../cust_op/ascendc_op/ai_core_op/relative_attn_bias_pos/v220/README.md) | Attention operator | Relative position bias forward pass (position part): computes the position bias in the HSTU model. Inputs: relative position encoding, the indicator matrix, and the valid length. Output: the position bias. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [relative_attn_bias_time](../../../cust_op/ascendc_op/ai_core_op/relative_attn_bias_time/v220/README.md) | Attention operator | Relative position bias forward pass (timestamp part): computes the timestamp bias in the HSTU model. Inputs: timestamps, timestamp weights, and the bucket partition factor. Output: the timestamp bias and bucket indexes. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [fused_lazy_adam](../../../cust_op/ascendc_op/ai_core_op/fused_lazy_adam/v220/README.md) | Optimizer operator | LazyAdam optimizer fusion: computes and updates the first-order moment estimate, second-order moment estimate, and parameters. Inputs: gradient, indexes, first-order moment, second-order moment, and learning rate. Output: the updated parameters. | ascend910b, ascend910_93, ascend950 |
| [fused_sgd](../../../cust_op/ascendc_op/ai_core_op/fused_sgd/v220/README.md) | Optimizer operator | SGD optimizer fusion: computes and updates parameters. Inputs: gradient, indexes, and learning rate. Output: the updated parameters. | ascend910b, ascend910_93, ascend950 |
| [gather_for_rank1](../../../cust_op/ascendc_op/ai_core_op/gather_for_rank1/v220/README.md) | Tensor operator | 1D index selection: selects elements from a 1D tensor according to the indexes. Inputs: 1D data and an index list. Output: the elements selected by index. | ascend910b, ascend910_93, ascend310p, ascend950 |
| [index_select_for_rank1_backward](../../../cust_op/ascendc_op/ai_core_op/index_select_for_rank1_backward/v220/README.md) | Tensor operator | Backpropagation for 1D index selection: aggregates gradients into the original tensor. Inputs: output gradient, original data, and indexes. Output: the gradient of the original data. | ascend910b, ascend910_93, ascend950 |
| [multislice_concat](../../../cust_op/ascendc_op/ai_core_op/multislice_concat/v220/README.md) | Tensor operator | 2D slice concatenation: slices by position and length and concatenates the slices into multiple tensors. Inputs: a 2D tensor and the slice configuration. Output: multiple tensors after concatenation. | ascend910b, ascend950 |
| [cust_op_by_addr](../../../cust_op/ascendc_op/ai_core_op/cust_op_by_addr/c310/README.md) | Lookup operator | Address-based embedding lookup: looks up embedding vectors by address, replaces `tf.gather`, and supports dynamic expansion. Inputs: an address list and the embedding dimension. Output: embedding vectors. | ascend910b, ascend910_93, ascend950 |
| [lccl](../../../cust_op/ascendc_op/ai_core_op/lccl/v220/README.md) | Communication operator | Collective communication: directly accesses the peer device memory across devices to perform `AllToAll`, `AllUss`, and `GatherAll` operations. Inputs: distributed tensors, the communication matrix, and shared memory. Output: the aggregated tensor. | ascend910b, ascend910_93 |
| [pcie_through](../../../cust_op/ascendc_op/ai_core_op/pcie_through/v220/README.md) | Communication operator | PCIe passthrough: transfers large amounts of data between the host and device, improving swap-in and swap-out performance. Inputs: swap-in indexes, swap-out indexes, table data, and shared memory addresses. Output: execution status. | ascend910b |
| [reverse_sequence](../../../cust_op/ascendc_op/ai_core_op/reverse_sequence/v220/README.md) | Sequence operator | Sequence reversal: reverses the second dimension according to the specified lengths. Inputs: 3D data and the length of each sequence. Output: the reversed data. | ascend910b, ascend910_93, ascend950 |

## Operator Directory Structure

```text
cust_op/ascendc_op/ai_core_op/
# ===== Fusion operators =====
├── concat_jagged_tensor/                              # Jagged Tensor concatenation
├── concat_jagged_tensor_grad/                         # Jagged Tensor concatenation backward
├── in_linear_silu/                                    # Linear + Silu fusion
├── in_linear_silu_backward/                           # Linear + Silu backward
├── ln_mul/                                            # LayerNorm + Multiply fusion
├── norm_multiply_dropout/                             # Norm + Multiply + Dropout fusion
├── norm_multiply_dropout_backward/                    # Norm + Multiply + Dropout backward
├── token_mixing/                                      # Token mixing normalization

# ===== Attention operators =====
├── disentangle_attention/                             # Disentangled attention (DeBERTa)
├── hstu_dense_forward/                                # HSTU forward
├── hstu_dense_backward/                               # HSTU backward
├── hstu_dense_forward_fuxi/                           # HSTU-Fuxi forward
├── hstu_dense_backward_fuxi/                          # HSTU-Fuxi backward
├── hstu_v2/                                           # HSTU V2
├── relative_attn_bias_pos/                            # Relative position bias (position)
├── relative_attn_bias_time/                           # Relative position bias (timestamp)
├── relative_attn_bias_backward/                       # Relative position bias backward

# ===== Optimizer operators =====
├── fused_lazy_adam/                                   # LazyAdam optimizer
├── fused_sgd/                                         # SGD optimizer

# ===== Tensor operators =====
├── gather_for_rank1/                                  # 1D index_select
├── index_select_for_rank1_backward/                   # index_select backward
├── multislice_concat/                                 # Multi-slice concatenation

# ===== Lookup operators =====
├── cust_op_by_addr/                                   # Address lookup (embedding table query)

# ===== Communication operators =====
├── lccl/                                              # Collective communication (AllToAll/AllUss/GatherAll)
├── pcie_through/                                      # PCIe passthrough data transfer

# ===== Sequence operators =====
└── reverse_sequence/                                  # Sequence reversal
```

## Version Compatibility

| Framework Component | Base Framework | Adaptation Status | Supported Operators |
|----------|---------|---------|---------|
| tf_rec_v1 | TensorFlow | Non-full offload | Basic operator set (lookup, optimizer, communication) |
| tf_rec_v2 | TensorFlow | Full offload | All operators (supports dynamic expansion) |
| torch_rec_v1 | PyTorch + TorchRec | Non-full offload | Basic operator set (lookup, optimizer, communication) |
| torch_rec_v2 | PyTorch + TorchRec | Full offload | All operators (supports Paged Attention) |

> **Note**:
>
> - **Non-full offload**: a hybrid mode in which some computing tasks run on the NPU and some run on the CPU. It applies to basic recommendation scenarios.
> - **Full offload**: all computing tasks are offloaded to the NPU for better performance and support for advanced features.
> - For specific installation and usage instructions of each framework component, see the corresponding documentation:
>   - [tf_rec_v1 documentation](../tensorflow/tf_rec_v1/recsdk_tf_installation_guide.md)
>   - [tf_rec_v2 documentation](../tensorflow/tf_rec_v2/recsdk_tf_installation_guide.md)
>   - [torch_rec_v1 documentation](../torch/torch_rec_v1/recsdk_torch_installation_guide.md)
>   - [torch_rec_v2 documentation](../torch/torch_rec_v2/recsdk_torch_installation_guide.md)
