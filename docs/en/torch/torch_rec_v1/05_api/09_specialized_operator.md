# Custom Operators

In recommendation training, some operators do not have an NPU implementation, or their existing NPU implementations perform poorly and cannot meet recommendation training requirements. Rec SDK Torch provides the following [Table 1 Custom operators](#table23231336143616) to support and accelerate NPU training for recommendation models.

Among these custom operators, some are already bound to open-source APIs, which forward the backend implementation of the open-source API to the NPU. After you import the Rec SDK Torch package, you can call these custom operators directly through the open-source APIs. The remaining operators must be called through the `mxrec` module registered in the PTA layer.

> [!NOTICE]
>
> 1. Custom operators are high-performance computing components. When you call a custom operator, make sure the input parameters satisfy the operator constraints, parameter types, parameter shapes, and other requirements. Otherwise, the operator may fail because of issues such as array out-of-bounds errors or insufficient device memory.
> 2. Because every custom operator has different parameter lists, types, and constraints, and each operator README already describes them in detail, this section does not elaborate on them and only lists the custom operators. **To view the detailed parameters, deployment, and calling methods of each custom operator, click the operator name hyperlink in the custom operator list to go to the README of the corresponding operator.**

The following describes the meanings of each value in the Supported via Open-Source API column of the custom operator list:

- Yes: Indicates that the operator has a corresponding open-source API and is bound to the open-source API. You can call it through the open-source API or through the `mxrec` module registered in the PTA layer.
- No: Indicates that the operator has a corresponding open-source API but is not bound to the open-source API. You can call it only through the `mxrec` module registered in the PTA layer.
- N/A: Indicates that the operator has no corresponding open-source API. You can call it only through the `mxrec` module registered in the PTA layer.

**Table 1** Custom operators
<a id="table23231336143616"></a>

| Operator | Whether a Corresponding Open-Source API Exists | Supported via Open-Source API | Description |
| --- | --- | --- | --- |
| [gather_for_rank1](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/gather_for_rank1/v220/README.md) | torch.index_select | No | Implements `index_select` when the `input` parameter has shape 1. This operator is optimized for that special shape. |
| [index_select_for_rank1_backward](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/index_select_for_rank1_backward/v220/README.md) | torch.index_select | No | Backward operator for `gather_for_rank1`. |
| [hstu_dense_forward](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/hstu_dense_forward/README.md) | No | N/A | Uses the `hstu` operator to implement the attention mechanism in recommendation scenarios. This operator is the forward operator of `hstu`. The `hstu` operator includes forward and backward passes and is already bound to Autograd. |
| [hstu_dense_backward](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/hstu_dense_backward/v220/README.md) | No | N/A | Backward operator of the `hstu` operator. |
| [hstu_dense_forward_fuxi](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/hstu_dense_forward_fuxi/v220/README.md) | No | N/A | Adapts the `hstu` forward operator to the `fuxi-alpha` model. |
| [hstu_dense_backward_fuxi](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/hstu_dense_backward_fuxi/v220/README.md) | No | N/A | Adapts the `hstu` backward operator to the `fuxi-alpha` model. |
| [relative_attn_bias_pos](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/relative_attn_bias_pos/v220/README.md) | No | N/A | Fused `relative_attn_bias` operator with a `pos` parameter. |
| [relative_attn_bias_time](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/relative_attn_bias_time/v220/README.md) | No | N/A | Fused `relative_attn_bias` operator with a `time` parameter. |
| [relative_attn_bias_backward](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/relative_attn_bias_backward/v220/README.md) | No | N/A | Backward operator of `relative_attn_bias`. |
| [recops_split_embedding_codegen_forward_unweighted](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/recops_split_embedding_codegen_forward_unweighted/v220/README.md) | torch.ops.mxrec.split_embedding_codegen_forward_unweighted | No | Implements the forward lookup function of `HybridSplitTableBatchedEmbeddingBagsCodegen`. The API must be called through the `HybridSplitTableBatchedEmbeddingBagsCodegen` wrapper class. |
| [recops_backward_codegen_adagrad_unweighted_exact](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/recops_backward_codegen_adagrad_unweighted_exact/v220/README.md) | torch.ops.mxrec.split_embedding_codegen_lookup_adagrad_function | No | Implements the backward update function of `HybridSplitTableBatchedEmbeddingBagsCodegen`. Different optimizers use different update algorithms. The API must be called through the `HybridSplitTableBatchedEmbeddingBagsCodegen` wrapper class. |
| [disentangle_attention](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/disentangle_attention/v220/README.md) | No | N/A | Implements `disentangle_attention` in the DeBERTa model to decouple attention. |
| [concat_jagged_tensor](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/concat_jagged_tensor/v220/README.md) | No | N/A | Concatenates two tensors in sequence according to their respective offset positions to form one tensor. |
| [concat_jagged_tensor_grad](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/concat_jagged_tensor_grad/v220/README.md) | No | N/A | Backward operator of `concat_jagged_tensor`. It splits one tensor into two tensors in sequence according to the two offset positions. |
| [in_linear_silu](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/in_linear_silu/v220/README.md) | No | N/A | Applies a linear transform to the input tensor, activates it with the `silu` function, and splits the result into four tensors. |
| [in_linear_silu_backward](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/in_linear_silu_backward/v220/README.md) | No | N/A | Backward operator of `in_linear_silu`. |
| [reverse_sequence](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/reverse_sequence/v220/README.md) | No | N/A | Reverses the input tensor. |
| [norm_multiply_dropout](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/norm_multiply_dropout/v220/README.md) | No | N/A | Implements the forward propagation of `layer_norm + multiply + dropout`. |
| [norm_multiply_dropout_backward](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/norm_multiply_dropout_backward/v220/README.md) | No | N/A | Implements the backward propagation of `layer_norm + multiply + dropout`. |

>[!NOTE]
>
>The table lists only some custom operators for PyTorch scenarios in the [Rec SDK Torch custom operators](https://gitcode.com/Ascend/RecSDK/tree/develop/cust_op/ascendc_op/ai_core_op) directory.
