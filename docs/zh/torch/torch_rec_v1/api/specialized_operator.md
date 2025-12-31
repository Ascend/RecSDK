# 自定义算子<a name="ZH-CN_TOPIC_0000002452078416"></a>

在推荐训练中，存在部分算子无NPU实现，或已有NPU实现但性能较差，不能满足推荐训练需求。Rec SDK Torch提供了以下[表1 自定义算子列表](#table23231336143616)，用于支持/加速推荐模型NPU训练。

其中，部分自定义算子已绑定到开源API（将开源API的backend实现转发到NPU），在导入Rec SDK Torch软件包后，可直接通过开源API调用到自定义算子。其余算子则需通过PTA层注册的mxrec模块进行调用。详情可参考[Rec SDK Torch代码库](https://gitcode.com/Ascend/RecSDK)中RecSDK/cust\_op/ascendc\_op/ai\_core\_op/路径下各算子目录下的README文件。

>[!NOTICE] 须知
>自定义算子为高性能计算，用户调用自定义算子时需自行确保输入的参数满足算子约束条件、参数类型、参数shape等要求，否则可能会出现数组越界，显存不够等问题导致算子执行失败。

**表 1**  自定义算子列表
<a id="table23231336143616"></a>

|算子名称|对应开源API|是否支持通过开源API调用|功能介绍|
|--|--|--|--|
|asynchronous_complete_cumsum|torch.ops.fbgemm.asynchronous_complete_cumsum|是|实现对输入offset的累加功能。|
|dense_to_jagged|torch.ops.fbgemm.dense_to_jagged|是|实现将padded dense tensor转为jagged tensor的功能。|
|gather_for_rank1|torch.index_select|否|实现input参数shape为1时的index_select功能。该算子为特殊shape优化。|
|index_select_for_rank1_backward|torch.index_select|否|gather_for_rank1的反向算子。|
|hstu_dense_forward|无|-|推荐场景下使用hstu算子实现注意力机制。该算子为hstu算子的前向算子。hstu算子包含前向和反向，且已绑定到Autograd。|
|hstu_dense_backward|无|-|hstu算子的反向算子。|
|hstu_dense_forward_fuxi|无|-|基于hstu前向算子实现适配fuxi-alpha模型。|
|hstu_dense_backward_fuxi|无|-|基于hstu反向算子实现适配fuxi-alpha模型。|
|jagged_to_padded_dense|torch.ops.fbgemm.jagged_to_padded_dense|是|实现将jagged tensor转为padded dense tensor的功能。该算子与dense_to_jagged算子互为前反向算子，且已绑定Autograd。|
|permute_1D_sparse_data|torch.ops.fbgemm.permute_1D_sparse_data|是|实现对1D稀疏数据转置。该算子无单独目录，为调用2D实现。该实现方式与FBGEMM相同。调用该算子时，可参考fbgemm接口文档。|
|permute2d_sparse_data|torch.ops.fbgemm.permute_2D_sparse_data|是|实现对2D稀疏数据转置。|
|relative_attn_bias_pos|无|-|带pos参数的relative_attn_bias融合算子。|
|relative_attn_bias_time|无|-|带time参数的relative_attn_bias融合算子。|
|relative_attn_bias_backward|无|-|relative_attn_bias的反向算子。|
|split_embedding_codegen_lookup_adagrad_function|fbgemm_gpu.split_table_batched_embeddings_ops_training.SplitTableBatchedEmbeddingBagsCodegen|是|实现FBGEMM库中SplitTableBatchedEmbeddingBagsCodegen查表的前向功能。|
|backward_codegen_adagrad_unweighted_exact|fbgemm_gpu.split_table_batched_embeddings_ops_training.SplitTableBatchedEmbeddingBagsCodegen|是|实现FBGEMM库中SplitTableBatchedEmbeddingBagsCodegen查表的反向更新功能。不同优化器采用不同的更新算法。|
|disetangle_attention|无|-|实现DeBERTa模型中的disetangle_attention，解耦注意力的功能。|
|dense_embedding_codegen_lookup_function|torch.ops.fbgemm. dense_embedding_nobag_codegen_forward_unweighted_cuda|是|实现FBGEMM GPU库dense_embedding_nobag_codegen_forward_unweighted_cuda查表的前向功能。|
|dense_embedding_codegen_lookup_function_grad|无|-|实现FBGEMM GPU库dense_embedding_nobag_codegen_forward_unweighted_cuda查表的反向梯度累计功能。|


>[!NOTE]  说明
>表格中仅列举ai\_core\_op路径下PyTroch场景的自定义算子。

