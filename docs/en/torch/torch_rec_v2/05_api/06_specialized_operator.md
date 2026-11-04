# Custom Operators

In recommendation training, some operators do not have an NPU implementation, or their existing NPU implementations perform poorly and cannot meet recommendation training requirements. For dynamic sparse tables, Rec SDK Torch provides the following [Table 1 Custom operators](#table23231336143616) based on HKV to support and accelerate NPU training for recommendation models.

After you install the Rec SDK Torch software package for dynamic sparse tables, you can call the custom operators directly through the `dynamic_emb_extensions` operator library.

>[!NOTE]
>Custom operators are high-performance computing components. When you call a custom operator, make sure the input parameters satisfy the operator constraints, parameter types, parameter shapes, and other requirements. Otherwise, the operator may fail because of issues such as array out-of-bounds errors or insufficient device memory.

**Table 1** Custom operators

<a id="table23231336143616"></a>

|Operator|Description|
|--------|----------|
|block_bucketize_sparse_features|Buckets sparse feature inputs (`keys`) into blocks.|
|dynamic_emb_Adam_with_pointer|Uses the Adam optimizer to locate and update a specific embedding vector through a pointer.|
|dynamic_emb_adamW_with_pointer|Uses the AdamW optimizer to locate and update a specific embedding vector through a pointer.|
|dynamic_emb_adagrad_with_pointer|Uses the Adagrad optimizer to locate and update a specific embedding vector through a pointer.|
|dynamic_emb_adagrad_with_table|Uses the Adagrad optimizer to locate and update a specific embedding vector using table pointer management.|
|dynamic_emb_adagrad_fused|Uses the Adagrad optimizer to locate and update a specific embedding vector using tensor management.|
|dynamic_emb_rowwise_adagrad_with_pointer|Uses the RowWiseAdagrad optimizer to locate and update a specific embedding vector through a pointer.|
|dynamic_emb_rowwise_adagrad_with_table|Uses the RowWiseAdagrad optimizer to locate and update a specific embedding vector using table pointer management.|
|dynamic_emb_rowwise_adagrad_fused|Uses the RowWiseAdagrad optimizer to locate and update a specific embedding vector using tensor management.|
|find_pointers|Finds the value pointer corresponding to a key in a dynamic embedding table and indicates whether the lookup succeeds.|
|unique_op|Deduplicates a one-dimensional `indices` tensor.|
|segmented_unique_op|Deduplicates data after segmenting `indices` across multiple tables.|
|get_table_range_op|Calculates table ranges from `offsets` and `featureOffsets`.|
|dedup_input_indices_op|Deduplicates input indices and combines segment deduplication, range calculation, length updates, and other logic.|
|gather_embedding|Aggregates embedding vectors. It gathers the embedding values of the corresponding dimensions from `inputs` according to `indices`.|
|reduce_grads|Reduces gradients according to the `indices` tensor.|
|load_from_pointer|Loads data from a pointer address into the target tensor.|
|device_timestamp|Gets the timestamp of the NPU device.|

**Table 2** DynamicEmb configuration classes

|Class|Description|Class Methods/Parameters|
|--------|--------|--------|
|DynamicEmbTable|Dynamic embedding table management class|<ul><li>`get_key_type`: Gets the data type of table keys.</li><li>`get_value_type`: Gets the data type of table values.</li><li>`get_evict_strategy`: Gets the eviction strategy of the table.</li><li>`get_max_capacity`: Gets the maximum capacity of the table.</li><li>`get_initializer_args`: Gets the initialization arguments of the table.</li><li>`optstate_dim`: Gets the total dimension of optimizer states.</li><li>`get_emb_cols`: Gets the number of columns in the embedding table.</li><li>`load`: Loads key-value pairs into the embedding table.</li><li>`update`: Updates key-value pairs in the table.</li><li>`export_batch`: Exports key-value pairs in batches from the table.</li></ul>
|InitializerArgs|Initialization argument class for dynamic embedding tables|<ul><li>`mode`: Initialization mode.</li><li>`mean`: Mean of the normal distribution.</li><li>`std_dev`: Standard deviation of the normal distribution.</li><li>`lower`: Lower bound of the uniform distribution.</li><li>`upper`: Upper bound of the uniform distribution.</li><li>`value`: Fixed value for constant initialization.</li></ul>|
|DynamicEmbDataType|Enumeration class that identifies dynamic embedding data types|<ul><li>`Float32`: 32-bit single-precision floating-point number.</li><li>`BFloat16`: 16-bit brain floating-point number.</li><li>`Float16`: 16-bit half-precision floating-point number.</li><li>`Int64`: 64-bit signed integer.</li><li>`UInt64`: 64-bit unsigned integer.</li><li>`Int32`: 32-bit signed integer.</li><li>`UInt32`: 32-bit unsigned integer.</li><li>`Size_t`: Unsigned integer.</li></ul>|
|EvictStrategy|Enumeration class for dynamic embedding table eviction strategies|<ul><li>`kLru`: Least recently used.</li><li>`kLfu`: Least frequently used.</li><li>`kEpochLru`: Epoch-based LRU.</li><li>`kEpochLfu`: Epoch-based LFU.</li><li>`kCustomized`: Custom eviction strategy.</li></ul>|
|OptimizerType|Enumeration class for dynamic embedding table optimizers|<ul><li>`Null`: No optimizer.</li><li>`SGD`: Stochastic gradient descent.</li><li>`Adam`: Adaptive moment estimation.</li><li>`AdamW`: Adam with weight decay.</li><li>`AdaGrad`: Adaptive gradient algorithm.</li></ul>|
