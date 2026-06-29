# API List

## API Overview

- [API Description](api_description.md)
- [Table Creation APIs](table_creation_apis.md)
- [Data APIs](data_apis.md)
- [Optimizer APIs](optimizers_apis.md)
- [Sharding APIs](subtable_apis.md)
- [Pipeline APIs](pipeline_apis.md)
- [Multi-Level Cache Management APIs](multilevel_cache_management_apis.md)
- [Admission and Eviction Management APIs](access_and_elimination_management_apis.md)
- [Custom Operators](specialized_operator.md)

## API Details

Note: In the following APIs, the `(TorchRec)` tag indicates a native TorchRec API. All other APIs are provided by Rec SDK Torch.

### Table Creation APIs

|API | Description|
|--|--|
|[HashEmbeddingBagConfig](table_creation_apis.md#hashembeddingbagconfig)|The input for `HashEmbeddingBagCollection`. Use it to configure table size, `dim`, data type, and other settings.|
|[HashEmbeddingBagCollection](table_creation_apis.md#hashembeddingbagcollection)|Creates a single-node table object with pooling and hash mapping.|
|[EmbCacheEmbeddingBagConfig](table_creation_apis.md#embcacheembeddingbagconfig)|The input for `EmbCacheEmbeddingBagCollection`. Use it to configure table size, `dim`, data type, cache policy, and other settings.|
|[EmbCacheEmbeddingBagCollection](table_creation_apis.md#embcacheembeddingbagcollection)|Creates a single-node table object with pooling, hash mapping, and multi-level cache.|
|[EmbCacheEmbeddingConfig](table_creation_apis.md#embcacheembeddingconfig)|The configuration class for `EmbCacheEmbeddingCollection`. Use it to configure table size, `dim`, data type, and other settings.|
|[EmbCacheEmbeddingCollection](table_creation_apis.md#embcacheembeddingcollection)|Creates a single-node table object with hash mapping and multi-level cache.|

### Data APIs

|API | Description|
|--|--|
|[JaggedTensor(TorchRec)](data_apis.md#jaggedtensor-torchrec)|A class that holds sparse IDs and feature lengths for table lookup.|
|[KeyedJaggedTensor(TorchRec)](data_apis.md#keyedjaggedtensor-torchrec)|A tensor with jagged dimensions whose slices can have different lengths. The first dimension contains keys, and the last dimension is a jagged dimension.|

### Optimizer APIs

|API | Description|
|--|--|
|[apply_optimizer_in_backward(TorchRec)](optimizers_apis.md#apply_optimizer_in_backward-torchrec)|Specifies the optimizer used during backpropagation. It is generally used to specify the optimizer for backpropagation of sparse table parameters.|
|[in_backward_optimizer_filter(TorchRec)](optimizers_apis.md#in_backward_optimizer_filter-torchrec)|Filters out parameters that are specified as `backward_optimizer`.|
|[KeyedOptimizerWrapper(TorchRec)](optimizers_apis.md#keyedoptimizerwrapper-torchrec)|Wraps an optimizer that filters out table parameters.|
|[CombinedOptimizer(TorchRec)](optimizers_apis.md#combinedoptimizer-torchrec)|Combines multiple optimizers into one.|

### Sharding APIs

|API | Description|
|--|--|
|[ShardingEnv(TorchRec)](subtable_apis.md#shardingenv-torchrec)|Stores distributed-related parameters.|
|[Topology(TorchRec)](subtable_apis.md#topology-torchrec)|Stores topology parameters for network devices in a distributed environment.|
|[ParameterConstraints(TorchRec)](subtable_apis.md#parameterconstraints-torchrec)|Specifies the query scope of the sharding plan.|
|[get_default_hybrid_sharders](subtable_apis.md#get_default_hybrid_sharders)|Gets the sharder.|
|[EmbeddingShardingPlanner(TorchRec)](subtable_apis.md#embeddingshardingplanner-torchrec)|Creates a sharding planner to search for the most suitable sharding plan.|
|[DistributedModelParallel(TorchRec)](subtable_apis.md#distributedmodelparallel-torchrec)|Converts the input `Module` into a distributed `Module` and executes the sharding plan.|
|[EmbCacheEmbeddingBagCollectionSharder](subtable_apis.md#embcacheembeddingbagcollectionsharder)|Creates an `EmbCacheEmbeddingBagCollectionSharder` sharder for sharding `EmbCacheEmbeddingBagCollection` across different devices.|
|[EmbCacheEmbeddingCollectionSharder](subtable_apis.md#embcacheembeddingcollectionsharder)|Initializes an `EmbCacheEmbeddingCollectionSharder` sharder for sharding `EmbCacheEmbeddingCollection` across different devices.|

### Pipeline APIs

|API | Description|
|--|--|
|[HybridTrainPipelineSparseDist](pipeline_apis.md#hybridtrainpipelinesparsedist)|Creates a pipelined table lookup in pure GPU memory mode.|
|[EmbCacheTrainPipelineSparseDist](pipeline_apis.md#embcachetrainpipelinesparsedist)|Creates a pipelined table lookup with multi-level cache.|

### Multi-Level Cache Management APIs

|API | Description|
|--|--|
|[InitializerType](multilevel_cache_management_apis.md#initializertype)|An enumeration of weight initialization types. It defines how embedding table weights are initialized.|
|[Saver](multilevel_cache_management_apis.md#saver)|A save and load utility for sparse tables in multi-level cache. It provides save and load interfaces for sparse table data in multi-level cache, such as sparse table embeddings and the optimizer parameters corresponding to the embeddings.|

### Admission and Eviction Management APIs

|API | Description|
|--|--|
|[AdmitAndEvictPolicyType](access_and_elimination_management_apis.md#admitandevictpolicytype)|An enumeration of admission and eviction policy types. It defines the policy type for feature admission and eviction in embedding tables.|
|[ShowClickParams](access_and_elimination_management_apis.md#showclickparams)|This interface defines the parameter settings for the show/click policy. The class provides parameters for configuring show/click admission and eviction, allowing users to control feature admission and eviction by impression count and click count.|
|[AdmitAndEvictConfig](access_and_elimination_management_apis.md#admitandevictconfig)|This interface defines the admission and eviction configuration for a single embedding table. The class provides parameters for configuring feature admission and eviction in embedding tables, allowing users to control feature admission and eviction behavior under specific conditions. It supports two policy types: count-based policy (`POLICY_COUNT`) and show/click-based policy (`POLICY_SHOWCLICK`).|
|[JaggedTensorWithTimestamp](access_and_elimination_management_apis.md#jaggedtensorwithtimestamp)|A class that extends `JaggedTensor` and represents a jagged tensor with timestamp information. It adds an `_timestamps` attribute to `JaggedTensor` to store the timestamps corresponding to `values`. It is used to calculate elapsed time during feature eviction.|
|[KeyedJaggedTensorWithTimestamp](access_and_elimination_management_apis.md#keyedjaggedtensorwithtimestamp)|A class that extends `KeyedJaggedTensor` and represents a keyed jagged tensor with timestamp information. It adds an `_timestamps` attribute to `KeyedJaggedTensor` to store the timestamps corresponding to `values`. It is used to calculate elapsed time during feature eviction.|

### Custom Operators

Rec SDK Torch provides some custom operators for processing sparse table data and accelerating model training. For details, see [Custom Operators](specialized_operator.md).
