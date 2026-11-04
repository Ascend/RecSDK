# API List

Note: This API list is organized by API type. To view API usage from the perspective of functional features, see [Functions and Features](../04_migration_and_training/migration_and_training.md#functional_features_description).

## API Overview

- [API Description](01_api_description.md)
- [Table Creation APIs](02_table_creation_apis.md)
- [Data APIs](03_data_apis.md)
- [Optimizer APIs](04_optimizers_apis.md)
- [Sharding APIs](05_subtable_apis.md)
- [Pipeline APIs](06_pipeline_apis.md)
- [Multi-Level Cache Management APIs](07_multilevel_cache_management_apis.md)
- [Admission and Eviction Management APIs](08_access_and_elimination_management_apis.md)
- [Custom Operators](09_specialized_operator.md)

## API Details

Note: In the following APIs, the `(TorchRec)` tag indicates a native TorchRec API. All other APIs are provided by Rec SDK Torch.

### Table Creation APIs

|API | Description|
|--|--|
|[HashEmbeddingBagConfig](02_table_creation_apis.md#hashembeddingbagconfig)|Input to `HashEmbeddingBagCollection`. Use it to configure table size, `dim`, data type, and other settings.|
|[HashEmbeddingBagCollection](02_table_creation_apis.md#hashembeddingbagcollection)|Creates a sparse table object with pooling and hash mapping.|
|[EmbCacheEmbeddingBagConfig](02_table_creation_apis.md#embcacheembeddingbagconfig)|Input to `EmbCacheEmbeddingBagCollection`. Use it to configure table size, `dim`, data type, and other settings.|
|[EmbCacheEmbeddingBagCollection](02_table_creation_apis.md#embcacheembeddingbagcollection)|Creates a sparse table object with pooling, hash mapping, and multi-level cache.|
|[EmbCacheEmbeddingConfig](02_table_creation_apis.md#embcacheembeddingconfig)|Configuration class for `EmbCacheEmbeddingCollection`. Use it to configure table size, `dim`, data type, and other settings.|
|[EmbCacheEmbeddingCollection](02_table_creation_apis.md#embcacheembeddingcollection)|Creates a sparse table object with hash mapping and multi-level cache.|

### Data APIs

|API | Description|
|--|--|
|[JaggedTensor(TorchRec)](03_data_apis.md#jaggedtensor-torchrec)|A class that holds sparse IDs and feature lengths for table lookup.|
|[KeyedJaggedTensor(TorchRec)](03_data_apis.md#keyedjaggedtensor-torchrec)|Extends `JaggedTensor` by introducing keys (usually feature names) to mark different feature groups.|

### Optimizer APIs

|API | Description|
|--|--|
|[apply_optimizer_in_backward(TorchRec)](04_optimizers_apis.md#apply_optimizer_in_backward-torchrec)|Specifies the optimizer used by the model during backpropagation. It is generally used to specify the optimizer for backpropagation of sparse table parameters.|
|[in_backward_optimizer_filter(TorchRec)](04_optimizers_apis.md#in_backward_optimizer_filter-torchrec)|Filters out parameters that are specified as `backward_optimizer`.|
|[KeyedOptimizerWrapper(TorchRec)](04_optimizers_apis.md#keyedoptimizerwrapper-torchrec)|Wraps an optimizer that filters out table parameters.|
|[CombinedOptimizer(TorchRec)](04_optimizers_apis.md#combinedoptimizer-torchrec)|Combines multiple optimizers into one.|

### Sharding APIs

|API | Description|
|--|--|
|[ShardingEnv(TorchRec)](05_subtable_apis.md#shardingenv-torchrec)|Stores distributed-related parameters.|
|[Topology(TorchRec)](05_subtable_apis.md#topology-torchrec)|Stores topology parameters for network devices in a distributed environment.|
|[ParameterConstraints(TorchRec)](05_subtable_apis.md#parameterconstraints-torchrec)|Specifies the query scope of the sharding plan.|
|[get_default_hybrid_sharders](05_subtable_apis.md#get_default_hybrid_sharders)|Gets the sharder.|
|[EmbeddingShardingPlanner(TorchRec)](05_subtable_apis.md#embeddingshardingplanner-torchrec)|Creates a sharding planner to search for the most suitable sharding plan.|
|[EmbCacheEmbeddingBagCollectionSharder](05_subtable_apis.md#embcacheembeddingbagcollectionsharder)|Creates an `EmbCacheEmbeddingBagCollectionSharder` sharder for sharding `EmbCacheEmbeddingBagCollection` across different devices.|
|[EmbCacheEmbeddingCollectionSharder](05_subtable_apis.md#embcacheembeddingcollectionsharder)|Creates an `EmbCacheEmbeddingCollectionSharder` sharder for sharding `EmbCacheEmbeddingCollection` across different devices.|
|[DistributedModelParallel(TorchRec)](05_subtable_apis.md#distributedmodelparallel-torchrec)|Converts the input `Module` into a distributed `Module` and executes the sharding plan.|

### Pipeline APIs

|API | Description|
|--|--|
|[HybridTrainPipelineSparseDist](06_pipeline_apis.md#hybridtrainpipelinesparsedist)|Creates a distributed training data pipeline scheduler in full-device memory mode.|
|[EmbCacheTrainPipelineSparseDist](06_pipeline_apis.md#embcachetrainpipelinesparsedist)|Creates a distributed training data pipeline scheduler in multi-level cache mode.|

### Multi-Level Cache Management APIs

|API | Description|
|--|--|
|[InitializerType](07_multilevel_cache_management_apis.md#initializertype)|An enumeration of weight initialization types. It defines how embedding table weights are initialized.|
|[Saver](07_multilevel_cache_management_apis.md#saver)|A save and load utility for sparse tables in multi-level cache. It provides save and load interfaces for sparse table data in multi-level cache, such as sparse table embeddings and the optimizer parameters corresponding to the embeddings.|

### Admission and Eviction Management APIs

|API | Description|
|--|--|
|[JaggedTensorWithTimestamp](08_access_and_elimination_management_apis.md#jaggedtensorwithtimestamp)|A class that extends `JaggedTensor` and represents a jagged tensor with timestamp information. It adds an `_timestamps` attribute to `JaggedTensor` to store the timestamps corresponding to `values`. It is used to calculate elapsed time during feature eviction.|
|[KeyedJaggedTensorWithTimestamp](08_access_and_elimination_management_apis.md#keyedjaggedtensorwithtimestamp)|A class that extends `KeyedJaggedTensor` and represents a keyed jagged tensor with timestamp information. It adds an `_timestamps` attribute to `KeyedJaggedTensor` to store the timestamps corresponding to `values`. It is used to calculate elapsed time during feature eviction.|
|[AdmitAndEvictPolicyType](08_access_and_elimination_management_apis.md#admitandevictpolicytype)|An enumeration of admission and eviction policy types. It defines the policy type for feature admission and eviction in embedding tables.|
|[ShowClickParams](08_access_and_elimination_management_apis.md#showclickparams)|This interface defines the parameter settings for the show/click policy. The class provides parameters for configuring show/click admission and eviction, allowing users to control feature admission and eviction by impression count and click count.|
|[AdmitAndEvictConfig](08_access_and_elimination_management_apis.md#admitandevictconfig)|This interface defines the admission and eviction configuration for a single embedding table. The class provides parameters for configuring feature admission and eviction in embedding tables, allowing users to control feature admission and eviction behavior under specific conditions. It supports two policy types: count-based policy (`POLICY_COUNT`) and show/click-based policy (`POLICY_SHOWCLICK`).|

### Custom Operators

Rec SDK Torch provides some custom operators for processing sparse table data and accelerating model training. For details, see the list of custom operators in [Custom Operators](09_specialized_operator.md).
