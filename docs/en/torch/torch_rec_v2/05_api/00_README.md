# API List

## API Overview

- [API Description](01_api_description.md)
- [Table Creation APIs](02_table_creation_apis.md)
- [Data APIs](03_data_apis.md)
- [Sharding APIs](04_subtable_apis.md)
- [Saving and Loading APIs](05_dump_load_apis.md)
- [Custom Operators](06_specialized_operator.md)
- [API Support](07_api_compatibility.md)

## API Details

Note: In the following APIs, the `(TorchRec)` tag indicates a native TorchRec API. All other APIs are provided by Rec SDK Torch.

### Table Creation APIs

|API | Description|
|--|--|
|[EmbeddingConfig(TorchRec)](02_table_creation_apis.md#embeddingconfig)|Input to `EmbeddingCollection`. Use it to configure table size, `dim`, data type, and other settings.|
|[EmbeddingCollection(TorchRec)](02_table_creation_apis.md#embeddingcollection)|Creates a single-node table object.|

### Data APIs

|API | Description|
|--|--|
|[JaggedTensor(TorchRec)](03_data_apis.md#jaggedtensor-torchrec)|A class that holds sparse IDs and feature lengths for table lookup.|
|[KeyedJaggedTensor(TorchRec)](03_data_apis.md#keyedjaggedtensor-torchrec)|Extends `JaggedTensor` by introducing keys (usually feature names) to identify different feature groups.|

### Sharding APIs

|API | Description|
|--|--|
|[ShardingEnv(TorchRec)](04_subtable_apis.md#shardingenv-torchrec)|Stores distributed-related parameters.|
|[Topology(TorchRec)](04_subtable_apis.md#topology-torchrec)|Stores topology parameters for network devices in a distributed environment.|
|[DynamicEmbeddingCollectionSharder](04_subtable_apis.md#dynamicembeddingcollectionsharder)|Creates a dynamic embedding table sharder without pooling.|
|[DynamicEmbeddingBagCollectionSharder](04_subtable_apis.md#dynamicembeddingbagcollectionsharder)|Creates a dynamic embedding table sharder with pooling.|
|[DynamicEmbParameterConstraints](04_subtable_apis.md#dynamicembparameterconstraints)|Configures sharding constraints for dynamic embedding tables.|
|[DynamicEmbTableOptions](04_subtable_apis.md#dynamicembtableoptions)|Configures dynamic embedding table parameters.|
|[DynamicEmbInitializerMode](04_subtable_apis.md#dynamicembinitializermode)|An enumeration of initialization modes for embedding vectors in dynamic embedding tables.|
|[DynamicEmbInitializerArgs](04_subtable_apis.md#dynamicembinitializerargs)|Configures initialization parameters for embedding vectors in dynamic embedding tables.|
|[DynamicEmbScoreStrategy](04_subtable_apis.md#dynamicembscorestrategy)|An enumeration of scoring strategies for dynamic embedding tables, used for sparse feature eviction.|
|[EmbOptimType](04_subtable_apis.md#emboptimtype)|An enumeration of optimizer types for dynamic embedding tables.|
|[DynamicEmbeddingEnumerator](04_subtable_apis.md#dynamicembeddingenumerator)|Enumerates sharding plans for dynamic embedding tables.|
|[DynamicEmbeddingShardingPlanner](04_subtable_apis.md#dynamicembeddingshardingplanner)|Creates a dynamic embedding table sharding planner to search for sharding plans.|
|[DynamicEmbCheckMode](04_subtable_apis.md#dynamicembcheckmode)|An enumeration of safe-check modes for dynamic embedding table index insertion.|

### Saving and Loading APIs

|API | Description|
|--|--|
|[DynamicEmbDump](05_dump_load_apis.md#dynamicembdump)|Saves dynamic embedding table data in the model to the file system in parallel.|
|[set_score](05_dump_load_apis.md#set_score)|Sets the scores of dynamic embedding tables in the model.|
|[get_score](05_dump_load_apis.md#get_score)|Gets the scores of dynamic embedding tables in the model.|
|[DynamicEmbLoad](05_dump_load_apis.md#dynamicembload)|Loads dynamic embedding table data from the file system and writes it to the model.|
|[incremental_dump](05_dump_load_apis.md#incremental_dump)|Incrementally exports dynamic embedding table data from the model based on score thresholds.|

### Custom Operators

Rec SDK Torch provides some custom operators for processing dynamic sparse table data and accelerating model training. For details, see the list of custom operators in [Custom Operators](06_specialized_operator.md).
