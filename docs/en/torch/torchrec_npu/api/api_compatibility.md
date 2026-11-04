# TorchRec_npu API

## Support for Native TorchRec APIs

### Overview

This section describes the support and restrictions of the TorchRec native APIs on the Ascend NPU. For details about how to use the TorchRec native APIs, see the [TorchRec community documentation](https://meta-pytorch.org/torchrec/api.html). The support and restrictions of the native APIs on the Ascend NPU are as follows:

- If an API is supported and the Restriction column is `-`, the support for this API is the same as that for the native API.
- If an API is supported and the Restriction column is not `-`, the support for this API is the same as that for the native API when the restriction content is a version description (for example, added or removed in a specific version). In this case, the restriction only supplements the version in which the API was introduced. Otherwise, the support for this API is inconsistent with that for the native API. Pay attention to the support on the Ascend NPU.
- If an API is not supported and the Restriction column is `-`, the API is currently not supported on the Ascend NPU.
- Some APIs are added in specific TorchRec versions. The Restriction column marks the TorchRec version that introduced such an API. APIs without a version marker are supported in v1.2.0 and later versions.

The general constraints of the `torchrec_npu` framework are as follows:

- SSD and UVM scenarios are not supported.
- The split/dense table lookup operators are supported only in some scenarios (weighted/unweighted + codegen + SGD/Adagrad/Adam/RowWiseAdagrad optimizers).

### Data Types

#### `torchrec.sparse.jagged_tensor.JaggedTensor`

| API                                    | Supported | Restriction     |
| -------------------------------------- | --------- | --------------- |
| torchrec.sparse.jagged_tensor.JaggedTensor | Yes | - |
| JaggedTensor.copy_ | Yes | Added in v1.6.0 |
| JaggedTensor.device | Yes | - |
| JaggedTensor.empty | Yes | - |
| JaggedTensor.empty_like | Yes | Added in v1.6.0 |
| JaggedTensor.from_dense | Yes | - |
| JaggedTensor.from_dense_lengths | Yes | - |
| JaggedTensor.lengths | Yes | - |
| JaggedTensor.lengths_or_none | Yes | - |
| JaggedTensor.offsets | Yes | - |
| JaggedTensor.offsets_or_none | Yes | - |
| JaggedTensor.record_stream | Yes | - |
| JaggedTensor.size_in_bytes | Yes | Added in v1.6.0 |
| JaggedTensor.to | Yes | - |
| JaggedTensor.to_dense | Yes | - |
| JaggedTensor.to_dense_weights | Yes | - |
| JaggedTensor.to_padded_dense | Yes | - |
| JaggedTensor.to_padded_dense_weights | Yes | - |
| JaggedTensor.values | Yes | - |
| JaggedTensor.weights | Yes | - |
| JaggedTensor.weights_or_none | Yes | - |

#### `torchrec.sparse.jagged_tensor.KeyedJaggedTensor`

| API                                         | Supported | Restriction     |
| ------------------------------------------- | --------- | --------------- |
| torchrec.sparse.jagged_tensor.KeyedJaggedTensor | Yes | - |
| KeyedJaggedTensor.clear_storage | Yes | Added in v1.7.0 |
| KeyedJaggedTensor.concat | Yes | - |
| KeyedJaggedTensor.copy_ | Yes | Added in v1.5.0 |
| KeyedJaggedTensor.device | Yes | - |
| KeyedJaggedTensor.empty | Yes | - |
| KeyedJaggedTensor.empty_like | Yes | - |
| KeyedJaggedTensor.from_jt_dict | Yes | - |
| KeyedJaggedTensor.from_lengths_sync | Yes | - |
| KeyedJaggedTensor.from_offsets_sync | Yes | - |
| KeyedJaggedTensor.index_per_key | Yes | - |
| KeyedJaggedTensor.inverse_indices | Yes | - |
| KeyedJaggedTensor.inverse_indices_or_none | Yes | - |
| KeyedJaggedTensor.keys | Yes | - |
| KeyedJaggedTensor.length_per_key | Yes | - |
| KeyedJaggedTensor.length_per_key_or_none | Yes | - |
| KeyedJaggedTensor.lengths | Yes | - |
| KeyedJaggedTensor.lengths_offset_per_key | Yes | - |
| KeyedJaggedTensor.lengths_or_none | Yes | - |
| KeyedJaggedTensor.offset_per_key | Yes | - |
| KeyedJaggedTensor.offset_per_key_or_none | Yes | - |
| KeyedJaggedTensor.offsets | Yes | - |
| KeyedJaggedTensor.offsets_or_none | Yes | - |
| KeyedJaggedTensor.permute | Yes | - |
| KeyedJaggedTensor.record_stream | Yes | - |
| KeyedJaggedTensor.size_in_bytes | Yes | Added in v1.6.0 |
| KeyedJaggedTensor.split | Yes | - |
| KeyedJaggedTensor.stride | Yes | - |
| KeyedJaggedTensor.stride_per_key | Yes | - |
| KeyedJaggedTensor.stride_per_key_per_rank | Yes | - |
| KeyedJaggedTensor.sync | Yes | - |
| KeyedJaggedTensor.to | Yes | - |
| KeyedJaggedTensor.to_dict | Yes | - |
| KeyedJaggedTensor.unsync | Yes | - |
| KeyedJaggedTensor.values | Yes | - |
| KeyedJaggedTensor.variable_stride_per_key | Yes | - |
| KeyedJaggedTensor.weights | Yes | - |
| KeyedJaggedTensor.weights_or_none | Yes | - |

#### `torchrec.sparse.jagged_tensor.KeyedTensor`

| API                                   | Supported | Restriction     |
| ------------------------------------- | --------- | --------------- |
| torchrec.sparse.jagged_tensor.KeyedTensor | Yes | - |
| KeyedTensor.device | Yes | - |
| KeyedTensor.from_tensor_list | Yes | - |
| KeyedTensor.key_dim | Yes | - |
| KeyedTensor.keys | Yes | - |
| KeyedTensor.length_per_key | Yes | - |
| KeyedTensor.offset_per_key | Yes | - |
| KeyedTensor.record_stream | Yes | - |
| KeyedTensor.regroup | Yes | - |
| KeyedTensor.regroup_as_dict | Yes | - |
| KeyedTensor.size_in_bytes | Yes | Added in v1.6.0 |
| KeyedTensor.to | Yes | - |
| KeyedTensor.to_dict | Yes | - |
| KeyedTensor.values | Yes | - |

### Modules

#### `torchrec.modules.embedding_configs.BaseEmbeddingConfig`

| API                                                | Supported | Restriction              |
| -------------------------------------------------- | --------- | ------------------------ |
| torchrec.modules.embedding_configs.BaseEmbeddingConfig | Yes | - |
| BaseEmbeddingConfig.num_embeddings | Yes | - |
| BaseEmbeddingConfig.embedding_dim | Yes | - |
| BaseEmbeddingConfig.name | Yes | - |
| BaseEmbeddingConfig.data_type | Yes | - |
| BaseEmbeddingConfig.feature_names | Yes | - |
| BaseEmbeddingConfig.weight_init_max | Yes | - |
| BaseEmbeddingConfig.weight_init_min | Yes | - |
| BaseEmbeddingConfig.num_embeddings_post_pruning | Yes | - |
| BaseEmbeddingConfig.init_fn | Yes | - |
| BaseEmbeddingConfig.need_pos | Yes | - |
| BaseEmbeddingConfig.input_dim | Yes | - |
| BaseEmbeddingConfig.total_num_buckets | Yes | Added in v1.5.0 |
| BaseEmbeddingConfig.use_virtual_table | Yes | Added in v1.5.0 |
| BaseEmbeddingConfig.virtual_table_eviction_policy | Yes | Added in v1.5.0 |
| BaseEmbeddingConfig.enable_embedding_update | Yes | Added in v1.5.0 |
| BaseEmbeddingConfig.stash_weights | Yes | Added in v1.6.0 |

#### `torchrec.modules.embedding_configs.EmbeddingBagConfig`

| API                                               | Supported | Restriction                |
| ------------------------------------------------- | --------- | -------------------------- |
| torchrec.modules.embedding_configs.EmbeddingBagConfig | Yes | Inherits `BaseEmbeddingConfig` |
| EmbeddingBagConfig.pooling | Yes | - |

#### `torchrec.modules.embedding_configs.EmbeddingConfig`

| API                                            | Supported | Restriction                |
| ---------------------------------------------- | --------- | -------------------------- |
| torchrec.modules.embedding_configs.EmbeddingConfig | Yes | Inherits `BaseEmbeddingConfig` |

#### `torchrec.modules.embedding_modules.EmbeddingBagCollection`

| API                                                   | Supported | Restriction |
| ----------------------------------------------------- | --------- | ----------- |
| torchrec.modules.embedding_modules.EmbeddingBagCollection | Yes | - |
| EmbeddingBagCollection.device | Yes | - |
| EmbeddingBagCollection.embedding_bag_configs | Yes | - |
| EmbeddingBagCollection.forward | Yes | - |
| EmbeddingBagCollection.is_weighted | Yes | - |
| EmbeddingBagCollection.reset_parameters | Yes | - |

#### `torchrec.modules.embedding_modules.EmbeddingCollection`

| API                                                | Supported | Restriction     |
| -------------------------------------------------- | --------- | --------------- |
| torchrec.modules.embedding_modules.EmbeddingCollection | Yes | - |
| EmbeddingCollection.device | Yes | - |
| EmbeddingCollection.embedding_configs | Yes | - |
| EmbeddingCollection.embedding_dim | Yes | - |
| EmbeddingCollection.embedding_names_by_table | Yes | - |
| EmbeddingCollection.forward | Yes | - |
| EmbeddingCollection.need_indices | Yes | - |
| EmbeddingCollection.reset_parameters | Yes | - |
| EmbeddingCollection.set_use_sorted_select | No | - |
| EmbeddingCollection.use_gather_select | Yes | Added in v1.5.0 |
| EmbeddingCollection.use_gather_select_per_sharding | No | - |
| EmbeddingCollection.use_sorted_select | No | - |

### Planner

#### `torchrec.distributed.types.ShardingPlan`

| API                                 | Supported | Restriction |
| ----------------------------------- | --------- | ----------- |
| torchrec.distributed.types.ShardingPlan | Yes | - |
| ShardingPlan.plan | Yes | - |
| ShardingPlan.get_plan_for_module | Yes | - |

#### `torchrec.distributed.planner.planners.EmbeddingShardingPlanner`

| API                                                      | Supported | Restriction                                     |
| -------------------------------------------------------- | --------- | ----------------------------------------------- |
| torchrec.distributed.planner.planners.EmbeddingShardingPlanner | Yes | Only the `FUSED` + `ROW_WISE` scenario is supported. |
| EmbeddingShardingPlanner.collective_plan | Yes | Only the `FUSED` + `ROW_WISE` scenario is supported. |
| EmbeddingShardingPlanner.get_search_space | No | - |
| EmbeddingShardingPlanner.get_selected_options | No | - |
| EmbeddingShardingPlanner.plan | Yes | Only the `FUSED` + `ROW_WISE` scenario is supported. |

#### `torchrec.distributed.planner.enumerators.EmbeddingEnumerator`

| API                                                      | Supported | Restriction |
| -------------------------------------------------------- | --------- | ----------- |
| torchrec.distributed.planner.enumerators.EmbeddingEnumerator | Yes | - |
| EmbeddingEnumerator.enumerate | Yes | - |
| EmbeddingEnumerator.populate_estimates | Yes | - |

#### `torchrec.distributed.planner.partitioners.GreedyPerfPartitioner`

| API                                                      | Supported | Restriction |
| -------------------------------------------------------- | --------- | ----------- |
| torchrec.distributed.planner.partitioners.GreedyPerfPartitioner | Yes | - |
| GreedyPerfPartitioner.partition | Yes | - |

#### `torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation`

| API                                                      | Supported | Restriction     |
| -------------------------------------------------------- | --------- | --------------- |
| torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation | Yes | - |
| HeuristicalStorageReservation.last_reserved_topology | Yes | Added in v1.5.0 |
| HeuristicalStorageReservation.reserve | Yes | - |

#### `torchrec.distributed.planner.proposers.GreedyProposer`

| API                                               | Supported | Restriction |
| ------------------------------------------------- | --------- | ----------- |
| torchrec.distributed.planner.proposers.GreedyProposer | Yes | - |
| GreedyProposer.feedback | Yes | - |
| GreedyProposer.load | Yes | - |
| GreedyProposer.propose | Yes | - |

#### `torchrec.distributed.planner.shard_estimators.EmbeddingPerfEstimator`

| API                                                      | Supported | Restriction                                           |
| -------------------------------------------------------- | --------- | ----------------------------------------------------- |
| torchrec.distributed.planner.shard_estimators.EmbeddingPerfEstimator | Yes | - |
| EmbeddingPerfEstimator.perf_func_emb_wall_time | Yes | Supported from v1.2.0 to v1.5.0, and removed starting from v1.6.0. |
| EmbeddingPerfEstimator.estimate | Yes | - |

#### `torchrec.distributed.planner.shard_estimators.EmbeddingStorageEstimator`

| API                                                      | Supported | Restriction |
| -------------------------------------------------------- | --------- | ----------- |
| torchrec.distributed.planner.shard_estimators.EmbeddingStorageEstimator | Yes | - |
| EmbeddingStorageEstimator.estimate | Yes | - |

### Model Parallel

#### `torchrec.distributed.model_parallel.DistributedModelParallel`

| API                                                      | Supported | Restriction                                     |
| -------------------------------------------------------- | --------- | ----------------------------------------------- |
| torchrec.distributed.model_parallel.DistributedModelParallel | Yes | Only the `FUSED` + `ROW_WISE` scenario is supported. |
| DistributedModelParallel.copy | Yes | - |
| DistributedModelParallel.forward | Yes | - |
| DistributedModelParallel.get_delta_tracker | Yes | Added in v1.5.0 |
| DistributedModelParallel.init_data_parallel | Yes | - |
| DistributedModelParallel.init_torchrec_delta_tracker | Yes | Added in v1.5.0 |
| DistributedModelParallel.load_state_dict | Yes | - |
| DistributedModelParallel.module | Yes | - |
| DistributedModelParallel.named_buffers | Yes | - |
| DistributedModelParallel.named_parameters | Yes | - |
| DistributedModelParallel.reshard | Yes | Added in v1.5.0 |
| DistributedModelParallel.state_dict | Yes | - |
| DistributedModelParallel.write | No | Added in v1.6.0 |

### Inference

#### `torchrec.inference.modules.quantize_inference_model`

| API                                             | Supported | Restriction |
| ----------------------------------------------- | --------- | ----------- |
| torchrec.inference.modules.quantize_inference_model | Yes | - |

#### `torchrec.inference.modules.shard_quant_model`

| API                                      | Supported | Restriction |
| ---------------------------------------- | --------- | ----------- |
| torchrec.inference.modules.shard_quant_model | Yes | - |
