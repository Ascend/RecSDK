# TorchRec-V2 API

## Support for Native TorchRec APIs

### TorchRec 1.2.0

#### Overview

This section describes the support and restrictions of the APIs (native to TorchRec 1.2.0) on the Ascend NPU. For details about how to use the APIs, see the [TorchRec community documentation](https://meta-pytorch.org/torchrec/api.html). The support and restrictions are classified into the following types:

- If an API is supported and `Restriction` is left blank, the support for this API is the same as that for the native API.
- If an API is supported and `Restriction` is specified, the support for the API is inconsistent with that for the native API. Pay attention to the support on the Ascend NPU.
- If an API is not supported and `Restriction` is left blank, the API is not supported on the Ascend NPU.
- Some APIs exist in the [TorchRec community documentation](https://meta-pytorch.org/torchrec/api.html) but do not appear in this document. These APIs are not part of TorchRec 1.2.0. Note that the APIs in the TorchRec community documentation are for the latest TorchRec version, not TorchRec 1.2.0.

#### `torchrec.sparse.jagged_tensor.JaggedTensor`

| API                              | Supported| Restriction                                             |
| ------------------------------------------ | -------- | ------------------------------------------------------- |
| torchrec.sparse.jagged_tensor.JaggedTensor | Yes      | The `to_padded_dense` and `to_padded_dense_weights` methods are not fully supported yet.|
| JaggedTensor.device                        | Yes      | -                                                       |
| JaggedTensor.empty                         | Yes      | -                                                       |
| JaggedTensor.from_dense                    | Yes      | -                                                       |
| JaggedTensor.from_dense_lengths            | Yes      | -                                                       |
| JaggedTensor.lengths                       | Yes      | -                                                       |
| JaggedTensor.lengths_or_none               | Yes      | -                                                       |
| JaggedTensor.offsets                       | Yes      | -                                                       |
| JaggedTensor.offsets_or_none               | Yes      | -                                                       |
| JaggedTensor.record_stream                 | Yes      | -                                                       |
| JaggedTensor.to                            | Yes      | -                                                       |
| JaggedTensor.to_dense                      | Yes      | -                                                       |
| JaggedTensor.to_dense_weights              | Yes      | -                                                       |
| JaggedTensor.to_padded_dense               | Yes      | The `fbgemm` operator `jagged_to_padded_dense` is partially supported.               |
| JaggedTensor.to_padded_dense_weights       | Yes      | The `fbgemm` operator `jagged_to_padded_dense` is partially supported.               |
| JaggedTensor.values                        | Yes      | -                                                       |
| JaggedTensor.weights                       | Yes      | -                                                       |
| JaggedTensor.weights_or_none               | Yes      | -                                                       |

#### `torchrec.sparse.jagged_tensor.KeyedJaggedTensor`

| API                                        | Supported| Restriction|
| ----------------------------------------------- | -------- | ---------- |
| torchrec.sparse.jagged_tensor.KeyedJaggedTensor | Yes      | -          |
| KeyedJaggedTensor.concat                        | Yes      | -          |
| KeyedJaggedTensor.device                        | Yes      | -          |
| KeyedJaggedTensor.empty                         | Yes      | -          |
| KeyedJaggedTensor.empty_like                    | Yes      | -          |
| KeyedJaggedTensor.from_jt_dict                  | Yes      | -          |
| KeyedJaggedTensor.from_lengths_sync             | Yes      | -          |
| KeyedJaggedTensor.from_offsets_sync             | Yes      | -          |
| KeyedJaggedTensor.index_per_key                 | Yes      | -          |
| KeyedJaggedTensor.inverse_indices               | Yes      | -          |
| KeyedJaggedTensor.inverse_indices_or_none       | Yes      | -          |
| KeyedJaggedTensor.keys                          | Yes      | -          |
| KeyedJaggedTensor.length_per_key                | Yes      | -          |
| KeyedJaggedTensor.length_per_key_or_none        | Yes      | -          |
| KeyedJaggedTensor.lengths                       | Yes      | -          |
| KeyedJaggedTensor.lengths_offset_per_key        | Yes      | -          |
| KeyedJaggedTensor.lengths_or_none               | Yes      | -          |
| KeyedJaggedTensor.offset_per_key                | Yes      | -          |
| KeyedJaggedTensor.offset_per_key_or_none        | Yes      | -          |
| KeyedJaggedTensor.offsets                       | Yes      | -          |
| KeyedJaggedTensor.offsets_or_none               | Yes      | -          |
| KeyedJaggedTensor.permute                       | Yes      | -          |
| KeyedJaggedTensor.record_stream                 | Yes      | -          |
| KeyedJaggedTensor.split                         | Yes      | -          |
| KeyedJaggedTensor.stride                        | Yes      | -          |
| KeyedJaggedTensor.stride_per_key                | Yes      | -          |
| KeyedJaggedTensor.stride_per_key_per_rank       | Yes      | -          |
| KeyedJaggedTensor.sync                          | Yes      | -          |
| KeyedJaggedTensor.to                            | Yes      | -          |
| KeyedJaggedTensor.to_dict                       | Yes      | -          |
| KeyedJaggedTensor.unsync                        | Yes      | -          |
| KeyedJaggedTensor.values                        | Yes      | -          |
| KeyedJaggedTensor.variable_stride_per_key       | Yes      | -          |
| KeyedJaggedTensor.weights                       | Yes      | -          |
| KeyedJaggedTensor.weights_or_none               | Yes      | -          |

#### `torchrec.sparse.jagged_tensor.KeyedTensor`

| API                                  | Supported| Restriction|
| ----------------------------------------- | -------- | ---------- |
| torchrec.sparse.jagged_tensor.KeyedTensor | Yes      | -          |
| KeyedTensor.device                        | Yes      | -          |
| KeyedTensor.from_tensor_list              | Yes      | -          |
| KeyedTensor.key_dim                       | Yes      | -          |
| KeyedTensor.keys                          | Yes      | -          |
| KeyedTensor.length_per_key                | Yes      | -          |
| KeyedTensor.offset_per_key                | Yes      | -          |
| KeyedTensor.record_stream                 | Yes      | -          |
| KeyedTensor.regroup                       | Yes      | -          |
| KeyedTensor.regroup_as_dict               | Yes      | -          |
| KeyedTensor.to                            | Yes      | -          |
| KeyedTensor.to_dict                       | Yes      | -          |
| KeyedTensor.values                        | Yes      | -          |

#### `torchrec.modules.embedding_configs.EmbeddingBagConfig`

| API                                              | Supported| Restriction|
| ----------------------------------------------------- | -------- | ---------- |
| torchrec.modules.embedding_configs.EmbeddingBagConfig | Yes      | -          |
| EmbeddingBagConfig.polling                            | Yes      | -          |

#### `torchrec.modules.embedding_configs.EmbeddingConfig`

| API                                           | Supported| Restriction|
| -------------------------------------------------- | -------- | ---------- |
| torchrec.modules.embedding_configs.EmbeddingConfig | Yes      | -          |

#### `torchrec.modules.embedding_configs.BaseEmbeddingConfig`

| API                                               | Supported| Restriction|
| ------------------------------------------------------ | -------- | ---------- |
| torchrec.modules.embedding_configs.BaseEmbeddingConfig | Yes      | -          |
| BaseEmbeddingConfig.num_embeddings                     | Yes      | -          |
| BaseEmbeddingConfig.embedding_dim                      | Yes      | -          |
| BaseEmbeddingConfig.name                               | Yes      | -          |
| BaseEmbeddingConfig.data_type                          | Yes      | -          |
| BaseEmbeddingConfig.feature_names                      | Yes      | -          |
| BaseEmbeddingConfig.weight_init_max                    | Yes      | -          |
| BaseEmbeddingConfig.weight_init_min                    | Yes      | -          |
| BaseEmbeddingConfig.num_embeddings_post_pruning        | Yes      | -          |
| BaseEmbeddingConfig.init_fn                            | Yes      | -          |
| BaseEmbeddingConfig.need_pos                           | Yes      | -          |

#### `torchrec.modules.embedding_modules.EmbeddingBagCollection`

| API                                                  | Supported| Restriction                             |
| --------------------------------------------------------- | -------- | --------------------------------------- |
| torchrec.modules.embedding_modules.EmbeddingBagCollection | Yes      | The `forward` method is not fully supported yet.                |
| EmbeddingBagCollection.device                             | Yes      | -                                       |
| EmbeddingBagCollection.embedding_bag_configs              | Yes      | -                                       |
| EmbeddingBagCollection.forward                            | Yes      | The `fbgemm` operator `group_index_select_dim0` is not supported.|
| EmbeddingBagCollection.is_weighted                        | Yes      | -                                       |
| EmbeddingBagCollection.reset_parameters                   | Yes      | -                                       |

#### `torchrec.modules.embedding_modules.EmbeddingCollection`

| API                                               | Supported| Restriction|
| ------------------------------------------------------ | -------- | ---------- |
| torchrec.modules.embedding_modules.EmbeddingCollection | Yes      | -          |
| EmbeddingCollection.device                             | Yes      | -          |
| EmbeddingCollection.embedding_configs                  | Yes      | -          |
| EmbeddingCollection.embedding_dim                      | Yes      | -          |
| EmbeddingCollection.embedding_names_by_table           | Yes      | -          |
| EmbeddingCollection.forward                            | Yes      | -          |
| EmbeddingCollection.need_indices                       | Yes      | -          |
| EmbeddingCollection.reset_parameters                   | Yes      | -          |

#### `torchrec.distributed.types.ShardingPlan`

| API                                | Supported| Restriction|
| --------------------------------------- | -------- | ---------- |
| torchrec.distributed.types.ShardingPlan | Yes      | -          |
| ShardingPlan.get_plan_for_module        | Yes      | -          |

#### `torchrec.distributed.planner.planners.EmbeddingShardingPlanner`

| API                                                     | Supported| Restriction              |
| ------------------------------------------------------------ | -------- | ------------------------ |
| torchrec.distributed.planner.planners.EmbeddingShardingPlanner | Yes      | Only the `FUSED` + `ROW_WISE` scenario is supported.|
| EmbeddingShardingPlanner.collective_plan                     | Yes      | Only the `FUSED` + `ROW_WISE` scenario is supported.|
| EmbeddingShardingPlanner.plan                                | Yes      | Only the `FUSED` + `ROW_WISE` scenario is supported.|

#### `torchrec.distributed.planner.enumerators.EmbeddingEnumerator`

| API                                                     | Supported| Restriction|
| ------------------------------------------------------------ | -------- | ---------- |
| torchrec.distributed.planner.enumerators.EmbeddingEnumerator | Yes      | -          |
| EmbeddingEnumerator.enumerate                                | Yes      | -          |
| EmbeddingEnumerator.populate_estimates                       | Yes      | -          |

#### `torchrec.distributed.planner.partitioners.GreedyPerfPartitioner`

| API                                                     | Supported| Restriction|
| ------------------------------------------------------------ | -------- | ---------- |
| torchrec.distributed.planner.partitioners.GreedyPerfPartitioner | Yes      | -          |
| GreedyPerfPartitioner.partition                              | Yes      | -          |

#### `torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation`

| API                                                     | Supported| Restriction|
| ------------------------------------------------------------ | -------- | ---------- |
| torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation | Yes      | -          |
| HeuristicalStorageReservation.reserve                        | Yes      | -          |

#### `torchrec.distributed.planner.proposers.GreedyProposer`

| API                                              | Supported| Restriction|
| ----------------------------------------------------- | -------- | ---------- |
| torchrec.distributed.planner.proposers.GreedyProposer | Yes      | -          |
| GreedyProposer.feedback                               | Yes      | -          |
| GreedyProposer.load                                   | Yes      | -          |
| GreedyProposer.propose                                | Yes      | -          |

#### `torchrec.distributed.planner.shard_estimators.EmbeddingPerfEstimator`

| API                                                     | Supported| Restriction|
| ------------------------------------------------------------ | -------- | ---------- |
| torchrec.distributed.planner.shard_estimators.EmbeddingPerfEstimator | Yes      | -          |
| EmbeddingPerfEstimator.perf_func_emb_wall_time               | Yes      | -          |
| EmbeddingPerfEstimator.estimate                              | Yes      | -          |

#### `torchrec.distributed.planner.shard_estimators.EmbeddingStorageEstimator`

| API                                                     | Supported| Restriction|
| ------------------------------------------------------------ | -------- | ---------- |
| torchrec.distributed.planner.shard_estimators.EmbeddingStorageEstimator | Yes      | -          |
| EmbeddingStorageEstimator.estimate                           | Yes      | -          |

#### `torchrec.distributed.model_parallel.DistributedModelParallel`

| API                                                     | Supported| Restriction              |
| ------------------------------------------------------------ | -------- | ------------------------ |
| torchrec.distributed.model_parallel.DistributedModelParallel | Yes      | Only the `FUSED` + `ROW_WISE` scenario is supported.|
| DistributedModelParallel.copy                                | No      | -                        |
| DistributedModelParallel.forward                             | Yes      | -                        |
| DistributedModelParallel.init_data_parallel                  | Yes      | -                        |
| DistributedModelParallel.load_state_dict                     | Yes      | -                        |
| DistributedModelParallel.module                              | Yes      | -                        |
| DistributedModelParallel.named_buffers                       | Yes      | -                        |
| DistributedModelParallel.named_parameters                    | Yes      | -                        |
| DistributedModelParallel.state_dict                          | Yes      | -                        |

#### `torchrec.inference.modules.quantize_inference_model`

| API                                            | Supported| Restriction|
| --------------------------------------------------- | -------- | ---------- |
| torchrec.inference.modules.quantize_inference_model | No      | -          |

#### `torchrec.inference.modules.shard_quant_model`

| API                                     | Supported| Restriction|
| -------------------------------------------- | -------- | ---------- |
| torchrec.inference.modules.shard_quant_model | No      | -          |

## Support for Native DynamicEmb APIs

### DynamicEmb v25.09

#### Overview

This section describes the support and restrictions of the APIs (native to DynamicEmb v25.09) on the Ascend NPU. For details about how to use the APIs, see the [DynamicEmb community documentation](https://github.com/NVIDIA/recsys-examples/blob/v25.09/corelib/dynamicemb/DynamicEmb_APIs.md). The support and restrictions are classified into the following types:

- If an API is supported and `Restriction` is left blank, the support for this API is the same as that for the native API.
- If an API is supported and `Restriction` is specified, the support for the API is inconsistent with that for the native API. Pay attention to the support on the Ascend NPU.
- If an API is not supported and `Restriction` is left blank, the API is not supported on the Ascend NPU.

#### `DynamicEmbParameterConstraints`

| API                                          | Supported| Restriction                                         |
| ------------------------------------------------- | -------- | --------------------------------------------------- |
| DynamicEmbParameterConstraints                    | Yes      | The `caching`, `score_strategy`, and `external_storage` scenarios are not supported.|
| DynamicEmbParameterConstraints.use_dynamicemb     | Yes      | -                                                   |
| DynamicEmbParameterConstraints.dynamicemb_options | Yes      | The `caching`, `score_strategy`, and `external_storage` scenarios are not supported.|

#### `DynamicEmbeddingEnumerator`

| API                             | Supported| Restriction|
| ------------------------------------ | -------- | ---------- |
| DynamicEmbeddingEnumerator           | Yes      | -          |
| DynamicEmbeddingEnumerator.enumerate | Yes      | -          |

#### `DynamicEmbeddingShardingPlanner`

| API                                        | Supported| Restriction|
| ----------------------------------------------- | -------- | ---------- |
| DynamicEmbeddingShardingPlanner                 | Yes      | -          |
| DynamicEmbeddingShardingPlanner.collective_plan | Yes      | -          |

#### `DynamicEmbeddingCollectionSharder`

| API                                | Supported| Restriction|
| --------------------------------------- | -------- | ---------- |
| DynamicEmbeddingCollectionSharder       | Yes      | -          |
| DynamicEmbeddingCollectionSharder.shard | Yes      | -          |

#### `DynamicEmbCheckMode`

| API                    | Supported| Restriction|
| --------------------------- | -------- | ---------- |
| DynamicEmbCheckMode         | Yes      | -          |
| DynamicEmbCheckMode.ERROR   | Yes      | -          |
| DynamicEmbCheckMode.WARNING | Yes      | -          |
| DynamicEmbCheckMode.IGNORE  | Yes      | -          |

#### `DynamicEmbInitializerMode`

| API                                   | Supported| Restriction|
| ------------------------------------------ | -------- | ---------- |
| DynamicEmbInitializerMode                  | Yes      | -          |
| DynamicEmbInitializerMode.NORMAL           | Yes      | -          |
| DynamicEmbInitializerMode.TRUNCATED_NORMAL | Yes      | -          |
| DynamicEmbInitializerMode.UNIFORM          | Yes      | -          |
| DynamicEmbInitializerMode.CONSTANT         | Yes      | -          |
| DynamicEmbInitializerMode.DEBUG            | Yes      | -          |

#### `DynamicEmbInitializerArgs`

| API                          | Supported| Restriction|
| --------------------------------- | -------- | ---------- |
| DynamicEmbInitializerArgs         | Yes      | -          |
| DynamicEmbInitializerArgs.mode    | Yes      | -          |
| DynamicEmbInitializerArgs.mean    | Yes      | -          |
| DynamicEmbInitializerArgs.std_dev | Yes      | -          |
| DynamicEmbInitializerArgs.lower   | Yes      | -          |
| DynamicEmbInitializerArgs.upper   | Yes      | -          |
| DynamicEmbInitializerArgs.value   | Yes      | -          |

#### `DynamicEmbScoreStrategy`

| API                           | Supported| Restriction|
| ---------------------------------- | -------- | ---------- |
| DynamicEmbScoreStrategy            | Yes      | -          |
| DynamicEmbScoreStrategy.TIMESTAMP  | Yes      | -          |
| DynamicEmbScoreStrategy.STEP       | Yes      | -          |
| DynamicEmbScoreStrategy.CUSTOMIZED | Yes      | -          |
| DynamicEmbScoreStrategy.LFU        | Yes      | -          |

#### `DynamicEmbTableOptions`

| API                                     | Supported| Restriction|
| -------------------------------------------- | -------- | ---------- |
| DynamicEmbTableOptions                       | Yes      | -          |
| DynamicEmbTableOptions.training              | Yes      | -          |
| DynamicEmbTableOptions.initializer_args      | Yes      | -          |
| DynamicEmbTableOptions.eval_initializer_args | Yes      | -          |
| DynamicEmbTableOptions.caching               | No      | -          |
| DynamicEmbTableOptions.init_capacity         | Yes      | -          |
| DynamicEmbTableOptions.max_load_factor       | Yes      | -          |
| DynamicEmbTableOptions.score_strategy        | No      | -          |
| DynamicEmbTableOptions.bucket_capacity       | Yes      | -          |
| DynamicEmbTableOptions.safe_check_mode       | Yes      | -          |
| DynamicEmbTableOptions.global_hbm_for_values | Yes      | -          |
| DynamicEmbTableOptions.external_storage      | No      | -          |
| DynamicEmbTableOptions.index_type            | Yes      | -          |

#### `DynamicEmbDump`

| API                   | Supported| Restriction|
| -------------------------- | -------- | ---------- |
| DynamicEmbDump             | Yes      | -          |
| DynamicEmbDump.path        | Yes      | -          |
| DynamicEmbDump.model       | Yes      | -          |
| DynamicEmbDump.table_names | Yes      | -          |
| DynamicEmbDump.optim       | Yes      | -          |
| DynamicEmbDump.pg          | Yes      | -          |

#### `DynamicEmbLoad`

| API                   | Supported| Restriction|
| -------------------------- | -------- | ---------- |
| DynamicEmbLoad             | Yes      | -          |
| DynamicEmbLoad.path        | Yes      | -          |
| DynamicEmbLoad.model       | Yes      | -          |
| DynamicEmbLoad.table_names | Yes      | -          |
| DynamicEmbLoad.optim       | Yes      | -          |
| DynamicEmbLoad.pg          | Yes      | -          |

#### `incremental_dump`

| API         | Supported| Restriction|
| ---------------- | -------- | ---------- |
| incremental_dump | No      | -          |

#### `get_score`

| API  | Supported| Restriction|
| --------- | -------- | ---------- |
| get_score | Yes      | -          |

#### `set_score`

| API  | Supported| Restriction|
| --------- | -------- | ---------- |
| set_score | Yes      | -          |
