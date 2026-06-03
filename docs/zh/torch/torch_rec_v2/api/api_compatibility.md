# TorchRec-V2 API

## TorchRec原生API支持度

### TorchRec 1.2.0版本

#### 概述

介绍TorchRec 1.2.0版本原生API接口在昇腾NPU上的支持情况与限制说明，TorchRec 1.2.0版本原生API接口具体使用方法请参考[TorchRec社区文档](https://meta-pytorch.org/torchrec/api.html)。原生API接口在昇腾NPU上的支持情况与限制说明可分为如下三类：

- API“是否支持“为“是“、“限制与说明“为“-“，说明此API和原生API支持度保持一致。
- API“是否支持“为“是“、“限制与说明“不为“-“，说明此API和原生API支持度不一致，请注意昇腾NPU上的支持度。
- API“是否支持“为“否“、“限制与说明“为“-“，说明在昇腾NPU上暂不支持此API。
- 部分API在[TorchRec社区文档](https://meta-pytorch.org/torchrec/api.html)中存在，但此文档中未承载，为TorchRec 1.2.0版本没有的API。注：TorchRec社区文档中API为TorchRec最新版本API，而非TorchRec 1.2.0版本API。

#### torchrec.sparse.jagged_tensor.JaggedTensor

| API名称                                    | 是否支持 | 限制与说明                                              |
| ------------------------------------------ | -------- | ------------------------------------------------------- |
| torchrec.sparse.jagged_tensor.JaggedTensor | 是       | to_padded_dense/to_padded_dense_weights方法暂未全量支持 |
| JaggedTensor.device                        | 是       | -                                                       |
| JaggedTensor.empty                         | 是       | -                                                       |
| JaggedTensor.from_dense                    | 是       | -                                                       |
| JaggedTensor.from_dense_lengths            | 是       | -                                                       |
| JaggedTensor.lengths                       | 是       | -                                                       |
| JaggedTensor.lengths_or_none               | 是       | -                                                       |
| JaggedTensor.offsets                       | 是       | -                                                       |
| JaggedTensor.offsets_or_none               | 是       | -                                                       |
| JaggedTensor.record_stream                 | 是       | -                                                       |
| JaggedTensor.to                            | 是       | -                                                       |
| JaggedTensor.to_dense                      | 是       | -                                                       |
| JaggedTensor.to_dense_weights              | 是       | -                                                       |
| JaggedTensor.to_padded_dense               | 是       | fbgemm算子jagged_to_padded_dense部分支持                |
| JaggedTensor.to_padded_dense_weights       | 是       | fbgemm算子jagged_to_padded_dense部分支持                |
| JaggedTensor.values                        | 是       | -                                                       |
| JaggedTensor.weights                       | 是       | -                                                       |
| JaggedTensor.weights_or_none               | 是       | -                                                       |

#### torchrec.sparse.jagged_tensor.KeyedJaggedTensor

| API名称                                         | 是否支持 | 限制与说明 |
| ----------------------------------------------- | -------- | ---------- |
| torchrec.sparse.jagged_tensor.KeyedJaggedTensor | 是       | -          |
| KeyedJaggedTensor.concat                        | 是       | -          |
| KeyedJaggedTensor.device                        | 是       | -          |
| KeyedJaggedTensor.empty                         | 是       | -          |
| KeyedJaggedTensor.empty_like                    | 是       | -          |
| KeyedJaggedTensor.from_jt_dict                  | 是       | -          |
| KeyedJaggedTensor.from_lengths_sync             | 是       | -          |
| KeyedJaggedTensor.from_offsets_sync             | 是       | -          |
| KeyedJaggedTensor.index_per_key                 | 是       | -          |
| KeyedJaggedTensor.inverse_indices               | 是       | -          |
| KeyedJaggedTensor.inverse_indices_or_none       | 是       | -          |
| KeyedJaggedTensor.keys                          | 是       | -          |
| KeyedJaggedTensor.length_per_key                | 是       | -          |
| KeyedJaggedTensor.length_per_key_or_none        | 是       | -          |
| KeyedJaggedTensor.lengths                       | 是       | -          |
| KeyedJaggedTensor.lengths_offset_per_key        | 是       | -          |
| KeyedJaggedTensor.lengths_or_none               | 是       | -          |
| KeyedJaggedTensor.offset_per_key                | 是       | -          |
| KeyedJaggedTensor.offset_per_key_or_none        | 是       | -          |
| KeyedJaggedTensor.offsets                       | 是       | -          |
| KeyedJaggedTensor.offsets_or_none               | 是       | -          |
| KeyedJaggedTensor.permute                       | 是       | -          |
| KeyedJaggedTensor.record_stream                 | 是       | -          |
| KeyedJaggedTensor.split                         | 是       | -          |
| KeyedJaggedTensor.stride                        | 是       | -          |
| KeyedJaggedTensor.stride_per_key                | 是       | -          |
| KeyedJaggedTensor.stride_per_key_per_rank       | 是       | -          |
| KeyedJaggedTensor.sync                          | 是       | -          |
| KeyedJaggedTensor.to                            | 是       | -          |
| KeyedJaggedTensor.to_dict                       | 是       | -          |
| KeyedJaggedTensor.unsync                        | 是       | -          |
| KeyedJaggedTensor.values                        | 是       | -          |
| KeyedJaggedTensor.variable_stride_per_key       | 是       | -          |
| KeyedJaggedTensor.weights                       | 是       | -          |
| KeyedJaggedTensor.weights_or_none               | 是       | -          |

#### torchrec.sparse.jagged_tensor.KeyedTensor

| API名称                                   | 是否支持 | 限制与说明 |
| ----------------------------------------- | -------- | ---------- |
| torchrec.sparse.jagged_tensor.KeyedTensor | 是       | -          |
| KeyedTensor.device                        | 是       | -          |
| KeyedTensor.from_tensor_list              | 是       | -          |
| KeyedTensor.key_dim                       | 是       | -          |
| KeyedTensor.keys                          | 是       | -          |
| KeyedTensor.length_per_key                | 是       | -          |
| KeyedTensor.offset_per_key                | 是       | -          |
| KeyedTensor.record_stream                 | 是       | -          |
| KeyedTensor.regroup                       | 是       | -          |
| KeyedTensor.regroup_as_dict               | 是       | -          |
| KeyedTensor.to                            | 是       | -          |
| KeyedTensor.to_dict                       | 是       | -          |
| KeyedTensor.values                        | 是       | -          |

#### torchrec.modules.embedding_configs.EmbeddingBagConfig

| API名称                                                 | 是否支持 | 限制与说明 |
|-------------------------------------------------------| -------- | ---------- |
| torchrec.modules.embedding_configs.EmbeddingBagConfig | 是       | -          |
| EmbeddingBagConfig.pooling                            | 是       | -          |

#### torchrec.modules.embedding_configs.EmbeddingConfig

| API名称                                            | 是否支持 | 限制与说明 |
| -------------------------------------------------- | -------- | ---------- |
| torchrec.modules.embedding_configs.EmbeddingConfig | 是       | -          |

#### torchrec.modules.embedding_configs.BaseEmbeddingConfig

| API名称                                                | 是否支持 | 限制与说明 |
| ------------------------------------------------------ | -------- | ---------- |
| torchrec.modules.embedding_configs.BaseEmbeddingConfig | 是       | -          |
| BaseEmbeddingConfig.num_embeddings                     | 是       | -          |
| BaseEmbeddingConfig.embedding_dim                      | 是       | -          |
| BaseEmbeddingConfig.name                               | 是       | -          |
| BaseEmbeddingConfig.data_type                          | 是       | -          |
| BaseEmbeddingConfig.feature_names                      | 是       | -          |
| BaseEmbeddingConfig.weight_init_max                    | 是       | -          |
| BaseEmbeddingConfig.weight_init_min                    | 是       | -          |
| BaseEmbeddingConfig.num_embeddings_post_pruning        | 是       | -          |
| BaseEmbeddingConfig.init_fn                            | 是       | -          |
| BaseEmbeddingConfig.need_pos                           | 是       | -          |

#### torchrec.modules.embedding_modules.EmbeddingBagCollection

| API名称                                                   | 是否支持 | 限制与说明                              |
| --------------------------------------------------------- | -------- | --------------------------------------- |
| torchrec.modules.embedding_modules.EmbeddingBagCollection | 是       | forward方法暂未全量支持                 |
| EmbeddingBagCollection.device                             | 是       | -                                       |
| EmbeddingBagCollection.embedding_bag_configs              | 是       | -                                       |
| EmbeddingBagCollection.forward                            | 是       | fbgemm算子group_index_select_dim0未支持 |
| EmbeddingBagCollection.is_weighted                        | 是       | -                                       |
| EmbeddingBagCollection.reset_parameters                   | 是       | -                                       |

#### torchrec.modules.embedding_modules.EmbeddingCollection

| API名称                                                | 是否支持 | 限制与说明 |
| ------------------------------------------------------ | -------- | ---------- |
| torchrec.modules.embedding_modules.EmbeddingCollection | 是       | -          |
| EmbeddingCollection.device                             | 是       | -          |
| EmbeddingCollection.embedding_configs                  | 是       | -          |
| EmbeddingCollection.embedding_dim                      | 是       | -          |
| EmbeddingCollection.embedding_names_by_table           | 是       | -          |
| EmbeddingCollection.forward                            | 是       | -          |
| EmbeddingCollection.need_indices                       | 是       | -          |
| EmbeddingCollection.reset_parameters                   | 是       | -          |

#### torchrec.distributed.types.ShardingPlan

| API名称                                 | 是否支持 | 限制与说明 |
| --------------------------------------- | -------- | ---------- |
| torchrec.distributed.types.ShardingPlan | 是       | -          |
| ShardingPlan.get_plan_for_module        | 是       | -          |

#### torchrec.distributed.planner.planners.EmbeddingShardingPlanner

| API名称                                                      | 是否支持 | 限制与说明               |
| ------------------------------------------------------------ | -------- | ------------------------ |
| torchrec.distributed.planner.planners.EmbeddingShardingPlanner | 是       | 仅支持FUSED+ROW_WISE场景 |
| EmbeddingShardingPlanner.collective_plan                     | 是       | 仅支持FUSED+ROW_WISE场景 |
| EmbeddingShardingPlanner.plan                                | 是       | 仅支持FUSED+ROW_WISE场景 |

#### torchrec.distributed.planner.enumerators.EmbeddingEnumerator

| API名称                                                      | 是否支持 | 限制与说明 |
| ------------------------------------------------------------ | -------- | ---------- |
| torchrec.distributed.planner.enumerators.EmbeddingEnumerator | 是       | -          |
| EmbeddingEnumerator.enumerate                                | 是       | -          |
| EmbeddingEnumerator.populate_estimates                       | 是       | -          |

#### torchrec.distributed.planner.partitioners.GreedyPerfPartitioner

| API名称                                                      | 是否支持 | 限制与说明 |
| ------------------------------------------------------------ | -------- | ---------- |
| torchrec.distributed.planner.partitioners.GreedyPerfPartitioner | 是       | -          |
| GreedyPerfPartitioner.partition                              | 是       | -          |

#### torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation

| API名称                                                      | 是否支持 | 限制与说明 |
| ------------------------------------------------------------ | -------- | ---------- |
| torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation | 是       | -          |
| HeuristicalStorageReservation.reserve                        | 是       | -          |

#### torchrec.distributed.planner.proposers.GreedyProposer

| API名称                                               | 是否支持 | 限制与说明 |
| ----------------------------------------------------- | -------- | ---------- |
| torchrec.distributed.planner.proposers.GreedyProposer | 是       | -          |
| GreedyProposer.feedback                               | 是       | -          |
| GreedyProposer.load                                   | 是       | -          |
| GreedyProposer.propose                                | 是       | -          |

#### torchrec.distributed.planner.shard_estimators.EmbeddingPerfEstimator

| API名称                                                      | 是否支持 | 限制与说明 |
| ------------------------------------------------------------ | -------- | ---------- |
| torchrec.distributed.planner.shard_estimators.EmbeddingPerfEstimator | 是       | -          |
| EmbeddingPerfEstimator.perf_func_emb_wall_time               | 是       | -          |
| EmbeddingPerfEstimator.estimate                              | 是       | -          |

#### torchrec.distributed.planner.shard_estimators.EmbeddingStorageEstimator

| API名称                                                      | 是否支持 | 限制与说明 |
| ------------------------------------------------------------ | -------- | ---------- |
| torchrec.distributed.planner.shard_estimators.EmbeddingStorageEstimator | 是       | -          |
| EmbeddingStorageEstimator.estimate                           | 是       | -          |

#### torchrec.distributed.model_parallel.DistributedModelParallel

| API名称                                                      | 是否支持 | 限制与说明               |
| ------------------------------------------------------------ | -------- | ------------------------ |
| torchrec.distributed.model_parallel.DistributedModelParallel | 是       | 仅支持FUSED+ROW_WISE场景 |
| DistributedModelParallel.copy                                | 否       | -                        |
| DistributedModelParallel.forward                             | 是       | -                        |
| DistributedModelParallel.init_data_parallel                  | 是       | -                        |
| DistributedModelParallel.load_state_dict                     | 是       | -                        |
| DistributedModelParallel.module                              | 是       | -                        |
| DistributedModelParallel.named_buffers                       | 是       | -                        |
| DistributedModelParallel.named_parameters                    | 是       | -                        |
| DistributedModelParallel.state_dict                          | 是       | -                        |

#### torchrec.inference.modules.quantize_inference_model

| API名称                                             | 是否支持 | 限制与说明 |
| --------------------------------------------------- | -------- | ---------- |
| torchrec.inference.modules.quantize_inference_model | 否       | -          |

#### torchrec.inference.modules.shard_quant_model

| API名称                                      | 是否支持 | 限制与说明 |
| -------------------------------------------- | -------- | ---------- |
| torchrec.inference.modules.shard_quant_model | 否       | -          |

## DynamicEmb原生API支持度

### DynamicEmb v25.09版本

#### 概述

介绍DynamicEmb v25.09版本原生API接口在昇腾NPU上的支持情况与限制说明，DynamicEmb v25.09版本原生API接口具体使用方法请参考[DynamicEmb社区文档](https://github.com/NVIDIA/recsys-examples/blob/v25.09/corelib/dynamicemb/DynamicEmb_APIs.md)。原生API接口在昇腾NPU上的支持情况与限制说明可分为如下三类：

- API“是否支持“为“是“，“限制与说明“为“-“，说明此API和原生API支持度保持一致。
- API“是否支持“为“是“，“限制与说明“不为“-“，说明此API和原生API支持度不一致，请注意昇腾NPU上的支持度。
- API“是否支持“为“否“，“限制与说明“为“-“，说明在昇腾NPU上暂不支持此API。

#### DynamicEmbParameterConstraints

| API名称                                           | 是否支持 | 限制与说明                                          |
| ------------------------------------------------- | -------- | --------------------------------------------------- |
| DynamicEmbParameterConstraints                    | 是       | caching、score_strategy、external_storage场景未支持 |
| DynamicEmbParameterConstraints.use_dynamicemb     | 是       | -                                                   |
| DynamicEmbParameterConstraints.dynamicemb_options | 是       | caching、score_strategy、external_storage场景未支持 |

#### DynamicEmbeddingEnumerator

| API名称                              | 是否支持 | 限制与说明 |
| ------------------------------------ | -------- | ---------- |
| DynamicEmbeddingEnumerator           | 是       | -          |
| DynamicEmbeddingEnumerator.enumerate | 是       | -          |

#### DynamicEmbeddingShardingPlanner

| API名称                                         | 是否支持 | 限制与说明 |
| ----------------------------------------------- | -------- | ---------- |
| DynamicEmbeddingShardingPlanner                 | 是       | -          |
| DynamicEmbeddingShardingPlanner.collective_plan | 是       | -          |

#### DynamicEmbeddingCollectionSharder

| API名称                                 | 是否支持 | 限制与说明 |
| --------------------------------------- | -------- | ---------- |
| DynamicEmbeddingCollectionSharder       | 是       | -          |
| DynamicEmbeddingCollectionSharder.shard | 是       | -          |

#### DynamicEmbCheckMode

| API名称                     | 是否支持 | 限制与说明 |
| --------------------------- | -------- | ---------- |
| DynamicEmbCheckMode         | 是       | -          |
| DynamicEmbCheckMode.ERROR   | 是       | -          |
| DynamicEmbCheckMode.WARNING | 是       | -          |
| DynamicEmbCheckMode.IGNORE  | 是       | -          |

#### DynamicEmbInitializerMode

| API名称                                    | 是否支持 | 限制与说明 |
| ------------------------------------------ | -------- | ---------- |
| DynamicEmbInitializerMode                  | 是       | -          |
| DynamicEmbInitializerMode.NORMAL           | 是       | -          |
| DynamicEmbInitializerMode.TRUNCATED_NORMAL | 是       | -          |
| DynamicEmbInitializerMode.UNIFORM          | 是       | -          |
| DynamicEmbInitializerMode.CONSTANT         | 是       | -          |
| DynamicEmbInitializerMode.DEBUG            | 是       | -          |

#### DynamicEmbInitializerArgs

| API名称                           | 是否支持 | 限制与说明 |
| --------------------------------- | -------- | ---------- |
| DynamicEmbInitializerArgs         | 是       | -          |
| DynamicEmbInitializerArgs.mode    | 是       | -          |
| DynamicEmbInitializerArgs.mean    | 是       | -          |
| DynamicEmbInitializerArgs.std_dev | 是       | -          |
| DynamicEmbInitializerArgs.lower   | 是       | -          |
| DynamicEmbInitializerArgs.upper   | 是       | -          |
| DynamicEmbInitializerArgs.value   | 是       | -          |

#### DynamicEmbScoreStrategy

| API名称                            | 是否支持 | 限制与说明 |
| ---------------------------------- | -------- | ---------- |
| DynamicEmbScoreStrategy            | 是       | -          |
| DynamicEmbScoreStrategy.TIMESTAMP  | 是       | -          |
| DynamicEmbScoreStrategy.STEP       | 是       | -          |
| DynamicEmbScoreStrategy.CUSTOMIZED | 是       | -          |
| DynamicEmbScoreStrategy.LFU        | 是       | -          |

#### DynamicEmbTableOptions

| API名称                                      | 是否支持 | 限制与说明 |
| -------------------------------------------- | -------- | ---------- |
| DynamicEmbTableOptions                       | 是       | -          |
| DynamicEmbTableOptions.training              | 是       | -          |
| DynamicEmbTableOptions.initializer_args      | 是       | -          |
| DynamicEmbTableOptions.eval_initializer_args | 是       | -          |
| DynamicEmbTableOptions.caching               | 否       | -          |
| DynamicEmbTableOptions.init_capacity         | 是       | -          |
| DynamicEmbTableOptions.max_load_factor       | 是       | -          |
| DynamicEmbTableOptions.score_strategy        | 否       | -          |
| DynamicEmbTableOptions.bucket_capacity       | 是       | -          |
| DynamicEmbTableOptions.safe_check_mode       | 是       | -          |
| DynamicEmbTableOptions.global_hbm_for_values | 是       | -          |
| DynamicEmbTableOptions.external_storage      | 否       | -          |
| DynamicEmbTableOptions.index_type            | 是       | -          |

#### DynamicEmbDump

| API名称                    | 是否支持 | 限制与说明 |
| -------------------------- | -------- | ---------- |
| DynamicEmbDump             | 是       | -          |
| DynamicEmbDump.path        | 是       | -          |
| DynamicEmbDump.model       | 是       | -          |
| DynamicEmbDump.table_names | 是       | -          |
| DynamicEmbDump.optim       | 是       | -          |
| DynamicEmbDump.pg          | 是       | -          |

#### DynamicEmbLoad

| API名称                    | 是否支持 | 限制与说明 |
| -------------------------- | -------- | ---------- |
| DynamicEmbLoad             | 是       | -          |
| DynamicEmbLoad.path        | 是       | -          |
| DynamicEmbLoad.model       | 是       | -          |
| DynamicEmbLoad.table_names | 是       | -          |
| DynamicEmbLoad.optim       | 是       | -          |
| DynamicEmbLoad.pg          | 是       | -          |

#### incremental_dump

| API名称          | 是否支持 | 限制与说明 |
| ---------------- | -------- | ---------- |
| incremental_dump | 否       | -          |

#### get_score

| API名称   | 是否支持 | 限制与说明 |
| --------- | -------- | ---------- |
| get_score | 是       | -          |

#### set_score

| API名称   | 是否支持 | 限制与说明 |
| --------- | -------- | ---------- |
| set_score | 是       | -          |
