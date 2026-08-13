# TorchRec_npu API

## TorchRec原生API支持度

### 概述

介绍TorchRec原生API接口在昇腾NPU上的支持情况与限制说明，TorchRec原生API接口具体使用方法请参考[TorchRec社区文档](https://meta-pytorch.org/torchrec/api.html)。原生API接口在昇腾NPU上的支持情况与限制说明如下：

- API“是否支持”为“是”、“限制与说明”为“-”，说明此API和原生API支持度保持一致。
- API“是否支持”为“是”、“限制与说明”不为“-”时，若内容为版本说明（如“某版本新增/移除”），则此API与原生API支持度保持一致，仅补充该API的引入版本信息；否则说明此API和原生API支持度不一致，请注意昇腾NPU上的支持度。
- API“是否支持”为“否”、“限制与说明”为“-”，说明在昇腾NPU上暂不支持此API。
- 部分API为特定TorchRec版本新增API，“限制与说明”中会标注引入该API的TorchRec版本，未标注版本的API在v1.2.0及以上版本均支持。

### Data Types

#### torchrec.sparse.jagged_tensor.JaggedTensor

| API名称                                    | 是否支持 | 限制与说明        |
| ------------------------------------------ | -------- | ----------------- |
| torchrec.sparse.jagged_tensor.JaggedTensor | 是       | -                 |
| JaggedTensor.copy_                         | 是       | v1.6.0版本新增    |
| JaggedTensor.device                        | 是       | -                 |
| JaggedTensor.empty                         | 是       | -                 |
| JaggedTensor.empty_like                    | 是       | v1.6.0版本新增    |
| JaggedTensor.from_dense                    | 是       | -                 |
| JaggedTensor.from_dense_lengths            | 是       | -                 |
| JaggedTensor.lengths                       | 是       | -                 |
| JaggedTensor.lengths_or_none               | 是       | -                 |
| JaggedTensor.offsets                       | 是       | -                 |
| JaggedTensor.offsets_or_none               | 是       | -                 |
| JaggedTensor.record_stream                 | 是       | -                 |
| JaggedTensor.size_in_bytes                 | 是       | v1.6.0版本新增    |
| JaggedTensor.to                            | 是       | -                 |
| JaggedTensor.to_dense                      | 是       | -                 |
| JaggedTensor.to_dense_weights              | 是       | -                 |
| JaggedTensor.to_padded_dense               | 是       | -                 |
| JaggedTensor.to_padded_dense_weights       | 是       | -                 |
| JaggedTensor.values                        | 是       | -                 |
| JaggedTensor.weights                       | 是       | -                 |
| JaggedTensor.weights_or_none               | 是       | -                 |

#### torchrec.sparse.jagged_tensor.KeyedJaggedTensor

| API名称                                         | 是否支持 | 限制与说明        |
| ----------------------------------------------- | -------- | ----------------- |
| torchrec.sparse.jagged_tensor.KeyedJaggedTensor | 是       | -                 |
| KeyedJaggedTensor.clear_storage                 | 是       | v1.7.0版本新增    |
| KeyedJaggedTensor.concat                        | 是       | -                 |
| KeyedJaggedTensor.copy_                         | 是       | v1.5.0版本新增    |
| KeyedJaggedTensor.device                        | 是       | -                 |
| KeyedJaggedTensor.empty                         | 是       | -                 |
| KeyedJaggedTensor.empty_like                    | 是       | -                 |
| KeyedJaggedTensor.from_jt_dict                  | 是       | -                 |
| KeyedJaggedTensor.from_lengths_sync             | 是       | -                 |
| KeyedJaggedTensor.from_offsets_sync             | 是       | -                 |
| KeyedJaggedTensor.index_per_key                 | 是       | -                 |
| KeyedJaggedTensor.inverse_indices               | 是       | -                 |
| KeyedJaggedTensor.inverse_indices_or_none       | 是       | -                 |
| KeyedJaggedTensor.keys                          | 是       | -                 |
| KeyedJaggedTensor.length_per_key                | 是       | -                 |
| KeyedJaggedTensor.length_per_key_or_none        | 是       | -                 |
| KeyedJaggedTensor.lengths                       | 是       | -                 |
| KeyedJaggedTensor.lengths_offset_per_key        | 是       | -                 |
| KeyedJaggedTensor.lengths_or_none               | 是       | -                 |
| KeyedJaggedTensor.offset_per_key                | 是       | -                 |
| KeyedJaggedTensor.offset_per_key_or_none        | 是       | -                 |
| KeyedJaggedTensor.offsets                       | 是       | -                 |
| KeyedJaggedTensor.offsets_or_none               | 是       | -                 |
| KeyedJaggedTensor.permute                       | 是       | -                 |
| KeyedJaggedTensor.record_stream                 | 是       | -                 |
| KeyedJaggedTensor.size_in_bytes                 | 是       | v1.6.0版本新增    |
| KeyedJaggedTensor.split                         | 是       | -                 |
| KeyedJaggedTensor.stride                        | 是       | -                 |
| KeyedJaggedTensor.stride_per_key                | 是       | -                 |
| KeyedJaggedTensor.stride_per_key_per_rank       | 是       | -                 |
| KeyedJaggedTensor.sync                          | 是       | -                 |
| KeyedJaggedTensor.to                            | 是       | -                 |
| KeyedJaggedTensor.to_dict                       | 是       | -                 |
| KeyedJaggedTensor.unsync                        | 是       | -                 |
| KeyedJaggedTensor.values                        | 是       | -                 |
| KeyedJaggedTensor.variable_stride_per_key       | 是       | -                 |
| KeyedJaggedTensor.weights                       | 是       | -                 |
| KeyedJaggedTensor.weights_or_none               | 是       | -                 |

#### torchrec.sparse.jagged_tensor.KeyedTensor

| API名称                                   | 是否支持 | 限制与说明        |
| ----------------------------------------- | -------- | ----------------- |
| torchrec.sparse.jagged_tensor.KeyedTensor | 是       | -                 |
| KeyedTensor.device                        | 是       | -                 |
| KeyedTensor.from_tensor_list              | 是       | -                 |
| KeyedTensor.key_dim                       | 是       | -                 |
| KeyedTensor.keys                          | 是       | -                 |
| KeyedTensor.length_per_key                | 是       | -                 |
| KeyedTensor.offset_per_key                | 是       | -                 |
| KeyedTensor.record_stream                 | 是       | -                 |
| KeyedTensor.regroup                       | 是       | -                 |
| KeyedTensor.regroup_as_dict               | 是       | -                 |
| KeyedTensor.size_in_bytes                 | 是       | v1.6.0版本新增    |
| KeyedTensor.to                            | 是       | -                 |
| KeyedTensor.to_dict                       | 是       | -                 |
| KeyedTensor.values                        | 是       | -                 |

### Modules

#### torchrec.modules.embedding_configs.BaseEmbeddingConfig

| API名称                                                | 是否支持 | 限制与说明        |
| ------------------------------------------------------ | -------- | ----------------- |
| torchrec.modules.embedding_configs.BaseEmbeddingConfig | 是       | -                 |
| BaseEmbeddingConfig.num_embeddings                     | 是       | -                 |
| BaseEmbeddingConfig.embedding_dim                      | 是       | -                 |
| BaseEmbeddingConfig.name                               | 是       | -                 |
| BaseEmbeddingConfig.data_type                          | 是       | -                 |
| BaseEmbeddingConfig.feature_names                      | 是       | -                 |
| BaseEmbeddingConfig.weight_init_max                    | 是       | -                 |
| BaseEmbeddingConfig.weight_init_min                    | 是       | -                 |
| BaseEmbeddingConfig.num_embeddings_post_pruning        | 是       | -                 |
| BaseEmbeddingConfig.init_fn                            | 是       | -                 |
| BaseEmbeddingConfig.need_pos                           | 是       | -                 |
| BaseEmbeddingConfig.input_dim                          | 是       | -                 |
| BaseEmbeddingConfig.total_num_buckets                  | 是       | v1.5.0版本新增    |
| BaseEmbeddingConfig.use_virtual_table                  | 是       | v1.5.0版本新增    |
| BaseEmbeddingConfig.virtual_table_eviction_policy      | 是       | v1.5.0版本新增    |
| BaseEmbeddingConfig.enable_embedding_update            | 是       | v1.5.0版本新增    |
| BaseEmbeddingConfig.stash_weights                      | 是       | v1.6.0版本新增    |

#### torchrec.modules.embedding_configs.EmbeddingBagConfig

| API名称                                               | 是否支持 | 限制与说明              |
| ----------------------------------------------------- | -------- | ----------------------- |
| torchrec.modules.embedding_configs.EmbeddingBagConfig | 是       | 继承BaseEmbeddingConfig |
| EmbeddingBagConfig.pooling                            | 是       | -                       |

#### torchrec.modules.embedding_configs.EmbeddingConfig

| API名称                                            | 是否支持 | 限制与说明              |
| -------------------------------------------------- | -------- | ----------------------- |
| torchrec.modules.embedding_configs.EmbeddingConfig | 是       | 继承BaseEmbeddingConfig |

#### torchrec.modules.embedding_modules.EmbeddingBagCollection

| API名称                                                   | 是否支持 | 限制与说明 |
| --------------------------------------------------------- | -------- | ---------- |
| torchrec.modules.embedding_modules.EmbeddingBagCollection | 是       | -          |
| EmbeddingBagCollection.device                             | 是       | -          |
| EmbeddingBagCollection.embedding_bag_configs              | 是       | -          |
| EmbeddingBagCollection.forward                            | 是       | -          |
| EmbeddingBagCollection.is_weighted                        | 是       | -          |
| EmbeddingBagCollection.reset_parameters                   | 是       | -          |

#### torchrec.modules.embedding_modules.EmbeddingCollection

| API名称                                                | 是否支持 | 限制与说明        |
| ------------------------------------------------------ | -------- | ----------------- |
| torchrec.modules.embedding_modules.EmbeddingCollection | 是       | -                 |
| EmbeddingCollection.device                             | 是       | -                 |
| EmbeddingCollection.embedding_configs                  | 是       | -                 |
| EmbeddingCollection.embedding_dim                      | 是       | -                 |
| EmbeddingCollection.embedding_names_by_table           | 是       | -                 |
| EmbeddingCollection.forward                            | 是       | -                 |
| EmbeddingCollection.need_indices                       | 是       | -                 |
| EmbeddingCollection.reset_parameters                   | 是       | -                 |
| EmbeddingCollection.set_use_sorted_select              | 否       | -                 |
| EmbeddingCollection.use_gather_select                  | 是       | v1.5.0版本新增    |
| EmbeddingCollection.use_gather_select_per_sharding     | 否       | -                 |
| EmbeddingCollection.use_sorted_select                  | 否       | -                 |

### Planner

#### torchrec.distributed.types.ShardingPlan

| API名称                                 | 是否支持 | 限制与说明 |
| --------------------------------------- | -------- | ---------- |
| torchrec.distributed.types.ShardingPlan | 是       | -          |
| ShardingPlan.plan                       | 是       | -          |
| ShardingPlan.get_plan_for_module        | 是       | -          |

#### torchrec.distributed.planner.planners.EmbeddingShardingPlanner

| API名称                                                      | 是否支持 | 限制与说明               |
| ------------------------------------------------------------ | -------- | ------------------------ |
| torchrec.distributed.planner.planners.EmbeddingShardingPlanner | 是       | 仅支持FUSED+ROW_WISE场景 |
| EmbeddingShardingPlanner.collective_plan                     | 是       | 仅支持FUSED+ROW_WISE场景 |
| EmbeddingShardingPlanner.get_search_space                    | 否       | -                        |
| EmbeddingShardingPlanner.get_selected_options                | 否       | -                        |
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

| API名称                                                      | 是否支持 | 限制与说明        |
| ------------------------------------------------------------ | -------- | ----------------- |
| torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation | 是       | -                 |
| HeuristicalStorageReservation.last_reserved_topology         | 是       | v1.5.0版本新增    |
| HeuristicalStorageReservation.reserve                        | 是       | -                 |

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
| EmbeddingPerfEstimator.perf_func_emb_wall_time               | 是       | v1.2.0/v1.5.0支持，v1.6.0起移除|
| EmbeddingPerfEstimator.estimate                              | 是       | -          |

#### torchrec.distributed.planner.shard_estimators.EmbeddingStorageEstimator

| API名称                                                      | 是否支持 | 限制与说明 |
| ------------------------------------------------------------ | -------- | ---------- |
| torchrec.distributed.planner.shard_estimators.EmbeddingStorageEstimator | 是       | -          |
| EmbeddingStorageEstimator.estimate                           | 是       | -          |

### Model Parallel

#### torchrec.distributed.model_parallel.DistributedModelParallel

| API名称                                                      | 是否支持 | 限制与说明               |
| ------------------------------------------------------------ | -------- | ------------------------ |
| torchrec.distributed.model_parallel.DistributedModelParallel | 是       | 仅支持FUSED+ROW_WISE场景 |
| DistributedModelParallel.copy                                | 否       | -                        |
| DistributedModelParallel.forward                             | 是       | -                        |
| DistributedModelParallel.get_delta_tracker                   | 否       | v1.5.0版本新增           |
| DistributedModelParallel.init_data_parallel                  | 是       | -                        |
| DistributedModelParallel.init_torchrec_delta_tracker         | 否       | v1.5.0版本新增           |
| DistributedModelParallel.load_state_dict                     | 是       | -                        |
| DistributedModelParallel.module                              | 是       | -                        |
| DistributedModelParallel.named_buffers                       | 是       | -                        |
| DistributedModelParallel.named_parameters                    | 是       | -                        |
| DistributedModelParallel.reshard                             | 否       | v1.5.0版本新增           |
| DistributedModelParallel.state_dict                          | 是       | -                        |
| DistributedModelParallel.write                               | 否       | v1.6.0版本新增           |

### Inference

#### torchrec.inference.modules.quantize_inference_model

| API名称                                             | 是否支持 | 限制与说明 |
| --------------------------------------------------- | -------- | ---------- |
| torchrec.inference.modules.quantize_inference_model | 是       | -          |

#### torchrec.inference.modules.shard_quant_model

| API名称                                      | 是否支持 | 限制与说明 |
| -------------------------------------------- | -------- | ---------- |
| torchrec.inference.modules.shard_quant_model | 否       | -          |
