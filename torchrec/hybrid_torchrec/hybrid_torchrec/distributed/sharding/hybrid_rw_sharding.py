#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, TypeVar, Tuple

import torch
import torch.distributed as dist
from hybrid_torchrec.modules.ids_process import HashMapBase

from hybrid_torchrec.distributed.embedding_lookup import (
    HybridGroupedPooledEmbeddingsLookup,
)
from hybrid_torchrec.distributed.sharding.post_input_dist import (
    SparseFeaturesPostDist,
    EMPTY_POST_INPUT_DIST,
    UniqueHashFeatureProcess,
    get_feature_len_groupby_table_name,
)

from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingLookup,
    BaseSparseFeaturesDist,
    EmbeddingShardingInfo,
)
from torchrec.distributed.embedding_sharding import (
    _fx_wrap_batch_size_per_feature,
    _fx_wrap_max_B,
    _fx_wrap_tensor_to_device_dtype,
    _fx_wrap_gen_list_n_times,
    _fx_wrap_stride,
    _fx_wrap_stride_per_key_per_rank,
)
from torchrec.distributed.embedding_types import (
    BaseGroupedFeatureProcessor,
)
from torchrec.distributed.sharding.rw_sharding import (
    RwPooledEmbeddingSharding,
    RwSparseFeaturesDist,
)
from torchrec.distributed.types import Awaitable
from torchrec.distributed.types import (
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingType,
)
from torchrec.fx.utils import assert_fx_safe
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable


C = TypeVar("C", bound=Multistreamable)
F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")
W = TypeVar("W")


class HybridRwPooledEmbeddingSharding(RwPooledEmbeddingSharding):

    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        host_env: ShardingEnv,
        device: Optional[torch.device] = None,
        need_pos: bool = False,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        self._host_pg = host_env.process_group
        super().__init__(sharding_infos, env, device, need_pos, qcomm_codecs_registry)

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        num_features = self._get_num_features()
        feature_hash_sizes = self._get_feature_hash_sizes()
        return RwSparseFeaturesDist(
            pg=self._host_pg,
            num_features=num_features,
            feature_hash_sizes=feature_hash_sizes,
            device=device if device is not None else self._device,
            is_sequence=False,
            has_feature_processor=self._has_feature_processor,
            need_pos=self._need_pos,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        return HybridGroupedPooledEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs,
            pg=self._pg,
            device=device if device is not None else self._device,
            feature_processor=feature_processor,
            sharding_type=ShardingType.ROW_WISE,
        )

    def create_post_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        return EMPTY_POST_INPUT_DIST


class HybridHashRwPooledEmbeddingSharding(HybridRwPooledEmbeddingSharding):
    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        table2hashmap: Dict[str, HashMapBase],
        env: ShardingEnv,
        host_env: ShardingEnv,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(sharding_infos, env, host_env, device, qcomm_codecs_registry)
        self.table2hashmap = table2hashmap

    def create_post_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:

        table_names, features_split_by_table_name = get_feature_len_groupby_table_name(
            self._grouped_embedding_configs
        )
        hashmaps = [self.table2hashmap[n] for n in table_names]
        feature_processor = UniqueHashFeatureProcess(
            table_names, features_split_by_table_name, hashmaps
        )
        return SparseFeaturesPostDist(feature_processor)
