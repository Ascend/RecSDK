#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Optional, TypeVar

import torch

from hybrid_torchrec.distributed.embedding_lookup import (
    HybridGroupedPooledEmbeddingsLookup,
)
from hybrid_torchrec.distributed.sharding.post_input_dist import (
    SparseFeaturesPostDist,
    EMPTY_POST_INPUT_DIST,
    UniqueHashFeatureProcess,
    get_feature_len_groupby_table_name,
)
from hybrid_torchrec.modules.hash_embeddingbag import HashMap
from torchrec.distributed.embedding_sharding import (
    BaseSparseFeaturesDist,
)
from torchrec.distributed.embedding_sharding import (
    EmbeddingShardingInfo,
)
from torchrec.distributed.sharding.tw_sharding import (
    TwPooledEmbeddingSharding,
    TwSparseFeaturesDist,
)
from torchrec.distributed.types import QuantizedCommCodecs, ShardingEnv
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable

C = TypeVar("C", bound=Multistreamable)
F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")
W = TypeVar("W")


class HybridTwPooledEmbeddingSharding(TwPooledEmbeddingSharding):
    """
    Shards embedding bags table-wise, which input dist with host computation and comunication
    """

    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        host_env: ShardingEnv,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        self._host_pg = host_env.process_group
        super().__init__(sharding_infos, env, device, qcomm_codecs_registry)

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        if self._pg is None:
            raise ValueError("Host pg is None")
        return TwSparseFeaturesDist(
            self._host_pg,
            self.features_per_rank(),
        )

    def create_lookup(self, device=None, fused_params=None, feature_processor=None):
        return HybridGroupedPooledEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs,
            pg=self._pg,
            device=device if device is not None else self._device,
            feature_processor=feature_processor,
        )

    def create_post_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        return EMPTY_POST_INPUT_DIST


class HybridHashTwPooledEmbeddingSharding(HybridTwPooledEmbeddingSharding):
    """
    Shards embedding bags table-wise, which input dist with host computation and comunication
    """

    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        table2hashmap: Dict[str, HashMap],
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
