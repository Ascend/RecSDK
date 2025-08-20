#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import torch

from hybrid_torchrec.modules.ids_process import HashMapBase
from hybrid_torchrec.distributed.embedding_lookup import HybridGroupedEmbeddingsLookup
from hybrid_torchrec.distributed.sharding.post_input_dist import (
    SparseFeaturesPostDist,
    UniqueHashFeatureProcess,
    get_feature_len_groupby_table_name,
)
from torchrec_embcache.distributed.sharding.rw_sharding import EmbCacheRwSparseFeaturesDist

from torchrec.distributed.types import QuantizedCommCodecs, ShardingEnv
from torchrec.distributed.sharding.rw_sequence_sharding import RwSequenceEmbeddingSharding
from torchrec.distributed.embedding_sharding import (
    EmbeddingShardingInfo,
    BaseSparseFeaturesDist,
    BaseEmbeddingLookup,
)
from torchrec.distributed.embedding_types import BaseGroupedFeatureProcessor
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class EmbCacheRwSequenceEmbeddingSharding(RwSequenceEmbeddingSharding):
    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        table2hashmap: Dict[str, HashMapBase],
        cpu_env: ShardingEnv,
        cpu_device: torch.device,
        npu_device: torch.device,
        npu_env: ShardingEnv,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        enable_admit: bool = False,
    ) -> None:
        super().__init__(
            sharding_infos=sharding_infos,
            env=npu_env,
            device=npu_device,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )
        self._cpu_device = cpu_device
        self._cpu_env = cpu_env
        self.table2hashmap = table2hashmap
        self._enable_admit = enable_admit

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        num_features = self._get_num_features()
        feature_hash_sizes = self._get_feature_hash_sizes()
        return EmbCacheRwSparseFeaturesDist(
            # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            pg=self._cpu_env.process_group,
            num_features=num_features,
            feature_hash_sizes=feature_hash_sizes,
            device=self._cpu_device,
            is_sequence=True,
            has_feature_processor=self._has_feature_processor,
            need_pos=self._need_pos,
            enable_admit=self._enable_admit,
            is_ec=True,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        return HybridGroupedEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs,
            pg=self._pg,
            device=device if device is not None else self._device,
        )

    def create_post_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        table_names, features_split_by_table_name = get_feature_len_groupby_table_name(
            self._grouped_embedding_configs
        )
        hashmaps = [self.table2hashmap[n] for n in table_names]
        feature_processor = UniqueHashFeatureProcess(
            table_names, features_split_by_table_name, hashmaps, self._enable_admit
        )
        return SparseFeaturesPostDist(feature_processor)
