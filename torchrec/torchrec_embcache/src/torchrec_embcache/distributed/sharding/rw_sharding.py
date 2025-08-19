#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from hybrid_torchrec.distributed.embedding_lookup import (
    HybridGroupedPooledEmbeddingsLookup,
)
from hybrid_torchrec.modules.ids_process import HashMapBase
from hybrid_torchrec.distributed.sharding.post_input_dist import (
    SparseFeaturesPostDist,
    UniqueHashFeatureProcess,
    get_feature_len_groupby_table_name,
)
from hybrid_torchrec.distributed.sharding.hybrid_rw_sharding import (
    bucketize_kjt_before_all2all,
)

from torchrec.distributed.types import (
    Awaitable,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingType,
)
from torchrec.distributed.sharding.rw_sharding import (
    RwPooledEmbeddingSharding,
    RwSparseFeaturesDist,
)
from torchrec.distributed.embedding_sharding import (
    EmbeddingShardingInfo,
    BaseSparseFeaturesDist,
    BaseEmbeddingLookup,
)
from torchrec.distributed.embedding_types import (
    BaseGroupedFeatureProcessor,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class EmbCacheInputDistThreadPoolExecutorSingleton:
    _instance: Optional["EmbCacheInputDistThreadPoolExecutorSingleton"] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(
                EmbCacheInputDistThreadPoolExecutorSingleton, cls
            ).__new__(cls, *args, **kwargs)
            cls.executor = ThreadPoolExecutor(2)
        return cls._instance


class EmbCacheRwSparseFeaturesDistAwaitable(Awaitable):
    def __init__(
        self, function, module, sparse_feature: KeyedJaggedTensor, context
    ) -> None:
        super().__init__()
        self.future = EmbCacheInputDistThreadPoolExecutorSingleton().executor.submit(
            function, sparse_feature
        )
        self.pg = module.pg
        self._context = context

    def _wait_impl(self) -> Any:
        result, unbucketize_permute_tensor = self.future.result()
        if self._context is not None:
            self._context.unbucketize_permute_tensor = unbucketize_permute_tensor
        return result


class EmbCacheRwSparseFeaturesDist(RwSparseFeaturesDist):
    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_features: int,
        feature_hash_sizes: List[int],
        feature_total_num_buckets: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        is_sequence: bool = False,
        has_feature_processor: bool = False,
        need_pos: bool = False,
        keep_original_indices: bool = False,
        enable_admit: bool = False,
        is_ec: bool = False,
    ) -> None:
        super().__init__(
            pg,
            num_features,
            feature_hash_sizes,
            feature_total_num_buckets,
            device,
            is_sequence,
            has_feature_processor,
            need_pos,
            keep_original_indices,
        )
        self.pg = pg

        # local unique只可用于EC(Embedding Collection / Sequence Embedding)
        yes_str = ("true", "1", "yes")
        self._do_unique = os.environ.get("DO_EC_LOCAL_UNIQUE", "False").lower() in yes_str and is_ec

        self._enable_admit = enable_admit

    def _forward_func(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        (
            bucketized_features,
            unbucketize_permute_tensor,
        ) = bucketize_kjt_before_all2all(
            sparse_features,
            num_buckets=self._world_size,
            block_sizes=self._feature_block_sizes_tensor,
            output_permute=self._is_sequence,
            bucketize_pos=(
                self._has_feature_processor
                if sparse_features.weights_or_none() is None
                else self._need_pos
            ),
            keep_original_indices=self._keep_original_indices,
            do_unique=self._do_unique,
            enable_admit=self._enable_admit,
        )
        result = self._dist(bucketized_features)
        return result, unbucketize_permute_tensor

    def forward(
        self, sparse_features: KeyedJaggedTensor, context=None
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        return EmbCacheRwSparseFeaturesDistAwaitable(
            self._forward_func, self, sparse_features, context
        )


class EmbCacheRwPooledEmbeddingSharding(RwPooledEmbeddingSharding):
    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        table2hashmap: Dict[str, HashMapBase],
        cpu_env: ShardingEnv,
        cpu_device: torch.device,
        npu_device: torch.device,
        npu_env: ShardingEnv,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
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

        table_names, features_split_by_table_name = get_feature_len_groupby_table_name(
            self._grouped_embedding_configs
        )
        hashmaps = [self.table2hashmap[n] for n in table_names]
        feature_processor = UniqueHashFeatureProcess(
            table_names, features_split_by_table_name, hashmaps
        )
        return SparseFeaturesPostDist(feature_processor)
