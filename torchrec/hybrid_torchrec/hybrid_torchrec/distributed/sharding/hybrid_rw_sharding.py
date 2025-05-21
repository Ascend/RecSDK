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
from concurrent.futures import ThreadPoolExecutor

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
from hybrid_torchrec.modules.ids_process import (
    block_bucketize_sparse_features_cpu,
    BucketParams,
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
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable
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

DEFAULT_POST_INPUT_THREADS = 6


class InputDistThreadPoolExecutorSingleton:
    _instance: "InputDistThreadPoolExecutorSingleton" = None

    def __new__(cls, *args, **kwargs):
        if cls._instance:
            return cls._instance
        cls._instance = super(InputDistThreadPoolExecutorSingleton, cls).__new__(
            cls, *args, **kwargs
        )
        max_threads = DEFAULT_POST_INPUT_THREADS
        input_dist_threads_str = os.environ["INPUT_DIST_THREADS"]
        if input_dist_threads_str is not None:
            try:
                max_threads = int(input_dist_threads_str)
            except ValueError as e:
                raise Exception("Environment variable INPUT_DIST_THREADS is not a valid integer.") from e
        cls.executor = ThreadPoolExecutor(max_threads)
        return cls._instance

C = TypeVar("C", bound=Multistreamable)
F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")
W = TypeVar("W")


def bucketize_kjt_before_all2all(
    kjt: KeyedJaggedTensor,
    num_buckets: int,
    block_sizes: torch.Tensor,
    output_permute: bool = False,
    bucketize_pos: bool = False,
    block_bucketize_row_pos: Optional[List[torch.Tensor]] = None,
    keep_original_indices: bool = False,
) -> Tuple[KeyedJaggedTensor, Optional[torch.Tensor]]:
    num_features = len(kjt.keys())
    assert_fx_safe(
        block_sizes.numel() == num_features,
        f"Expecting block sizes for {num_features} features, but {block_sizes.numel()} received.",
    )
    block_sizes_new_type = _fx_wrap_tensor_to_device_dtype(block_sizes, kjt.values())
    bucket_params = BucketParams( 
        kjt.lengths().view(-1),
        kjt.values(),
        bucketize_pos=bucketize_pos,
        sequence=output_permute,
        block_sizes=block_sizes_new_type,
        bucket_size=num_buckets,
        weights=kjt.weights_or_none(),
        batch_size_per_feature=_fx_wrap_batch_size_per_feature(kjt),
        max_b=_fx_wrap_max_B(kjt),
        block_bucketize_pos=block_bucketize_row_pos,  # each tensor should have the same dtype as kjt.lengths()
        keep_orig_idx=keep_original_indices,)
    (
        bucketized_lengths,
        bucketized_indices,
        bucketized_weights,
        pos,
        unbucketize_permute,
        _,
    ) = block_bucketize_sparse_features_cpu(bucket_params)
    return (
        KeyedJaggedTensor(
            # duplicate keys will be resolved by AllToAll
            keys=_fx_wrap_gen_list_n_times(kjt.keys(), num_buckets),
            values=bucketized_indices,
            weights=pos if bucketize_pos else bucketized_weights,
            lengths=bucketized_lengths.view(-1),
            offsets=None,
            stride=_fx_wrap_stride(kjt),
            stride_per_key_per_rank=_fx_wrap_stride_per_key_per_rank(kjt, num_buckets),
            length_per_key=None,
            offset_per_key=None,
            index_per_key=None,
        ),
        unbucketize_permute,
    )


class HashRwSparseFeaturesDistAwaitable(Awaitable):
    def __init__(self, function, module, sparse_feature: KeyedJaggedTensor) -> None:
        super().__init__()
        self.future = InputDistThreadPoolExecutorSingleton().executor.submit(
            function, sparse_feature
        )
        self.pg = module.pg

    def _wait_impl(self) -> Any:
        return self.future.result()


class HashRwSparseFeaturesDist(RwSparseFeaturesDist):
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

    def forward_function(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        (
            bucketized_features,
            self.unbucketize_permute_tensor,
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
        )
        result = self._dist(bucketized_features)
        if isinstance(result, Awaitable):
            result = result.wait()
        return result

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        return HashRwSparseFeaturesDistAwaitable(
            self.forward_function, self, sparse_features
        )


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

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        num_features = self._get_num_features()
        feature_hash_sizes = self._get_feature_hash_sizes()
        return HashRwSparseFeaturesDist(
            pg=self._host_pg,
            num_features=num_features,
            feature_hash_sizes=feature_hash_sizes,
            device="cpu",
            is_sequence=False,
            has_feature_processor=self._has_feature_processor,
            need_pos=self._need_pos,
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
