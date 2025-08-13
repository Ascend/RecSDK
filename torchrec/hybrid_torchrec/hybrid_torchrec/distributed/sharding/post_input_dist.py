#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, TypeVar, Optional

import torch

from hybrid_torchrec.modules.ids_process import (
    HashMapBase,
)
from hybrid_torchrec.sparse import KeyedJaggedTensorWithCount
from hybrid_torchrec.sparse.jagged_tensor_with_looup_helper import (
    KeyedJaggedTensorWithLookHelper,
)
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.types import Awaitable, LazyAwaitable
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable

logger = logging.getLogger(__name__)

C = TypeVar("C", bound=Multistreamable)
F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")
W = TypeVar("W")

DEFAULT_POST_INPUT_THREADS = 6
MAX_POST_INPUT_THREADS = 12


class ThreadPoolExecutorSingleton:
    _instance: "ThreadPoolExecutorSingleton" = None

    def __new__(cls, *args, **kwargs):
        if cls._instance:
            return cls._instance
        cls._instance = super(ThreadPoolExecutorSingleton, cls).__new__(
            cls, *args, **kwargs
        )
        try:
            max_threads = int(os.environ.get("POST_INPUT_THREADS", DEFAULT_POST_INPUT_THREADS))
            if max_threads <= 0 or max_threads > MAX_POST_INPUT_THREADS:
                raise ValueError(f"POST_INPUT_THREADS expected in range [1, {MAX_POST_INPUT_THREADS}],"
                                    f"but got {max_threads}.") 
            cls.executor = ThreadPoolExecutor(max_threads)
            return cls._instance
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise Exception("Environment variable POST_INPUT_THREADS is not a valid integer.") from e
            raise


def get_feature_len_groupby_table_name(grouped_embedding_configs):
    table_names = []
    features_len_by_table_name = [0]
    for group_config in grouped_embedding_configs:
        for table_config in group_config.embedding_tables:
            table_names.append(table_config.name)
            features_len_by_table_name.append(table_config.num_features())
    features_len_by_table_name = features_len_by_table_name[1:]
    return table_names, features_len_by_table_name


class PostInputKJTListAwaitable(LazyAwaitable[KJTList]):
    def __init__(
        self, feature_awaitables: List[LazyAwaitable[KeyedJaggedTensor]]
    ) -> None:
        super().__init__()
        self._feature_awaitables = feature_awaitables

    def _wait_impl(self) -> KJTList:
        features_list = []
        for w in self._feature_awaitables:
            features_list.append(w.wait())
        return KJTList(features_list)


class EmptyKJTAwaitable(LazyAwaitable[KeyedJaggedTensor]):
    def __init__(self, kjt: KeyedJaggedTensor) -> None:
        super().__init__()
        self._kjt = kjt

    def _wait_impl(self) -> KeyedJaggedTensor:
        return self._kjt


class BasePostInputProcess(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> KeyedJaggedTensor:
        return EmptyKJTAwaitable(sparse_features)


class SparseFeaturesPostDist(torch.nn.Module):
    def __init__(
        self, feature_processor: Optional[BasePostInputProcess] = None
    ) -> None:
        super().__init__()
        self._dist = feature_processor

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        return self._dist(sparse_features)


EMPTY_POST_INPUT_DIST = SparseFeaturesPostDist(BasePostInputProcess())


def do_unique_hash(
    origin_kjt: KeyedJaggedTensor,
    feature_split_by_table: List[int],
    hashmap_list: List[HashMapBase],
):
    feature_lens = []
    hash_indices_all_table = []
    unique_indices, unique_inverse, unique_offset_list = [], [], []
    start = 0
    for ind, features_shard in enumerate(
        origin_kjt.split(segments=feature_split_by_table)
    ):
        hash_indices, unique, inverse = hashmap_list[ind](
            features_shard.values()
        )
        feature_lens.append(len(features_shard.keys()))
        unique_indices.append(unique)
        unique_inverse.append(inverse)
        unique_offset_list.extend([start] * len(features_shard.keys()))
        hash_indices_all_table.append(hash_indices)
        start += unique.numel()
    unique_offset_list.append(start)
    unique_offset = torch.LongTensor(unique_offset_list)
    if len(unique_indices) > 0:
        unique_indices = torch.cat(unique_indices).pin_memory()
        unique_inverse = torch.cat(unique_inverse).pin_memory()
        hash_indices = torch.concat(hash_indices_all_table).pin_memory()
    else:
        unique_indices = None
        unique_inverse = None
        hash_indices = None
    return KeyedJaggedTensorWithLookHelper(
        keys=origin_kjt.keys(),
        values=origin_kjt.values().pin_memory(),
        hash_indices=hash_indices,
        unique_indices=unique_indices,
        unique_offset=unique_offset.pin_memory(),
        unique_offset_host=unique_offset_list,
        unique_inverse=unique_inverse,
        lengths=origin_kjt.lengths().pin_memory(),
        offsets=origin_kjt.offsets().pin_memory(),
        length_per_key=origin_kjt.length_per_key(),
        offset_per_key=origin_kjt.offset_per_key(),
        stride=origin_kjt.stride(),
    )


def split_keys_offset(origin_kjt: KeyedJaggedTensor, feature_split_by_table: List[int]):
    offsets = origin_kjt.offset_per_key()
    result = [0 for _ in range(len(feature_split_by_table) + 1)]
    start = 0
    for ind, feat_num in enumerate(feature_split_by_table):
        end = start + feat_num
        result[ind + 1] = offsets[end]
        start = end
    return torch.LongTensor(result)


def _unwrap_ids_mapper(hm):
    if hasattr(hm, "ids2indices_unique_out") and not isinstance(hm, torch.nn.Module):
        return hm
    
    if hasattr(hm, "ids_mapper"):
        return hm.ids_mapper
    
    raise TypeError(f"Cannot extract IdsMapper from hashmap type: {type(hm)}")


def do_unique_hash_out(
    origin_kjt: KeyedJaggedTensor | KeyedJaggedTensorWithCount,
    feature_split_by_table: List[int],
    hashmap_list: List[HashMapBase],
    enable_admit: bool = False,
):
    num_of_table = len(feature_split_by_table)
    ids = origin_kjt.values()
    hash_indices = torch.empty_like(ids, pin_memory=True)
    offsets = split_keys_offset(origin_kjt, feature_split_by_table)
    unique = torch.empty_like(ids, pin_memory=True)
    unique_ids = torch.empty_like(ids)
    unique_inverse = torch.empty_like(ids, pin_memory=True)
    unique_offset = torch.zeros(num_of_table + 1, dtype=torch.long)

    # 存在表开启准入时记录feature count
    if enable_admit:
        # 开启local unique时，会传入KeyedJaggedTensorWithCount,使用其counts属性；
        # 不包含counts属性时使用空tensor, 统计时会用1作为count
        counts = origin_kjt.counts if hasattr(origin_kjt, "counts") else torch.empty((0,), dtype=torch.int64)
        for table_i in range(num_of_table):
            hashmap_list[table_i].statistic_key_count(
                ids, offsets, counts, table_i
            )

    use_parallel = os.getenv("ENABLE_PARALLEL_GLOBAL_UNIQUE", "0") == "1"

    start_tm = time.perf_counter()

    if use_parallel:
        native_mappers = [_unwrap_ids_mapper(h) for h in hashmap_list]

        torch.classes.hybrid.IdsMapper.parallel_ids2indices_unique_out(
            native_mappers,
            ids,
            hash_indices,
            offsets,
            unique,
            unique_ids,
            unique_inverse,
            unique_offset,
        )
        logger.debug(
            "[do_unique_hash_out] run parallel batch implementation. Time cost: %.1f ms",
            (time.perf_counter() - start_tm) * 1000
        )
    else:
        for table_i in range(num_of_table):
            hashmap_list[table_i].ids2indices_unique_out(
                ids,
                hash_indices,
                offsets,
                unique,
                unique_ids,
                unique_inverse,
                unique_offset,
                table_i,
            )
        logger.debug(
            "[do_unique_hash_out] run serial for-loop implementation. Time cost: %.1f ms",
            (time.perf_counter() - start_tm) * 1000
        )

    unique_offset_list_single = unique_offset.tolist()
    unique_offset_list = []
    for table_i in range(num_of_table):
        unique_offset_list.extend(
            [unique_offset_list_single[table_i]] * feature_split_by_table[table_i]
        )
    unique_offset_list.append(unique_offset_list_single[-1])
    unique_offset = torch.LongTensor(unique_offset_list)
    unique.resize_(unique_offset_list[-1])

    return KeyedJaggedTensorWithLookHelper(
        keys=origin_kjt.keys(),
        values=hash_indices.pin_memory(),
        hash_indices=hash_indices.pin_memory(),
        unique_indices=unique.pin_memory(),
        unique_offset=unique_offset.pin_memory(),
        unique_offset_host=unique_offset_list,
        unique_offset_list_single=unique_offset_list_single,
        unique_ids=unique_ids,
        unique_inverse=unique_inverse.pin_memory(),
        lengths=origin_kjt.lengths().pin_memory(),
        offsets=origin_kjt.offsets().pin_memory(),
        length_per_key=origin_kjt.length_per_key(),
        offset_per_key=origin_kjt.offset_per_key(),
        stride=origin_kjt.stride(),
    )


class UniqueHashKJTAwaitable(LazyAwaitable[KeyedJaggedTensorWithLookHelper]):
    def __init__(
        self,
        origin_kjt: KeyedJaggedTensor,
        feature_split_by_table: List[int],
        hash_list: List[HashMapBase],
        enable_admit: bool = False,
    ) -> None:
        super().__init__()
        self.future = ThreadPoolExecutorSingleton().executor.submit(
            do_unique_hash_out, origin_kjt, feature_split_by_table, hash_list, enable_admit
        )

    def _wait_impl(self) -> KeyedJaggedTensorWithLookHelper:
        return self.future.result()


class UniqueHashFeatureProcess(BasePostInputProcess):
    def __init__(
        self,
        table_names: List[str],
        feature_split_by_table: List[int],
        hashmap_list: List[HashMapBase],
        enable_admit: bool = False,
    ) -> None:
        super().__init__()
        self.hashmap_list = hashmap_list
        self.table_names = table_names
        self.feature_split_by_table = feature_split_by_table
        self.enable_admit = enable_admit

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[KeyedJaggedTensorWithLookHelper]:
        return UniqueHashKJTAwaitable(
            sparse_features, self.feature_split_by_table, self.hashmap_list, self.enable_admit
        )
