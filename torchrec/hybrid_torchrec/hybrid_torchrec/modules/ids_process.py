#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import logging
from typing import List
from dataclasses import dataclass, asdict

import torch
from torch.autograd.profiler import record_function

torch.ops.load_library(os.path.join(os.path.dirname(__file__), "libhybrid_cpp.so"))


@dataclass
class BucketParams:
    lengths: torch.Tensor
    indices: torch.Tensor
    bucketize_pos: bool
    sequence: bool
    block_sizes: torch.Tensor
    bucket_size: int
    total_num_blocks: torch.Tensor = None
    weights: torch.Tensor = None
    batch_size_per_feature: torch.Tensor = None
    max_b: int = 0
    block_bucketize_pos: torch.Tensor = None
    return_bucket_mapping: bool = False
    keep_orig_idx: bool = False
    do_unique: bool = False
    return_count: bool = False


class HashMapBase(torch.nn.Module):
    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor]:
        pass


class IdsMapper(HashMapBase):
    """
    This class is primarily used for managing global ids.
    Its core functionality is to convert global ids into indices,
    which represent offsets in the embedding table stored on the NPU device.
    Additionally, it provides functionality for evicting ids,
    exporting all ids along with their corresponding indices,
    retrieving timestamps for the ids, and returning the current count
    of ids stored in the IdsMapper.
    """

    def __init__(self, embedding_num, only_device_memory: bool = True):
        super().__init__()
        self.ids_mapper = torch.classes.hybrid.IdsMapper(embedding_num, only_device_memory)
        self.embedding_num = embedding_num
        self._cache_mgr = None
    
    def set_cache_mgr(self, cache_mgr):
        self._cache_mgr = cache_mgr

    def statistic_key_count(self, ids: torch.Tensor, offset: torch.Tensor, counts: torch.Tensor, table_i: int):
        # cache manager内会判断表是否开启准入，开启时才记录count数据
        self._cache_mgr.statistics_key_count(
            ids, offset, counts, table_i
        )

    def forward(self, ids: torch.Tensor):
        with record_function("## ids2indices ##"):
            result, unique, unique_inverse = self.ids_mapper.ids2indices_unique(
                ids
            )
            return result, unique, unique_inverse

    def ids2indices_unique_out(
        self,
        ids: torch.Tensor,
        hash_indices: torch.Tensor,
        offset: torch.Tensor,
        unique: torch.Tensor,
        unique_ids: torch.Tensor,
        unique_inverse: torch.Tensor,
        unique_offset: List[int],
        table_id: int,
    ):
        self.ids_mapper.ids2indices_unique_out(
            ids, hash_indices, offset, unique, unique_ids, unique_inverse, unique_offset, table_id
        )


def block_bucketize_sparse_features_cpu(bucket_params: BucketParams):
    return torch.ops.hybrid.block_bucketize_sparse_features_cpu(bucket_params.lengths,
                                                                bucket_params.indices,
                                                                bucket_params.bucketize_pos,
                                                                bucket_params.sequence,
                                                                bucket_params.block_sizes,
                                                                bucket_params.bucket_size,
                                                                bucket_params.total_num_blocks,
                                                                bucket_params.weights,
                                                                bucket_params.batch_size_per_feature,
                                                                bucket_params.max_b,
                                                                bucket_params.block_bucketize_pos,
                                                                bucket_params.return_bucket_mapping,
                                                                bucket_params.keep_orig_idx,
                                                                bucket_params.do_unique,
                                                                bucket_params.return_count)
