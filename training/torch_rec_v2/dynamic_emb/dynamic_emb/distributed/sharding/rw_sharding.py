#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import distributed as dist
from torchrec.distributed.dist_data import KJTAllToAll
from torchrec.distributed.embedding_sharding import BaseSparseFeaturesDist
from torchrec.distributed.types import Awaitable
from torchrec.fx.utils import assert_fx_safe
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from dynamic_emb_extensions import block_bucketize_sparse_features


torch.fx.wrap("len")


@dataclass
class BucketizeParam:
    kjt: KeyedJaggedTensor
    num_buckets: int
    block_sizes: torch.Tensor
    output_permute: bool = False
    bucketize_pos: bool = False
    block_bucketize_row_pos: Optional[List[torch.Tensor]] = None
    dist_type_per_feature: Optional[Dict[str, str]] = None


# torch.Tensor.to can not be fx symbolic traced as it does not go through __torch_dispatch__ => fx.wrap it
@torch.fx.wrap
def _fx_wrap_tensor_to_device_dtype(t: torch.Tensor, tensor_device_dtype: torch.Tensor) -> torch.Tensor:
    return t.to(device=tensor_device_dtype.device, dtype=tensor_device_dtype.dtype)


@torch.fx.wrap
def _fx_wrap_stride(kjt: KeyedJaggedTensor) -> Optional[int]:
    return None if kjt.variable_stride_per_key() else kjt.stride()


@torch.fx.wrap
def _fx_wrap_stride_per_key_per_rank(kjt: KeyedJaggedTensor, num_buckets: int) -> Optional[List[List[int]]]:
    return kjt.stride_per_key_per_rank() * num_buckets if kjt.variable_stride_per_key() else None


@torch.fx.wrap
def _fx_wrap_gen_list_n_times(ls: List[str], n: int) -> List[str]:
    # Syntax for dynamo (instead of generator kjt.keys() * num_buckets)
    ret: List[str] = []
    for _ in range(n):
        ret.extend(ls)
    return ret


def bucketize_kjt_before_all2all(bucketize_param: BucketizeParam) -> Tuple[KeyedJaggedTensor, Optional[torch.Tensor]]:
    """
    Bucketizes the `values` in KeyedJaggedTensor into `num_buckets` buckets,
    `lengths` are readjusted based on the bucketization results.

    Note: This function should be used only for row-wise sharding before calling
    `KJTAllToAll`. use the custom AscendC Operator 'block_bucketize_sparse_features'.

    Args:
        bucketize_param (BucketizeParam):
            kjt (KeyedJaggedTensor): input KeyedJaggedTensor tensor.
            num_buckets (int): number of buckets to bucketize the values into.
            block_sizes: (torch.Tensor): bucket sizes for the keyed dimension.
            total_num_blocks: (Optional[torch.Tensor]): number of blocks per feature, useful for two-level
                bucketization output_permute (bool): output the memory location mapping from the unbucketized
                values to bucketized values or not.
            bucketize_pos (bool): output the changed position of the bucketized values or not.
            block_bucketize_row_pos (Optional[List[torch.Tensor]]): The offsets of shard size for each feature.
            dist_type_per_feature (Dict[str, str])：the Dict of distributor type for feature.

    Returns:
        Tuple[KeyedJaggedTensor, Optional[torch.Tensor]]: the bucketized `KeyedJaggedTensor` and
            the optional permute mapping from the unbucketized values to bucketized value.
    """

    num_features = len(bucketize_param.kjt.keys())
    assert_fx_safe(
        bucketize_param.block_sizes.numel() == num_features,
        f"Expecting block sizes for {num_features} features, but {bucketize_param.block_sizes.numel()} received.",
    )

    dist_type_list = []
    for key in bucketize_param.kjt.keys():
        if key not in bucketize_param.dist_type_per_feature:
            raise ValueError(f"key {key} not in {bucketize_param.dist_type_per_feature}")
        dist_type_str = bucketize_param.dist_type_per_feature[key]
        if dist_type_str == "continuous":
            dist_type = 0
        elif dist_type_str == "roundrobin":
            dist_type = 1
        else:
            raise ValueError(f"not support dist type of {dist_type_str}")
        dist_type_list.append(dist_type)

    dist_type_t = torch.tensor(dist_type_list, dtype=torch.int32, device=bucketize_param.kjt.values().device)

    (
        bucketized_lengths,
        bucketized_indices,
        bucketized_weights,
        pos,
        unbucketize_permute,
    ) = block_bucketize_sparse_features(
        bucketize_param.kjt.lengths().view(-1),
        bucketize_param.kjt.values(),
        bucketizePos=bucketize_param.bucketize_pos,
        sequence=bucketize_param.output_permute,
        distTypePerFeature=dist_type_t,
        blockSizes=_fx_wrap_tensor_to_device_dtype(bucketize_param.block_sizes, bucketize_param.kjt.values()),
        mySize=bucketize_param.num_buckets,
        weights=bucketize_param.kjt.weights_or_none(),
    )

    return (
        KeyedJaggedTensor(
            # duplicate keys will be resolved by AllToAll
            keys=_fx_wrap_gen_list_n_times(bucketize_param.kjt.keys(), bucketize_param.num_buckets),
            values=bucketized_indices,
            weights=pos if bucketize_param.bucketize_pos else bucketized_weights,
            lengths=bucketized_lengths.view(-1),
            offsets=None,
            stride=_fx_wrap_stride(bucketize_param.kjt),
            stride_per_key_per_rank=_fx_wrap_stride_per_key_per_rank(bucketize_param.kjt, bucketize_param.num_buckets),
            length_per_key=None,
            offset_per_key=None,
            index_per_key=None,
        ),
        unbucketize_permute,
    )


class DynamicEmbRwSparseFeaturesDist(BaseSparseFeaturesDist[KeyedJaggedTensor]):
    """
    Bucketizes sparse features in RW fashion and then redistributes with an AlltoAll
    collective operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        num_features (int): total number of features.
        feature_hash_sizes (List[int]): hash sizes of features.
        feature_total_num_buckets (Optional[List[int]]): total number of buckets, if provided will be >= world size.
        device (Optional[torch.device]): device on which buffers will be allocated.
        is_sequence (bool): if this is for a sequence embedding.
        has_feature_processor (bool): existence of feature processor (ie. position
            weighted features).
        need_pos(bool): if need position for features.
        dist_type_per_feature (Dict[str, str])：the Dict of distributor type for feature
    """

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
        dist_type_per_feature: Dict[str, str] = None,
    ) -> None:
        super().__init__()
        self.pg = pg
        self._world_size: int = pg.size()
        self._num_features = num_features

        feature_block_sizes: List[int] = []

        for i, hash_size in enumerate(feature_hash_sizes):
            block_divisor = self._world_size
            if feature_total_num_buckets is not None:
                if feature_total_num_buckets[i] % self._world_size != 0:
                    raise ValueError(
                        f"feature_total_num_buckets[{i}] must be divisible by world_size. "
                        f"Got {feature_total_num_buckets[i]} and {self._world_size}"
                    )
                block_divisor = feature_total_num_buckets[i]
            feature_block_sizes.append((hash_size + block_divisor - 1) // block_divisor)

        self.register_buffer(
            "_feature_block_sizes_tensor",
            torch.tensor(
                feature_block_sizes,
                device=device,
                dtype=torch.int64,
            ),
            persistent=False,
        )
        self._has_multiple_blocks_per_shard: bool = feature_total_num_buckets is not None
        if self._has_multiple_blocks_per_shard:
            self.register_buffer(
                "_feature_total_num_blocks_tensor",
                torch.tensor(
                    [feature_total_num_buckets],
                    device=device,
                    dtype=torch.int64,
                ),
                persistent=False,
            )

        self._dist = KJTAllToAll(
            pg=pg,
            splits=[self._num_features] * self._world_size,
        )
        self._is_sequence = is_sequence
        self._has_feature_processor = has_feature_processor
        self.unbucketize_permute_tensor: Optional[torch.Tensor] = None
        self._need_pos = need_pos
        self._dist_type_per_feature = dist_type_per_feature

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        """
        Bucketizes sparse feature values into world size number of buckets and then
        performs AlltoAll operation.
        use the custom AscendC Operator to bucketize KeyedJaggedTensor.
        Args:
            sparse_features (KeyedJaggedTensor): sparse features to bucketize and
                redistribute.

        Returns:
            Awaitable[Awaitable[KeyedJaggedTensor]]: awaitable of awaitable of KeyedJaggedTensor.
        """

        bucketize_param = BucketizeParam(
            kjt=sparse_features,
            num_buckets=self._world_size,
            block_sizes=self._feature_block_sizes_tensor,
            output_permute=self._is_sequence,
            bucketize_pos=(
                self._has_feature_processor if sparse_features.weights_or_none() is None else self._need_pos
            ),
            dist_type_per_feature=self._dist_type_per_feature,
        )
        (
            bucketized_features,
            self.unbucketize_permute_tensor,
        ) = bucketize_kjt_before_all2all(bucketize_param)

        return self._dist(bucketized_features)
