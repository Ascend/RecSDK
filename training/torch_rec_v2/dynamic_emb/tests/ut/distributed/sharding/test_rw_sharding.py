#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import sys
from typing import List, Optional
import unittest
from unittest.mock import patch, MagicMock

import torch
from torch import distributed as dist
from torch.futures import Future
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from dynamic_emb.distributed.sharding.rw_sharding import (
    BucketizeParam,
    bucketize_kjt_before_all2all,
    DynamicEmbRwSparseFeaturesDist,
)

_original_dynamic_emb_extensions = sys.modules.get("dynamic_emb_extensions")


def setup_module():
    mock_dynamicemb = MagicMock()
    sys.modules["dynamic_emb_extensions"] = mock_dynamicemb


def teardown_module():
    sys.modules["dynamic_emb_extensions"] = _original_dynamic_emb_extensions


def create_test_kjt(
    keys: Optional[List[str]] = None,
    lengths: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> KeyedJaggedTensor:
    if keys is None:
        keys = ["feature_0", "feature_1"]
    if lengths is None:
        lengths = torch.tensor([10, 10], device=device)
    if values is None:
        values = torch.arange(lengths.sum().item(), device=device)
    return KeyedJaggedTensor(
        keys=keys,
        values=values,
        lengths=lengths,
    )


class TestBucketizeKjtBeforeAll2All(unittest.TestCase):
    def test_basic_bucketize(self):
        num_buckets = 2
        kjt = create_test_kjt()
        block_sizes = torch.tensor([5, 5], device=kjt.device())  # 每个特征的块大小
        dist_type_per_feature = {"feature_0": "continuous", "feature_1": "roundrobin"}

        bucketized_lengths = torch.tensor([5, 5, 5, 5], device="cpu")
        bucketized_indices = torch.tensor(
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19], device="cpu"
        )
        bucketized_weights = None
        pos = None
        unbucketize_permute = torch.tensor(
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19], device="cpu"
        )

        with patch(
            "dynamic_emb.distributed.sharding.rw_sharding.block_bucketize_sparse_features"
        ) as mocked_block_bucketize_sparse_features:
            mocked_block_bucketize_sparse_features.return_value = (
                bucketized_lengths,
                bucketized_indices,
                bucketized_weights,
                pos,
                unbucketize_permute,
            )
            bucketize_param = BucketizeParam(
                kjt=kjt,
                num_buckets=num_buckets,
                block_sizes=block_sizes,
                output_permute=False,
                dist_type_per_feature=dist_type_per_feature,
            )
            bucketized_kjt, permute = bucketize_kjt_before_all2all(bucketize_param)
            mocked_block_bucketize_sparse_features.assert_called_once()

            self.assertEqual(len(bucketized_kjt.keys()), len(kjt.keys()) * num_buckets)
            self.assertEqual(bucketized_kjt.values().numel(), kjt.values().numel())
            self.assertEqual(len(bucketized_kjt.lengths()), len(kjt.lengths()) * num_buckets)

    def test_empty_input(self):
        empty_kjt = create_test_kjt(
            lengths=torch.tensor([], device="cpu"),
            values=torch.tensor([], device="cpu"),
        )
        block_sizes = torch.tensor([0, 0], device="cpu")
        dist_type = {"feature_0": "continuous", "feature_1": "continuous"}

        bucketized_lengths = torch.tensor([], device="cpu")
        bucketized_indices = torch.tensor([], device="cpu")
        bucketized_weights = None
        pos = None
        unbucketize_permute = torch.tensor([], device="cpu")

        with patch(
            "dynamic_emb.distributed.sharding.rw_sharding.block_bucketize_sparse_features"
        ) as mocked_block_bucketize_sparse_features:
            mocked_block_bucketize_sparse_features.return_value = (
                bucketized_lengths,
                bucketized_indices,
                bucketized_weights,
                pos,
                unbucketize_permute,
            )
            bucketize_param = BucketizeParam(
                kjt=empty_kjt,
                num_buckets=2,
                block_sizes=block_sizes,
                dist_type_per_feature=dist_type,
            )
            bucketized, permute = bucketize_kjt_before_all2all(bucketize_param)

            mocked_block_bucketize_sparse_features.assert_called_once()
            self.assertEqual(bucketized.values().numel(), 0)
            self.assertEqual(len(bucketized.lengths()), 0)


def setup_distributed():
    if not dist.is_available():
        return False
    if dist.is_initialized():
        return True
    try:
        dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:23457", world_size=1, rank=0)
        return True
    except Exception as e:
        return False


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class TestRwSparseFeaturesDist(unittest.TestCase):
    def setUp(self):
        self.is_dist_initialized = setup_distributed()
        if self.is_dist_initialized:
            self.pg = dist.group.WORLD
        else:
            self.pg = MagicMock()
            self.pg.size.return_value = 1
        self.device = torch.device("cpu")

    def tearDown(self):
        cleanup_distributed()

    def test_init_params(self):
        pg = dist.group.WORLD
        num_features = 2
        feature_hash_sizes = [10, 20]  # 两个特征的哈希大小
        dist_type = {"feature_0": "continuous", "feature_1": "roundrobin"}

        rw_dist = DynamicEmbRwSparseFeaturesDist(
            pg=pg,
            num_features=num_features,
            feature_hash_sizes=feature_hash_sizes,
            device="cpu",
            dist_type_per_feature=dist_type,
        )
        assert torch.all(rw_dist._feature_block_sizes_tensor == torch.tensor([10, 20])).item()
        assert rw_dist._world_size == 1

    def test_forward(self):
        pg = dist.group.WORLD
        kjt = create_test_kjt()
        dist_type = {"feature_0": "continuous", "feature_1": "roundrobin"}

        bucketized_lengths = torch.tensor([5, 5, 5, 5], device="cpu")
        bucketized_indices = torch.tensor(
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19], device="cpu"
        )
        bucketized_weights = None
        pos = None
        unbucketize_permute = torch.tensor(
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19], device="cpu"
        )

        mock_kjt_all_to_all_instance = MagicMock()
        mock_final_kjt_result = MagicMock(spec=KeyedJaggedTensor)
        mock_final_kjt_result.keys = ["feature_0", "feature_1", "feature_0", "feature_1"]
        inner_future = Future()
        inner_future.set_result(mock_final_kjt_result)
        outer_future = Future()
        outer_future.set_result(inner_future)
        mock_kjt_all_to_all_instance.side_effect = [outer_future]
        with patch("dynamic_emb.distributed.sharding.rw_sharding.KJTAllToAll") as MockKJTAllToAll:
            with patch(
                "dynamic_emb.distributed.sharding.rw_sharding.block_bucketize_sparse_features"
            ) as mocked_block_bucketize_sparse_features:
                mocked_block_bucketize_sparse_features.return_value = (
                    bucketized_lengths,
                    bucketized_indices,
                    bucketized_weights,
                    pos,
                    unbucketize_permute,
                )
                MockKJTAllToAll.return_value = mock_kjt_all_to_all_instance
                rw_dist = DynamicEmbRwSparseFeaturesDist(
                    pg=pg,
                    num_features=2,
                    feature_hash_sizes=[10, 20],
                    device="cpu",
                    dist_type_per_feature=dist_type,
                )

                result_awaitable = rw_dist.forward(kjt)
                result = result_awaitable.wait().wait()

                mocked_block_bucketize_sparse_features.assert_called_once()
                assert isinstance(result, KeyedJaggedTensor)
                assert len(result.keys) == 4
