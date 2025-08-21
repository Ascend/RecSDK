#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from unittest.mock import patch

import torch
from torch import empty_like
from hybrid_torchrec.distributed.sharding.post_input_dist import (
    split_keys_offset,
    do_unique_hash,
    do_unique_hash_out,
    HashMapBase,
    KeyedJaggedTensorWithLookHelper
)
from hybrid_torchrec.modules.ids_process import IdsMapper
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class MockHashMap(HashMapBase):
    def __call__(self, values):
        unique, inverse = torch.unique(values, return_inverse=True)
        hash_indices = torch.arange(len(values))
        return hash_indices, unique, inverse

    @staticmethod
    def ids2indices_unique_out(*args, **kwargs):
        pass


def empty_like_without_pin_memory(*args, **kwargs):
    kwargs['pin_memory'] = False
    return empty_like(*args, **kwargs)


class TestDoUniqueHash:
    @staticmethod
    @patch("torch.Tensor.pin_memory", new=lambda self, *args, **kwargs: self)
    def test_do_unique_hash_f0():
        kjt = KeyedJaggedTensor(
            keys=[],
            values=torch.tensor([]),
            lengths=torch.tensor([]),
            offsets=torch.tensor([0])
        )
        result = do_unique_hash(kjt, [], [])
        assert result.unique_indices is None
        assert result.unique_inverse is None

    @staticmethod
    @patch("torch.Tensor.pin_memory", new=lambda self, *args, **kwargs: self)
    def test_do_unique_hash_f1():
        kjt = KeyedJaggedTensor(
            keys=["f1"],
            values=torch.tensor([1, 1, 2]),
            lengths=torch.tensor([3]),
            offsets=torch.tensor([0, 3])
        )
        result = do_unique_hash(kjt, [1], [MockHashMap()])
        assert len(result.unique_indices) == 2  # [1, 2]

    @staticmethod
    @patch("torch.Tensor.pin_memory", new=lambda self, *args, **kwargs: self)
    def test_do_unique_hash_f2():
        kjt = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([1, 2, 3, 1, 2, 4]),
            lengths=torch.tensor([3, 3]),
            offsets=torch.tensor([0, 3, 6])
        )

        feature_split = [1, 1]
        hashmaps = [MockHashMap(), MockHashMap()]
        result = do_unique_hash(kjt, feature_split, hashmaps)

        assert isinstance(result, KeyedJaggedTensorWithLookHelper)
        assert result.keys() == ["f1", "f2"]
        assert torch.equal(result.values(), kjt.values())

    @staticmethod
    @patch("torch.Tensor.pin_memory", new=lambda self, *args, **kwargs: self)
    def test_do_unique_hash_f3():
        kjt = KeyedJaggedTensor(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            lengths=torch.tensor([3, 3, 3]),
            offsets=torch.tensor([0, 3, 6, 9])
        )
        result = do_unique_hash(kjt, [2, 1], [MockHashMap(), MockHashMap()])
        assert len(result.unique_indices) == 9


class TestDoUniqueHashOut:
    @staticmethod
    @patch("torch.empty_like", new=empty_like_without_pin_memory)
    @patch("torch.Tensor.pin_memory", new=lambda self, *args, **kwargs: self)
    def test_do_unique_hash_out_with_parallel():
        with patch.dict(os.environ, {"ENABLE_PARALLEL_GLOBAL_UNIQUE": "1"}):
            kjt = KeyedJaggedTensor(
                keys=["f1"],
                values=torch.tensor([1, 1, 2]),
                lengths=torch.tensor([3]),
                offsets=torch.tensor([0, 3])
            )
            result = do_unique_hash_out(kjt, [1], [IdsMapper(128)])
            assert len(result.unique_indices) == 2


class TestSplitKeysOffset:
    @staticmethod
    def test_split_key_offset_f0():
        kjt = KeyedJaggedTensor(
            keys=[],
            values=torch.tensor([]),
            lengths=torch.tensor([]),
            offsets=torch.tensor([0])
        )
        result = split_keys_offset(kjt, [])
        expected = torch.LongTensor([0])
        assert torch.equal(result, expected)

    @staticmethod
    def test_split_key_offset_f1():
        kjt = KeyedJaggedTensor(
            keys=["f1"],
            values=torch.tensor([1, 2]),
            lengths=torch.tensor([2]),
            offsets=torch.tensor([0, 2])
        )
        result = split_keys_offset(kjt, [1])
        expected = torch.LongTensor([0, 2])
        assert torch.equal(result, expected)

    @staticmethod
    def test_split_key_offset_f3():
        kjt = KeyedJaggedTensor(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([1, 2, 3, 4, 5, 6]),
            lengths=torch.tensor([2, 2, 2]),
            offsets=torch.tensor([0, 2, 4, 6])
        )
        result = split_keys_offset(kjt, [1, 2])
        expected = torch.LongTensor([0, 2, 6])
        assert torch.equal(result, expected)

    @staticmethod
    def test_split_key_offset_f4():
        kjt = KeyedJaggedTensor(
            keys=["f1", "f2", "f3", "f4"],
            values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
            lengths=torch.tensor([2, 2, 2, 2]),
            offsets=torch.tensor([0, 2, 4, 6, 8])
        )
        result = split_keys_offset(kjt, [2, 2])
        expected = torch.LongTensor([0, 4, 8])
        assert torch.equal(result, expected)
