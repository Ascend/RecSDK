#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import pytest
import torch

from hybrid_torchrec.modules.ids_process import IdsMapper
from torchrec import JaggedTensor, KeyedJaggedTensor

TEST_NUM = 10
IDS_RANGE_TIMES = 10


def verify_unique(indices, unique, unique_inverse):
    sets = set()
    for i in unique:
        assert i not in sets, "Unique ids is not unique"
        sets.add(i)
    result = torch.index_select(unique, index=unique_inverse, dim=0)
    assert (indices == result).all(), "Invalid inverse tensor"


def verify_mapper(id2indices, indices2id, input_ids, indices):
    for k, v in zip(input_ids.tolist(), indices.tolist()):
        if k in id2indices.keys():
            assert v == id2indices[k], "Two ids has the same indices"
        else:
            id2indices[k] = v

        if v in indices2id.keys():
            assert k == indices2id[v], "Two ids has the same indices"
        else:
            indices2id[v] = k


@pytest.mark.parametrize("input_size", [1000])
@pytest.mark.parametrize("high_precison", [True, False])
def test_ids2indices_sequential(input_size, high_precison):
    """Test ids2indices with sequential numbers"""
    logging.info("Testing sequential ids mapping")
    mapper = IdsMapper(input_size)
    id2indices = {}
    indices2id = {}
    for _ in range(TEST_NUM):
        input_ids = torch.randint(0, input_size * IDS_RANGE_TIMES, (input_size,))
        indices, unique, unique_inverse = mapper(input_ids, high_precison)
        verify_mapper(id2indices, indices2id, input_ids, indices)
        verify_unique(indices, unique, unique_inverse)


@pytest.mark.parametrize("input_size", [10000])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("num_mapper", [3])
def test_ids2indices_out(input_size, pin_memory, num_mapper):
    """Test ids2indices with sequential numbers"""
    logging.info("Testing sequential ids mapping")
    mappers = [IdsMapper(input_size * IDS_RANGE_TIMES) for _ in range(num_mapper)]
    id2indices = [{} for _ in range(num_mapper)]
    indices2id = [{} for _ in range(num_mapper)]

    for _ in range(TEST_NUM):
        input_ids = [
            torch.randint(0, input_size * IDS_RANGE_TIMES, (input_size,))
            for _ in range(num_mapper)
        ]

        ids = torch.concat(input_ids)
        hash_indices = torch.empty_like(ids, pin_memory=pin_memory)
        offsets = torch.LongTensor([0, input_size, input_size * 2, input_size * 3])
        unique = torch.empty_like(ids, pin_memory=pin_memory)
        unique_inverse = torch.empty_like(ids, pin_memory=pin_memory)
        unique_offset = torch.LongTensor([0 for _ in range(num_mapper + 1)])
        for i in range(num_mapper):
            mappers[i].ids2indices_unique_out(
                ids, hash_indices, offsets, unique, unique_inverse, unique_offset, i
            )

            start = offsets[i].item()
            end = offsets[i + 1].item()
            input_id = ids[start:end]
            indices = hash_indices[start:end]
            verify_mapper(id2indices[i], indices2id[i], input_id, indices)
            unique_start = unique_offset[i].item()
            unique_end = unique_offset[i + 1].item()
            unique_this = unique[unique_start:unique_end]
            unique_inverse_this = unique_inverse[start:end]
            verify_unique(indices, unique_this, unique_inverse_this)
