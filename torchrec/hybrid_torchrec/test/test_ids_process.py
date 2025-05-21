#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
import pytest
import torch

from hybrid_torchrec.modules.ids_process import (
    IdsMapper,
    block_bucketize_sparse_features_cpu,
    BucketParams,
)

from torchrec import JaggedTensor, KeyedJaggedTensor

TEST_NUM = 10
IDS_RANGE_TIMES = 10


@dataclass
class BucketResult:
    bucketized_lengths: torch.Tensor
    bucketized_indices: torch.Tensor
    origin_len: torch.Tensor
    origin_indices: torch.Tensor
    feat_num: int
    bucket_size: int


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


def check_bucketized_valid(params: BucketResult):
    batch_size = params.bucketized_lengths.numel() // params.bucket_size // params.feat_num
    bucketized_offset = 0
    for rank in range(params.bucket_size):
        this_rank_length = params.bucketized_lengths[
            rank * params.feat_num * batch_size: (rank + 1) * params.feat_num * batch_size
        ]
        origin_batch_offset = 0
        for feat_id in range(params.feat_num):
            this_feat_length = this_rank_length[
                feat_id * batch_size: (feat_id + 1) * batch_size
            ]
            for ind in range(batch_size):
                this_indices_len = this_feat_length[ind].item()

                origin_indices_len = params.origin_len[feat_id * batch_size + ind]
                origin_index = params.origin_indices[
                    origin_batch_offset: origin_batch_offset + origin_indices_len
                ]
                for _ in range(this_indices_len):
                    ids = params.bucketized_indices[bucketized_offset]
                    assert (
                        ids % params.bucket_size
                    ) == rank, \
                    f"bucketized_indices {ids} in invalid bucket {rank} bucketized_offset {bucketized_offset}"
                    assert (
                        ids in origin_index
                    ), f"bucketized_indices {ids} in invalid position {origin_batch_offset} \
                    origin_index {origin_index} bucketized_offset {bucketized_offset}"
                    bucketized_offset += 1
                origin_batch_offset += origin_indices_len


@pytest.mark.parametrize("input_size", [1000])
def test_ids2indices_sequential(input_size):
    """Test ids2indices with sequential numbers"""
    logging.info("Testing sequential ids mapping")
    mapper = IdsMapper(input_size)
    id2indices = {}
    indices2id = {}
    for _ in range(TEST_NUM):
        input_ids = torch.randint(0, input_size * IDS_RANGE_TIMES, (input_size,))
        indices, unique, unique_inverse = mapper(input_ids)
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


@pytest.mark.parametrize("input_size", [10000])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("num_mapper", [3])
def test_ids2indices_out_ids_max_than_table_size(input_size, pin_memory, num_mapper):
    """Test ids2indices with sequential numbers"""
    logging.info("Testing sequential ids mapping")
    mappers = [IdsMapper(input_size) for _ in range(num_mapper)]
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
        with pytest.raises(RuntimeError):
            mappers[i].ids2indices_unique_out(
                ids, hash_indices, offsets, unique, unique_inverse, unique_offset, i
            )


@pytest.mark.parametrize("input_size", [10000])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("num_mapper", [3])
def test_ids2indices_out_ids_smaller_than_0(input_size, pin_memory, num_mapper):
    """Test ids2indices with sequential numbers"""
    logging.info("Testing sequential ids mapping")
    mappers = [IdsMapper(input_size * IDS_RANGE_TIMES) for _ in range(num_mapper)]

    input_ids = [
        torch.randint(-input_size, input_size, (input_size,))
        for _ in range(num_mapper)
    ]

    ids = torch.concat(input_ids)
    hash_indices = torch.empty_like(ids, pin_memory=pin_memory)
    offsets = torch.LongTensor([0, input_size, input_size * 2, input_size * 3])
    unique = torch.empty_like(ids, pin_memory=pin_memory)
    unique_inverse = torch.empty_like(ids, pin_memory=pin_memory)
    unique_offset = torch.LongTensor([0 for _ in range(num_mapper + 1)])
    for i in range(num_mapper):
        with pytest.raises(RuntimeError):
            mappers[i].ids2indices_unique_out(
                ids, hash_indices, offsets, unique, unique_inverse, unique_offset, i
            )


@pytest.mark.parametrize("input_size", [10000])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("num_mapper", [3])
def test_ids2indices_out_ids_unique_is_none(input_size, pin_memory, num_mapper):
    """Test ids2indices with sequential numbers"""
    logging.info("Testing sequential ids mapping")
    mappers = [IdsMapper(input_size * IDS_RANGE_TIMES) for _ in range(num_mapper)]

    input_ids = [
        torch.randint(0, input_size * IDS_RANGE_TIMES, (input_size,))
        for _ in range(num_mapper)
    ]

    ids = torch.concat(input_ids)
    hash_indices = torch.empty_like(ids, pin_memory=pin_memory)
    offsets = torch.LongTensor([0, input_size, input_size * 2, input_size * 3])
    unique = None
    unique_inverse = torch.empty_like(ids, pin_memory=pin_memory)
    unique_offset = torch.LongTensor([0 for _ in range(num_mapper + 1)])
    for i in range(num_mapper):
        with pytest.raises(RuntimeError):
            mappers[i].ids2indices_unique_out(
                ids, hash_indices, offsets, unique, unique_inverse, unique_offset, i
            )


@pytest.mark.parametrize("input_size", [10000])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("num_mapper", [3])
def test_ids2indices_out_ids_is_none(input_size, pin_memory, num_mapper):
    """Test ids2indices with sequential numbers"""
    logging.info("Testing sequential ids mapping")
    mappers = [IdsMapper(input_size * IDS_RANGE_TIMES) for _ in range(num_mapper)]
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
        with pytest.raises(RuntimeError):
            mappers[i].ids2indices_unique_out(
                None, hash_indices, offsets, unique, unique_inverse, unique_offset, i
            )


@pytest.mark.parametrize("input_size", [10000])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("num_mapper", [3])
def test_ids2indices_out_ids_invalid_offset(input_size, pin_memory, num_mapper):
    """Test ids2indices with sequential numbers"""
    logging.info("Testing sequential ids mapping")
    mappers = [IdsMapper(input_size * IDS_RANGE_TIMES) for _ in range(num_mapper)]

    input_ids = [
        torch.randint(0, input_size * IDS_RANGE_TIMES, (input_size,))
        for _ in range(num_mapper)
    ]

    ids = torch.concat(input_ids)
    hash_indices = torch.empty_like(ids, pin_memory=pin_memory)
    offsets = torch.LongTensor([0, input_size])
    unique = torch.empty_like(ids, pin_memory=pin_memory)
    unique_inverse = torch.empty_like(ids, pin_memory=pin_memory)
    unique_offset = torch.LongTensor([0 for _ in range(num_mapper + 1)])
    with pytest.raises(RuntimeError):
        for i in range(num_mapper):
            mappers[i].ids2indices_unique_out(
                ids, hash_indices, offsets, unique, unique_inverse, unique_offset, i
            )


@pytest.mark.parametrize("input_size", [10000])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("num_mapper", [3])
def test_ids2indices_out_ids_hash_indices_is_none(input_size, pin_memory, num_mapper):
    """Test ids2indices with sequential numbers"""
    logging.info("Testing sequential ids mapping")
    mappers = [IdsMapper(input_size * IDS_RANGE_TIMES) for _ in range(num_mapper)]

    input_ids = [
        torch.randint(0, input_size * IDS_RANGE_TIMES, (input_size,))
        for _ in range(num_mapper)
    ]

    ids = torch.concat(input_ids)
    hash_indices = None
    offsets = torch.LongTensor([0, input_size, input_size * 2, input_size * 3])
    unique = torch.empty_like(ids, pin_memory=pin_memory)
    unique_inverse = torch.empty_like(ids, pin_memory=pin_memory)
    unique_offset = torch.LongTensor([0 for _ in range(num_mapper + 1)])
    for i in range(num_mapper):
        with pytest.raises(RuntimeError):
            mappers[i].ids2indices_unique_out(
                ids, hash_indices, offsets, unique, unique_inverse, unique_offset, i
            )


@pytest.mark.parametrize("input_size", [1000])
@pytest.mark.parametrize("mutil_hots", [[1, 2, 3, 4]])
@pytest.mark.parametrize("bucket_size", [4])
def test_block_bucketize_sparse_features_cpu(input_size, mutil_hots, bucket_size):
    for _ in range(TEST_NUM):
        jt_dict = {}
        for ind, mutil_hot in enumerate(mutil_hots):
            v = torch.randint(0, input_size, (input_size * mutil_hot,))
            jt_dict[f"feat{ind}"] = JaggedTensor(
                values=v, lengths=torch.ones(input_size, dtype=torch.int64) * mutil_hot
            )
        kjt = KeyedJaggedTensor.from_jt_dict(jt_dict)

        lengths = kjt.lengths().view(-1)
        values = kjt.values()
        block_size = torch.Tensor([100 for _ in range(len(mutil_hots))]).long()
        params_in = BucketParams(
            lengths,
            values,
            bucketize_pos=False,
            sequence=True,
            block_sizes=block_size,
            bucket_size=bucket_size,
            weights=kjt.weights_or_none(),
            batch_size_per_feature=None,
            max_b=-1,
            block_bucketize_pos=None,)

        (
            bucketized_lengths,
            bucketized_indices,
            bucketized_weights,
            pos,
            unbucketize_permute,
            _,
        ) = block_bucketize_sparse_features_cpu(params_in)

        params = BucketResult(bucketized_lengths,
                              bucketized_indices,
                              lengths,
                              values,
                              len(mutil_hots),
                              bucket_size)
        check_bucketized_valid(params)

        inverse_result = torch.index_select(
            bucketized_indices, dim=0, index=unbucketize_permute
        )
        assert (inverse_result == values).all(), "unbucketize_permute is invalid"


@pytest.mark.parametrize("input_size", [1000])
@pytest.mark.parametrize("mutil_hots", [[1, 2, 3, 4]])
@pytest.mark.parametrize("bucket_size", [0])
def test_block_bucketize_sparse_features_cpu_invalid_bucket_size(input_size, mutil_hots, bucket_size):
    jt_dict = {}
    for ind, mutil_hot in enumerate(mutil_hots):
        v = torch.randint(0, input_size, (input_size * mutil_hot,))
        jt_dict[f"feat{ind}"] = JaggedTensor(
            values=v, lengths=torch.ones(input_size, dtype=torch.int64) * mutil_hot
        )
    kjt = KeyedJaggedTensor.from_jt_dict(jt_dict)

    lengths = kjt.lengths().view(-1)
    values = kjt.values()
    with pytest.raises(RuntimeError):
        block_size = torch.Tensor([100 for _ in range(len(mutil_hots))]).long()
        params_in = BucketParams(
            lengths,
            values,
            bucketize_pos=False,
            sequence=True,
            block_sizes=block_size,
            bucket_size=bucket_size,
            weights=kjt.weights_or_none(),
            batch_size_per_feature=None,
            max_b=-1,
            block_bucketize_pos=None,)

        (
            bucketized_lengths,
            bucketized_indices,
            bucketized_weights,
            pos,
            unbucketize_permute,
            _,
        ) = block_bucketize_sparse_features_cpu(params_in)


@pytest.mark.parametrize("input_size", [1000])
@pytest.mark.parametrize("mutil_hots", [[1, 2, 3, 4]])
@pytest.mark.parametrize("bucket_size", [4])
def test_block_bucketize_sparse_features_cpu_invalid_block_size(input_size, mutil_hots, bucket_size):
    jt_dict = {}
    for ind, mutil_hot in enumerate(mutil_hots):
        v = torch.randint(0, input_size, (input_size * mutil_hot,))
        jt_dict[f"feat{ind}"] = JaggedTensor(
            values=v, lengths=torch.ones(input_size, dtype=torch.int64) * mutil_hot
        )
    kjt = KeyedJaggedTensor.from_jt_dict(jt_dict)

    lengths = kjt.lengths().view(-1)
    values = kjt.values()
    with pytest.raises(RuntimeError):
        block_size = None
        params_in = BucketParams(
            lengths,
            values,
            bucketize_pos=False,
            sequence=True,
            block_sizes=block_size,
            bucket_size=bucket_size,
            weights=kjt.weights_or_none(),
            batch_size_per_feature=None,
            max_b=-1,
            block_bucketize_pos=None,)

        (
            bucketized_lengths,
            bucketized_indices,
            bucketized_weights,
            pos,
            unbucketize_permute,
            _,
        ) = block_bucketize_sparse_features_cpu(params_in)
