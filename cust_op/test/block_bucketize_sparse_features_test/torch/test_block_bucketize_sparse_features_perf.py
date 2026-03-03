#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights
# reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random
import sysconfig
import unittest
from dataclasses import dataclass

import fbgemm_gpu
import numpy as np
import pytest
import torch
import torch_npu


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(10000)

from block_bucketize_sparse_features_perf_cases import (
    PERF_CASES,
    PerfCase,
    _generate_case_tensors,
    _op_kwargs,
    _generate_total_num_blocks_tensors,
    _generate_batch_size_per_feature_and_max_B,
    _generate_block_bucketize_pos,
    GenTotalNumsBlocksType
)

DEVICE = "npu:0"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def _validate_out_of_order_output(
        expected: torch.Tensor,
        actual: torch.Tensor,
        lengths: torch.Tensor,
        is_int: bool = True
    ) -> None:
    tester = unittest.TestCase()
    tester.assertEqual(actual.numel(), expected.numel())
    tester.assertEqual(torch.sum(lengths).item(), actual.numel())
    expected_list = expected.tolist()
    actual_list = actual.tolist()
    offset_list = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths).tolist()

    for i in range(len(offset_list) - 1):
        expected_sample = sorted(expected_list[offset_list[i]: offset_list[i + 1]])
        actual_sample = sorted(actual_list[offset_list[i]: offset_list[i + 1]])
        if is_int:
            tester.assertEqual(expected_sample, actual_sample)
        else:
            for left, right in zip(expected_sample, actual_sample):
                tester.assertAlmostEqual(left, right)
    return


def _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu):
    cpu_out = torch.ops.fbgemm.block_bucketize_sparse_features(
        **kwargs_cpu
    )
    npu_out = torch.ops.mxrec.block_bucketize_sparse_features(
        **kwargs_npu
    )

    assert torch.equal(cpu_out[0], npu_out[0].cpu())
    sequence = bool(kwargs_cpu.get("sequence", False))
    lengths = cpu_out[0]

    def _compare_optional(cpu_val, npu_val, is_int=True):
        if cpu_val is None:
            assert npu_val is None
            return
        if sequence:
            assert torch.equal(cpu_val, npu_val.cpu())
        else:
            _validate_out_of_order_output(cpu_val, npu_val.cpu(), lengths, is_int=is_int)

    _compare_optional(cpu_out[1], npu_out[1], is_int=True)
    _compare_optional(cpu_out[2], npu_out[2], is_int=False)
    _compare_optional(cpu_out[3], npu_out[3], is_int=True)

    # unbucketize_permute 只在 sequence=true 时有语义
    if sequence:
        _compare_optional(cpu_out[4], npu_out[4], is_int=True)
    else:
        assert cpu_out[4] is None
        assert npu_out[4] is None


@pytest.mark.parametrize("need_weights", [False, True])
@pytest.mark.parametrize("bucketize_pos", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("keep_orig_idx", [False, True])
@pytest.mark.parametrize("case", PERF_CASES, ids=lambda case: case.name)
# 覆盖基础路径：所有 PerfCase × 权重/位置/序列/原序号四维参数交叉，校验 CPU 与 NPU 一致
def test_block_bucketize_sparse_features_performance(
    case: PerfCase, need_weights: bool, bucketize_pos: bool, sequence: bool, keep_orig_idx: bool):
    lengths, indices, block_sizes, weights = _generate_case_tensors(case, need_weights)

    kwargs_cpu = _op_kwargs(
        lengths=lengths,
        indices=indices,
        block_sizes=block_sizes,
        my_size=case.my_size,
        weights=weights if need_weights else None,
        bucketize_pos=bucketize_pos,
        sequence=sequence,
        keep_orig_idx=keep_orig_idx,
    )

    kwargs_npu = _op_kwargs(**kwargs_cpu)
    kwargs_npu['lengths'] = lengths.to(DEVICE)
    kwargs_npu['indices'] = indices.to(DEVICE)
    kwargs_npu['block_sizes'] = block_sizes.to(DEVICE)
    kwargs_npu['weights'] = weights.to(DEVICE) if need_weights else None

    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)


@pytest.mark.parametrize("total_num_blocks_type",
    [GenTotalNumsBlocksType.RAND_TYPE, GenTotalNumsBlocksType.MULTIPLE_TYPE, GenTotalNumsBlocksType.NONE_TYPE])
@pytest.mark.parametrize("case", PERF_CASES, ids=lambda case: case.name)
# 覆盖 total_num_blocks 分支：倍数/非倍数/None 三类，与所有 PerfCase 组合
def test_block_bucketize_sparse_features_performance_with_total_num_blocks(
        case: PerfCase,
        total_num_blocks_type: GenTotalNumsBlocksType
    ):
    lengths, indices, block_sizes, _ = _generate_case_tensors(case, False)
    total_num_blocks = _generate_total_num_blocks_tensors(block_sizes, case.my_size, total_num_blocks_type)

    kwargs_cpu = _op_kwargs(
        lengths=lengths,
        indices=indices,
        block_sizes=block_sizes,
        my_size=case.my_size,
        total_num_blocks=total_num_blocks.cpu() if total_num_blocks is not None else None,
    )

    kwargs_npu = _op_kwargs(**kwargs_cpu)
    kwargs_npu['lengths'] = lengths.to(DEVICE)
    kwargs_npu['indices'] = indices.to(DEVICE)
    kwargs_npu['block_sizes'] = block_sizes.to(DEVICE)
    kwargs_npu['total_num_blocks'] = total_num_blocks.to(DEVICE) if total_num_blocks is not None else None

    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)


@pytest.mark.parametrize("case", PERF_CASES, ids=lambda case: case.name)
# 覆盖 batch_size_per_feature/max_B 分支：所有 PerfCase 下变批量的路径
def test_block_bucketize_sparse_features_performance_with_batch_size_per_feature_and_max_B(case: PerfCase):
    lengths, indices, block_sizes, _ = _generate_case_tensors(case, False)
    batch_size_per_feature, max_B = _generate_batch_size_per_feature_and_max_B(block_sizes, case.batch_size)

    kwargs_cpu = _op_kwargs(
        lengths=lengths,
        indices=indices,
        block_sizes=block_sizes,
        my_size=case.my_size,
        batch_size_per_feature=batch_size_per_feature,
        max_B=max_B,
    )

    kwargs_npu = _op_kwargs(**kwargs_cpu)
    kwargs_npu['lengths'] = lengths.to(DEVICE)
    kwargs_npu['indices'] = indices.to(DEVICE)
    kwargs_npu['block_sizes'] = block_sizes.to(DEVICE)
    kwargs_npu['batch_size_per_feature'] = batch_size_per_feature.to(DEVICE)

    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)


@pytest.mark.parametrize("case", PERF_CASES, ids=lambda case: case.name)
# 覆盖 block_bucketize_pos 分支：提供位置表但保持其它参数默认
def test_block_bucketize_sparse_features_performance_with_block_bucketize_pos(case: PerfCase):
    lengths, indices, block_sizes, _ = _generate_case_tensors(case, False)
    block_bucketize_pos_cpu_list, block_bucketize_pos_npu_list = \
        _generate_block_bucketize_pos(block_sizes, case.my_size, DEVICE)

    kwargs_cpu = _op_kwargs(
        lengths=lengths,
        indices=indices,
        block_sizes=block_sizes,
        my_size=case.my_size,
        block_bucketize_pos=block_bucketize_pos_cpu_list,
    )

    kwargs_npu = _op_kwargs(**kwargs_cpu)
    kwargs_npu['lengths'] = lengths.to(DEVICE)
    kwargs_npu['indices'] = indices.to(DEVICE)
    kwargs_npu['block_sizes'] = block_sizes.to(DEVICE)
    kwargs_npu['block_bucketize_pos'] = block_bucketize_pos_npu_list

    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)


@pytest.mark.parametrize("need_weights", [False, True])
@pytest.mark.parametrize("case", PERF_CASES, ids=lambda case: case.name)
# 覆盖 block_bucketize_pos + bucketize_pos/sequence/keep_orig_idx + 权重组合
def test_block_bucketize_sparse_features_block_bucketize_pos_with_flags(case: PerfCase, need_weights: bool):
    lengths, indices, block_sizes, weights = _generate_case_tensors(case, need_weights)
    block_bucketize_pos_cpu_list, block_bucketize_pos_npu_list = _generate_block_bucketize_pos(
        block_sizes, case.my_size, DEVICE
    )

    kwargs_cpu = _op_kwargs(
        lengths=lengths,
        indices=indices,
        block_sizes=block_sizes,
        my_size=case.my_size,
        weights=weights if need_weights else None,
        bucketize_pos=True,
        sequence=True,
        keep_orig_idx=True,
        block_bucketize_pos=block_bucketize_pos_cpu_list,
    )

    kwargs_npu = _op_kwargs(**kwargs_cpu)
    kwargs_npu['lengths'] = lengths.to(DEVICE)
    kwargs_npu['indices'] = indices.to(DEVICE)
    kwargs_npu['block_sizes'] = block_sizes.to(DEVICE)
    kwargs_npu['weights'] = weights.to(DEVICE) if need_weights else None
    kwargs_npu['block_bucketize_pos'] = block_bucketize_pos_npu_list

    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)


@pytest.mark.parametrize(
    "total_num_blocks_type", [GenTotalNumsBlocksType.RAND_TYPE, GenTotalNumsBlocksType.MULTIPLE_TYPE]
)
@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("keep_orig_idx", [False, True])
@pytest.mark.parametrize("dtype_combo", [
    pytest.param((None, None), id="same_dtype"),
    pytest.param((torch.int32, torch.int64), id="off32_idx64"),
    pytest.param((torch.int64, torch.int32), id="off64_idx32"),
])
@pytest.mark.parametrize("case", PERF_CASES, ids=lambda case: case.name)
def test_block_bucketize_sparse_features_with_all_optionals(
    case: PerfCase, dtype_combo,
    total_num_blocks_type: GenTotalNumsBlocksType, sequence: bool, keep_orig_idx: bool
):
    offset_dtype, index_dtype = dtype_combo
    lengths, indices, block_sizes, weights = _generate_case_tensors(
        case, True, offset_dtype=offset_dtype, index_dtype=index_dtype
    )
    batch_size_per_feature, max_B = _generate_batch_size_per_feature_and_max_B(
        block_sizes.to(offset_dtype or case.dtype), case.batch_size
    )
    total_num_blocks = _generate_total_num_blocks_tensors(block_sizes, case.my_size, total_num_blocks_type)
    block_bucketize_pos_cpu_list, block_bucketize_pos_npu_list = _generate_block_bucketize_pos(
        block_sizes, case.my_size, DEVICE
    )

    kwargs_cpu = _op_kwargs(
        lengths=lengths,
        indices=indices,
        block_sizes=block_sizes,
        my_size=case.my_size,
        weights=weights,
        bucketize_pos=True,
        sequence=sequence,
        keep_orig_idx=keep_orig_idx,
        batch_size_per_feature=batch_size_per_feature,
        max_B=max_B,
        total_num_blocks=total_num_blocks,
        block_bucketize_pos=block_bucketize_pos_cpu_list,
    )

    kwargs_npu = _op_kwargs(**kwargs_cpu)
    kwargs_npu['lengths'] = lengths.to(DEVICE)
    kwargs_npu['indices'] = indices.to(DEVICE)
    kwargs_npu['block_sizes'] = block_sizes.to(DEVICE)
    kwargs_npu['weights'] = weights.to(DEVICE)
    kwargs_npu['batch_size_per_feature'] = batch_size_per_feature.to(DEVICE)
    kwargs_npu['total_num_blocks'] = total_num_blocks.to(DEVICE)
    kwargs_npu['block_bucketize_pos'] = block_bucketize_pos_npu_list

    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)


@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("keep_orig_idx", [False, True])
def test_block_bucketize_sparse_features_int64_extreme(sequence, keep_orig_idx):
    """
    indices 包含超出 int32 表示范围的极值，验证 int64 通路无截断。
    - lengths 使用 int32（覆盖 OffsetT=int32 / IndexT=int64 混合 dtype 路径）
    - indices 覆盖 int32 边界值（INT32_MAX, INT32_MAX+1）和极大值（2^40, 2^62）
    - 多 feature 使用不同 block_size，验证多 feature 场景下分桶一致性
    """
    int32_max = int(np.iinfo(np.int32).max)
    lengths = torch.tensor([3, 2, 2], dtype=torch.int32)
    indices = torch.tensor([
        int32_max, int32_max + 1, int32_max * 2,
        2**33 + 7, 2**40 + 13,
        2**50, 2**62,
    ], dtype=torch.int64)
    block_sizes = torch.tensor([2**16, 2**18, 2**20], dtype=torch.int64)
    my_size = 4

    weights = torch.randn(indices.numel(), dtype=torch.float32).uniform_(-1.0, 1.0)

    kwargs_cpu = _op_kwargs(
        lengths=lengths,
        indices=indices,
        block_sizes=block_sizes,
        my_size=my_size,
        weights=weights,
        bucketize_pos=False,
        sequence=sequence,
        keep_orig_idx=keep_orig_idx,
    )

    kwargs_npu = _op_kwargs(**kwargs_cpu)
    kwargs_npu['lengths'] = lengths.to(DEVICE)
    kwargs_npu['indices'] = indices.to(DEVICE)
    kwargs_npu['block_sizes'] = block_sizes.to(DEVICE)
    kwargs_npu['weights'] = weights.to(DEVICE)

    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)
