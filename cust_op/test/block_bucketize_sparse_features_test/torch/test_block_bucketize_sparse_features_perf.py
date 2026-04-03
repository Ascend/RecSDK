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
    # 因为 batch_size_per_feature 要求各 feature 有不同的 batch_size，而 batch_size=1 时 lengths 总长只有 num_features 个，无法支持变长 batch 分配。
    if case.batch_size <= 1:
        pytest.skip("batch_size_per_feature requires batch_size > 1")
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
    # 因为 batch_size_per_feature 要求各 feature 有不同的 batch_size，而 batch_size=1 时 lengths 总长只有 num_features 个，无法支持变长 batch 分配。
    if case.batch_size <= 1:
        pytest.skip("batch_size_per_feature requires batch_size > 1")
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


# =====================================================================
# 固定数据精确校验用例：使用独立设计的测试数据覆盖各分支路径
# =====================================================================
@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("keep_orig_idx", [False, True])
@pytest.mark.parametrize("my_size", [2, 3, 5, 7])
def test_block_bucketize_sparse_features_negative_indices(sequence, keep_orig_idx, my_size):
    """
    indices 包含负值，验证负数转无符号后 bucket/new_index 计算正确。
    负数 int64 转 uint64 后接近上限，曾触发快速除法（UintDiv）精度溢出。
    覆盖 pow2 和非 pow2 的 my_size，以及 pooled / sequence 两条路径。
    """
    lengths = torch.tensor([4, 3], dtype=torch.int32)
    indices = torch.tensor([
        -1, -8, -3, 10,
        -100, 7, -9223372036854775808,
    ], dtype=torch.int64)
    block_sizes = torch.tensor([4, 8], dtype=torch.int64)

    weights = torch.randn(indices.numel(), dtype=torch.float32).uniform_(-1.0, 1.0)

    kwargs_cpu = _op_kwargs(
        lengths=lengths,
        indices=indices,
        block_sizes=block_sizes,
        my_size=my_size,
        weights=weights,
        bucketize_pos=True,
        sequence=sequence,
        keep_orig_idx=keep_orig_idx,
    )

    kwargs_npu = _op_kwargs(**kwargs_cpu)
    kwargs_npu['lengths'] = lengths.to(DEVICE)
    kwargs_npu['indices'] = indices.to(DEVICE)
    kwargs_npu['block_sizes'] = block_sizes.to(DEVICE)
    kwargs_npu['weights'] = weights.to(DEVICE)

    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)


def _make_npu_kwargs(kwargs_cpu, device, tensor_keys, list_keys=None):
    """从 CPU kwargs 构造 NPU kwargs 的通用辅助函数。"""
    kwargs_npu = _op_kwargs(**kwargs_cpu)
    for key in tensor_keys:
        val = kwargs_cpu.get(key)
        kwargs_npu[key] = val.to(device) if val is not None else None
    for key in (list_keys or []):
        val = kwargs_cpu.get(key)
        kwargs_npu[key] = [t.to(device) for t in val] if val is not None else None
    return kwargs_npu


@pytest.mark.parametrize("index_type", [torch.int, torch.long])
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("bucketize_pos", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
def test_fixed_multi_feature_basic(index_type, has_weight, bucketize_pos, sequence):
    """
    5 features, batch_size=2, my_size=3, 含零长度行。
    覆盖：多 feature 不同 block_size + 零长度行 + 跨桶 indices。
    """
    lengths = torch.tensor([1, 3, 0, 2, 3, 1, 2, 0, 1, 4], dtype=index_type)
    indices = torch.tensor([
        2,
        7, 14, 0,
        9, 3,
        25, 33, 44,
        50,
        61, 72,
        100,
        130, 155, 170, 199,
    ], dtype=index_type)
    block_sizes = torch.tensor([6, 8, 12, 20, 35], dtype=index_type)
    my_size = 3
    weights = torch.linspace(0.1, 1.7, indices.numel(), dtype=torch.float32) if has_weight else None

    kwargs_cpu = _op_kwargs(
        lengths=lengths, indices=indices, block_sizes=block_sizes,
        my_size=my_size, weights=weights,
        bucketize_pos=bucketize_pos, sequence=sequence,
    )
    kwargs_npu = _make_npu_kwargs(kwargs_cpu, DEVICE, ['lengths', 'indices', 'block_sizes', 'weights'])
    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)


@pytest.mark.parametrize("index_type", [torch.int, torch.long])
@pytest.mark.parametrize("keep_orig_idx", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("bucketize_pos", [False, True])
def test_fixed_long_and_negative_indices(index_type, keep_orig_idx, sequence, bucketize_pos):
    """
    2 features, batch_size=3, my_size=5。
    int32 路径使用常规小值；int64 路径混入负数和超大正整数。
    覆盖：负数 indices 分桶 + 超大正整数越界分桶 + keep_orig_idx 分支。
    """
    my_size = 5
    block_sizes = torch.tensor([7, 11], dtype=index_type)
    if index_type == torch.int:
        lengths = torch.tensor([2, 1, 3, 0, 2, 1], dtype=index_type)
        indices = torch.tensor([0, 13, 20, 5, 30, 34, 8, 21, 10], dtype=index_type)
    else:
        lengths = torch.tensor([3, 2, 1, 2, 3, 1], dtype=index_type)
        indices = torch.tensor([
            5, -3, 200043781927513,
            -17, 42,
            0,
            -1, 100029876543210,
            11, 33, -9223372036854775807,
            54,
        ], dtype=index_type)

    kwargs_cpu = _op_kwargs(
        lengths=lengths, indices=indices, block_sizes=block_sizes,
        my_size=my_size, bucketize_pos=bucketize_pos, sequence=sequence,
        keep_orig_idx=keep_orig_idx,
    )
    kwargs_npu = _make_npu_kwargs(kwargs_cpu, DEVICE, ['lengths', 'indices', 'block_sizes'])
    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)


@pytest.mark.parametrize("index_type", [torch.int, torch.long])
@pytest.mark.parametrize("keep_orig_idx", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
def test_fixed_total_num_blocks_even_split(index_type, keep_orig_idx, sequence):
    """
    2 features, batch_size=3, my_size=4, total_num_blocks 为 my_size 的整数倍。
    覆盖：total_num_blocks 均匀可整除场景。
    """
    my_size = 4
    block_sizes = torch.tensor([3, 5], dtype=index_type)
    total_num_blocks = torch.tensor([8, 12], dtype=index_type)
    lengths = torch.tensor([1, 2, 3, 2, 1, 0], dtype=index_type)
    indices = torch.tensor([5, 0, 11, 2, 14, 23, 9, 55, 3], dtype=index_type)

    kwargs_cpu = _op_kwargs(
        lengths=lengths, indices=indices, block_sizes=block_sizes,
        my_size=my_size, sequence=sequence, keep_orig_idx=keep_orig_idx,
        total_num_blocks=total_num_blocks,
    )
    kwargs_npu = _make_npu_kwargs(kwargs_cpu, DEVICE, ['lengths', 'indices', 'block_sizes', 'total_num_blocks'])
    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)


@pytest.mark.parametrize("index_type", [torch.int, torch.long])
@pytest.mark.parametrize("keep_orig_idx", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
def test_fixed_total_num_blocks_raw_id_mode(index_type, keep_orig_idx, sequence):
    """
    3 features, batch_size=2, my_size=3, block_sizes 全 0 (raw id 模式)。
    覆盖：block_size=0 时按 total_num_blocks 直接对 raw id 分桶。
    """
    my_size = 3
    block_sizes = torch.tensor([0, 0, 0], dtype=index_type)
    total_num_blocks = torch.tensor([9, 12, 6], dtype=index_type)
    lengths = torch.tensor([3, 0, 2, 1, 4, 2], dtype=index_type)
    indices = torch.tensor([0, 8, 3, 11, 5, 2, 1, 4, 3, 5, 0, 1], dtype=index_type)

    kwargs_cpu = _op_kwargs(
        lengths=lengths, indices=indices, block_sizes=block_sizes,
        my_size=my_size, sequence=sequence, keep_orig_idx=keep_orig_idx,
        total_num_blocks=total_num_blocks,
    )
    kwargs_npu = _make_npu_kwargs(kwargs_cpu, DEVICE, ['lengths', 'indices', 'block_sizes', 'total_num_blocks'])
    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)


@pytest.mark.parametrize("index_type", [torch.int, torch.long])
@pytest.mark.parametrize("keep_orig_idx", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
def test_fixed_uneven_pos_with_block_sizes(index_type, keep_orig_idx, sequence):
    """
    2 features, batch_size=3, my_size=3, 非均匀 block_bucketize_pos + 非零 block_sizes。
    覆盖：block_bucketize_pos 做变长桶边界 + total_num_blocks 联合路径。
    """
    my_size = 3
    block_sizes = torch.tensor([3, 5], dtype=index_type)
    total_num_blocks = torch.tensor([9, 9], dtype=index_type)
    lengths = torch.tensor([2, 0, 3, 1, 2, 1], dtype=index_type)
    indices = torch.tensor([1, 7, 4, 12, 26, 0, 8, 20, 3], dtype=index_type)
    block_bucketize_pos_cpu = [
        torch.tensor([0, 3, 10, 15], dtype=index_type),
        torch.tensor([0, 6, 14, 21], dtype=index_type),
    ]

    kwargs_cpu = _op_kwargs(
        lengths=lengths, indices=indices, block_sizes=block_sizes,
        my_size=my_size, sequence=sequence, keep_orig_idx=keep_orig_idx,
        total_num_blocks=total_num_blocks, block_bucketize_pos=block_bucketize_pos_cpu,
    )
    kwargs_npu = _make_npu_kwargs(
        kwargs_cpu, DEVICE,
        ['lengths', 'indices', 'block_sizes', 'total_num_blocks'],
        list_keys=['block_bucketize_pos'],
    )
    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)


@pytest.mark.parametrize("index_type", [torch.int, torch.long])
@pytest.mark.parametrize("keep_orig_idx", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
def test_fixed_uneven_pos_raw_id_mode(index_type, keep_orig_idx, sequence):
    """
    2 features, batch_size=2, my_size=4, block_sizes 全 0 + block_bucketize_pos。
    覆盖：raw id 模式下使用 pos 表做非均匀分桶。
    """
    my_size = 4
    block_sizes = torch.tensor([0, 0], dtype=index_type)
    total_num_blocks = torch.tensor([8, 16], dtype=index_type)
    lengths = torch.tensor([3, 2, 1, 4], dtype=index_type)
    indices = torch.tensor([0, 7, 3, 15, 1, 10, 2, 14, 6, 9], dtype=index_type)
    block_bucketize_pos_cpu = [
        torch.tensor([0, 1, 3, 6, 8], dtype=index_type),
        torch.tensor([0, 2, 7, 11, 16], dtype=index_type),
    ]

    kwargs_cpu = _op_kwargs(
        lengths=lengths, indices=indices, block_sizes=block_sizes,
        my_size=my_size, sequence=sequence, keep_orig_idx=keep_orig_idx,
        total_num_blocks=total_num_blocks, block_bucketize_pos=block_bucketize_pos_cpu,
    )
    kwargs_npu = _make_npu_kwargs(
        kwargs_cpu, DEVICE,
        ['lengths', 'indices', 'block_sizes', 'total_num_blocks'],
        list_keys=['block_bucketize_pos'],
    )
    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)


@pytest.mark.parametrize("index_type", [torch.int, torch.long])
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("bucketize_pos", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
def test_fixed_variable_batch_sizes(index_type, has_weight, bucketize_pos, sequence):
    """
    3 features, 各 feature batch_size 不同 (2,3,1), my_size=3。
    覆盖：batch_size_per_feature 变长 + 不同 feature 不同行数。
    """
    lengths = torch.tensor([1, 3, 2, 0, 1, 2], dtype=index_type)
    indices = torch.tensor([4, 13, 7, 20, 1, 9, 25, 6, 15], dtype=index_type)
    batch_sizes = torch.tensor([2, 3, 1], dtype=index_type)
    block_sizes = torch.tensor([6, 9, 14], dtype=index_type)
    my_size = 3
    max_B = int(batch_sizes.max().item())
    weights = torch.linspace(-0.5, 0.5, indices.numel(), dtype=torch.float32) if has_weight else None

    kwargs_cpu = _op_kwargs(
        lengths=lengths, indices=indices, block_sizes=block_sizes,
        my_size=my_size, weights=weights, bucketize_pos=bucketize_pos,
        sequence=sequence, batch_size_per_feature=batch_sizes, max_B=max_B,
    )
    kwargs_npu = _make_npu_kwargs(
        kwargs_cpu, DEVICE,
        ['lengths', 'indices', 'block_sizes', 'weights', 'batch_size_per_feature'],
    )
    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)


@pytest.mark.parametrize("index_type", [torch.int, torch.long])
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("bucketize_pos", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
def test_fixed_variable_batch_with_pos(index_type, has_weight, bucketize_pos, sequence):
    """
    2 features, batch_size_per_feature=[3,2], my_size=2, 带 block_bucketize_pos。
    覆盖：变长 batch + pos 表联合路径。
    """
    lengths = torch.tensor([2, 0, 1, 3, 1], dtype=index_type)
    indices = torch.tensor([3, 11, 5, 14, 19, 7, 2], dtype=index_type)
    batch_sizes = torch.tensor([3, 2], dtype=index_type)
    block_sizes = torch.tensor([7, 12], dtype=index_type)
    my_size = 2
    max_B = int(batch_sizes.max().item())
    weights = torch.linspace(0.2, 1.4, indices.numel(), dtype=torch.float32) if has_weight else None
    block_bucketize_pos_cpu = [
        torch.tensor([0, 4, 10], dtype=index_type),
        torch.tensor([0, 8, 18], dtype=index_type),
    ]

    kwargs_cpu = _op_kwargs(
        lengths=lengths, indices=indices, block_sizes=block_sizes,
        my_size=my_size, weights=weights, bucketize_pos=bucketize_pos,
        sequence=sequence, batch_size_per_feature=batch_sizes, max_B=max_B,
        block_bucketize_pos=block_bucketize_pos_cpu,
    )
    kwargs_npu = _make_npu_kwargs(
        kwargs_cpu, DEVICE,
        ['lengths', 'indices', 'block_sizes', 'weights', 'batch_size_per_feature'],
        list_keys=['block_bucketize_pos'],
    )
    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)


@pytest.mark.parametrize("index_type", [torch.int, torch.long])
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("bucketize_pos", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("my_size", [3, 128, 200, 512])
def test_fixed_large_random_stress(index_type, has_weight, bucketize_pos, sequence, my_size):
    """
    6 features, 1536 行, 平均长度 ~36, 随机 indices。
    覆盖：大数据量压力测试 + 大 my_size（含 pow2 与非 pow2）。
    """
    num_features = 6
    block_size = 7
    num_rows = 1536
    avg_len = 36
    length_list = [max(0, int(random.gauss(mu=avg_len, sigma=2.0))) for _ in range(num_rows)]
    total_len = sum(length_list)
    block_sizes = torch.tensor([block_size] * num_features, dtype=index_type)
    lengths = torch.tensor(length_list, dtype=index_type)
    indices = torch.randint(0, my_size * block_size, (total_len,), dtype=index_type)
    weights = torch.rand((total_len,), dtype=torch.float32) if has_weight else None

    kwargs_cpu = _op_kwargs(
        lengths=lengths, indices=indices, block_sizes=block_sizes,
        my_size=my_size, weights=weights,
        bucketize_pos=bucketize_pos, sequence=sequence,
    )
    kwargs_npu = _make_npu_kwargs(kwargs_cpu, DEVICE, ['lengths', 'indices', 'block_sizes', 'weights'])
    _validate_npu_matches_cpu(kwargs_cpu, kwargs_npu)
