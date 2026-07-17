#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import itertools
import random
import sysconfig
from dataclasses import dataclass

import pytest
import torch
import fbgemm_gpu  # noqa: F401
import torch_npu  # noqa: F401
import numpy as np

DEVICE = "npu:0"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)


set_seed(10000)


@dataclass
class TestData:
    """测试数据封装类，用于封装 expand_into_jagged_permute 的测试数据。"""

    permute_list: list
    length_per_table: list
    input_offsets: list
    output_offsets: list

    @property
    def output_size(self):
        """自动计算输出大小。"""
        return self.input_offsets[-1]


def expand_into_jagged_permute_ref(permute, length):
    """Python 参考实现。"""
    offsets = [0] + list(itertools.accumulate(length))
    output_permute = []
    for r in permute:
        output_permute.extend(range(offsets[r], offsets[r + 1]))
    return output_permute


def _calc_case_seed(num_tables, length_range):
    start, end = length_range
    seed = (num_tables * 1315423911 + start * 1000003 + end * 16807) & 0xFFFFFFFF
    return seed


def build_offsets(lengths):
    return [0] + list(itertools.accumulate(lengths))


def build_test_case(num_tables, length_range):
    rng = random.Random(_calc_case_seed(num_tables, length_range))
    min_length, max_length = length_range
    lengths = [rng.randint(min_length, max_length) for _ in range(num_tables)]
    permute_list = list(range(len(lengths)))
    rng.shuffle(permute_list)
    permuted_lengths = [lengths[idx] for idx in permute_list]
    input_offsets = build_offsets(lengths)
    output_offsets = build_offsets(permuted_lengths)
    return permute_list, lengths, input_offsets, output_offsets


def run_expand_into_jagged_permute(
    test_data: TestData,
    is_mxrec: bool,
    dtype=torch.int32,
):
    permute_tensor = torch.tensor(test_data.permute_list, dtype=dtype)
    input_offsets_tensor = torch.tensor(test_data.input_offsets, dtype=dtype)
    output_offsets_tensor = torch.tensor(test_data.output_offsets, dtype=dtype)

    ref_tensor = torch.tensor(
        expand_into_jagged_permute_ref(test_data.permute_list, test_data.length_per_table),
        dtype=dtype,
    )

    cpu_result = _call_op_cpu(
        permute_tensor,
        input_offsets_tensor,
        output_offsets_tensor,
        test_data.output_size,
    )

    npu_result = _call_op_npu(
        permute_tensor,
        input_offsets_tensor,
        output_offsets_tensor,
        test_data.output_size,
        is_mxrec=is_mxrec,
    )

    return cpu_result, npu_result.cpu(), ref_tensor


def _call_op_cpu(permute, input_offsets, output_offsets, output_size):
    result = torch.ops.fbgemm.expand_into_jagged_permute(permute, input_offsets, output_offsets, output_size)

    return result


def _call_op_npu(permute, input_offsets, output_offsets, output_size, is_mxrec):
    torch.npu.set_device(DEVICE)
    permute = permute.to(DEVICE)
    input_offsets = input_offsets.to(DEVICE)
    output_offsets = output_offsets.to(DEVICE)

    if is_mxrec:
        result = torch.ops.mxrec.expand_into_jagged_permute(permute, input_offsets, output_offsets, output_size)
    else:
        result = torch.ops.fbgemm.expand_into_jagged_permute(permute, input_offsets, output_offsets, output_size)

    return result.cpu()


# 表的数量范围
NUM_TABLES_LIST = [1, 5, 10, 20, 40, 60, 80, 100]
# 每张表的长度范围
LENGTH_RANGE_LIST = [
    (1, 200),
    (200, 1000),
    (1000, 2000),
    (2000, 5000),
    (5000, 10000),
]
# 数据类型
DTYPE_LIST = [torch.int32, torch.int64]
# 是否使用 mxrec 命名空间
IS_MXREC_LIST = [True, False]


@pytest.mark.parametrize("num_tables", NUM_TABLES_LIST)
@pytest.mark.parametrize("length_range", LENGTH_RANGE_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize("is_mxrec", IS_MXREC_LIST)
def test_expand_into_jagged_permute(num_tables, length_range, dtype, is_mxrec):
    """主测试：每张表的长度在 length_range 内随机生成。"""
    permute_list, length_per_table, input_offsets, output_offsets = build_test_case(num_tables, length_range)

    test_data = TestData(
        permute_list=permute_list,
        length_per_table=length_per_table,
        input_offsets=input_offsets,
        output_offsets=output_offsets,
    )

    cpu_result, npu_result, ref_tensor = run_expand_into_jagged_permute(
        test_data,
        is_mxrec,
        dtype=dtype,
    )

    assert torch.equal(cpu_result, ref_tensor), (
        f"CPU result mismatch: num_tables={num_tables}, length_range={length_range}, dtype={dtype}, is_mxrec={is_mxrec}"
    )
    assert torch.equal(npu_result, ref_tensor), (
        f"NPU result mismatch: num_tables={num_tables}, length_range={length_range}, dtype={dtype}, is_mxrec={is_mxrec}"
    )


# ruff: off
def test_expand_into_jagged_permute_exceptions():
    """测试 expand_into_jagged_permute 算子的异常校验"""
    # pylint: disable=unused-variable,line-too-long
    # 默认合法参数规格
    permute = torch.tensor([1, 0, 2], dtype=torch.int32, device=DEVICE)
    input_offsets = torch.tensor([0, 5, 10, 15], dtype=torch.int32, device=DEVICE)
    output_offsets = torch.tensor([0, 5, 10, 15], dtype=torch.int32, device=DEVICE)
    output_size = 15

    # 1. 测试维度检查（例如 permute 传入 2D 张量）
    permute_2d = torch.tensor([[1, 0, 2]], dtype=torch.int32, device=DEVICE)
    with pytest.raises(Exception) as ctx:
        torch.ops.mxrec.expand_into_jagged_permute(permute_2d, input_offsets, output_offsets, output_size)
    assert "The permute should be 1D" in str(ctx.value)

    # 2. 测试尺寸匹配（例如 input_offsets 长度不匹配 permute.numel() + 1）
    input_offsets_err = torch.tensor([0, 5, 10], dtype=torch.int32, device=DEVICE)
    with pytest.raises(Exception) as ctx:
        torch.ops.mxrec.expand_into_jagged_permute(permute, input_offsets_err, output_offsets, output_size)
    assert "must equal input_offsets.numel() - 1" in str(ctx.value)

    # 3. 测试 dtype 校验（例如 permute 为 int64，offsets 为 int32）
    permute_int64 = permute.to(torch.int64)
    with pytest.raises(Exception) as ctx:
        torch.ops.mxrec.expand_into_jagged_permute(permute_int64, input_offsets, output_offsets, output_size)
    assert "must have the same dtype" in str(ctx.value)


# ruff: on
