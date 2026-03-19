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

import sysconfig
from typing import NamedTuple
import pytest
import torch
import torch_npu
import fbgemm_gpu

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

SEED = 123


def set_seed(seed: int):
    torch.manual_seed(seed)

set_seed(SEED)


def make_offsets(num_segments: int, max_step: int, allow_empty: bool = True, dtype=torch.int32):
    """
    构造合法 offsets:
    - int32
    - 非递减
    - offsets[0]=0
    - 第 i 段区间: [offsets[i], offsets[i+1])，最后一段: [offsets[-1], range_size)
    - range_size = sum(steps)
    """
    g = torch.Generator(device="cpu")

    low = 0 if allow_empty else 1
    steps = torch.randint(low, max_step + 1, (num_segments,), generator=g, dtype=dtype)

    offsets = torch.zeros((num_segments,), dtype=dtype)
    if num_segments > 1:
        offsets[1:] = torch.cumsum(steps[:-1], dim=0)

    range_size = int(torch.sum(steps).item())
    if range_size == 0:
        range_size = 1
    return offsets, range_size


def get_golden(offsets: torch.Tensor, range_size: int):
    """
    - 输出 shape = [range_size], dtype 与 offsets 一致
    - 对每段 i:
        start = offsets[i]
        end   = offsets[i+1] if (i < n-1) else range_size
        result[start:end] = 0..(end-start-1)
    """
    offsets_cpu = offsets.to("cpu")   # 不改 dtype
    out_dtype = offsets_cpu.dtype
    n = offsets_cpu.numel()

    result = torch.zeros((range_size,), dtype=out_dtype)

    if range_size == 0:
        return result.cpu()

    for i in range(n):
        s = int(offsets_cpu[i].item())
        e = int(offsets_cpu[i + 1].item()) if i < n - 1 else int(range_size)
        if e > s:
            result[s:e] = torch.arange(e - s, dtype=out_dtype)

    return result.cpu()


def get_op(offsets: torch.Tensor, range_size: int, device: str):
    offsets = offsets.to(device)
    result = torch.ops.mxrec.offsets_range(offsets, range_size)
    return result.cpu().detach()


# ===================== 特殊场景：行长度分布极不均衡 =====================

def offsets_from_steps(steps: torch.Tensor, dtype: torch.dtype):
    """
    输入：
      steps: [num_segments]，每一行的长度，可包含 0
    返回：
      offsets: [num_segments]，满足 offsets[0]=0 且非递减
      range_size: sum(steps)；若为 0，则置为 1（保持与现有行为一致）
    """
    steps = steps.to("cpu").to(dtype)

    num_segments = steps.numel()
    offsets = torch.zeros((num_segments,), dtype=dtype)
    if num_segments > 1:
        offsets[1:] = torch.cumsum(steps[:-1], dim=0)

    range_size = int(torch.sum(steps).item())
    if range_size == 0:
        range_size = 1
    return offsets, range_size


class OneHugeRestSmallInput(NamedTuple):
    num_segments: int
    huge_len: int
    small_len: int = 1
    allow_empty: bool = True
    dtype: torch.dtype = torch.int32


class LongTailInput(NamedTuple):
    num_segments: int
    long_k: int
    long_len: int
    short_len: int = 1
    zero_ratio: float = 0.0
    allow_empty: bool = True
    dtype: torch.dtype = torch.int32


def make_offsets_one_huge_rest_small(inputs: OneHugeRestSmallInput):
    """
    场景1：一行特别长，其余行很短（当 allow_empty 且 small_len=0 时其余行可为空）
    """
    steps = torch.full((inputs.num_segments,), inputs.small_len, dtype=inputs.dtype)
    steps[0] = inputs.huge_len
    return offsets_from_steps(steps, inputs.dtype)


def make_offsets_long_tail(inputs: LongTailInput):
    """
    场景2：长尾分布
      - 前 long_k 行长度为 long_len（较长）
      - 其余行长度为 short_len（较短），或按 zero_ratio 的比例置为 0
    """
    if not inputs.allow_empty:
        zero_ratio = 0.0  # 不允许空行时，强制不置零
    else:
        zero_ratio = inputs.zero_ratio

    steps = torch.full((inputs.num_segments,), inputs.short_len, dtype=inputs.dtype)
    steps[:inputs.long_k] = inputs.long_len

    if inputs.allow_empty and zero_ratio > 0:
        tail_n = inputs.num_segments - inputs.long_k
        zero_n = int(tail_n * zero_ratio)
        if zero_n > 0:
            # 将尾部前 zero_n 行置 0（确定性处理，不引入随机性）
            steps[inputs.long_k:inputs.long_k + zero_n] = 0

    return offsets_from_steps(steps, inputs.dtype)


SPECIAL_CASES = [
    # ================== 场景1：单行超长，其余短行 ==================
    # 更小的 huge_len：测试“只有一点不均衡”
    ("one_huge_1024_h32K_rest1", lambda dtype: make_offsets_one_huge_rest_small(
        OneHugeRestSmallInput(num_segments=1024, huge_len=1 << 15, small_len=32, allow_empty=True, dtype=dtype)
    )),
    ("one_huge_4096_h128K_rest1", lambda dtype: make_offsets_one_huge_rest_small(
        OneHugeRestSmallInput(num_segments=4096, huge_len=1 << 17, small_len=32, allow_empty=True, dtype=dtype)
    )),

    # 更极端 huge_len：测试“单行超长”
    ("one_huge_4096_h1M_rest1", lambda dtype: make_offsets_one_huge_rest_small(
        OneHugeRestSmallInput(num_segments=4096, huge_len=1 << 20, small_len=32, allow_empty=True, dtype=dtype)
    )),
    ("one_huge_8192_h2M_rest1", lambda dtype: make_offsets_one_huge_rest_small(
        OneHugeRestSmallInput(num_segments=8192, huge_len=1 << 21, small_len=32, allow_empty=True, dtype=dtype)
    )),

    # 测试短行略长时的行为
    ("one_huge_4096_h256K_rest2", lambda dtype: make_offsets_one_huge_rest_small(
        OneHugeRestSmallInput(num_segments=4096, huge_len=1 << 18, small_len=64, allow_empty=True, dtype=dtype)
    )),
    ("one_huge_8192_h256K_rest4", lambda dtype: make_offsets_one_huge_rest_small(
        OneHugeRestSmallInput(num_segments=8192, huge_len=1 << 18, small_len=128, allow_empty=True, dtype=dtype)
    )),

    # rest=0（大量空行），huge 在中等水平：测试 allow_empty + 单行大块
    ("one_huge_16384_h64K_rest0", lambda dtype: make_offsets_one_huge_rest_small(
        OneHugeRestSmallInput(num_segments=1 << 14, huge_len=1 << 16, small_len=0, allow_empty=True, dtype=dtype)
    )),

    # ================== 场景2：长尾分布，少数长行+大量短行/空行 ==================
    # long_k 更小：极少数长行
    ("long_tail_4096_k1_L256K_short1_zero50%", lambda dtype: make_offsets_long_tail(
        LongTailInput(num_segments=4096, long_k=1, long_len=1 << 18, short_len=32, 
                      zero_ratio=0.5, allow_empty=True, dtype=dtype)
    )),
    ("long_tail_8192_k2_L128K_short1_zero80%", lambda dtype: make_offsets_long_tail(
        LongTailInput(num_segments=8192, long_k=2, long_len=1 << 17, short_len=32, 
                      zero_ratio=0.8, allow_empty=True, dtype=dtype)
    )),

    # long_k 中等：一些长行 + 大量短行
    ("long_tail_8192_k8_L64K_short1_zero50%", lambda dtype: make_offsets_long_tail(
        LongTailInput(num_segments=8192, long_k=8, long_len=1 << 16, short_len=32, 
                      zero_ratio=0.5, allow_empty=True, dtype=dtype)
    )),
    ("long_tail_16384_k16_L32K_short1_zero70%", lambda dtype: make_offsets_long_tail(
        LongTailInput(num_segments=1 << 14, long_k=16, long_len=1 << 15, short_len=32, 
                      zero_ratio=0.7, allow_empty=True, dtype=dtype)
    )),

    # short_len=0：尾部绝大多数为空，只有少数行有内容（长尾极端）
    ("long_tail_8192_k8_L32K_short0_zero90%", lambda dtype: make_offsets_long_tail(
        LongTailInput(num_segments=8192, long_k=8, long_len=1 << 15, short_len=0, 
                      zero_ratio=0.9, allow_empty=True, dtype=dtype)
    )),
    ("long_tail_16384_k32_L16K_short0_zero95%", lambda dtype: make_offsets_long_tail(
        LongTailInput(num_segments=1 << 14, long_k=32, long_len=1 << 14, short_len=0, 
                      zero_ratio=0.95, allow_empty=True, dtype=dtype)
    )),

    # long_len 较小但 long_k 较大：更“平缓”的长尾（很多中长行）
    ("long_tail_8192_k256_L1024_short1_zero0%", lambda dtype: make_offsets_long_tail(
        LongTailInput(num_segments=8192, long_k=256, long_len=1024, short_len=32, 
                      zero_ratio=0.0, allow_empty=True, dtype=dtype)
    )),
    ("long_tail_16384_k512_L2048_short1_zero20%", lambda dtype: make_offsets_long_tail(
        LongTailInput(num_segments=1 << 14, long_k=512, long_len=2048, short_len=32, 
                      zero_ratio=0.2, allow_empty=True, dtype=dtype)
    )),
]


@pytest.mark.parametrize("case_name,case_builder", SPECIAL_CASES)
@pytest.mark.parametrize("dtype", [torch.int32])
@pytest.mark.parametrize("device", ["npu"])
def test_offsets_range_special_cases(case_name, case_builder, dtype, device):
    offsets, range_size = case_builder(dtype)

    # 计算基准结果
    y_golden = get_golden(offsets, range_size)
    torch.npu.synchronize()

    # 自定义算子
    y_op = get_op(offsets, range_size, device)
    torch.npu.synchronize()

    # 结果校验
    assert torch.equal(y_golden, y_op)
