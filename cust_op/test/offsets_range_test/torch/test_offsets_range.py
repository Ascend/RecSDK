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
import random
import pytest
import torch
import torch_npu
import fbgemm_gpu

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

SEED = 123


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)


def make_offsets(num_segments: int, max_step: int, allow_empty: bool = True, dtype=torch.int32):
    """
    根据给定分段数和最大步长构造合法 offsets:
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
    offsets_cpu = offsets.to("cpu")
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


def get_op(offsets: torch.Tensor, range_size: int, namespace: str = "mxrec"):
    offsets = offsets.to("npu")
    result = getattr(torch.ops, namespace).offsets_range(offsets, range_size)
    return result.cpu().detach()


@pytest.mark.parametrize("num_segments", [1, 8, 64, 512, 1024, 2048, 4096, 1 << 13, 1 << 14])
@pytest.mark.parametrize("max_step", [4, 16, 64, 256, 1024, 4096, 8192, 1 << 14, 1 << 16, 1 << 18])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_offsets_range(dtype, num_segments, max_step):
    # 构造随机输入
    offsets, range_size = make_offsets(
        num_segments=num_segments,
        max_step=max_step,
        allow_empty=True,
        dtype=dtype
    )

    # 计算基准结果
    y_golden = get_golden(offsets, range_size)
    torch.npu.synchronize()

    # 自定义算子
    y_op = get_op(offsets, range_size)
    torch.npu.synchronize()

    # 结果校验
    assert torch.equal(y_golden, y_op)


@pytest.mark.parametrize("num_segments,max_step", [(1, 4), (1024, 256), (1 << 14, 1 << 18)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_offsets_range_fbgemm(dtype, num_segments, max_step):
    # 验证通过 fbgemm 命名空间调用 offsets_range 算子的结果正确性
    offsets, range_size = make_offsets(
        num_segments=num_segments,
        max_step=max_step,
        allow_empty=True,
        dtype=dtype
    )

    y_golden = get_golden(offsets, range_size)
    torch.npu.synchronize()

    y_op = get_op(offsets, range_size, namespace="fbgemm")
    torch.npu.synchronize()

    assert torch.equal(y_golden, y_op)


def test_offsets_range_max_range_size():
    # 验证 README 中声明的 range_size 上界 2^32 在大内存场景下可执行
    offsets = torch.tensor([0, 1 << 31, (1 << 32) - 1], dtype=torch.int64)
    range_size = 1 << 32

    y_golden = get_golden(offsets, range_size)
    torch.npu.synchronize()

    y_op = get_op(offsets, range_size)
    torch.npu.synchronize()

    assert torch.equal(y_golden, y_op)
