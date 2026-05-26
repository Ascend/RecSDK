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
import pytest
import torch
import fbgemm_gpu

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

DEVICE = "npu:0"


def get_golden(batch_size: int, csr_seg: torch.Tensor, values: torch.Tensor):
    """
    使用 FBGEMM CPU 实现生成 golden 结果。

    对于 FP16/BF16 输入，先用 FP32 计算再转回目标精度。
    因为 FBGEMM CPU 实现直接在低精度中累加（误差大），而 NPU 内部先 cast 到
    FP32 做 WholeReduceSum 再截断。大 shape 下两者差异会显著放大，因此用
    FP32 作为中间精度生成 golden，以匹配 NPU 的实际计算精度。
    """
    if values.dtype in (torch.float16, torch.bfloat16):
        golden_fp32 = torch.ops.fbgemm.segment_sum_csr(batch_size, csr_seg, values.float())
        return golden_fp32.to(values.dtype)
    return torch.ops.fbgemm.segment_sum_csr(batch_size, csr_seg, values)


def get_op(batch_size: int, csr_seg: torch.Tensor, values: torch.Tensor):
    """
    调用 NPU 实现。
    """
    return torch.ops.mxrec.segment_sum_csr(batch_size, csr_seg, values)


def generate_random_segment_sum_data(device, csr_type, v_type):
    """
    为 segment_sum_csr 算子生成随机测试数据。

    参数:
        device: torch.device 对象或字符串，如 "npu:0", "cpu"
        csr_type: torch.dtype 对象，csr_seg 张量的数据类型（如 torch.int32）
        v_type: torch.dtype 对象，values 张量的数据类型（如 torch.float32）

    返回:
        batch_size: int，batch 大小
        csr_seg: CSR 分段指针，shape=[num_segments+1], dtype=csr_type
        values: 数据张量，shape=[batch_size * total_elements_per_batch], dtype=v_type
    """
    batch_size_val = torch.randint(1, 33, (1,), device=device, dtype=csr_type).item()
    num_segments = torch.randint(2, 101, (1,), device=device, dtype=csr_type).item()
    segment_lengths = torch.randint(1, 101, (num_segments,), device=device, dtype=csr_type)

    csr_seg = torch.cat([torch.tensor([0], device=device, dtype=csr_type), segment_lengths.cumsum(dim=0)], dim=0)

    total_elements_per_batch = csr_seg[-1].item()
    total_values_length = batch_size_val * total_elements_per_batch

    if v_type.is_floating_point:
        values = torch.empty(total_values_length, device=device, dtype=v_type).uniform_(-5, 5)
    else:
        values = torch.randint(-5, 6, (total_values_length,), device=device, dtype=v_type)

    return batch_size_val, csr_seg, values


def _compute_tol(v_type: torch.dtype):
    if v_type == torch.float16:
        return 2 ** (-7), 2 ** (-7)
    elif v_type == torch.bfloat16:
        return 2 ** (-6), 2 ** (-6)
    elif v_type in (torch.int32, torch.int64):
        return 0, 0
    else:
        return 2 ** (-9), 2 ** (-9)


# -----------------------------------------------------------------------------
# 基础功能测试
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("csr_type", [torch.int32, torch.int64])
@pytest.mark.parametrize("v_type", [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64])
def test_segment_sum_csr_basic(csr_type, v_type):
    """batch_size=1 的基本分段求和测试。"""
    torch.npu.set_device(DEVICE)
    batch_size = 1
    csr_seg = torch.tensor([0, 2, 3, 6], dtype=csr_type)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=v_type)

    golden = get_golden(batch_size, csr_seg, values)
    npu_out = get_op(batch_size, csr_seg.to(DEVICE), values.to(DEVICE))

    rtol, atol = _compute_tol(v_type)
    torch.testing.assert_close(npu_out.cpu(), golden, rtol=rtol, atol=atol)


@pytest.mark.parametrize("csr_type", [torch.int32, torch.int64])
@pytest.mark.parametrize("v_type", [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64])
def test_segment_sum_csr_batch_size_gt1(csr_type, v_type):
    """batch_size > 1 的分段求和测试，验证 reshape 语义。"""
    torch.npu.set_device(DEVICE)
    batch_size = 2
    csr_seg = torch.tensor([0, 2, 3], dtype=csr_type)
    # 共 3 * 2 = 6 个 values，对应 2 段
    # 段 0: csr_seg[0]=0 -> csr_seg[1]=2, 取 values[0:4] = [1,2,3,4] -> sum=10
    # 段 1: csr_seg[1]=2 -> csr_seg[2]=3, 取 values[4:6] = [5,6] -> sum=11
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=v_type)

    golden = get_golden(batch_size, csr_seg, values)
    npu_out = get_op(batch_size, csr_seg.to(DEVICE), values.to(DEVICE))

    expected = torch.tensor([10.0, 11.0], dtype=v_type)
    torch.testing.assert_close(golden, expected, rtol=0, atol=0)

    rtol, atol = _compute_tol(v_type)
    torch.testing.assert_close(npu_out.cpu(), golden, rtol=rtol, atol=atol)


@pytest.mark.parametrize("csr_type", [torch.int32, torch.int64])
@pytest.mark.parametrize("v_type", [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64])
def test_segment_sum_csr_empty_input(csr_type, v_type):
    """空输入测试：batch_size=0, csr_seg=[0], values=[]。"""
    torch.npu.set_device(DEVICE)
    batch_size = 0
    csr_seg = torch.tensor([0], dtype=csr_type)
    values = torch.tensor([], dtype=v_type)

    golden = get_golden(batch_size, csr_seg, values)
    npu_out = get_op(batch_size, csr_seg.to(DEVICE), values.to(DEVICE))

    assert golden.numel() == 0
    assert npu_out.cpu().numel() == 0


@pytest.mark.parametrize("v_type", [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64])
def test_segment_sum_csr_single_segment(v_type):
    """仅有一段的分段求和。"""
    torch.npu.set_device(DEVICE)
    batch_size = 1
    csr_seg = torch.tensor([0, 5], dtype=torch.int32)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=v_type)

    golden = get_golden(batch_size, csr_seg, values)
    npu_out = get_op(batch_size, csr_seg.to(DEVICE), values.to(DEVICE))

    expected = torch.tensor([15.0], dtype=v_type)
    torch.testing.assert_close(golden, expected, rtol=0, atol=0)

    rtol, atol = _compute_tol(v_type)
    torch.testing.assert_close(npu_out.cpu(), golden, rtol=rtol, atol=atol)


# -----------------------------------------------------------------------------
# 随机参数化压力测试
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("csr_type", [torch.int32, torch.int64])
@pytest.mark.parametrize("v_type", [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64])
def test_segment_sum_csr_random(csr_type, v_type):
    """随机生成数据与 FBGEMM CPU golden 对比。"""
    torch.npu.set_device(DEVICE)
    batch_size, csr_seg, values = generate_random_segment_sum_data("cpu", csr_type, v_type)

    golden = get_golden(batch_size, csr_seg, values.to(torch.float32)).to(v_type)
    npu_out = get_op(batch_size, csr_seg.to(DEVICE), values.to(DEVICE))

    rtol, atol = _compute_tol(v_type)
    torch.testing.assert_close(npu_out.cpu(), golden, rtol=rtol, atol=atol)


# -----------------------------------------------------------------------------
# 大 shape 压力测试
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("v_type", [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64])
def test_segment_sum_csr_large_shape(v_type):
    """大 shape 压力测试：1000 段，每段约 100 个元素，batch_size=16。"""
    torch.npu.set_device(DEVICE)
    batch_size = 16
    num_segments = 1000
    elems_per_segment = 100

    segment_lengths = torch.full((num_segments,), elems_per_segment, dtype=torch.int32)
    csr_seg = torch.cat([torch.tensor([0], dtype=torch.int32), segment_lengths.cumsum(dim=0)], dim=0)

    total_values = batch_size * csr_seg[-1].item()
    if v_type.is_floating_point:
        values = torch.randn(total_values, dtype=v_type)
    else:
        values = torch.randint(-100, 101, (total_values,), dtype=v_type)

    golden = get_golden(batch_size, csr_seg, values)
    npu_out = get_op(batch_size, csr_seg.to(DEVICE), values.to(DEVICE))

    rtol, atol = _compute_tol(v_type)
    torch.testing.assert_close(npu_out.cpu(), golden, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
