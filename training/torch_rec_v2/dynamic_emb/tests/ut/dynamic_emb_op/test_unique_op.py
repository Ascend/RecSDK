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


import sysconfig
import logging

import pytest
import torch
import torch_npu

import dynamic_emb_extensions


logging.basicConfig(level=logging.NOTSET)


@pytest.mark.parametrize("dtype", [torch.int64])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize("length", [1, 10, 100, 1000, 10000])
@pytest.mark.parametrize("range_max", [5, 50, 100])
def test_unique_op_consistency(dtype, device, length, range_max):
    """
    测试自定义 unique 算子结果的一致性。

    1. 验证 a[b] 是否能完全恢复原始输入。
    2. 验证 count c 是否与 a 中每个元素在恢复后张量中的出现次数一致。
    """
    device_id = int(device.split(":")[-1])
    # 1. 生成测试数据
    tensor_1d = torch.randint(0, range_max, (length,), dtype=dtype, device=device)

    # 2. 使用自定义算子计算结果
    a, b, c = dynamic_emb_extensions.unique_op(tensor_1d)
    # 校验 1: 验证可恢复性
    tensor_1d_cpu = tensor_1d.cpu()
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    c_cpu = c.cpu()
    s_cpu = a_cpu[b_cpu]

    assert (
        s_cpu.shape == tensor_1d_cpu.shape
    ), f"Shape mismatch after restoration: s.shape={s_cpu.shape}, tensor_1d.shape={tensor_1d_cpu.shape}"
    restore_match = torch.equal(s_cpu, tensor_1d_cpu)

    # 校验 2: 验证计数正确性
    count_match = True
    if restore_match:
        unique_elements, counts = torch.unique(s_cpu, sorted=False, return_counts=True)
        count_map = {elem.item(): cnt.item() for elem, cnt in zip(unique_elements, counts)}

        for i, a_item in enumerate(a_cpu):
            key = a_item.item()
            actual_count = c_cpu[i].item()

            if key not in count_map or count_map[key] != actual_count:
                count_match = False
                logging.error(f"Count mismatch for key {key}: got {actual_count}, expected {count_map.get(key, 0)}")
                break
    else:
        count_match = False

    # 打印调试信息（校验失败）
    if not restore_match or not count_match:
        logging.error(f"\nTest Failed for length={length}, range_max={range_max}, dtype={dtype}")

        if not restore_match:
            logging.error("  FAIL: Restoration check failed (a[b] != tensor_1d).")
            # 找出第一个不匹配的位置
            diff_indices = torch.where(s_cpu != tensor_1d_cpu)[0]
            if len(diff_indices) > 0:
                first_diff_idx = diff_indices[0].item()
                logging.error(f"    First mismatch at index {first_diff_idx}:")
                logging.error(f"      tensor_1d[{first_diff_idx}] = {tensor_1d_cpu[first_diff_idx]}")
                logging.error(f"      s[{first_diff_idx}] (a[b[{first_diff_idx}]]) = {s_cpu[first_diff_idx]}")
                logging.error(
                    f"      b[{first_diff_idx}] = {b_cpu[first_diff_idx]}, \
                    a[{b_cpu[first_diff_idx].item()}] = {a_cpu[b_cpu[first_diff_idx].item()]}"
                )
        if not count_match:
            logging.error("  FAIL: Count check failed (c does not match counts in restored tensor).")
            logging.error(f"    unique keys (a): {a_cpu}")
            logging.error(f"    counts (c):      {c_cpu}")
            logging.error(f"    counts in s:     {[(k, v) for k, v in sorted(count_map.items())]}")

    # 所有校验都通过
    assert (
        restore_match
    ), "Restoration check failed: The tensor reconstructed by a[b] is not identical to the original tensor."
    assert (
        count_match
    ), "Count check failed: The counts in 'c' do not match the occurrences of unique keys in the restored tensor."
