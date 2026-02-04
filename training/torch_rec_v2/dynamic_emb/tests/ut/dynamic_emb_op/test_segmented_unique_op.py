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
@pytest.mark.parametrize("length", [10, 100, 1000, 10000])
@pytest.mark.parametrize("range_max", [5, 50, 100])
@pytest.mark.parametrize("segment_num", [1, 3, 5])
def test_segmented_unique_op_consistency(dtype, device, length, range_max, segment_num):
    """
    测试分段去重算子(segmented_unique_op)结果正确性。

    1. 验证每个分段内部的unique操作是否正确。
    2. 验证 inverse_idx 是否能正确地从对应的分段唯一Key中恢复出原始的 keys。
    3. 验证 h_unique_indices_table_range 是否准确记录了各段唯一Key的范围。
    """
    device_id = int(device.split(":")[-1])

    # 1. 生成测试数据
    keys = torch.randint(0, range_max, (length,), dtype=dtype, device=device)
    segment_size = length // segment_num
    segment_offsets = [i * segment_size for i in range(segment_num + 1)]
    segment_offsets[-1] = length
    segment_range = torch.tensor(segment_offsets, dtype=torch.int64, device=device)

    # 2. 使用自定义算子计算结果
    unique_keys, inverse_idx, d_unique_indices_table_range, _ = dynamic_emb_extensions.segmented_unique_op(
        keys, segment_range
    )
    keys_cpu = keys.cpu()
    unique_keys_cpu = unique_keys.cpu()
    inverse_idx_cpu = inverse_idx.cpu()
    h_unique_indices_table_range_cpu = d_unique_indices_table_range.cpu()
    segment_range_cpu = segment_range.cpu()

    # 校验 1: 验证 inverse_idx 的正确性
    restored_keys_cpu = unique_keys_cpu[inverse_idx_cpu]
    # 每段还原后的数据与原始数据对比
    restore_match = torch.equal(restored_keys_cpu, keys_cpu)

    # 校验 2: 验证分段处理和 h_unique_indices_table_range 的正确性
    segment_match = True
    for i in range(segment_num):
        start = segment_range_cpu[i].item()
        end = segment_range_cpu[i + 1].item()

        if start == end:
            expected_unique_count = 0
        else:
            segment_keys = keys_cpu[start:end]
            expected_unique, _ = torch.unique(segment_keys, sorted=False, return_inverse=True)
            expected_unique_count = len(expected_unique)

            segment_unique_start = h_unique_indices_table_range_cpu[i].item()
            segment_unique_end = h_unique_indices_table_range_cpu[i + 1].item()
            actual_segment_unique = unique_keys_cpu[segment_unique_start:segment_unique_end]

            if not torch.equal(torch.sort(expected_unique)[0], torch.sort(actual_segment_unique)[0]):
                segment_match = False
                logging.error(f"Segment {i} unique keys mismatch.")
                logging.error(f"  Expected unique (sorted): {torch.sort(expected_unique)[0]}")
                logging.error(f"  Actual unique (sorted):   {torch.sort(actual_segment_unique)[0]}")
                break

        actual_unique_count = (
            h_unique_indices_table_range_cpu[i + 1].item() - h_unique_indices_table_range_cpu[i].item()
        )
        if actual_unique_count != expected_unique_count:
            segment_match = False
            logging.error(f"Segment {i} unique count mismatch.")
            logging.error(f"  Expected count: {expected_unique_count}")
            logging.error(f"  Actual count:   {actual_unique_count}")
            break

    # 4. 打印调试信息
    if not restore_match or not segment_match:
        logging.error(f"\nTest Failed for length={length}, range_max={range_max}, segment_num={segment_num}")

        if not restore_match:
            logging.error("  FAIL: Restoration check failed.")
            diff_indices = torch.where(restored_keys_cpu != keys_cpu)[0]
            if len(diff_indices) > 0:
                first_diff_idx = diff_indices[0].item()
                logging.error(f"    First mismatch at index {first_diff_idx}:")
                logging.error(f"      Original keys[{first_diff_idx}] = {keys_cpu[first_diff_idx]}")
                logging.error(f"      Restored keys[{first_diff_idx}] = {restored_keys_cpu[first_diff_idx]}")

                # 找出这个错误属于哪个分段
                seg_idx = next(
                    j for j in range(segment_num) if segment_range_cpu[j] <= first_diff_idx < segment_range_cpu[j + 1]
                )

                seg_unique_keys = unique_keys_cpu[
                    h_unique_indices_table_range_cpu[seg_idx]:h_unique_indices_table_range_cpu[seg_idx + 1]
                ]

                logging.error(f"    This index belongs to segment {seg_idx}.")
                logging.error(f"    Segment {seg_idx} unique keys: {seg_unique_keys}")

        if not segment_match:
            logging.error("  FAIL: Segment processing or h_unique_indices_table_range check failed.")

    # 断言所有校验都通过
    assert (
        restore_match
    ), "Restoration check failed: The tensor reconstructed \
          by per-segment unique_keys and inverse_idx is not identical to the original tensor 'keys'."
    assert (
        segment_match
    ), "Segment processing check failed: The unique keys in a segment do not match \
          the expected result, or h_unique_indices_table_range is incorrect."
