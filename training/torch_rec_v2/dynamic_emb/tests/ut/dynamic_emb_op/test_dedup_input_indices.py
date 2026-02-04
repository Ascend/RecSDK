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
import time
from typing import Optional, Tuple, Dict, List

import pytest
import torch
import torch_npu
import numpy as np

import dynamic_emb_extensions


def binary_search_last_le_idx(arr: np.ndarray, target: int) -> int:
    start = 0
    end = len(arr)
    while start < end:
        middle = start + (end - start) // 2
        if arr[middle] <= target:
            start = middle + 1
        else:
            end = middle
    return start - 1 if start > 0 else -1


def get_table_range_cpu(offsets: torch.Tensor, feature_offsets: torch.Tensor) -> torch.Tensor:
    offsets_np = offsets.cpu().numpy().astype(np.int64)
    feature_offsets_np = feature_offsets.cpu().numpy().astype(np.int64)
    table_num = len(feature_offsets_np) - 1
    feature_num_x_batch = len(offsets_np) - 1

    table_range = np.zeros_like(feature_offsets_np)
    if table_num == 0:
        table_range[0] = 0
        return torch.tensor(table_range, dtype=torch.int64)

    num_feature = feature_offsets_np[table_num]
    batch = feature_num_x_batch // num_feature if num_feature != 0 else 0

    for global_tid in range(table_num + 1):
        if global_tid < table_num + 1:
            feature_offset = feature_offsets_np[global_tid]
            feature_per_batch_offset = int(feature_offset * batch)
            table_range[global_tid] = offsets_np[feature_per_batch_offset]

    return torch.tensor(table_range, dtype=torch.int64)


def unique_op_impl_cpu(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keys_np = keys.cpu().numpy().astype(np.int64)
    total_length = len(keys_np)

    hash_table = dict()  # key: (unique_idx, count)
    unique_key_list = []
    restore_index = np.zeros(total_length, dtype=np.int64)
    count = []
    unique_src_indices = []
    global_counter = 0

    for idx in range(total_length):
        key = keys_np[idx]
        if key not in hash_table:
            hash_table[key] = (global_counter, 1)
            unique_key_list.append(key)
            unique_src_indices.append(idx)
            count.append(1)
            restore_index[idx] = global_counter
            global_counter += 1
        else:
            unique_idx, cnt = hash_table[key]
            hash_table[key] = (unique_idx, cnt + 1)
            count[unique_idx] += 1
            restore_index[idx] = unique_idx

    unique_key = torch.tensor(unique_key_list, dtype=torch.int64)
    restore_index = torch.tensor(restore_index, dtype=torch.int64)
    count = torch.tensor(count, dtype=torch.int64)
    unique_src_indices = torch.tensor(unique_src_indices, dtype=torch.int32)

    return unique_key, restore_index, count, unique_src_indices


def segmented_unique_op_cpu(keys: torch.Tensor, segment_range: torch.Tensor) -> tuple:
    """对齐NPU的segmented_unique_op逻辑"""
    keys = keys.cpu().to(torch.int64)
    segment_range = segment_range.cpu().to(torch.int64)
    table_num = segment_range.numel() - 1
    h_segment_range = segment_range.clone()
    keys_num = keys.size(0)

    tmp_unique_indices = [torch.empty_like(keys) for _ in range(table_num)]
    unique_pos_vec = []
    unique_cnt_vec = []

    d_unique_nums = torch.empty(table_num, dtype=torch.int64)
    inverse_idx = torch.empty(keys_num, dtype=torch.int64)
    h_unique_indices_table_range = torch.zeros(table_num + 1, dtype=torch.int64)

    # 分段去重
    global_offset = 0
    for i in range(table_num):
        indices_begin = h_segment_range[i].item()
        indices_end = h_segment_range[i + 1].item()
        indices_length = indices_end - indices_begin

        if indices_length == 0:
            d_unique_nums[i] = 0
            h_unique_indices_table_range[i + 1] = h_unique_indices_table_range[i]
        else:
            tmp_indices = keys[indices_begin:indices_end].contiguous()
            tmp_inverse_idx = inverse_idx[indices_begin:indices_end]
            tmp_unique, tmp_restore_index, tmp_count, tmp_pos = unique_op_impl_cpu(tmp_indices)
            global_restore_idx = tmp_restore_index + global_offset
            tmp_inverse_idx.copy_(global_restore_idx)

            tmp_unique_num = tmp_unique.size(0)
            tmp_unique_indices[i][:tmp_unique_num].copy_(tmp_unique)
            unique_pos_vec.append(tmp_pos)
            unique_cnt_vec.append(tmp_count)
            d_unique_nums[i] = tmp_unique_num
            h_unique_indices_table_range[i + 1] = h_unique_indices_table_range[i] + tmp_unique_num
            global_offset += tmp_unique_num

    d_unique_indices_table_range = torch.zeros(table_num + 1, dtype=torch.int64)
    d_unique_indices_table_range.copy_(h_unique_indices_table_range)

    # 合并唯一keys
    num_unique_total = h_unique_indices_table_range[-1].item()
    unique_keys = torch.empty(num_unique_total, dtype=torch.int64)
    unique_src_indices = torch.empty(num_unique_total, dtype=torch.int32)
    unique_cnts = torch.empty(num_unique_total, dtype=torch.int64)
    unique_embs_offset = 0

    for i in range(table_num):
        tmp_unique_num = h_unique_indices_table_range[i + 1].item() - h_unique_indices_table_range[i].item()
        if tmp_unique_num != 0:
            unique_keys[unique_embs_offset: unique_embs_offset + tmp_unique_num] = \
                tmp_unique_indices[i][:tmp_unique_num]
            unique_src_indices[unique_embs_offset: unique_embs_offset + tmp_unique_num] = \
                unique_pos_vec[i][:tmp_unique_num]
            unique_cnts[unique_embs_offset: unique_embs_offset + tmp_unique_num] = \
                unique_cnt_vec[i][:tmp_unique_num]
            unique_embs_offset += tmp_unique_num

    return (
        unique_keys,
        inverse_idx,
        d_unique_indices_table_range,
        h_unique_indices_table_range,
        unique_cnts,
        unique_src_indices,
    )


def get_new_length_and_offsets_cpu(
    d_unique_offsets: torch.Tensor,
    d_table_offsets_in_feature: torch.Tensor,
    new_offsets: torch.Tensor,
    new_lengths: torch.Tensor,
    local_batch_size: int,
):
    d_unique_offsets_np = d_unique_offsets.cpu().numpy().astype(np.int64)
    d_table_offsets_in_feature_np = d_table_offsets_in_feature.cpu().numpy().astype(np.int64)
    new_lengths_size = new_lengths.size(0)
    table_num = len(d_table_offsets_in_feature_np) - 1
    new_offsets_np = new_offsets.cpu().numpy()
    new_lengths_np = new_lengths.cpu().numpy()

    for i in range(new_lengths_size):
        # 1. 计算feature_id和table_id
        feature_id = i // local_batch_size
        table_id = binary_search_last_le_idx(d_table_offsets_in_feature_np, feature_id)

        # 2. 计算table_feature_count/table_buckets/bucket_id
        table_feature_count = d_table_offsets_in_feature_np[table_id + 1] - d_table_offsets_in_feature_np[table_id]
        table_buckets = table_feature_count * local_batch_size
        bucket_id = i - (d_table_offsets_in_feature_np[table_id] * local_batch_size)

        # 3. 计算unique_num
        unique_num = d_unique_offsets_np[table_id + 1] - d_unique_offsets_np[table_id]

        # 4. 分配bucket长度和偏移
        bucket_base = unique_num // table_buckets
        bucket_remainder = unique_num % table_buckets
        tmp_length = bucket_base
        tmp_offset = d_unique_offsets_np[table_id]

        if bucket_id < bucket_remainder:
            tmp_length += 1

        tmp_offset += (bucket_id * bucket_base) + (bucket_id if bucket_id < bucket_remainder else bucket_remainder)
        new_lengths_np[i] = tmp_length
        new_offsets_np[i] = tmp_offset

        if i == new_lengths_size - 1:
            new_offsets_np[new_lengths_size] = tmp_offset + tmp_length

    new_offsets.copy_(torch.tensor(new_offsets_np, dtype=new_offsets.dtype))
    new_lengths.copy_(torch.tensor(new_lengths_np, dtype=new_lengths.dtype))


def dedup_input_indices_cpu(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    d_table_offsets_in_feature: torch.Tensor,
    table_num: int,
    local_batch_size: int,
    reverse_idx: torch.Tensor,
    d_unique_nums: torch.Tensor,
    d_unique_offsets: torch.Tensor,
    unique_idx: list[torch.Tensor],
    new_offsets: torch.Tensor,
    new_lengths: torch.Tensor,
):
    # 计算segment_range
    segment_range = get_table_range_cpu(offsets, d_table_offsets_in_feature)

    # 分段去重
    seg_unique_result = segmented_unique_op_cpu(indices, segment_range)
    unique_keys = seg_unique_result[0]
    inverse_idx = seg_unique_result[1]
    d_unique_indices_range = seg_unique_result[2]
    h_unique_indices_range = seg_unique_result[3]
    reverse_idx.copy_(inverse_idx)

    # 填充unique_idx
    for i in range(table_num):
        start = h_unique_indices_range[i].item()
        end = h_unique_indices_range[i + 1].item()
        len_ = end - start
        if len_ > 0:
            unique_idx[i][:len_].copy_(unique_keys[start:end])

    # 计算d_unique_nums
    for i in range(table_num):
        start = h_unique_indices_range[i].item()
        end = h_unique_indices_range[i + 1].item()
        unique_num = end - start
        d_unique_nums[i] = unique_num

    d_unique_offsets.copy_(d_unique_indices_range)
    get_new_length_and_offsets_cpu(
        d_unique_offsets, d_table_offsets_in_feature, new_offsets, new_lengths, local_batch_size
    )

    return (reverse_idx, d_unique_nums, d_unique_offsets, unique_idx, new_offsets, new_lengths)


def dedup_input_indices_npu(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    d_table_offsets_in_feature: torch.Tensor,
    table_num: int,
    local_batch_size: int,
    reverse_idx: torch.Tensor,
    d_unique_nums: torch.Tensor,
    d_unique_offsets: torch.Tensor,
    unique_idx: list[torch.Tensor],
    new_offsets: torch.Tensor,
    new_lengths: torch.Tensor,
):

    indices_npu = indices.npu()
    offsets_npu = offsets.npu()
    d_table_offsets_in_feature_npu = d_table_offsets_in_feature.npu()
    reverse_idx_npu = reverse_idx.npu()
    d_unique_nums_npu = d_unique_nums.npu()
    d_unique_offsets_npu = d_unique_offsets.npu()
    unique_idx_npu = [t.npu() for t in unique_idx]
    new_offsets_npu = new_offsets.npu()
    new_lengths_npu = new_lengths.npu()

    dynamic_emb_extensions.dedup_input_indices_op(
        indices=indices_npu,
        offsets=offsets_npu,
        d_table_offsets_in_feature=d_table_offsets_in_feature_npu,
        table_num=table_num,
        local_batch_size=local_batch_size,
        reverse_idx=reverse_idx_npu,
        d_unique_nums=d_unique_nums_npu,
        d_unique_offsets=d_unique_offsets_npu,
        unique_idx=unique_idx_npu,
        new_offsets=new_offsets_npu,
        new_lengths=new_lengths_npu,
    )

    reverse_idx.copy_(reverse_idx_npu.cpu())
    d_unique_nums.copy_(d_unique_nums_npu.cpu())
    d_unique_offsets.copy_(d_unique_offsets_npu.cpu())
    unique_idx = [t.cpu() for t in unique_idx_npu]
    new_offsets.copy_(new_offsets_npu.cpu())
    new_lengths.copy_(new_lengths_npu.cpu())

    return (reverse_idx, d_unique_nums, d_unique_offsets, unique_idx, new_offsets, new_lengths)


def validate_reverse_idx(
    indices: torch.Tensor,
    reverse_idx: torch.Tensor,
    global_unique_keys: torch.Tensor,
    segment_range: torch.Tensor,
    table_num: int,
    d_unique_nums: torch.Tensor,
):
    indices_np = indices.cpu().numpy()
    reverse_idx_np = reverse_idx.cpu().numpy()
    d_unique_nums_np = d_unique_nums.cpu().numpy()

    valid_unique_segments = []
    for i in range(table_num):
        unique_num = d_unique_nums_np[i]
        if unique_num > 0:
            valid_segment = global_unique_keys[i][:unique_num].cpu()
            valid_unique_segments.append(valid_segment)

    if valid_unique_segments:
        global_unique_keys_cat = torch.cat(valid_unique_segments, dim=0)
        global_unique_keys_np = global_unique_keys_cat.numpy()
    else:
        global_unique_keys_np = np.array([], dtype=np.int64)

    restored_indices_np = global_unique_keys_np[reverse_idx_np]

    if not np.array_equal(restored_indices_np, indices_np):
        print(f"\n===== 全局reverse_idx还原失败 =====")
        print(f"原始前10个值: {indices_np[:10]}")
        print(f"还原前10个值: {restored_indices_np[:10]}")
        # 打印前5个差异位置
        diff_mask = restored_indices_np != indices_np
        diff_indices = np.where(diff_mask)[0]
        if len(diff_indices) > 0:
            top5_diff_indices = diff_indices[:5]
            print(f"前5个差异索引: {top5_diff_indices}")
            print(f"原始差异值: {indices_np[top5_diff_indices]}")
            print(f"还原差异值: {restored_indices_np[top5_diff_indices]}")
            print(f"对应reverse_idx值: {reverse_idx_np[top5_diff_indices]}")
            valid_reverse_idx = reverse_idx_np[top5_diff_indices]
            valid_mask = valid_reverse_idx < len(global_unique_keys_np)
            if np.any(valid_mask):
                print(f"对应unique_keys值: {global_unique_keys_np[valid_reverse_idx[valid_mask]]}")
            else:
                print(
                    f"差异位置的reverse_idx({valid_reverse_idx})超出global_unique_keys长度({len(global_unique_keys_np)})"
                )
        assert False, "全局reverse_idx还原后与原始数据不一致"
    else:
        print(f"✅ 全局reverse_idx还原校验通过（总长度: {len(indices_np)}）")


@pytest.mark.parametrize("table_num", [1, 10, 50, 100])
@pytest.mark.parametrize("local_batch_size", [4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.int64])
def test_dedup_input_indices_cpu_npu(table_num, local_batch_size, dtype):
    torch.manual_seed(42)

    # table_num=1 → [0,2]；table_num=2 → [0,2,4]；table_num=3 → [0,2,4,6]
    d_table_offsets_in_feature = torch.tensor([i * 2 for i in range(table_num + 1)], dtype=dtype)
    num_feature_total = d_table_offsets_in_feature[-1].item()

    # 构造offsets：长度=num_feature_total * local_batch_size + 1，步长10
    offsets_length = num_feature_total * local_batch_size + 1
    offsets = torch.arange(0, offsets_length * 10, 10, dtype=dtype)

    # 构造indices：长度=offsets[-1].item()，填充重复值
    indices_length = offsets[-1].item()
    indices = torch.randint(0, 10, (indices_length,), dtype=dtype)
    indices_original = indices.clone()

    reverse_idx_cpu = torch.empty_like(indices, dtype=dtype)
    reverse_idx_npu = torch.empty_like(indices, dtype=dtype)

    d_unique_nums_cpu = torch.empty(table_num, dtype=dtype)
    d_unique_nums_npu = torch.empty(table_num, dtype=dtype)

    d_unique_offsets_cpu = torch.empty(table_num + 1, dtype=dtype)
    d_unique_offsets_npu = torch.empty(table_num + 1, dtype=dtype)

    unique_idx_cpu = [torch.empty_like(indices, dtype=dtype) for _ in range(table_num)]
    unique_idx_npu = [torch.empty_like(indices, dtype=dtype) for _ in range(table_num)]

    new_lengths_size = num_feature_total * local_batch_size
    new_offsets_size = new_lengths_size + 1
    new_offsets_cpu = torch.empty(new_offsets_size, dtype=torch.int32)
    new_offsets_npu = torch.empty(new_offsets_size, dtype=torch.int32)
    new_lengths_cpu = torch.empty(new_lengths_size, dtype=torch.int32)
    new_lengths_npu = torch.empty(new_lengths_size, dtype=torch.int32)

    cpu_outputs = dedup_input_indices_cpu(
        indices=indices,
        offsets=offsets,
        d_table_offsets_in_feature=d_table_offsets_in_feature,
        table_num=table_num,
        local_batch_size=local_batch_size,
        reverse_idx=reverse_idx_cpu,
        d_unique_nums=d_unique_nums_cpu,
        d_unique_offsets=d_unique_offsets_cpu,
        unique_idx=unique_idx_cpu,
        new_offsets=new_offsets_cpu,
        new_lengths=new_lengths_cpu,
    )
    (reverse_idx_cpu, d_unique_nums_cpu, d_unique_offsets_cpu, unique_idx_cpu, new_offsets_cpu, new_lengths_cpu) = (
        cpu_outputs
    )

    npu_outputs = dedup_input_indices_npu(
        indices=indices,
        offsets=offsets,
        d_table_offsets_in_feature=d_table_offsets_in_feature,
        table_num=table_num,
        local_batch_size=local_batch_size,
        reverse_idx=reverse_idx_npu,
        d_unique_nums=d_unique_nums_npu,
        d_unique_offsets=d_unique_offsets_npu,
        unique_idx=unique_idx_npu,
        new_offsets=new_offsets_npu,
        new_lengths=new_lengths_npu,
    )
    (reverse_idx_npu, d_unique_nums_npu, d_unique_offsets_npu, unique_idx_npu, new_offsets_npu, new_lengths_npu) = (
        npu_outputs
    )

    def assert_tensor_equal(cpu_tensor, npu_tensor, name):
        cpu_tensor = cpu_tensor.to(torch.int64)
        npu_tensor = npu_tensor.to(torch.int64)
        if not torch.equal(cpu_tensor, npu_tensor):
            print(f"\n===== {name} 对比失败 =====")
            print(f"CPU前10个值: {cpu_tensor[:10]}")
            print(f"NPU前10个值: {npu_tensor[:10]}")
            diff_mask = cpu_tensor != npu_tensor
            diff_indices = torch.where(diff_mask)[0][:5]
            print(f"差异索引: {diff_indices}")
            print(f"CPU差异值: {cpu_tensor[diff_indices]}")
            print(f"NPU差异值: {npu_tensor[diff_indices]}")
            assert False, f"{name} CPU/NPU结果不一致"
        else:
            print(f"✅ {name} CPU/NPU结果一致")

    segment_range = get_table_range_cpu(offsets, d_table_offsets_in_feature)
    print("\n===== CPU reverse_idx还原校验 =====")
    validate_reverse_idx(indices_original, reverse_idx_cpu, unique_idx_cpu, segment_range, table_num, d_unique_nums_cpu)
    print("\n===== NPU reverse_idx还原校验 =====")
    validate_reverse_idx(indices_original, reverse_idx_npu, unique_idx_npu, segment_range, table_num, d_unique_nums_npu)

    assert_tensor_equal(d_unique_nums_cpu, d_unique_nums_npu, "d_unique_nums")
    assert_tensor_equal(d_unique_offsets_cpu, d_unique_offsets_npu, "d_unique_offsets")
    assert_tensor_equal(new_offsets_cpu, new_offsets_npu, "new_offsets")
    assert_tensor_equal(new_lengths_cpu, new_lengths_npu, "new_lengths")

    # unique_idx逐表对比
    for i in range(table_num):
        start = d_unique_offsets_cpu[i].item()
        end = d_unique_offsets_cpu[i + 1].item()
        len_unique = end - start

        if len_unique > 0:
            cpu_unique = unique_idx_cpu[i][:len_unique]
            npu_unique = unique_idx_npu[i][:len_unique]

            cpu_unique_sorted = torch.sort(cpu_unique)[0]
            npu_unique_sorted = torch.sort(npu_unique)[0]

            assert_tensor_equal(cpu_unique_sorted, npu_unique_sorted, f"unique_idx[table={i}] (sorted)")
        else:
            print(f"✅ unique_idx[table={i}] 无有效数据，跳过对比")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
