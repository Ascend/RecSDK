#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import torch


class GenTotalNumsBlocksType(Enum):
    NONE_TYPE = 0
    RAND_TYPE = 1
    MULTIPLE_TYPE = 2


@dataclass(frozen=True)
class PerfCase:
    name: str
    num_features: int
    batch_size: int
    min_length: int
    max_length: int
    block_size_range: Tuple[int, int]
    my_size: int
    dtype: torch.dtype


def _op_kwargs(**kwargs):
    args = dict(
        bucketize_pos=False,
        sequence=False,
        weights=None,
        batch_size_per_feature=None,
        max_B=-1,
        block_bucketize_pos=None,
        keep_orig_idx=False,
        total_num_blocks=None,
    )
    args.update(kwargs)
    return args


def _generate_case_tensors(case: PerfCase, need_weights: bool = False,
                           offset_dtype: torch.dtype = None, index_dtype: torch.dtype = None):
    o_dtype = offset_dtype if offset_dtype is not None else case.dtype
    i_dtype = index_dtype if index_dtype is not None else case.dtype

    lengths = torch.randint(
        case.min_length,
        case.max_length + 1,
        (case.num_features * case.batch_size,),
        dtype=o_dtype,
    )
    block_low, block_high = case.block_size_range
    block_sizes = torch.randint(
        block_low,
        block_high + 1,
        (case.num_features,),
        dtype=i_dtype,
    ).clamp_min(1)

    lengths_2d = lengths.view(case.num_features, case.batch_size)
    segments = []
    for feature_idx in range(case.num_features):
        limit = int(block_sizes[feature_idx].item()) * case.my_size
        limit = max(limit, 1)
        feature_lengths = lengths_2d[feature_idx]
        for row_length in feature_lengths:
            count = int(row_length.item())
            if count == 0:
                continue
            segments.append(torch.randint(0, limit, (count,), dtype=i_dtype))

    if segments:
        indices = torch.cat(segments)
    else:
        indices = torch.zeros(0, dtype=i_dtype)

    weights = None
    if need_weights == True:
        weights = torch.randn_like(indices, dtype=torch.float32).uniform_(-1.0, 1.0)

    return lengths, indices, block_sizes, weights


def _generate_total_num_blocks_tensors(block_sizes, my_size, genType: GenTotalNumsBlocksType):
    """
    生成 total_num_blocks 张量：
    - RAND_TYPE: 每个 feature 在 [my_size, 2*my_size] 内随机，尽量选非整倍数（my_size==1 时无法避免倍数）。
    - MULTIPLE_TYPE: 每个 feature 在 [1, 20] 的随机倍率上乘 my_size。
    """
    num_features = block_sizes.numel()
    device = block_sizes.device
    dtype = block_sizes.dtype
    base = int(my_size)
    base = base if base > 0 else 1

    if genType == GenTotalNumsBlocksType.RAND_TYPE:
        low = base
        high = max(base * 2, low + 1)
        vals = torch.randint(low, high + 1, (num_features,), device=device, dtype=dtype)
        # 尽量打散整倍数，my_size==1 时全部都会是倍数，保持现状即可
        if base > 1:
            multiples = (vals % base == 0)
            if torch.any(multiples):
                vals[multiples] = torch.clamp(vals[multiples] + 1, max=high)
    elif genType == GenTotalNumsBlocksType.MULTIPLE_TYPE:
        multipliers = torch.randint(1, 21, (num_features,), device=device, dtype=dtype)
        vals = multipliers * base
    elif genType == GenTotalNumsBlocksType.NONE_TYPE:
        return None
    else:
        raise ValueError(f"Unsupported GenTotalNumsBlocksType: {genType}")

    return vals.contiguous()


# batch_size_per_feature
# 形状：1D 张量，长度必须等于 block_sizes.numel()（每个 feature 一个值）。
# 类型：dtype 必须与 lengths/indices/block_sizes 一致（int32 或 int64）。
# 数值：所有值 > 0，且所有元素之和必须等于 lengths.numel()（因为 lengths 按 feature 展平成一维）。
def _generate_batch_size_per_feature_and_max_B(block_sizes, batch_size: int, max_scale: int = 2):
    """
    生成 batch_size_per_feature 张量及对应的 max_B：
    - 产出的总和固定为 num_features * batch_size，与现有 length 生成逻辑保持一致；
    - 每个 feature 的 batch size 在 [max(1, batch_size/2), batch_size*max_scale] 之间随机；
    - 最后通过比例缩放和微调，保证总和一致且每个值 >=1。
    """
    num_features = block_sizes.numel()
    device = block_sizes.device
    dtype = block_sizes.dtype
    target_total = int(batch_size) * num_features
    if target_total <= 0 or num_features <= 0:
        raise ValueError("batch_size must be positive and num_features must be > 0")

    low = max(1, batch_size // 2)
    high = max(low + 1, batch_size * max_scale)
    rand_vals = torch.randint(low, high + 1, (num_features,), device=device, dtype=dtype)

    current_total = int(rand_vals.sum().item())

    # 将每个随机值乘以一个缩放因子（目标总和 / 当前总和），使其总和接近 target_total
    scaled = (rand_vals.float() * (target_total / current_total)).round().clamp_min(1)
    batch_size_per_feature = scaled.to(dtype)

    # 调整差值，确保总和精确匹配
    diff = target_total - int(batch_size_per_feature.sum().item())
    if diff != 0:
        step = 1 if diff > 0 else -1
        idx = 0
        while diff != 0:
            batch_size_per_feature[idx] = batch_size_per_feature[idx] + step
            if batch_size_per_feature[idx] < 1:
                batch_size_per_feature[idx] = torch.tensor(1, device=device, dtype=dtype)
            diff -= step
            idx = (idx + 1) % num_features

    max_B = int(batch_size_per_feature.max().item())
    return batch_size_per_feature.contiguous(), max_B


def _generate_block_bucketize_pos(block_sizes, my_size, append_device, min_step: int = 1):
    """
    为每个 feature 生成一组 block_bucketize_pos（长度固定为 my_size + 1，单调递增）。
    - 步长在 [min_step, max(block_size, 1)] 内随机，block_size 为 0 时使用步长下限；
    - 第一个位置为 0，其余为累加步长，保持有序。
    返回值为 python list，每个元素为 1D Tensor，dtype/device 与 block_sizes 一致。
    """
    num_features = block_sizes.numel()
    device = block_sizes.device
    dtype = block_sizes.dtype
    pos_list = []
    pos_append_dev_list = []
    for idx in range(num_features):
        blk_size = int(block_sizes[idx].item())
        step_high = max(blk_size, 1)
        steps = torch.randint(min_step, step_high + 1, (my_size,), device=device, dtype=dtype)
        pos = torch.empty((my_size + 1,), device=device, dtype=dtype)
        pos[0] = 0
        pos[1:] = torch.cumsum(steps, dim=0)
        pos_list.append(pos)
        pos_append_dev_list.append(pos.to(append_device))
    return pos_list, pos_append_dev_list


PERF_CASES = (
    PerfCase("small_batch_uniform_blocks", 4, 64, 0, 16, (8, 16), 2, torch.int32),
    PerfCase("medium_batch_large_blocks", 8, 128, 8, 32, (32, 64), 4, torch.int32),
    PerfCase("many_features_high_rank", 16, 64, 2, 48, (8, 24), 8, torch.int64),
    PerfCase("long_sequences_deep_batch", 2, 512, 64, 128, (64, 128), 2, torch.int64),
    PerfCase("tiny_batch_sparse", 2, 16, 0, 4, (2, 4), 1, torch.int32),
    PerfCase("tiny_batch_dense", 2, 16, 4, 12, (4, 12), 2, torch.int32),
    PerfCase("wide_feature_low_rank", 24, 32, 0, 8, (4, 8), 2, torch.int32),
    PerfCase("wide_feature_high_rank", 24, 32, 4, 20, (16, 32), 6, torch.int32),
    PerfCase("deep_batch_moderate_blocks", 6, 256, 4, 20, (16, 32), 3, torch.int32),
    PerfCase("deep_batch_small_blocks", 6, 256, 0, 6, (4, 8), 2, torch.int32),
    PerfCase("jagged_lengths_focus", 12, 128, 0, 64, (8, 16), 4, torch.int64),
    PerfCase("long_seq_many_features", 32, 64, 32, 96, (32, 64), 8, torch.int64),
    PerfCase("row_partition_heavy", 48, 32, 8, 40, (16, 24), 12, torch.int64),
    PerfCase("high_rank_sparse", 8, 96, 0, 12, (4, 8), 16, torch.int64),
    PerfCase("high_rank_dense", 8, 96, 16, 64, (16, 32), 16, torch.int64),
    PerfCase("mixed_precision_small", 10, 48, 2, 18, (6, 10), 3, torch.int32),
    PerfCase("mixed_precision_large", 10, 48, 8, 40, (24, 40), 3, torch.int32),
    PerfCase("memory_stress_indices", 4, 200, 0, 80, (64, 96), 4, torch.int64),
    PerfCase("memory_stress_offsets", 4, 200, 20, 120, (32, 64), 4, torch.int64),
    PerfCase("balanced_midrange", 12, 120, 4, 24, (12, 24), 6, torch.int32),
    PerfCase("balanced_high_dim", 12, 120, 16, 36, (24, 48), 6, torch.int32),
    PerfCase("high_variance_lengths", 5, 180, 0, 128, (16, 24), 5, torch.int64),
    PerfCase("long_tail_blocks", 5, 180, 0, 32, (32, 96), 5, torch.int64),
    PerfCase("massive_features_dense_rank8", 96, 128, 32, 128, (32, 64), 8, torch.int32),
    PerfCase("massive_features_dense_rank16", 96, 128, 64, 192, (64, 128), 16, torch.int32),
    PerfCase("ultra_batch_sparse_rank4", 8, 4096, 0, 64, (16, 32), 4, torch.int64),
    PerfCase("ultra_batch_dense_rank8", 8, 4096, 64, 256, (32, 96), 8, torch.int64),
    PerfCase("extreme_seq_rank16", 4, 2048, 128, 512, (64, 128), 16, torch.int64),
    PerfCase("extreme_seq_rank32", 4, 2048, 128, 512, (64, 128), 32, torch.int64),
    PerfCase("huge_batch_variable", 1, 4096, 0, 64, (16, 64), 4, torch.int32),
    PerfCase("mega_features_super_dense_rank32", 128, 256, 128, 512, (64, 128), 32, torch.int64),
    PerfCase("mega_batch_huge_blocks", 16, 8192, 128, 512, (128, 256), 16, torch.int64),
    PerfCase("extreme_rank64", 4, 4096, 32, 256, (64, 128), 64, torch.int64),
    PerfCase("extreme_dense_rank64", 12, 2048, 256, 768, (96, 192), 64, torch.int64),
)


__all__ = [
    "PerfCase",
    "PERF_CASES",
    "_op_kwargs",
    "_generate_case_tensors",
    "_generate_total_num_blocks_tensors",
    "_generate_batch_size_per_feature_and_max_B",
    "_generate_block_bucketize_pos",
    "GenTotalNumsBlocksType",
]
