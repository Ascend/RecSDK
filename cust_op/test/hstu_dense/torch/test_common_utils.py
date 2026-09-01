#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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
import dataclasses
import random
import sysconfig
from enum import Enum
import logging

import numpy as np
import torch
import torch_npu

BLOCK_HEIGHT: int = 256
MAX_NUM_TARGET: int = 512

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

device_id: int = 0
torch.npu.set_device(device_id)


class MaskType(int, Enum):
    TRIL = 0  # 下三角掩码
    TRIU = 1  # 上三角掩码
    NONE = 2  # 无掩码
    CUSTOM = 3  # 自定义掩码


def get_chip():
    return False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True  # 确保CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False  # 关闭CuDNN自动优化


def allclose(tensor: torch.Tensor, other: torch.Tensor, atol: float, ratio: float) -> bool:
    assert tensor.shape == other.shape
    diff = torch.abs(tensor - other) > atol
    diff_count = torch.sum(diff).tolist()
    return (diff_count / tensor.numel()) < ratio


def hstu_close_double(actual, out_ref, fp32_ref, try_allclose: bool = False, multiplier: int = 2) -> bool:
    # 算子输出
    actual = actual.reshape(-1)
    # 原生golden
    out_ref = out_ref.reshape(-1)
    # 高精度原生golden
    fp32_ref = fp32_ref.reshape(-1)
    assert fp32_ref.dtype == torch.float32, "fp32_ref should be float32"
    if try_allclose:
        try_allclose = torch.allclose(actual, out_ref)

    left_abs_max = (actual - fp32_ref).abs().max().item()
    right_abs_max = (out_ref - fp32_ref).abs().max().item()
    mid_max = (out_ref - actual).abs().max().item()

    if left_abs_max > multiplier * right_abs_max:
        print("left_abs_max=", left_abs_max)
        print("right_abs_max=", right_abs_max)
        print("actual-out_ref=", mid_max)
        print(
            f"[HSTU_CLOSE] assert fail: diff abs max: {left_abs_max:.6f}, threshold: {multiplier * right_abs_max:.6f},"
            f" multiplier: {multiplier}"
        )
    return (left_abs_max <= multiplier * right_abs_max) or (try_allclose)


def jagged_to_dense(jagged_tensor, seq_lens, head_nums, attn_dim):
    need_pad_seq = []
    offset = 0
    for seq_len in seq_lens:
        src_tensor = jagged_tensor[offset : offset + seq_len, :, :].reshape(seq_len, head_nums, attn_dim)
        need_pad_seq.append(src_tensor)
        offset = offset + seq_len

    dense_tensor = torch.nn.utils.rnn.pad_sequence(need_pad_seq, batch_first=True)
    return dense_tensor


def dense_to_jagged(q, dense_tensor, seq_lens):
    dense_dim = dense_tensor.shape[3]
    # tensor: [b_s, n, d]
    tensor = torch.zeros(q.shape[0], q.shape[1], dense_dim).cpu()

    offset = 0
    for batch_id, seq_len in enumerate(seq_lens):
        tensor[offset : offset + seq_len, :, :] = dense_tensor[batch_id, 0:seq_len, :, :]
        offset = offset + seq_len

    return tensor


def show_diff(golden: torch.Tensor, result: torch.Tensor, atol: float):
    if golden is None or result is None:
        return
    diff = torch.abs(golden - result) > atol

    cnt = 0
    last_offset = last_head = -1
    for offset, head, dim in torch.nonzero(diff):
        if offset == last_offset and head == last_head:
            continue
        last_offset, last_head, cnt = offset, head, cnt + 1
        logging.info("===== (%s, %s, %s) =====", offset, head, dim)
        logging.info("%s", golden[offset, head, dim : dim + 16])
        logging.info("%s", result[offset, head, dim : dim + 16])
        if cnt >= 5:
            break


@dataclasses.dataclass
class QKVShapeInfo:
    float_type: torch.dtype
    int_type: torch.dtype
    batch_size: int
    num_heads_q: int
    num_heads_k: int
    head_dim_qk: int
    head_dim_v: int
    max_seq_len: int
    min_seq_len: int = 1


@dataclasses.dataclass
class MaskGenInfo:
    mask_type: int | MaskType = MaskType.TRIL
    target_group_size: int = 0
    max_num_context: int = 0
    max_num_target: int = 0
    min_num_context: int = 0
    min_num_target: int = 0


def _check_int_valid(num):
    if not isinstance(num, int):
        return False
    if num <= 0:
        return False
    return True


def create_target_mask(num_target, target_group_size):
    row_indices = torch.arange(num_target).view(-1, 1)
    col_indices = torch.arange(num_target).view(1, -1)
    block_row = row_indices // target_group_size
    block_col = col_indices // target_group_size
    mask = (block_row == block_col).int()
    tril = torch.tril(torch.ones(num_target, num_target), diagonal=0).int()
    return tril & mask


def create_causal_mask(seqlen_q, seqlen_k=None, num_context=None, num_target=None, target_group_size=None):
    if seqlen_k is None:
        seqlen_k = seqlen_q
    mask = torch.tril(torch.ones(seqlen_q, seqlen_k), diagonal=(seqlen_k - seqlen_q))
    if _check_int_valid(num_context):
        num_target = 0 if num_target is None else num_target
        mask[:num_context, : seqlen_k - num_target] = 1
    if _check_int_valid(target_group_size) and _check_int_valid(num_target):
        mask[-num_target:, -num_target:] = create_target_mask(num_target, target_group_size)
    return mask


def get_golden(q, k, v, mask=None, attn_bias=None, mask_type=0, max_seq_len=None, silu_scale=None):
    """numpy 实现 HSTU dense 前向：out = (silu(qk^T + bias) * scale * mask) @ v（fp32 精度）

    参数顺序与算子接口 hstu_dense(q, k, v, mask, attn_bias, mask_type, max_seq_len, silu_scale) 对齐。
    """
    q_f = q.astype(np.float32)
    k_f = k.astype(np.float32)
    v_f = v.astype(np.float32)
    B, S, N_q, D = q.shape
    scale = silu_scale if silu_scale not in (None, 0.0) else 1.0 / max_seq_len

    # GQA：N_q != N_k 时，k/v 的 head 维按 N_q // N_k 扩展（repeat_interleave 语义）
    N_k = k.shape[2]
    if N_k != N_q:
        assert N_q % N_k == 0, f"head_nums_q ({N_q}) must be divisible by head_nums_k ({N_k})"
        h_qk_ratio = N_q // N_k
        k_f = np.repeat(k_f, h_qk_ratio, axis=2)
        v_f = np.repeat(v_f, h_qk_ratio, axis=2)

    qk = np.einsum("bnhd,bmhd->bhnm", q_f, k_f)
    if attn_bias is not None:
        qk = qk + attn_bias.astype(np.float32)
    qk = qk * (1.0 / (1.0 + np.exp(-qk))) * scale  # silu
    if mask is not None:
        qk = qk * mask.astype(np.float32)
    elif mask_type == 0:
        qk = qk * np.tril(np.ones((S, S), dtype=np.float32))[None, None, :, :]
    out = np.einsum("bhnm,bmhd->bnhd", qk, v_f)
    return out.astype(q.dtype)


def get_golden_paged(q, k, v, kv_cache, seq_offset, page_offsets, page_ids, last_page_len, num_target, silu_scale):
    """numpy 实现 HSTU paged 前向 golden（fp32），针对 MASK_TRIL（mask_type=0）。

    语义依据 op_kernel/atlas950/hstu_paged_forward_kernel.h（FetchKvMayFromCache 的
    cache/input 拼接）+ hstu_dense_causal_mask.h（GenCausalMask/GenTargetMask），并与
    RecSDK cust_op/test/hstu_dense/torch/test_hstu_paged_forward.py 参考 golden 交叉确认：
      - q 行 = 每个 batch 的全部 token（历史 + target，由 seq_offset 分段），输出 shape (total, N, D)
      - K/V 列 = kv_cache 解页后的历史 KV（H 行，由 page_offsets/page_ids/last_page_len 索引）
        + 输入 k/v 的 target 段（T 行），共 H+T 行；输入 k/v 中的历史段不作为 KV 使用
      - mask：历史行 i 仅见历史列 [0, i]（下三角）；target 行 H+j 见全部历史列 [0, H)
        + 自身列 H+j（eye，target 之间互不 attend）
      - 输出 = silu(qk) * silu_scale * mask @ v_con，按 q 的 token 顺序拼接
    """
    q_f = q.astype(np.float32)
    k_f = k.astype(np.float32)
    v_f = v.astype(np.float32)
    outs = []
    for b in range(len(seq_offset) - 1):
        q0, q1 = seq_offset[b], seq_offset[b + 1]
        S = int(q1 - q0)  # 该 batch 全部 q 行（历史 + target）
        T = int(num_target[b])
        H = S - T  # 历史行/列数 == kv_cache 覆盖长度
        # 1) 解页 kv_cache：每 batch 的 page 依序拼接，末页只取 last_page_len 行
        cache_k, cache_v = [], []
        for p in range(int(page_offsets[b]), int(page_offsets[b + 1])):
            pid = int(page_ids[p])
            is_last = p == int(page_offsets[b + 1]) - 1
            n = int(last_page_len[b]) if is_last else kv_cache.shape[2]
            cache_k.append(kv_cache[pid, 0, :n])
            cache_v.append(kv_cache[pid, 1, :n])
        cache_k = np.concatenate(cache_k, axis=0)  # (H, N, D)
        cache_v = np.concatenate(cache_v, axis=0)
        assert len(cache_k) == H, f"batch {b}: cache len {len(cache_k)} != H={H} (seq_offset 与 page 元数据不一致)"
        # 2) K/V 全序列 = cache 历史 + 输入 k/v 的 target 段（第 H..S 行）
        k_con = np.concatenate([cache_k, k_f[q0 + H : q1]], axis=0)  # (H+T, N, D)
        v_con = np.concatenate([cache_v, v_f[q0 + H : q1]], axis=0)
        # 3) 注意力 + 掩码
        qb = q_f[q0:q1]
        qk = np.einsum("nhd,mhd->hnm", qb, k_con)
        qk = qk * (1.0 / (1.0 + np.exp(-qk))) * silu_scale
        mask = np.zeros((S, S), dtype=np.float32)
        mask[:, :H] = np.tril(np.ones((S, H), dtype=np.float32), k=H + T - S)  # 历史列：下三角
        mask[H:, H:] = np.eye(T, dtype=np.float32)  # target 列：自身
        qk = qk * mask[None, :, :]
        outs.append(np.einsum("hnm,mhd->nhd", qk, v_con))
    return np.concatenate(outs, axis=0).astype(q.dtype)
