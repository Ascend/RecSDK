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
import dataclasses
import logging
import random
import sysconfig
from enum import Enum

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

device_id: int = random.choice(range(torch.npu.device_count()))
torch.npu.set_device(device_id)


class MaskType(int, Enum):
    TRIL = 0  # 下三角掩码
    TRIU = 1  # 上三角掩码
    NONE = 2  # 无掩码
    CUSTOM = 3  # 自定义掩码


@dataclasses.dataclass
class QKVShapeInfo:
    float_type: torch.dtype = torch.float16
    int_type: torch.dtype = torch.int32
    batch_size: int = 32
    num_heads_q: int = 4
    num_heads_k: int = 4
    head_dim_qk: int = 128
    head_dim_v: int = 128
    max_seq_len: int = 2048
    min_seq_len: int = 1


@dataclasses.dataclass
class MaskGenInfo:
    mask_type: int | MaskType = MaskType.TRIL
    target_group_size: int = 0
    num_context: int = 0
    num_target: int = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True  # 确保CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False  # 关闭CuDNN自动优化


def allclose(tensor: torch.Tensor, other: torch.Tensor) -> bool:
    assert tensor.shape == other.shape
    assert tensor.dtype == other.dtype
    precision_maps = {torch.float32: 1e-4,
                      torch.float16: 1e-3,
                      torch.bfloat16: 5e-3,
                      torch.float8_e4m3fn: 5e-3}
    atol = ratio = precision_maps.get(tensor.dtype, 1e-8)
    diff = (torch.abs(tensor - other) > atol)
    diff_count = torch.sum(diff).tolist()
    show_diff(tensor, other, atol)
    return (diff_count / tensor.numel()) < ratio


def show_diff(golden: torch.Tensor, result: torch.Tensor, atol: float):
    if golden is None or result is None:
        return
    diff = (torch.abs(golden - result) > atol)

    cnt = 0
    last_offset = last_head = -1
    for (offset, head, dim) in torch.nonzero(diff):
        if offset == last_offset and head == last_head:
            continue
        last_offset, last_head, cnt = offset, head, cnt + 1
        logging.info(f"===== ({offset, head, dim}) =====")
        logging.info(golden[offset, head, dim: dim + 16])
        logging.info(result[offset, head, dim: dim + 16])
        if cnt >= 5:
            break


class MaskGen:
    def __init__(self):
        pass
    
    @staticmethod
    def check_init_valid(num: int):
        if not isinstance(num, int):
            return False
        if num <= 0:
            return False
        return True
    
    @staticmethod
    def create_target_mask(num_target: int, target_group_size: int) -> torch.Tensor:
        row_indices = torch.arange(num_target, device="npu").view(-1, 1)
        col_indices = torch.arange(num_target, device="npu").view(1, -1)
    
        block_row = row_indices // target_group_size
        block_col = col_indices // target_group_size
    
        mask = (block_row == block_col).int()
        tril = torch.tril(torch.ones(num_target, num_target, device="npu"), diagonal=0).int()
        return tril & mask

    def create_mask(self,
                    seqlen_q: int,
                    seqlen_k: int = None,
                    num_context: int = None,
                    num_target: int = None,
                    target_group_size: int = None) -> torch.Tensor:
        if seqlen_k is None:
            seqlen_k = seqlen_q
        # causal mask
        mask = torch.tril(torch.ones(seqlen_q, seqlen_k, device="npu"), diagonal=(seqlen_k - seqlen_q))
        # context mask
        if self.check_init_valid(num_context):
            num_target = 0 if num_target is None else num_target
            mask[:num_context, :seqlen_k - num_target] = 1
        # target mask
        if self.check_init_valid(target_group_size) and self.check_init_valid(num_target):
            mask[-num_target:, -num_target:] = self.create_target_mask(num_target, target_group_size)
        return mask


def create_offset(qkv_shape_info: QKVShapeInfo, mask_info: MaskGenInfo) -> (torch.Tensor, torch.Tensor):
    min_seq_len = qkv_shape_info.min_seq_len
    if mask_info.num_context is not None:
        min_seq_len += mask_info.num_context
    if mask_info.num_target is not None:
        min_seq_len += mask_info.num_target
    min_seq_len = max(min_seq_len, qkv_shape_info.min_seq_len)
    max_seq_len = qkv_shape_info.max_seq_len
    b = qkv_shape_info.batch_size

    seq_lens_q = torch.randint(min_seq_len, max_seq_len + 1, (b,), dtype=qkv_shape_info.int_type)
    seq_lens_k = torch.randint(min_seq_len, max_seq_len + 1, (b,), dtype=qkv_shape_info.int_type)
    seq_lens_q = torch.where(seq_lens_k < seq_lens_q, seq_lens_k, seq_lens_q)

    seq_offset_q = torch.concat((torch.zeros((1,), dtype=qkv_shape_info.int_type), torch.cumsum(seq_lens_q, axis=0)))
    seq_offset_k = torch.concat((torch.zeros((1,), dtype=qkv_shape_info.int_type), torch.cumsum(seq_lens_k, axis=0)))
    return seq_offset_q.to("npu"), seq_offset_k.to("npu")


def create_grad_qkvb(qkv_shape_info: QKVShapeInfo,
                     mask_info: MaskGenInfo,
                     seq_offset_q: torch.Tensor,
                     seq_offset_k: torch.Tensor,
                     enable_bias: bool) -> \
        (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    total_len_q = seq_offset_q[-1].item()
    total_len_k = seq_offset_k[-1].item()
    grad = torch.rand(total_len_q, qkv_shape_info.num_heads_q, qkv_shape_info.head_dim_v, device="npu",
                      dtype=qkv_shape_info.float_type).uniform_(-1, 1)
    q = torch.rand(total_len_q, qkv_shape_info.num_heads_q, qkv_shape_info.head_dim_qk, device="npu",
                   dtype=qkv_shape_info.float_type).uniform_(-1, 1)
    k = torch.rand(total_len_k, qkv_shape_info.num_heads_k, qkv_shape_info.head_dim_qk, device="npu",
                   dtype=qkv_shape_info.float_type).uniform_(-1, 1)
    v = torch.rand(total_len_k, qkv_shape_info.num_heads_k, qkv_shape_info.head_dim_v, device="npu",
                   dtype=qkv_shape_info.float_type).uniform_(-1, 1)

    bias = None
    if enable_bias:
        b, n, s = qkv_shape_info.batch_size, qkv_shape_info.num_heads_q, qkv_shape_info.max_seq_len
        bias = torch.rand(b, n, s, s, device="npu", dtype=qkv_shape_info.float_type).uniform_(-1, 1)
    return grad, q, k, v, bias


def create_mask(qkv_shape_info: QKVShapeInfo,
                mask_info: MaskGenInfo,
                seq_offset_q: torch.Tensor,
                seq_offset_k: torch.Tensor) -> torch.Tensor:
    mask_gen = MaskGen()
    b, n, s = qkv_shape_info.batch_size, qkv_shape_info.num_heads_q, qkv_shape_info.max_seq_len
    if mask_info.mask_type == MaskType.TRIL:
        mask = torch.zeros(b, n, s, s, device="npu")
        _offset_q, _offset_k = 0, 0
        for bid, (offset_q, offset_k) in enumerate(zip(seq_offset_q[1:], seq_offset_k[1:])):
            seqlen_q, seqlen_k = offset_q - _offset_q, offset_k - _offset_k
            _offset_q, _offset_k = offset_q, offset_k
            mask[bid, :, :seqlen_q, :seqlen_k] = mask_gen.create_mask(seqlen_q,
                                                                      seqlen_k,
                                                                      mask_info.num_context,
                                                                      mask_info.num_target,
                                                                      mask_info.target_group_size)
    elif mask_info.mask_type == MaskType.TRIU:
        raise ValueError(f"Not support mask type: {mask_info.mask_type}")
    elif mask_info.mask_type == MaskType.NONE:
        mask = None
    elif mask_info.mask_type == MaskType.CUSTOM:
        mask = torch.randint(0, 2, (b, n, s, s), device="npu", dtype=qkv_shape_info.float_type)
    else:
        raise ValueError(f"Not support mask type: {mask_info.mask_type}")
    return mask


def create_num_context(qkv_shape_info: QKVShapeInfo,
                       mask_info: MaskGenInfo,
                       seq_offset_q: torch.Tensor,
                       seq_offset_k: torch.Tensor) -> torch.Tensor:
    num_context = None
    if isinstance(mask_info.num_context, int):
        num_context = torch.ones(qkv_shape_info.batch_size, device="npu", dtype=qkv_shape_info.int_type)
        num_context *= mask_info.num_context
    return num_context


def create_num_target(qkv_shape_info: QKVShapeInfo,
                      mask_info: MaskGenInfo,
                      seq_offset_q: torch.Tensor,
                      seq_offset_k: torch.Tensor) -> torch.Tensor:
    num_target = None
    if isinstance(mask_info.num_target, int):
        num_target = torch.ones(qkv_shape_info.batch_size, device="npu", dtype=qkv_shape_info.int_type)
        num_target *= mask_info.num_target
    return num_target


def tnd_to_bsnd(tnd_tensor, seq_lens, bsnd):
    bsnd_tensor = torch.zeros(*bsnd, device=tnd_tensor.device, dtype=tnd_tensor.dtype)

    offset = 0
    for batch_id, seq_len in enumerate(seq_lens):
        bsnd_tensor[batch_id, :seq_len, :, :] = tnd_tensor[offset: offset + seq_len, :, :]
        offset = offset + seq_len

    return bsnd_tensor


def bsnd_to_tnd(bsnd_tensor, seq_lens, tnd):
    tnd_tensor = torch.zeros(*tnd, device=bsnd_tensor.device, dtype=bsnd_tensor.dtype)

    offset = 0
    for batch_id, seq_len in enumerate(seq_lens):
        tnd_tensor[offset: offset + seq_len, :, :] = bsnd_tensor[batch_id, 0: seq_len, :, :]
        offset = offset + seq_len

    return tnd_tensor


def hstu_fwd_gold(q: torch.Tensor,
                  k: torch.Tensor,
                  v: torch.Tensor,
                  mask: torch.Tensor,
                  bias: torch.Tensor,
                  mask_type: MaskType,
                  max_seqlen_q: int,
                  max_seqlen_k: int,
                  silu_scale: float,
                  offset_q: torch.Tensor,
                  offset_k: torch.Tensor,
                  num_context: torch.Tensor,
                  num_target: torch.Tensor,
                  target_group_size: int,
                  alpha: float,
                  deterministic: bool) -> torch.Tensor:
    total_len_q, nhead_q, dim_qk = q.shape  # (T, Nq, Dqk)
    _, nhead_k, _ = k.shape  # (T, Nk, Dqk)
    _, _, dim_v = v.shape  # (T, Nk, Dv)
    seqlen_q, seqlen_k = offset_q[1:] - offset_q[:-1], offset_k[1:] - offset_k[:-1]
    batchsize = offset_q.shape[0] - 1

    assert nhead_q % nhead_k == 0, f"head_nums_q ({nhead_q}) must be divisible by head_nums_k({nhead_k}) "
    
    use_fp8 = bool(q.dtype == torch.float8_e4m3fn)
    dtype = torch.float32 if use_fp8 else q.dtype
    out_dtype = torch.float16 if use_fp8 else q.dtype
    q_dens = tnd_to_bsnd(q, seqlen_q, bsnd=(batchsize, max_seqlen_q, nhead_q, dim_qk)).to(dtype)
    k_dens = tnd_to_bsnd(k, seqlen_k, bsnd=(batchsize, max_seqlen_k, nhead_k, dim_qk)).to(dtype)
    v_dens = tnd_to_bsnd(v, seqlen_k, bsnd=(batchsize, max_seqlen_k, nhead_k, dim_v)).to(dtype)

    gqa_qk_ratio = nhead_q // nhead_k
    q_dens = q_dens.permute(0, 2, 1, 3)  # b, nq, sq, dqk
    k_dens = k_dens.repeat_interleave(gqa_qk_ratio, dim=2).permute(0, 2, 3, 1)  # b, nq, dqk, sk
    v_dens = v_dens.repeat_interleave(gqa_qk_ratio, dim=2).permute(0, 2, 1, 3)  # b, nq, sk, dv

    qk = torch.matmul(q_dens, k_dens).to(torch.float32)  # b, nq, sq, sk

    if isinstance(bias, torch.Tensor):
        bias = bias.to(torch.float32)
        qk += bias

    silu_scale = 1 / max_seqlen_q if silu_scale == 0 else silu_scale
    qk *= alpha
    F.silu(qk, inplace=True)
    qk *= silu_scale

    if isinstance(mask, torch.Tensor):
        mask = mask.to(torch.float32)
        qk *= mask

    qk = qk.to(q.dtype).to(dtype)
    # (b, nq, sq, sk) x (b, nq, sk, dv) -> (b, nq, sq, dv) -> (b, sq, nq, dv)
    out_dense = torch.matmul(qk, v_dens).permute(0, 2, 1, 3).cpu()
    out = bsnd_to_tnd(out_dense, seqlen_q, tnd=(total_len_q, nhead_q, dim_v)).to(out_dtype)
    torch.npu.synchronize()
    return out


def hstu_fwd_op(q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor,
                bias: torch.Tensor,
                mask_type: MaskType,
                max_seqlen_q: int,
                max_seqlen_k: int,
                silu_scale: float,
                offset_q: torch.Tensor,
                offset_k: torch.Tensor,
                num_context: torch.Tensor,
                num_target: torch.Tensor,
                target_group_size: int,
                alpha: float,
                deterministic: bool) -> torch.Tensor:

    output = torch.ops.mxrec.hstu_jagged(q,
                                         k,
                                         v,
                                         mask,
                                         bias,
                                         mask_type,
                                         max_seqlen_q,
                                         max_seqlen_k,
                                         silu_scale,
                                         offset_q,
                                         offset_k,
                                         num_context,
                                         num_target,
                                         target_group_size,
                                         alpha,
                                         deterministic)
    torch.npu.synchronize()
    return output.cpu()
