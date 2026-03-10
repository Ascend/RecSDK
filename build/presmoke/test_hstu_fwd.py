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
import random

import pytest

from hstu_common import (
    create_offset,
    create_grad_qkvb,
    create_mask,
    create_num_context,
    create_num_target,
    hstu_fwd_gold,
    hstu_fwd_op,
    allclose,
    QKVShapeInfo,
    MaskGenInfo,
    MaskType
)


def fwd(qkv_shape_info: QKVShapeInfo,
        mask_info: MaskGenInfo,
        enable_bias: bool = False,
        silu_scale: float = .0,
        alpha: float = 0.5,
        deterministic: bool = False):
    # create data
    seq_offset_q, seq_offset_k = create_offset(qkv_shape_info, mask_info)
    _, q, k, v, bias = create_grad_qkvb(qkv_shape_info, mask_info, seq_offset_q, seq_offset_k, enable_bias)
    mask = create_mask(qkv_shape_info, mask_info, seq_offset_q, seq_offset_k)
    num_context = create_num_context(qkv_shape_info, mask_info, seq_offset_q, seq_offset_k)
    num_target = create_num_target(qkv_shape_info, mask_info, seq_offset_q, seq_offset_k)
    # compute
    gold = hstu_fwd_gold(q, k, v, mask, bias, mask_info.mask_type, qkv_shape_info.max_seq_len,
                         qkv_shape_info.max_seq_len, silu_scale, seq_offset_q, seq_offset_k, num_context, num_target,
                         mask_info.target_group_size, alpha, deterministic)
    ops = hstu_fwd_op(q, k, v, mask, bias, mask_info.mask_type, qkv_shape_info.max_seq_len,
                      qkv_shape_info.max_seq_len, silu_scale, seq_offset_q, seq_offset_k, num_context, num_target,
                      mask_info.target_group_size, alpha, deterministic)
    # checkout
    assert allclose(gold, ops)


@pytest.mark.parametrize("batch_size, max_seq_len", [
    (1, 2048),
    (8, 2048),
    (16, 2048),
    (64, 2048),
    (128, 2048),
    (256, 1024),
])
def test_hstu_batch_size(batch_size, max_seq_len):
    fwd(QKVShapeInfo(batch_size=batch_size,
                     max_seq_len=max_seq_len),
        MaskGenInfo())


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("num_heads_q", range(1, 17))
@pytest.mark.parametrize("num_heads_k", range(1, 17))
def test_hstu_nhead(batch_size, num_heads_q, num_heads_k):
    if num_heads_q % num_heads_k != 0:
        return
    fwd(QKVShapeInfo(batch_size=batch_size,
                     num_heads_q=num_heads_q,
                     num_heads_k=num_heads_k),
        MaskGenInfo())


@pytest.mark.parametrize("head_dim_v", range(16, 513, 16))
def test_hstu_head_dim(head_dim_v):
    head_dim_qk = random.randint(1, 512)
    fwd(QKVShapeInfo(head_dim_qk=head_dim_qk,
                     head_dim_v=head_dim_v),
        MaskGenInfo())


@pytest.mark.parametrize("mask_type, target_group_size, num_context, num_target", [
    (MaskType.NONE, 0, 0, 0),
    (MaskType.CUSTOM, 0, 0, 0),
    (MaskType.TRIL, 1, 0, 30),
    (MaskType.TRIL, 3, 0, 30),
    (MaskType.TRIL, 1, 6, 0),
    (MaskType.TRIL, 3, 6, 0),
    (MaskType.TRIL, 1, 6, 30),
    (MaskType.TRIL, 3, 6, 30),
])
def test_hstu_mask(mask_type, target_group_size, num_context, num_target):
    fwd(QKVShapeInfo(),
        MaskGenInfo(mask_type, target_group_size, num_context, num_target))


@pytest.mark.parametrize("max_seq_len", range(2048, 8192 + 1, 2048))
def test_hstu_seqlen(max_seq_len):
    fwd(QKVShapeInfo(batch_size=4, max_seq_len=max_seq_len),
        MaskGenInfo())


def test_hstu_bias():
    fwd(QKVShapeInfo(),
        MaskGenInfo(),
        enable_bias=True)


def test_hstu_silu():
    fwd(QKVShapeInfo(),
        MaskGenInfo(),
        silu_scale=1 / 256)


def test_hstu_deterministic():
    fwd(QKVShapeInfo(),
        MaskGenInfo(),
        deterministic=True)
