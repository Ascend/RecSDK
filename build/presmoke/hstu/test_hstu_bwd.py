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
    hstu_bwd_gold,
    hstu_bwd_op,
    allclose,
    QKVShapeInfo,
    MaskGenInfo,
    MaskType,
)


def bwd(
    qkv_shape_info: QKVShapeInfo,
    mask_info: MaskGenInfo,
    enable_bias: bool = False,
    silu_scale: float = 0.0,
    alpha: float = 0.5,
):
    # pylint: disable=duplicate-code
    # create data
    seq_offset_q, seq_offset_k = create_offset(qkv_shape_info, mask_info)
    grad, q, k, v, bias = create_grad_qkvb(qkv_shape_info, mask_info, seq_offset_q, seq_offset_k, enable_bias)
    mask = create_mask(qkv_shape_info, mask_info, seq_offset_q, seq_offset_k)
    num_context = create_num_context(qkv_shape_info, mask_info, seq_offset_q, seq_offset_k)
    num_target = create_num_target(qkv_shape_info, mask_info, seq_offset_q, seq_offset_k)

    # compute
    gold_dq, gold_dk, gold_dv, gold_dbias = hstu_bwd_gold(
        grad,
        q,
        k,
        v,
        mask,
        bias,
        mask_info.mask_type,
        qkv_shape_info.max_seq_len,
        qkv_shape_info.max_seq_len,
        silu_scale,
        seq_offset_q,
        seq_offset_k,
        num_context,
        num_target,
        mask_info.target_group_size,
        alpha,
    )
    ops_dq, ops_dk, ops_dv, ops_dbias = hstu_bwd_op(
        grad,
        q,
        k,
        v,
        mask,
        bias,
        mask_info.mask_type,
        qkv_shape_info.max_seq_len,
        qkv_shape_info.max_seq_len,
        silu_scale,
        seq_offset_q,
        seq_offset_k,
        num_context,
        num_target,
        mask_info.target_group_size,
        alpha,
    )

    # checkout
    assert allclose(gold_dq, ops_dq), "dq mismatch"
    assert allclose(gold_dk, ops_dk), "dk mismatch"
    assert allclose(gold_dv, ops_dv), "dv mismatch"
    if enable_bias:
        assert allclose(gold_dbias, ops_dbias), "dbias mismatch"
    # pylint: enable=duplicate-code


@pytest.mark.parametrize(
    "batch_size, max_seq_len",
    [
        (1, 2048),
        (8, 2048),
        (16, 2048),
    ],
)
def test_hstu_bwd_batch_size(batch_size, max_seq_len):
    bwd(QKVShapeInfo(batch_size=batch_size, max_seq_len=max_seq_len), MaskGenInfo())


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("num_heads_q", range(1, 17))
@pytest.mark.parametrize("num_heads_k", range(1, 17))
def test_hstu_bwd_nhead(batch_size, num_heads_q, num_heads_k):
    if num_heads_q % num_heads_k != 0:
        return
    bwd(QKVShapeInfo(batch_size=batch_size, num_heads_q=num_heads_q, num_heads_k=num_heads_k), MaskGenInfo())


@pytest.mark.parametrize("head_dim_v", range(16, 513, 16))
def test_hstu_bwd_head_dim(head_dim_v):
    head_dim_qk = random.randint(1, 512)
    bwd(QKVShapeInfo(head_dim_qk=head_dim_qk, head_dim_v=head_dim_v), MaskGenInfo())


@pytest.mark.parametrize(
    "mask_type, target_group_size, num_context, num_target",
    [
        (MaskType.NONE, None, None, None),
        (MaskType.CUSTOM, None, None, None),
        (MaskType.TRIL, 1, 0, 30),
        (MaskType.TRIL, 3, 0, 30),
        (MaskType.TRIL, 1, 6, 30),
        (MaskType.TRIL, 3, 6, 30),
    ],
)
def test_hstu_bwd_mask(mask_type, target_group_size, num_context, num_target):
    bwd(QKVShapeInfo(), MaskGenInfo(mask_type, target_group_size, num_context, num_target))


@pytest.mark.parametrize("max_seq_len", range(2048, 8192 + 1, 2048))
def test_hstu_bwd_seqlen(max_seq_len):
    bwd(QKVShapeInfo(batch_size=4, max_seq_len=max_seq_len), MaskGenInfo())


def test_hstu_bwd_bias():
    bwd(QKVShapeInfo(), MaskGenInfo(), enable_bias=True)


def test_hstu_bwd_silu():
    bwd(QKVShapeInfo(), MaskGenInfo(), silu_scale=1 / 256)
