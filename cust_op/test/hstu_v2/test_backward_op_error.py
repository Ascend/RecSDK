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
# pylint: disable=redefined-outer-name

import pytest
import torch
import torch_npu
import sysconfig

DEVICE = "npu:0"

# ---------------------------------------------------------------------------
# 模块级初始化：加载算子库
# ---------------------------------------------------------------------------
torch_npu.npu.set_device(DEVICE)
try:
    torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
except Exception:
    pass  # 库可能已加载


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _call_op(
    grad,
    q,
    k,
    v,
    max_seqlen_q,
    max_seqlen_k,
    seq_offset_q,
    seq_offset_k,
    rab=None,
    num_context=None,
    num_target=None,
    scale=0.0,
    target_group_size=0,
    alpha=1.0,
):
    """调用 hstu_backward_v2 算子。"""
    return torch.ops.mxrec.hstu_backward_v2(
        grad,
        q,
        k,
        v,
        max_seqlen_q,
        max_seqlen_k,
        seq_offset_q,
        seq_offset_k,
        rab,
        num_context,
        num_target,
        scale,
        target_group_size,
        alpha,
    )


# ===================================================================
# Fixture：模块级合法默认输入，各测试复用
# ===================================================================
@pytest.fixture(scope="module")
def d():
    """返回包含全部默认合法 tensor 及尺寸元信息的 dict。

    各测试从此 dict 取出合法输入，仅修改待校验的字段即可触发参数报错。
    """
    b = 2  # batchSize
    h = 1  # heads
    dim_qk = 32  # dimQK (multiple of 16, <= 256)
    dim_v = 32  # dimGV
    max_q = 128  # maxSeqLenQ
    max_k = 128  # maxSeqLenK

    total_q = max_q * b
    total_k = max_k * b

    return {
        "grad": torch.randn(total_q, h, dim_v, dtype=torch.float16, device=DEVICE),
        "q": torch.randn(total_q, h, dim_qk, dtype=torch.float16, device=DEVICE),
        "k": torch.randn(total_k, h, dim_qk, dtype=torch.float16, device=DEVICE),
        "v": torch.randn(total_k, h, dim_v, dtype=torch.float16, device=DEVICE),
        "seq_offset_q": torch.tensor([i * max_q for i in range(b + 1)], dtype=torch.int32, device=DEVICE),
        "seq_offset_k": torch.tensor([i * max_k for i in range(b + 1)], dtype=torch.int32, device=DEVICE),
        "rab": torch.randn(b, h, max_q, max_k, dtype=torch.float16, device=DEVICE),
        # 元信息
        "b": b,
        "h": h,
        "dim_qk": dim_qk,
        "dim_v": dim_v,
        "total_q": total_q,
        "total_k": total_k,
        "max_q": max_q,
        "max_k": max_k,
    }


# ===================================================================
# 1. Tensor 维度校验 (grad/q/k/v 必须为 3D, seq_offset 必须为 1D)
# ===================================================================
@pytest.mark.parametrize("bad_dim", [2, 4])
def test_grad_dim(d, bad_dim):
    grad = d["grad"]
    if bad_dim == 2:
        grad_bad = grad.reshape(grad.shape[0], -1)
    else:
        grad_bad = grad.unsqueeze(0)
    with pytest.raises(RuntimeError):
        _call_op(grad_bad, d["q"], d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


@pytest.mark.parametrize("bad_dim", [2, 4])
def test_q_dim(d, bad_dim):
    q = d["q"]
    if bad_dim == 2:
        q_bad = q.reshape(q.shape[0], -1)
    else:
        q_bad = q.unsqueeze(0)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], q_bad, d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


@pytest.mark.parametrize("bad_dim", [2, 4])
def test_k_dim(d, bad_dim):
    k = d["k"]
    if bad_dim == 2:
        k_bad = k.reshape(k.shape[0], -1)
    else:
        k_bad = k.unsqueeze(0)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], k_bad, d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


@pytest.mark.parametrize("bad_dim", [2, 4])
def test_v_dim(d, bad_dim):
    v = d["v"]
    if bad_dim == 2:
        v_bad = v.reshape(v.shape[0], -1)
    else:
        v_bad = v.unsqueeze(0)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], d["k"], v_bad, d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


def test_seq_offset_q_dim(d):
    soq_bad = d["seq_offset_q"].unsqueeze(0)  # 1D -> 2D
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], d["k"], d["v"], d["max_q"], d["max_k"], soq_bad, d["seq_offset_k"])


def test_seq_offset_k_dim(d):
    sok_bad = d["seq_offset_k"].unsqueeze(0)  # 1D -> 2D
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], sok_bad)


# ===================================================================
# 2. Tensor Shape 一致性校验
# ===================================================================
def test_grad_shape_dim0(d):
    """grad.size(0) 必须等于 totalSeqLenQ (= q.size(0))"""
    grad_bad = d["grad"][:-1, :, :]
    with pytest.raises(RuntimeError):
        _call_op(grad_bad, d["q"], d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


def test_grad_shape_dim1(d):
    """grad.size(1) 必须等于 heads (= q.size(1))"""
    grad_bad = torch.randn(d["total_q"], d["h"] + 1, d["dim_v"], dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(grad_bad, d["q"], d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


def test_grad_shape_dim2(d):
    """grad.size(2) 必须等于 dimGV (= v.size(2))"""
    grad_bad = torch.randn(d["total_q"], d["h"], d["dim_v"] + 16, dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(grad_bad, d["q"], d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


def test_k_shape_heads(d):
    """k.size(1) 必须等于 heads"""
    k_bad = torch.randn(d["total_k"], d["h"] + 1, d["dim_qk"], dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], k_bad, d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


def test_k_shape_dim_qk(d):
    """k.size(2) 必须等于 dimQK"""
    k_bad = torch.randn(d["total_k"], d["h"], d["dim_qk"] + 16, dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], k_bad, d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


def test_v_shape_dim0(d):
    """v.size(0) 必须等于 totalSeqLenK (= k.size(0))"""
    v_bad = d["v"][:-1, :, :]
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], d["k"], v_bad, d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


def test_v_shape_dim1(d):
    """v.size(1) 必须等于 heads"""
    v_bad = torch.randn(d["total_k"], d["h"] + 1, d["dim_v"], dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], d["k"], v_bad, d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


def test_v_shape_dim2(d):
    """v.size(2) 必须等于 dimGV"""
    v_bad = torch.randn(d["total_k"], d["h"], d["dim_v"] + 16, dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], d["k"], v_bad, d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


def test_seq_offset_size_mismatch(d):
    """seq_offset_q 和 seq_offset_k 的 size(0) 必须相等"""
    sok_bad = d["seq_offset_k"][:-1]
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], sok_bad)


# ===================================================================
# 3. 参数取值范围校验
# ===================================================================
def test_batch_size_zero(d):
    """batchSize = seq_offset.size(0) - 1 必须 > 0"""
    soq = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    sok = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    # 配套构造 k/v: totalSeqLenK = 1
    k1 = torch.randn(1, d["h"], d["dim_qk"], dtype=torch.float16, device=DEVICE)
    v1 = torch.randn(1, d["h"], d["dim_v"], dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], k1, v1, d["max_q"], d["max_k"], soq, sok)


@pytest.mark.parametrize("bad_heads", [0, 17])
def test_heads_out_of_range(d, bad_heads):
    """heads 必须在 [1, 16] 范围内"""
    # 需要配套修改 q/grad/k/v 的 heads 维度，使 shape 一致性检查先通过
    q_bad = torch.randn(d["total_q"], bad_heads, d["dim_qk"], dtype=torch.float16, device=DEVICE)
    grad_bad = torch.randn(d["total_q"], bad_heads, d["dim_v"], dtype=torch.float16, device=DEVICE)
    k_bad = torch.randn(d["total_k"], bad_heads, d["dim_qk"], dtype=torch.float16, device=DEVICE)
    v_bad = torch.randn(d["total_k"], bad_heads, d["dim_v"], dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(grad_bad, q_bad, k_bad, v_bad, d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


@pytest.mark.parametrize("bad_dim", [1, 17, 272])
def test_dim_qk_invalid(d, bad_dim):
    """dimQK 必须是 16 的倍数且 <= 256"""
    q_bad = torch.randn(d["total_q"], d["h"], bad_dim, dtype=torch.float16, device=DEVICE)
    k_bad = torch.randn(d["total_k"], d["h"], bad_dim, dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], q_bad, k_bad, d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


@pytest.mark.parametrize("bad_dim", [1, 17, 272])
def test_dim_v_invalid(d, bad_dim):
    """dimGV 必须是 16 的倍数且 <= 256"""
    v_bad = torch.randn(d["total_k"], d["h"], bad_dim, dtype=torch.float16, device=DEVICE)
    grad_bad = torch.randn(d["total_q"], d["h"], bad_dim, dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(grad_bad, d["q"], d["k"], v_bad, d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


@pytest.mark.parametrize("bad_val", [0, -1])
def test_max_seqlen_q_invalid(d, bad_val):
    """maxSeqLenQ 必须 > 0"""
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], d["k"], d["v"], bad_val, d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


@pytest.mark.parametrize("bad_val", [0, -1])
def test_max_seqlen_k_invalid(d, bad_val):
    """maxSeqLenK 必须 > 0"""
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], d["k"], d["v"], d["max_q"], bad_val, d["seq_offset_q"], d["seq_offset_k"])


# ===================================================================
# 4. Mask 校验（当前仅支持全 mask，即 num_context 和 num_target 均为 None）
# ===================================================================
def test_num_context_not_none(d):
    """num_context 不为 None 时报错"""
    num_ctx = torch.tensor([d["max_q"]], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(
            d["grad"],
            d["q"],
            d["k"],
            d["v"],
            d["max_q"],
            d["max_k"],
            d["seq_offset_q"],
            d["seq_offset_k"],
            num_context=num_ctx,
        )


def test_num_target_not_none(d):
    """num_target 不为 None 时报错"""
    num_tgt = torch.tensor([d["max_q"]], dtype=torch.int32, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(
            d["grad"],
            d["q"],
            d["k"],
            d["v"],
            d["max_q"],
            d["max_k"],
            d["seq_offset_q"],
            d["seq_offset_k"],
            num_target=num_tgt,
        )


# ===================================================================
# 5. RAB 校验
# ===================================================================
@pytest.mark.parametrize("bad_dim", [3, 5])
def test_rab_dim(d, bad_dim):
    """rab 必须为 4D"""
    shape = [d["b"], d["h"], d["max_q"], d["max_k"]]
    if bad_dim == 3:
        shape = shape[:-1]
    else:
        shape = [1] + shape
    rab_bad = torch.randn(*shape, dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(
            d["grad"], d["q"], d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"], rab=rab_bad
        )


def test_rab_shape_batch(d):
    """rab.size(0) 必须等于 batchSize"""
    rab_bad = torch.randn(d["b"] + 1, d["h"], d["max_q"], d["max_k"], dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(
            d["grad"], d["q"], d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"], rab=rab_bad
        )


def test_rab_shape_heads(d):
    """rab.size(1) 必须等于 heads"""
    rab_bad = torch.randn(d["b"], d["h"] + 1, d["max_q"], d["max_k"], dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(
            d["grad"], d["q"], d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"], rab=rab_bad
        )


def test_rab_shape_max_q(d):
    """rab.size(2) 必须等于 maxSeqLenQ"""
    rab_bad = torch.randn(d["b"], d["h"], d["max_q"] + 1, d["max_k"], dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(
            d["grad"], d["q"], d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"], rab=rab_bad
        )


def test_rab_shape_max_k(d):
    """rab.size(3) 必须等于 maxSeqLenK"""
    rab_bad = torch.randn(d["b"], d["h"], d["max_q"], d["max_k"] + 1, dtype=torch.float16, device=DEVICE)
    with pytest.raises(RuntimeError):
        _call_op(
            d["grad"], d["q"], d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"], rab=rab_bad
        )


# ===================================================================
# 6. 数据类型校验
# ===================================================================
@pytest.mark.parametrize("bad_dtype", [torch.float32, torch.int32, torch.int64])
def test_grad_dtype_invalid(d, bad_dtype):
    """grad 必须为 float16 或 bfloat16"""
    grad_bad = d["grad"].to(bad_dtype)
    with pytest.raises(RuntimeError):
        _call_op(grad_bad, d["q"], d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


@pytest.mark.parametrize("bad_dtype", [torch.float32, torch.int32, torch.int64])
def test_q_dtype_invalid(d, bad_dtype):
    """q 必须为 float16 或 bfloat16"""
    q_bad = d["q"].to(bad_dtype)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], q_bad, d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


@pytest.mark.parametrize("bad_dtype", [torch.float32, torch.int32, torch.int64])
def test_k_dtype_invalid(d, bad_dtype):
    """k 必须为 float16 或 bfloat16"""
    k_bad = d["k"].to(bad_dtype)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], k_bad, d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


@pytest.mark.parametrize("bad_dtype", [torch.float32, torch.int32, torch.int64])
def test_v_dtype_invalid(d, bad_dtype):
    """v 必须为 float16 或 bfloat16"""
    v_bad = d["v"].to(bad_dtype)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], d["k"], v_bad, d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


def test_grad_q_dtype_mismatch(d):
    """grad/q/k/v 必须 dtype 一致"""
    q_bad = d["q"].to(torch.bfloat16)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], q_bad, d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


def test_grad_k_dtype_mismatch(d):
    """grad 与 k dtype 不一致时报错"""
    k_bad = d["k"].to(torch.bfloat16)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], k_bad, d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


def test_grad_v_dtype_mismatch(d):
    """grad 与 v dtype 不一致时报错"""
    v_bad = d["v"].to(torch.bfloat16)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], d["k"], v_bad, d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"])


@pytest.mark.parametrize("bad_dtype", [torch.int64, torch.float32])
def test_seq_offset_q_dtype_invalid(d, bad_dtype):
    """seqOffsetQ 必须为 int32"""
    soq_bad = d["seq_offset_q"].to(bad_dtype)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], d["k"], d["v"], d["max_q"], d["max_k"], soq_bad, d["seq_offset_k"])


@pytest.mark.parametrize("bad_dtype", [torch.int64, torch.float32])
def test_seq_offset_k_dtype_invalid(d, bad_dtype):
    """seqOffsetK 必须为 int32"""
    sok_bad = d["seq_offset_k"].to(bad_dtype)
    with pytest.raises(RuntimeError):
        _call_op(d["grad"], d["q"], d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], sok_bad)


@pytest.mark.parametrize("bad_dtype", [torch.float32, torch.int32, torch.int64])
def test_rab_dtype_invalid(d, bad_dtype):
    """rab 必须为 float16 或 bfloat16"""
    rab_bad = d["rab"].to(bad_dtype)
    with pytest.raises(RuntimeError):
        _call_op(
            d["grad"], d["q"], d["k"], d["v"], d["max_q"], d["max_k"], d["seq_offset_q"], d["seq_offset_k"], rab=rab_bad
        )
