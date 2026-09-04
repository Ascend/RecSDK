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
# pylint: disable=redefined-outer-name, duplicate-code, disable=too-many-lines

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_common_utils import allclose, jagged_to_dense, dense_to_jagged, MaskType, QKVShapeInfo, MaskGenInfo, set_seed
from test_target_mask import create_causal_mask


def jagged_data_gen(
    qkv_shape_info: QKVShapeInfo, mask_info: MaskGenInfo, enable_bias: bool, repeat_offset: bool = False
):
    int_type = qkv_shape_info.int_type
    batch_size = qkv_shape_info.batch_size
    max_num_context, max_num_target = mask_info.max_num_context, mask_info.max_num_target
    min_num_context, min_num_target = mask_info.max_num_context, mask_info.max_num_target
    num_context = torch.randint(min_num_context, max_num_context + 1, (batch_size,), dtype=int_type)
    num_target = torch.randint(min_num_target, max_num_target + 1, (batch_size,), dtype=int_type)

    float_type = qkv_shape_info.float_type
    min_seq_len, max_seq_len = qkv_shape_info.min_seq_len, qkv_shape_info.max_seq_len

    seq_lens = torch.randint(min_seq_len, max_seq_len + 1, (batch_size,))
    if mask_info.mask_type == MaskType.TRIL:
        seq_lens += num_context + num_target
    seq_offset = torch.concat((torch.zeros((1,)), torch.cumsum(seq_lens, dim=0))).to(int_type)
    if repeat_offset:
        seq_offset = torch.cat((seq_offset, seq_offset[-1:]), dim=0)
        num_context = torch.cat((num_context, num_context[-1:]), dim=0)
        num_target = torch.cat((num_target, num_target[-1:]), dim=0)
    max_seq_len, total_seqs = max(seq_lens.tolist()), sum(seq_lens.tolist())

    num_heads_q, num_heads_k, head_dim_qk, head_dim_v = (
        qkv_shape_info.num_heads_q,
        qkv_shape_info.num_heads_k,
        qkv_shape_info.head_dim_qk,
        qkv_shape_info.head_dim_v,
    )

    if float_type == torch.float8_e4m3fn:
        data_type = torch.float16
    else:
        data_type = float_type

    q = torch.rand(total_seqs, num_heads_q, head_dim_qk, dtype=data_type).uniform_(-1, 1)
    k = torch.rand(total_seqs, num_heads_k, head_dim_qk, dtype=data_type).uniform_(-1, 1)
    v = torch.rand(total_seqs, num_heads_k, head_dim_v, dtype=data_type).uniform_(-1, 1)

    rel_attn_bias = (
        torch.rand(batch_size, num_heads_q, max_seq_len, max_seq_len).to(float_type) if enable_bias else None
    )

    if mask_info.mask_type == MaskType.TRIL:
        mask = torch.zeros(batch_size, num_heads_q, max_seq_len, max_seq_len)
        seqs = zip(seq_lens.tolist(), num_context.tolist(), num_target.tolist())
        for batch_id, (seq_len, ctx, tar) in enumerate(seqs):
            mask_tensor = create_causal_mask(seq_len, seq_len, ctx, tar, mask_info.target_group_size)
            mask[batch_id, :, :seq_len, :seq_len] = mask_tensor
        mask = mask.cpu().to(float_type)
    elif mask_info.mask_type == MaskType.CUSTOM:
        mask = torch.randint(0, 2, size=(batch_size, num_heads_q, max_seq_len, max_seq_len))
        mask = mask.cpu().to(float_type)
    else:
        mask = None

    if float_type == torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)

    qkv_tensors = (q, k, v, seq_offset)
    mask_tensors = (mask_info.mask_type, mask, num_context, num_target, mask_info.target_group_size)
    return qkv_tensors, mask_tensors, rel_attn_bias, max_seq_len


class TestHstuJaggedDemo:
    @staticmethod
    def custom_op_exec(qkv_tensors, mask_tensors, rel_attn_bias, silu_scale, max_seq_len, deterministic=False):
        q, k, v, seq_offset = qkv_tensors
        mask_type, mask, num_context, num_target, target_group_size = mask_tensors

        q_npu = q.to("npu")
        k_npu = k.to("npu")
        v_npu = v.to("npu")
        bias_npu = rel_attn_bias.to("npu") if isinstance(rel_attn_bias, torch.Tensor) else None
        mask_npu = mask.to("npu") if mask_type is MaskType.CUSTOM else None
        seq_offset = seq_offset.to("npu")
        num_context = num_context.to("npu")
        num_target = num_target.to("npu")
        # 函数重载：hstu_jagged -> hstu_jagged.equal
        output = torch.ops.mxrec.hstu_jagged.equal(
            q_npu,
            k_npu,
            v_npu,
            mask=mask_npu,
            attn_bias=bias_npu,
            mask_type=mask_type,
            max_seq_len=max_seq_len,
            silu_scale=silu_scale,
            seq_offset=seq_offset,
            num_context=num_context,
            num_target=num_target,
            target_group_size=target_group_size,
            deterministic=deterministic,
        )
        torch.npu.synchronize()
        return output.cpu()

    @staticmethod
    def golden_op_exec(qkv_tensors, mask_tensors, rel_attn_bias, silu_scale, max_seq_len, repeat_offset):
        q, k, v, seq_offset = qkv_tensors
        mask_type, mask, _, _, _ = mask_tensors

        (_, head_nums_q, head_dim), data_type = q.shape, q.dtype
        (_, head_nums_k, head_dim) = k.shape
        head_dim_v = v.shape[2]
        batch_size = seq_offset.shape[0] - 1 - int(repeat_offset)

        if head_nums_q != head_nums_k:
            assert head_nums_q % head_nums_k == 0, (
                f"head_nums_q ({head_nums_q}) must be divisible by head_nums_k({head_nums_k}) "
            )
        h_qk_ratio = head_nums_q // head_nums_k

        seq_lens = np.zeros((batch_size,)).astype(np.int32)
        for batch_id in range(batch_size):
            seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]

        silu_scale = 1 / max_seq_len if silu_scale == 0 else silu_scale

        use_float8 = data_type == torch.float8_e4m3fn
        if use_float8:
            q_dens = jagged_to_dense(q, seq_lens, head_nums_q, head_dim).to(torch.float32).to("npu")
            k_dens = jagged_to_dense(k, seq_lens, head_nums_k, head_dim).to(torch.float32).to("npu")
            v_dens = jagged_to_dense(v, seq_lens, head_nums_k, head_dim_v).to(torch.float32).to("npu")
        else:
            q_dens = jagged_to_dense(q, seq_lens, head_nums_q, head_dim).to(data_type).to("npu")
            k_dens = jagged_to_dense(k, seq_lens, head_nums_k, head_dim).to(data_type).to("npu")
            v_dens = jagged_to_dense(v, seq_lens, head_nums_k, head_dim_v).to(data_type).to("npu")

        k_dens_expanded = k_dens.repeat_interleave(h_qk_ratio, dim=2)
        v_dens_expanded = v_dens.repeat_interleave(h_qk_ratio, dim=2)

        q_dens = q_dens.permute(0, 2, 1, 3)
        k_dens = k_dens_expanded.permute(0, 2, 3, 1)
        qk_attn = torch.matmul(q_dens, k_dens).to(torch.float32)

        if rel_attn_bias is not None:
            rel_attn_bias = rel_attn_bias.to(torch.float32).to("npu")
            qk_attn += rel_attn_bias

        F.silu(qk_attn, inplace=True)
        qk_attn *= silu_scale

        if mask_type != MaskType.NONE:
            mask = mask.to(torch.float32).to("npu")
            qk_attn *= mask

        v_dens = v_dens_expanded.permute(0, 2, 1, 3)

        if use_float8:
            qk_attn = qk_attn.to(torch.float8_e4m3fn)
            qk_attn = qk_attn.to(torch.float32)
        else:
            qk_attn = qk_attn.to(data_type)

        attn_output = torch.matmul(qk_attn, v_dens)
        attn_output = attn_output.permute(0, 2, 1, 3).cpu()
        attn_output = dense_to_jagged(q, attn_output, seq_lens)

        if use_float8:
            attn_output = attn_output.to(torch.float16)

        torch.npu.synchronize()
        return attn_output.to(torch.float16) if use_float8 else attn_output.to(data_type)

    def execute(self, qkv_shape_info, mask_info, enable_bias, silu_scale, repeat_offset=False, deterministic=False):
        qkv_tensors, mask_tensors, rel_attn_bias, max_seq_len = jagged_data_gen(
            qkv_shape_info, mask_info, enable_bias, repeat_offset
        )

        output = self.custom_op_exec(qkv_tensors, mask_tensors, rel_attn_bias, silu_scale, max_seq_len, deterministic)
        golden = self.golden_op_exec(qkv_tensors, mask_tensors, rel_attn_bias, silu_scale, max_seq_len, repeat_offset)

        data_type = qkv_shape_info.float_type
        if data_type == torch.bfloat16:
            res = allclose(output, golden, 5e-3, 5e-3)
        elif data_type == torch.float8_e4m3fn:
            res = allclose(output, golden, 1e-3, 1e-3)
        elif data_type == torch.float16:
            res = allclose(output, golden, 1e-3, 1e-3)
        else:
            res = allclose(output, golden, 1e-4, 1e-4)
        assert res

    @pytest.mark.parametrize(
        "batch_size, head_num, max_seq_len, head_dim",
        [
            (1, 2, 15, 16),
            (16, 4, 1024, 128),
            (1, 4, 1024, 16),
            (16, 2, 15, 128),
        ],
    )
    @pytest.mark.parametrize("enable_bias, silu_scale", [(False, 0), (True, 1 / 1024)])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int32])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 1, 0, 30),
            (MaskType.TRIL, 3, 0, 30),
            (MaskType.TRIL, 1, 6, 0),
            (MaskType.TRIL, 3, 6, 0),
            (MaskType.TRIL, 1, 6, 30),
            (MaskType.TRIL, 3, 6, 30),
        ],
    )
    def test_hstu_jagged_forward(
        self,
        batch_size,
        head_num,
        max_seq_len,
        head_dim,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        int_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=int_data_type,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads_q=head_num,
            num_heads_k=head_num,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize(
        "target_group_size, max_num_context, max_num_target",
        [(0, 0, 0), (1, 255, 30), (1, 256, 30), (1, 257, 30), (3, 20, 511), (3, 20, 512)],
    )
    def test_hstu_jagged_forward_mask(self, target_group_size, max_num_context, max_num_target):
        qkv_shape_info = QKVShapeInfo(
            float_type=torch.float16,
            int_type=torch.int32,
            batch_size=1,
            max_seq_len=16,
            num_heads_q=1,
            num_heads_k=1,
            head_dim_qk=16,
            head_dim_v=16,
        )
        mask_info = MaskGenInfo(
            mask_type=MaskType.TRIL,
            max_num_context=max_num_context,
            min_num_context=max_num_context,
            max_num_target=max_num_target,
            min_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, False, 0)

    @pytest.mark.parametrize(
        "batch_size, head_num, max_seq_len, head_dim, repeat_offset",
        [
            (1, 1, 15, 16, False),
            (2, 7, 256, 32, True),
            (1, 16, 256, 16, True),
            (2, 1, 15, 32, False),
        ],
    )
    # bias×silu 在此补全另外两组组合（test_hstu_jagged_forward 已覆盖 (F,0)/(T,1/1024)）
    @pytest.mark.parametrize("enable_bias, silu_scale", [(True, 0), (False, 1 / 1024)])
    @pytest.mark.parametrize("mask_type", [MaskType.NONE, MaskType.CUSTOM, MaskType.TRIL])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int32])
    def test_hstu_jagged_forward_head16(
        self,
        batch_size,
        head_num,
        max_seq_len,
        head_dim,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        int_data_type,
        repeat_offset,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=int_data_type,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads_q=head_num,
            num_heads_k=head_num,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(mask_type=mask_type, max_num_context=0, max_num_target=0, target_group_size=0)
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale, repeat_offset)

    @pytest.mark.parametrize("head_num", [2])
    @pytest.mark.parametrize("max_seq_len", [2570])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias, silu_scale", [(False, 0), (True, 1 / 1024)])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int32])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 1, 6, 30),
        ],
    )
    def test_hstu_jagged_forward_128bs(
        self,
        head_num,
        max_seq_len,
        head_dim,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        int_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=int_data_type,
            batch_size=128,
            max_seq_len=max_seq_len,
            num_heads_q=head_num,
            num_heads_k=head_num,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize("head_num", [2])
    @pytest.mark.parametrize("max_seq_len", [16])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias, silu_scale", [(False, 0), (True, 1 / 1024)])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 3, 6, 30),
        ],
    )
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    def test_hstu_jagged_forward_2048bs(
        self,
        head_num,
        max_seq_len,
        head_dim,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=torch.int32,
            batch_size=2048,
            max_seq_len=max_seq_len,
            num_heads_q=head_num,
            num_heads_k=head_num,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    ## GQA测试（head_q=8, batch=4 固定；head_k 覆盖 8/4/2/1 全部 GQA 比例）
    @pytest.mark.parametrize(
        "head_num_k, max_seq_len, head_dim",
        [
            (8, 15, 16),
            (4, 1024, 128),
            (2, 15, 128),
            (1, 1024, 16),
        ],
    )
    @pytest.mark.parametrize("enable_bias, silu_scale", [(False, 0), (True, 1 / 1024)])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int32])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 3, 6, 30),
        ],
    )
    def test_hstu_jagged_forward_GQA(
        self,
        head_num_k,
        max_seq_len,
        head_dim,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        int_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=int_data_type,
            batch_size=4,
            max_seq_len=max_seq_len,
            num_heads_q=8,
            num_heads_k=head_num_k,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize("head_num_k", [2, 1])
    @pytest.mark.parametrize("enable_bias, silu_scale", [(False, 0), (True, 1 / 1024)])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int32])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 1, 6, 30),
        ],
    )
    def test_hstu_jagged_forward_128bs_GQA(
        self,
        head_num_k,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        int_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=int_data_type,
            batch_size=128,
            max_seq_len=2570,
            num_heads_q=2,
            num_heads_k=head_num_k,
            head_dim_qk=256,
            head_dim_v=256,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize("head_num_k", [4, 2, 1])
    @pytest.mark.parametrize("enable_bias, silu_scale", [(False, 0), (True, 1 / 1024)])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 3, 6, 30),
        ],
    )
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    def test_hstu_jagged_forward_2048bs_GQA(
        self,
        head_num_k,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=torch.int32,
            batch_size=2048,
            max_seq_len=16,
            num_heads_q=4,
            num_heads_k=head_num_k,
            head_dim_qk=256,
            head_dim_v=256,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    # qk_dim != v_dim（head_q=4 固定；覆盖 batch 4/128/1024、head_k 4/2/1、dim_qk 64/96/128、
    # dim_v 16/32/48/64/80，含 qk>v、qk=v、qk<v 三种关系）
    @pytest.mark.parametrize(
        "batch_size, head_num_k, head_dim_qk, head_dim_v",
        [
            (4, 4, 64, 16),
            (128, 2, 96, 48),
            (1024, 1, 128, 80),
            (4, 1, 64, 32),
            (128, 4, 128, 64),
            (4, 2, 64, 80),
        ],
    )
    @pytest.mark.parametrize("enable_bias, silu_scale", [(False, 0), (True, 1 / 1024)])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int32])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 1, 6, 30),
            (MaskType.TRIL, 3, 6, 0),
        ],
    )
    def test_hstu_jagged_forward_VDA(
        self,
        batch_size,
        head_num_k,
        head_dim_qk,
        head_dim_v,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        int_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=int_data_type,
            batch_size=batch_size,
            max_seq_len=15,
            num_heads_q=4,
            num_heads_k=head_num_k,
            head_dim_qk=head_dim_qk,
            head_dim_v=head_dim_v,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize(
        "batch_size, head_num, max_seq_len, head_dim",
        [
            (1, 2, 15, 16),
            (16, 4, 1024, 128),
            (1, 4, 1024, 16),
            (16, 2, 15, 128),
        ],
    )
    @pytest.mark.parametrize("enable_bias, silu_scale", [(False, 0), (True, 1 / 1024)])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int64])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 1, 6, 30),
            (MaskType.TRIL, 3, 6, 0),
        ],
    )
    def test_hstu_jagged_deterministic_forward(
        self,
        batch_size,
        head_num,
        max_seq_len,
        head_dim,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        int_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=int_data_type,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads_q=head_num,
            num_heads_k=head_num,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale, deterministic=True)

    # fp8
    is_a5 = False
    if is_a5:

        @pytest.mark.parametrize("batch_size", [4, 128, 256])
        @pytest.mark.parametrize("head_num_q", [8])
        @pytest.mark.parametrize("head_num_k", [8, 4, 2, 1])
        @pytest.mark.parametrize("max_seq_len", [15, 256])
        @pytest.mark.parametrize("head_dim_qk", [64, 96, 128])
        @pytest.mark.parametrize("head_dim_v", [16, 32, 48, 64, 80])
        @pytest.mark.parametrize("enable_bias", [True, False])
        @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
        @pytest.mark.parametrize("float_data_type", [torch.float8_e4m3fn])
        @pytest.mark.parametrize("int_data_type", [torch.int32])
        @pytest.mark.parametrize(
            "mask_type, target_group_size, max_num_context, max_num_target",
            [
                (MaskType.NONE, 0, 0, 0),
                (MaskType.CUSTOM, 0, 0, 0),
                (MaskType.TRIL, 1, 0, 30),
                (MaskType.TRIL, 3, 0, 30),
                (MaskType.TRIL, 1, 6, 0),
                (MaskType.TRIL, 3, 6, 0),
                (MaskType.TRIL, 1, 6, 30),
                (MaskType.TRIL, 3, 6, 30),
            ],
        )
        def test_hstu_jagged_forward_fp8(
            self,
            batch_size,
            head_num_q,
            head_num_k,
            max_seq_len,
            head_dim_qk,
            head_dim_v,
            enable_bias,
            mask_type,
            silu_scale,
            float_data_type,
            int_data_type,
            target_group_size,
            max_num_context,
            max_num_target,
        ):
            qkv_shape_info = QKVShapeInfo(
                float_type=float_data_type,
                int_type=int_data_type,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_heads_q=head_num_q,
                num_heads_k=head_num_k,
                head_dim_qk=head_dim_qk,
                head_dim_v=head_dim_v,
            )
            mask_info = MaskGenInfo(
                mask_type=mask_type,
                max_num_context=max_num_context,
                max_num_target=max_num_target,
                target_group_size=target_group_size,
            )

            self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)


def deltaqk_data_gen(qkv_shape_info: QKVShapeInfo, mask_info: MaskGenInfo, enable_bias: bool):
    int_type = qkv_shape_info.int_type
    batch_size = qkv_shape_info.batch_size
    min_seq_len, max_seq_len = qkv_shape_info.min_seq_len, qkv_shape_info.max_seq_len
    seq_lens = np.random.randint(min_seq_len, max_seq_len + 1, batch_size)
    seq_lens_k = seq_lens + np.random.randint(0, max_seq_len - seq_lens + 1)

    seq_offset = torch.concat((torch.zeros((1,)), torch.cumsum(torch.from_numpy(seq_lens), axis=0))).to(int_type)
    seq_offset_k = torch.concat((torch.zeros((1,)), torch.cumsum(torch.from_numpy(seq_lens_k), axis=0))).to(int_type)
    max_seq_len = np.max(seq_lens)
    max_seq_len_k = np.max(seq_lens_k)
    total_seqs_q = np.sum(seq_lens)
    total_seqs_k = np.sum(seq_lens_k)

    float_type = qkv_shape_info.float_type
    head_num_q, head_num_k, head_dim_qk, head_dim_v = (
        qkv_shape_info.num_heads_q,
        qkv_shape_info.num_heads_k,
        qkv_shape_info.head_dim_qk,
        qkv_shape_info.head_dim_v,
    )

    if float_type == torch.float8_e4m3fn:
        data_type = torch.float16
    else:
        data_type = float_type
    q = torch.rand(total_seqs_q, head_num_q, head_dim_qk, dtype=data_type).uniform_(-1, 1)
    k = torch.rand(total_seqs_k, head_num_k, head_dim_qk, dtype=data_type).uniform_(-1, 1)
    v = torch.rand(total_seqs_k, head_num_k, head_dim_v, dtype=data_type).uniform_(-1, 1)

    if float_type == torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)

    bias = torch.rand(batch_size, head_num_q, max_seq_len, max_seq_len_k).to(float_type) if enable_bias else None

    if mask_info.mask_type == MaskType.TRIL:
        mask = torch.zeros(size=(batch_size, head_num_q, max_seq_len, max_seq_len_k))
        for i, (seq_len, seq_len_k) in enumerate(zip(seq_lens, seq_lens_k)):
            delta_qk = seq_len_k - seq_len
            mask[i, :, :, :] = torch.tril(torch.ones(1, head_num_q, max_seq_len, max_seq_len_k), diagonal=delta_qk)
        mask = mask.cpu().to(float_type)
    elif mask_info.mask_type == MaskType.CUSTOM:
        mask = torch.randint(0, 2, dtype=data_type, size=(batch_size, head_num_q, max_seq_len, max_seq_len_k))
        mask = mask.cpu().to(float_type)
    else:
        mask = None

    qkv_tensors = (q, k, v, seq_offset, seq_offset_k)
    mask_tensors = (mask_info.mask_type, mask)
    max_seq_lens = (max_seq_len, max_seq_len_k)
    return qkv_tensors, mask_tensors, bias, max_seq_lens


class TestHstuDeltaqkDemo:
    @staticmethod
    def custom_op_exec(qkv_tensors, mask_tensors, bias, silu_scale, max_seq_lens, deterministic):
        q, k, v, seq_offset, seq_offset_k = qkv_tensors
        max_seq_len_q, max_seq_len_k = max_seq_lens
        mask_type, invalid_attn_mask = mask_tensors

        q_npu = q.to("npu")
        k_npu = k.to("npu")
        v_npu = v.to("npu")
        bias_npu = bias.to("npu") if isinstance(bias, torch.Tensor) else None
        mask_npu = invalid_attn_mask.to("npu") if mask_type is MaskType.CUSTOM else None
        seq_offset = seq_offset.to("npu")
        seq_offset_k = seq_offset_k.to("npu")

        output = torch.ops.mxrec.hstu_jagged.delta(
            q_npu,
            k_npu,
            v_npu,
            mask=mask_npu,
            attn_bias=bias_npu,
            mask_type=mask_type,
            max_seq_len=max_seq_len_q,
            max_seq_len_k=max_seq_len_k,
            silu_scale=silu_scale,
            seq_offset=seq_offset,
            seq_offset_k=seq_offset_k,
            deterministic=deterministic,
        )
        torch.npu.synchronize()
        return output.cpu().reshape(-1)

    @staticmethod
    def golden_op_exec(qkv_tensors, mask_tensors, bias, silu_scale, max_seq_lens):
        q, k, v, seq_offset, seq_offset_k = qkv_tensors
        max_seq_len_q, max_seq_len_k = max_seq_lens
        mask_type, mask = mask_tensors
        (_, head_nums_q, head_dim), data_type = q.shape, q.dtype
        head_nums_k = k.shape[1]
        head_dim_v = v.shape[2]
        batch_size = seq_offset.shape[0] - 1

        if head_nums_q != head_nums_k:
            assert head_nums_q % head_nums_k == 0, (
                f"head_num_q ({head_nums_q}) must be divisible by head_num_k({head_nums_k}) "
            )
        h_qk_ratio = head_nums_q // head_nums_k

        seq_lens = np.zeros((batch_size,)).astype(np.int32)
        seq_lens_k = np.zeros((batch_size,)).astype(np.int32)
        for batch_id in range(batch_size):
            seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]
            seq_lens_k[batch_id] = seq_offset_k[batch_id + 1] - seq_offset_k[batch_id]

        silu_scale = 1 / max_seq_len_q if silu_scale == 0 else silu_scale

        use_float8 = data_type == torch.float8_e4m3fn
        if use_float8:
            q_dens = jagged_to_dense(q, seq_lens, head_nums_q, head_dim).to(torch.float32).to("npu")
            k_dens = jagged_to_dense(k, seq_lens_k, head_nums_k, head_dim).to(torch.float32).to("npu")
            v_dens = jagged_to_dense(v, seq_lens_k, head_nums_k, head_dim_v).to(torch.float32).to("npu")
            mask = mask.to(torch.float32).to("npu") if isinstance(mask, torch.Tensor) else None
            bias = bias.to(torch.float32).to("npu") if isinstance(bias, torch.Tensor) else None
        else:
            q_dens = jagged_to_dense(q, seq_lens, head_nums_q, head_dim).to(data_type).to("npu")
            k_dens = jagged_to_dense(k, seq_lens_k, head_nums_k, head_dim).to(data_type).to("npu")
            v_dens = jagged_to_dense(v, seq_lens_k, head_nums_k, head_dim_v).to(data_type).to("npu")
            mask = mask.to(data_type).to("npu") if isinstance(mask, torch.Tensor) else None
            bias = bias.to(data_type).to("npu") if isinstance(bias, torch.Tensor) else None

        k_dens_expanded = k_dens.repeat_interleave(h_qk_ratio, dim=2)
        v_dens_expanded = v_dens.repeat_interleave(h_qk_ratio, dim=2)

        q_dens = q_dens.permute(0, 2, 1, 3)
        k_dens = k_dens_expanded.permute(0, 2, 3, 1)
        qk_attn = torch.matmul(q_dens, k_dens).to(torch.float32)

        if isinstance(bias, torch.Tensor):
            bias = bias.to(torch.float32)
            qk_attn = qk_attn + bias

        qk_attn = F.silu(qk_attn) * silu_scale

        if isinstance(mask, torch.Tensor):
            mask = mask.to(torch.float32)
            qk_attn = qk_attn * mask

        v_dens = v_dens_expanded.permute(0, 2, 1, 3)

        if use_float8:
            qk_attn = qk_attn.to(torch.float8_e4m3fn)
            qk_attn = qk_attn.to(torch.float32)
        else:
            qk_attn = qk_attn.to(data_type)
        attn_output = torch.matmul(qk_attn, v_dens)

        if use_float8:
            attn_output = attn_output.to(torch.float16)

        attn_output = attn_output.permute(0, 2, 1, 3).cpu()

        attn_output = dense_to_jagged(q, attn_output, seq_lens)

        torch.npu.synchronize()
        return attn_output.to(torch.float16).reshape(-1) if use_float8 else attn_output.to(data_type).reshape(-1)

    def execute(self, qkv_shape_info, mask_info, enable_bias, silu_scale, deterministic=False):
        qkv_tensors, mask_tensors, bias, max_seq_lens = deltaqk_data_gen(qkv_shape_info, mask_info, enable_bias)
        output = self.custom_op_exec(qkv_tensors, mask_tensors, bias, silu_scale, max_seq_lens, deterministic)
        golden = self.golden_op_exec(qkv_tensors, mask_tensors, bias, silu_scale, max_seq_lens)

        if qkv_shape_info.float_type == torch.bfloat16:
            res = allclose(output, golden, 5e-3, 5e-3)
        elif qkv_shape_info.float_type == torch.float8_e4m3fn:
            res = allclose(output, golden, 1e-3, 1e-3)
        elif qkv_shape_info.float_type == torch.float16:
            res = allclose(output, golden, 1e-3, 1e-3)
        else:
            res = allclose(output, golden, 1e-4, 1e-4)
        assert res

    @pytest.mark.parametrize(
        "batch_size, head_num, max_seq_len, head_dim",
        [
            (1, 2, 15, 16),
            (16, 4, 1024, 128),
            (1, 4, 1024, 16),
            (16, 2, 15, 128),
        ],
    )
    @pytest.mark.parametrize("enable_bias, silu_scale", [(False, 0), (True, 1 / 1024)])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    def test_hstu_deltaqk_forward(
        self, batch_size, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale, data_type
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=data_type,
            int_type=torch.int32,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads_q=head_num,
            num_heads_k=head_num,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(mask_type=mask_type, max_num_context=0, max_num_target=0, target_group_size=0)
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize("enable_bias, silu_scale", [(False, 0), (True, 1 / 1024)])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    def test_hstu_deltaqk_forward_128bs(self, enable_bias, mask_type, silu_scale, data_type):
        qkv_shape_info = QKVShapeInfo(
            float_type=data_type,
            int_type=torch.int32,
            batch_size=128,
            max_seq_len=2570,
            num_heads_q=2,
            num_heads_k=2,
            head_dim_qk=256,
            head_dim_v=256,
        )
        mask_info = MaskGenInfo(mask_type=mask_type, max_num_context=0, max_num_target=0, target_group_size=0)
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize("enable_bias, silu_scale", [(False, 0), (True, 1 / 1024)])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    def test_hstu_deltaqk_forward_2048bs(self, enable_bias, mask_type, silu_scale, data_type):
        qkv_shape_info = QKVShapeInfo(
            float_type=data_type,
            int_type=torch.int32,
            batch_size=2048,
            max_seq_len=16,
            num_heads_q=2,
            num_heads_k=2,
            head_dim_qk=256,
            head_dim_v=256,
        )
        mask_info = MaskGenInfo(mask_type=mask_type, max_num_context=0, max_num_target=0, target_group_size=0)
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    ## GQA（batch=4 固定；head_q/head_k 覆盖 8:8/8:4/8:2/8:1 全部比例）
    @pytest.mark.parametrize(
        "head_num_q, head_num_k, max_seq_len, head_dim",
        [
            (8, 8, 16, 32),
            (8, 4, 1024, 64),
            (8, 2, 16, 64),
            (8, 1, 1024, 32),
        ],
    )
    @pytest.mark.parametrize("enable_bias, silu_scale", [(False, 0), (True, 1 / 1024)])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    def test_hstu_deltaqk_forward_GQA(
        self, head_num_q, head_num_k, max_seq_len, head_dim, enable_bias, mask_type, silu_scale, data_type
    ):
        set_seed(42)
        qkv_shape_info = QKVShapeInfo(
            float_type=data_type,
            int_type=torch.int32,
            batch_size=4,
            max_seq_len=max_seq_len,
            num_heads_q=head_num_q,
            num_heads_k=head_num_k,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(mask_type=mask_type, max_num_context=0, max_num_target=0, target_group_size=0)
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    ## GQA + 大 batch（batch=128, seq=16 固定）
    @pytest.mark.parametrize(
        "head_num_k, head_dim",
        [
            (4, 32),
            (2, 64),
            (1, 32),
        ],
    )
    @pytest.mark.parametrize("enable_bias, silu_scale", [(False, 0), (True, 0)])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    def test_hstu_deltaqk_forward_128bs_GQA(self, head_num_k, head_dim, enable_bias, mask_type, silu_scale, data_type):
        set_seed(42)
        qkv_shape_info = QKVShapeInfo(
            float_type=data_type,
            int_type=torch.int32,
            batch_size=128,
            max_seq_len=16,
            num_heads_q=4,
            num_heads_k=head_num_k,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(mask_type=mask_type, max_num_context=0, max_num_target=0, target_group_size=0)
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize(
        "batch_size, head_num_q, head_num_k, max_seq_len, head_dim_qk, head_dim_v",
        [
            (4, 4, 2, 15, 64, 80),
            (32, 4, 1, 128, 96, 48),
            (128, 6, 3, 1024, 128, 16),
            (4, 8, 4, 15, 128, 32),
            (32, 4, 2, 128, 64, 64),
            (128, 4, 1, 1024, 96, 80),
        ],
    )
    @pytest.mark.parametrize("enable_bias, silu_scale", [(True, 0), (False, 1 / 1024)])
    @pytest.mark.parametrize("mask_type", [MaskType.NONE, MaskType.CUSTOM, MaskType.TRIL])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    def test_hstu_varlen_forward(
        self,
        batch_size,
        head_num_q,
        head_num_k,
        max_seq_len,
        head_dim_qk,
        head_dim_v,
        enable_bias,
        mask_type,
        silu_scale,
        data_type,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=data_type,
            int_type=torch.int32,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads_q=head_num_q,
            num_heads_k=head_num_k,
            head_dim_qk=head_dim_qk,
            head_dim_v=head_dim_v,
        )
        mask_info = MaskGenInfo(mask_type=mask_type, max_num_context=0, max_num_target=0, target_group_size=0)
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize(
        "batch_size, head_num, max_seq_len, head_dim",
        [
            (1, 2, 15, 16),
            (16, 4, 1024, 128),
            (1, 4, 1024, 16),
            (16, 2, 15, 128),
        ],
    )
    @pytest.mark.parametrize("enable_bias, silu_scale", [(False, 0), (True, 1 / 1024)])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    def test_hstu_deltaqk_deterministic_forward(
        self, batch_size, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale, data_type
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=data_type,
            int_type=torch.int64,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads_q=head_num,
            num_heads_k=head_num,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(mask_type=mask_type, max_num_context=0, max_num_target=0, target_group_size=0)
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale, deterministic=True)

    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.parametrize("head_num", [4])
    @pytest.mark.parametrize("max_seq_len", [512])
    @pytest.mark.parametrize("head_dim_qk", [72])
    @pytest.mark.parametrize("head_dim_v", [64, 80])
    @pytest.mark.parametrize("enable_bias", [False])
    @pytest.mark.parametrize("mask_type", [MaskType.NONE])
    @pytest.mark.parametrize("silu_scale", [1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    def test_hstu_deltaqk_forward_MASK_NONE(
        self, batch_size, head_num, max_seq_len, head_dim_qk, head_dim_v, enable_bias, mask_type, silu_scale, data_type
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=data_type,
            int_type=torch.int64,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads_q=head_num,
            num_heads_k=head_num,
            head_dim_qk=head_dim_qk,
            head_dim_v=head_dim_v,
        )
        mask_info = MaskGenInfo(mask_type=mask_type, max_num_context=0, max_num_target=0, target_group_size=0)

        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    # fp8
    is_a5 = False
    if is_a5:

        @pytest.mark.parametrize("batch_size", [4, 32, 128])
        @pytest.mark.parametrize(
            "head_num_q, head_num_k",
            [
                (4, 2),
                (4, 1),
                (6, 3),
                (8, 4),
            ],
        )
        @pytest.mark.parametrize("max_seq_len", [15, 256])
        @pytest.mark.parametrize("head_dim_qk", [64, 72, 96, 128])
        @pytest.mark.parametrize("head_dim_v", [80, 64, 48, 32, 16])
        @pytest.mark.parametrize("enable_bias", [True, False])
        @pytest.mark.parametrize("mask_type", [MaskType.NONE, MaskType.CUSTOM, MaskType.TRIL])
        @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
        @pytest.mark.parametrize("data_type", [torch.float8_e4m3fn])
        def test_hstu_varlen_forward_fp8(
            self,
            batch_size,
            head_num_q,
            head_num_k,
            max_seq_len,
            head_dim_qk,
            head_dim_v,
            enable_bias,
            mask_type,
            silu_scale,
            data_type,
        ):
            qkv_shape_info = QKVShapeInfo(
                float_type=data_type,
                int_type=torch.int32,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_heads_q=head_num_q,
                num_heads_k=head_num_k,
                head_dim_qk=head_dim_qk,
                head_dim_v=head_dim_v,
            )
            mask_info = MaskGenInfo(mask_type=mask_type, max_num_context=0, max_num_target=0, target_group_size=0)
            self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
