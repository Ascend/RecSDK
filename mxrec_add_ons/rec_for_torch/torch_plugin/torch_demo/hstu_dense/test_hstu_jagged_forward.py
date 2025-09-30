#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_common_utils import allclose, jagged_to_dense, dense_to_jagged, MaskType, QKVShapeInfo, MaskGenInfo
from test_target_mask import cached_create_causal_mask, ScoreShapeParam


def jagged_data_gen(qkv_shape_info: QKVShapeInfo, mask_info: MaskGenInfo, enable_bias: bool):
    int_type = qkv_shape_info.int_type
    batch_size = qkv_shape_info.batch_size
    max_num_context, max_num_target = mask_info.max_num_context, mask_info.max_num_target
    num_context = torch.randint(0, max_num_context + 1, (batch_size,), dtype=int_type)
    num_target = torch.randint(0, max_num_target + 1, (batch_size,), dtype=int_type)

    float_type = qkv_shape_info.float_type
    min_seq_len, max_seq_len = qkv_shape_info.min_seq_len, qkv_shape_info.max_seq_len

    seq_lens = torch.randint(min_seq_len, max_seq_len + 1, (batch_size,))
    if mask_info.mask_type == MaskType.TRIL:
        seq_lens += num_context + num_target
    seq_offset = torch.concat((torch.zeros((1,)), torch.cumsum(seq_lens, dim=0))).to(int_type)
    max_seq_len, total_seqs = max(seq_lens.tolist()), sum(seq_lens.tolist())

    num_heads, attention_dim = qkv_shape_info.num_heads, qkv_shape_info.attention_dim
    q = torch.rand(total_seqs, num_heads, attention_dim).to(float_type)
    q = q.uniform_(-1, 1)
    k = torch.rand(total_seqs, num_heads, attention_dim).to(float_type)
    k = k.uniform_(-1, 1)
    v = torch.rand(total_seqs, num_heads, attention_dim).to(float_type)
    v = v.uniform_(-1, 1)

    rel_attn_bias = torch.rand(batch_size, num_heads, max_seq_len, max_seq_len).to(float_type) \
        if enable_bias else None

    if mask_info.mask_type == MaskType.TRIL:
        mask = torch.zeros(batch_size, num_heads, max_seq_len, max_seq_len)
        for batch_id, seq_len in enumerate(seq_lens.tolist()):
            score_shape_param = ScoreShapeParam(
                seq_len=seq_len,
                num_target=num_target[batch_id],
                num_context=num_context[batch_id],
                num_history=seq_len - num_target[batch_id],
                target_group_size=mask_info.target_group_size
            )
            mask[batch_id, :, :seq_len, :seq_len] = cached_create_causal_mask(score_shape_param)
        mask = mask.cpu().to(float_type)
    elif mask_info.mask_type == MaskType.CUSTOM:
        mask = torch.randint(0, 2, size=(batch_size, num_heads, max_seq_len, max_seq_len))
        mask = mask.cpu().to(float_type)
    else:
        mask = None

    qkv_tensors = (q, k, v, seq_offset)
    mask_tensors = (mask_info.mask_type, mask, num_context, num_target, mask_info.target_group_size)
    return qkv_tensors, mask_tensors, rel_attn_bias, max_seq_len


class TestHstuJaggedDemo:
    @staticmethod
    def custom_op_exec(qkv_tensors, mask_tensors, rel_attn_bias, silu_scale, max_seq_len):
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
        output = torch.ops.mxrec.hstu_jagged(
            q_npu, k_npu, v_npu, mask_npu, bias_npu, mask_type, max_seq_len, silu_scale, seq_offset,
            num_context, num_target, target_group_size
        )
        torch.npu.synchronize()
        return output.cpu()

    @staticmethod
    def golden_op_exec(qkv_tensors, mask_tensors, rel_attn_bias, silu_scale, max_seq_len):
        q, k, v, seq_offset = qkv_tensors
        mask_type, mask, _, _, _ = mask_tensors

        (_, head_nums, head_dim), data_type = q.shape, q.dtype
        batch_size = seq_offset.shape[0] - 1

        seq_lens = np.zeros((batch_size,)).astype(np.int64)
        for batch_id in range(batch_size):
            seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]

        silu_scale = 1 / max_seq_len if silu_scale == 0 else silu_scale

        q_dens = jagged_to_dense(q, seq_lens, head_nums, head_dim).to(data_type).to("npu")
        k_dens = jagged_to_dense(k, seq_lens, head_nums, head_dim).to(data_type).to("npu")
        v_dens = jagged_to_dense(v, seq_lens, head_nums, head_dim).to(data_type).to("npu")

        q_dens = q_dens.permute(0, 2, 1, 3)
        k_dens = k_dens.permute(0, 2, 3, 1)
        qk_attn = torch.matmul(q_dens, k_dens).to(torch.float32)

        if rel_attn_bias is not None:
            rel_attn_bias = rel_attn_bias.to(torch.float32).to("npu")
            qk_attn = qk_attn + rel_attn_bias

        qk_attn = F.silu(qk_attn) * silu_scale

        if mask_type != MaskType.NONE:
            mask = mask.to(torch.float32).to("npu")
            qk_attn = qk_attn * mask

        v_dens = v_dens.permute(0, 2, 1, 3)

        qk_attn = qk_attn.to(data_type)
        attn_output = torch.matmul(qk_attn, v_dens)
        attn_output = attn_output.permute(0, 2, 1, 3).cpu()
        attn_output = dense_to_jagged(q, attn_output, seq_lens)

        torch.npu.synchronize()
        return attn_output.to(data_type)

    def execute(self, qkv_shape_info, mask_info, enable_bias, silu_scale):
        qkv_tensors, mask_tensors, rel_attn_bias, max_seq_len = jagged_data_gen(qkv_shape_info, mask_info, enable_bias)

        output = self.custom_op_exec(qkv_tensors, mask_tensors, rel_attn_bias, silu_scale, max_seq_len)
        golden = self.golden_op_exec(qkv_tensors, mask_tensors, rel_attn_bias, silu_scale, max_seq_len)

        data_type = qkv_shape_info.float_type
        if data_type == torch.bfloat16:
            res = allclose(output, golden, 1e-2, 1e-2)
        elif data_type == torch.float16:
            res = allclose(output, golden, 1e-3, 1e-3)
        else:
            res = allclose(output, golden, 1e-4, 1e-4)
        assert res

    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("max_seq_len", [15, 1024])
    @pytest.mark.parametrize("head_dim", [16, 128])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int64])
    @pytest.mark.parametrize("mask_type, target_group_size, max_num_context, max_num_target", [
        (MaskType.NONE, 0, 0, 0),
        (MaskType.CUSTOM, 0, 0, 0),
        (MaskType.TRIL, 1, 0, 30),
        (MaskType.TRIL, 3, 0, 30),
        (MaskType.TRIL, 1, 6, 0),
        (MaskType.TRIL, 3, 6, 0),
        (MaskType.TRIL, 1, 6, 30),
        (MaskType.TRIL, 3, 6, 30),
    ])
    def test_hstu_jagged_forward(self, batch_size, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                                 float_data_type, int_data_type, target_group_size, max_num_context, max_num_target):
        qkv_shape_info = QKVShapeInfo(float_type=float_data_type,
                                      int_type=int_data_type,
                                      batch_size=batch_size,
                                      max_seq_len=max_seq_len,
                                      num_heads=head_num,
                                      attention_dim=head_dim)
        mask_info = MaskGenInfo(mask_type=mask_type,
                                max_num_context=max_num_context,
                                max_num_target=max_num_target,
                                target_group_size=target_group_size)
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize("head_num", [2])
    @pytest.mark.parametrize("max_seq_len", [2570])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int64])
    @pytest.mark.parametrize("mask_type, target_group_size, max_num_context, max_num_target", [
        (MaskType.NONE, 0, 0, 0),
        (MaskType.CUSTOM, 0, 0, 0),
        (MaskType.TRIL, 1, 0, 30),
        (MaskType.TRIL, 3, 0, 30),
        (MaskType.TRIL, 1, 6, 0),
        (MaskType.TRIL, 3, 6, 0),
        (MaskType.TRIL, 1, 6, 30),
        (MaskType.TRIL, 3, 6, 30),
    ])
    def test_hstu_jagged_forward_128bs(self, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                                       float_data_type, int_data_type, target_group_size, max_num_context,
                                       max_num_target):
        qkv_shape_info = QKVShapeInfo(float_type=float_data_type,
                                      int_type=int_data_type,
                                      batch_size=128,
                                      max_seq_len=max_seq_len,
                                      num_heads=head_num,
                                      attention_dim=head_dim)
        mask_info = MaskGenInfo(mask_type=mask_type,
                                max_num_context=max_num_context,
                                max_num_target=max_num_target,
                                target_group_size=target_group_size)
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize("head_num", [2])
    @pytest.mark.parametrize("max_seq_len", [16])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("mask_type, target_group_size, max_num_context, max_num_target", [
        (MaskType.NONE, 0, 0, 0),
        (MaskType.CUSTOM, 0, 0, 0),
        (MaskType.TRIL, 3, 6, 30),
    ])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    def test_hstu_jagged_forward_2048bs(self, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                                        float_data_type, target_group_size, max_num_context,
                                        max_num_target):
        qkv_shape_info = QKVShapeInfo(float_type=float_data_type,
                                      int_type=torch.int64,
                                      batch_size=2048,
                                      max_seq_len=max_seq_len,
                                      num_heads=head_num,
                                      attention_dim=head_dim)
        mask_info = MaskGenInfo(mask_type=mask_type,
                                max_num_context=max_num_context,
                                max_num_target=max_num_target,
                                target_group_size=target_group_size)
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)
