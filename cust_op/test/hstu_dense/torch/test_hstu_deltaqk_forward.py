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

from test_common_utils import get_chip, allclose, MaskType, jagged_to_dense, dense_to_jagged, QKVShapeInfo, MaskGenInfo


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
    num_heads, attention_dim = qkv_shape_info.num_heads, qkv_shape_info.attention_dim
    q = torch.rand(total_seqs_q, num_heads, attention_dim).to(float_type).uniform_(-1, 1)
    k = torch.rand(total_seqs_k, num_heads, attention_dim).to(float_type).uniform_(-1, 1)
    v = torch.rand(total_seqs_k, num_heads, attention_dim).to(float_type).uniform_(-1, 1)

    bias = torch.rand(batch_size, num_heads, max_seq_len, max_seq_len_k).to(float_type) \
        if enable_bias else None

    if mask_info.mask_type == MaskType.TRIL:
        mask = torch.zeros(size=(batch_size, num_heads, max_seq_len, max_seq_len_k))
        for i, (seq_len, seq_len_k) in enumerate(zip(seq_lens, seq_lens_k)):
            delta_qk = seq_len_k - seq_len
            mask[i, :, :, :] = torch.tril(torch.ones(1, num_heads, max_seq_len, max_seq_len_k),
                                          diagonal=delta_qk)
    elif mask_info.mask_type == MaskType.CUSTOM:
        mask = torch.randint(0, 2, dtype=float_type,
                             size=(batch_size, num_heads, max_seq_len, max_seq_len_k))
    else:
        mask = None

    qkv_tensors = (q, k, v, seq_offset, seq_offset_k)
    mask_tensors = (mask_info.mask_type, mask)
    max_seq_lens = (max_seq_len, max_seq_len_k)
    return qkv_tensors, mask_tensors, bias, max_seq_lens


class TestHstuDeltaqkDemo:
    @staticmethod
    def custom_op_exec(qkv_tensors, mask_tensors, bias, silu_scale, max_seq_lens):
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

        output = torch.ops.mxrec.hstu_jagged(
            q_npu, k_npu, v_npu, mask_npu, bias_npu, mask_type, max_seq_len_q, max_seq_len_k, silu_scale,
            seq_offset, seq_offset_k
        )
        torch.npu.synchronize()
        return output.cpu().reshape(-1)

    @staticmethod
    def golden_op_exec(qkv_tensors, mask_tensors, bias, silu_scale, max_seq_lens):
        q, k, v, seq_offset, seq_offset_k = qkv_tensors
        max_seq_len_q, max_seq_len_k = max_seq_lens
        mask_type, mask = mask_tensors

        (_, head_nums, head_dim), data_type = q.shape, q.dtype
        batch_size = seq_offset.shape[0] - 1

        seq_lens = np.zeros((batch_size,)).astype(np.int64)
        seq_lens_k = np.zeros((batch_size,)).astype(np.int64)
        for batch_id in range(batch_size):
            seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]
            seq_lens_k[batch_id] = seq_offset_k[batch_id + 1] - seq_offset_k[batch_id]

        silu_scale = 1 / max_seq_len_q if silu_scale == 0 else silu_scale

        q_dens = jagged_to_dense(q, seq_lens, head_nums, head_dim).to(data_type).to("npu")
        k_dens = jagged_to_dense(k, seq_lens_k, head_nums, head_dim).to(data_type).to("npu")
        v_dens = jagged_to_dense(v, seq_lens_k, head_nums, head_dim).to(data_type).to("npu")
        mask = mask.to(data_type).to("npu") if isinstance(mask, torch.Tensor) else None
        bias = bias.to(data_type).to("npu") if isinstance(bias, torch.Tensor) else None

        q_dens = q_dens.permute(0, 2, 1, 3)
        k_dens = k_dens.permute(0, 2, 3, 1)
        qk_attn = torch.matmul(q_dens, k_dens).to(torch.float32)

        if isinstance(bias, torch.Tensor):
            bias = bias.to(torch.float32)
            qk_attn = qk_attn + bias

        qk_attn = F.silu(qk_attn) * silu_scale

        if isinstance(mask, torch.Tensor):
            mask = mask.to(torch.float32)
            qk_attn = qk_attn * mask

        v_dens = v_dens.permute(0, 2, 1, 3)

        qk_attn = qk_attn.to(data_type)
        attn_output = torch.matmul(qk_attn, v_dens)
        attn_output = attn_output.permute(0, 2, 1, 3).cpu()
        attn_output = dense_to_jagged(q, attn_output, seq_lens)

        torch.npu.synchronize()
        return attn_output.to(data_type).reshape(-1)

    def execute(self, qkv_shape_info, mask_info, enable_bias, silu_scale):
        qkv_tensors, mask_tensors, bias, max_seq_lens = deltaqk_data_gen(qkv_shape_info, mask_info, enable_bias)
        output = self.custom_op_exec(qkv_tensors, mask_tensors, bias, silu_scale, max_seq_lens)
        golden = self.golden_op_exec(qkv_tensors, mask_tensors, bias, silu_scale, max_seq_lens)

        if qkv_shape_info.float_type == torch.bfloat16:
            res = allclose(output, golden, 1e-2, 1e-2)
        elif qkv_shape_info.float_type == torch.float16:
            res = allclose(output, golden, 1e-3, 1e-3)
        else:
            res = allclose(output, golden, 1e-4, 1e-4)
        assert res

    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("max_seq_len", [15, 1024])
    @pytest.mark.parametrize("head_dim", [16, 128])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_deltaqk_forward(self, batch_size, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                                  data_type):
        qkv_shape_info = QKVShapeInfo(float_type=data_type,
                                      int_type=torch.int64,
                                      batch_size=batch_size,
                                      max_seq_len=max_seq_len,
                                      num_heads=head_num,
                                      attention_dim=head_dim)
        mask_info = MaskGenInfo(mask_type=mask_type,
                                max_num_context=0,
                                max_num_target=0,
                                target_group_size=0)
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize("head_num", [2])
    @pytest.mark.parametrize("max_seq_len", [2570])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_deltaqk_forward_128bs(self, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                                        data_type):
        qkv_shape_info = QKVShapeInfo(float_type=data_type,
                                      int_type=torch.int64,
                                      batch_size=128,
                                      max_seq_len=max_seq_len,
                                      num_heads=head_num,
                                      attention_dim=head_dim)
        mask_info = MaskGenInfo(mask_type=mask_type,
                                max_num_context=0,
                                max_num_target=0,
                                target_group_size=0)
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize("head_num", [2])
    @pytest.mark.parametrize("max_seq_len", [16])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_deltaqk_forward_2048bs(self, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                                         data_type):
        qkv_shape_info = QKVShapeInfo(float_type=data_type,
                                      int_type=torch.int64,
                                      batch_size=2048,
                                      max_seq_len=max_seq_len,
                                      num_heads=head_num,
                                      attention_dim=head_dim)
        mask_info = MaskGenInfo(mask_type=mask_type,
                                max_num_context=0,
                                max_num_target=0,
                                target_group_size=0)
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)
