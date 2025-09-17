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

from test_common_utils import init, get_chip, allclose, mask_tril, mask_triu, mask_none, mask_custom

init()


def jagged_data_gen(batch_size, max_seq_len, num_heads, attention_dim, data_type, mask_type):
    seq_lens = np.random.randint(1, max_seq_len + 1, (batch_size))

    seq_offset = torch.concat((torch.zeros((1,), dtype=torch.int64), \
                               torch.cumsum(torch.from_numpy(seq_lens), axis=0))).to(torch.int64)

    # num_context\num_target
    num_context = torch.randint(0, 129, (batch_size,), dtype=torch.int64)
    num_target = torch.randint(0, 513, (batch_size,), dtype=torch.int64)

    max_seq_len = np.max(seq_lens)
    total_seqs = np.sum(seq_lens)

    q = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32)
    q = q.uniform_(-1, 1)
    k = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32)
    k = k.uniform_(-1, 1)
    v = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32)
    v = v.uniform_(-1, 1)

    rel_attn_bias = torch.zeros(batch_size, num_heads, max_seq_len, max_seq_len).to(torch.float32)
    for batch_id in range(batch_size):
        seq_len = seq_lens[batch_id]
        rel_attn_bias[batch_id, :, 0:seq_len, 0:seq_len] = torch.rand(seq_len, seq_len).to(torch.float32)

    if mask_type == mask_tril:
        invalid_attn_mask = 1 - torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len), diagonal=1)
    else:
        invalid_attn_mask = torch.randint(0, 2, size=(batch_size, num_heads, max_seq_len, max_seq_len))
    invalid_attn_mask = invalid_attn_mask.cpu().to(torch.float32)

    return q, k, v, seq_offset, rel_attn_bias, invalid_attn_mask, max_seq_len, num_context, num_target


class TestHstuJaggedDemo:
    @staticmethod
    def jagged_to_dense(jagged_tensor, seq_lens, head_nums, atten_dim):
        need_pad_seq = []
        offset = 0
        for seq_len in seq_lens:
            src_tensor = jagged_tensor[offset: offset + seq_len, :, :].reshape(seq_len, head_nums, atten_dim)
            need_pad_seq.append(src_tensor)
            offset = offset + seq_len

        dense_tensor = torch.nn.utils.rnn.pad_sequence(need_pad_seq, batch_first=True)
        return dense_tensor

    @staticmethod
    def dense_to_jagged(q, dense_tensor, seq_lens):
        tensor = torch.zeros_like(q).cpu()

        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            tensor[offset: offset + seq_len, :, :] = dense_tensor[batch_id, 0: seq_len, :, :]
            offset = offset + seq_len

        return tensor

    @staticmethod
    def custom_op_exec(q, k, v, seq_offset, bias, mask, max_seq_len, enable_bias, mask_type, silu_scale,
                       data_type, num_context, num_target, target_group_size):
        q_npu = q.to("npu").to(data_type)
        k_npu = k.to("npu").to(data_type)
        v_npu = v.to("npu").to(data_type)
        bias_npu = bias.to("npu").to(data_type)
        mask_npu = mask.to("npu").to(data_type)
        seq_offset = seq_offset.to("npu")
        num_context = num_context.to("npu")
        num_target = num_target.to("npu")

        if enable_bias:
            output = torch.ops.mxrec.hstu_jagged(
                q_npu, k_npu, v_npu, mask_npu, bias_npu, mask_type, max_seq_len, silu_scale, seq_offset,
                num_context, num_target, target_group_size
            )
        else:
            output = torch.ops.mxrec.hstu_jagged(
                q_npu, k_npu, v_npu, mask_npu, None, mask_type, max_seq_len, silu_scale, seq_offset,
                num_context, num_target, target_group_size
            )
        torch.npu.synchronize()
        return output.cpu().to(data_type).reshape(-1)

    def gloden(self, q, k, v, seq_offset, bias, mask, max_seq_len, enable_bias, mask_type, silu_scale,
               data_type):
        head_nums = q.shape[1]
        head_dim = q.shape[2]
        batch_size = bias.shape[0]

        seq_lens = np.zeros((batch_size,)).astype(np.int64)
        for batch_id in range(batch_size):
            seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]

        silu_scale = 1 / max_seq_len if silu_scale == 0 else silu_scale

        q_dens = self.jagged_to_dense(q, seq_lens, head_nums, head_dim).to(data_type).to("npu")
        k_dens = self.jagged_to_dense(k, seq_lens, head_nums, head_dim).to(data_type).to("npu")
        v_dens = self.jagged_to_dense(v, seq_lens, head_nums, head_dim).to(data_type).to("npu")
        mask = mask.reshape(batch_size, head_nums, max_seq_len, max_seq_len).to(data_type).to("npu")
        attn_bias = bias.reshape(batch_size, head_nums, max_seq_len, max_seq_len).to(data_type).to("npu")

        q_dens = q_dens.permute(0, 2, 1, 3)
        k_dens = k_dens.permute(0, 2, 3, 1)
        qk_attn = torch.matmul(q_dens, k_dens)

        qk_attn = qk_attn.to(torch.float32)
        attn_bias = attn_bias.to(torch.float32)
        mask = mask.to(torch.float32)
        if enable_bias:
            qk_attn = qk_attn + attn_bias

        qk_attn = F.silu(qk_attn) * silu_scale

        if mask_type != mask_none:
            qk_attn = qk_attn * mask

        v_dens = v_dens.permute(0, 2, 1, 3)

        qk_attn = qk_attn.to(data_type)
        atten_output = torch.matmul(qk_attn, v_dens)
        atten_output = atten_output.permute(0, 2, 1, 3).cpu()
        atten_output = self.dense_to_jagged(q, atten_output, seq_lens)

        torch.npu.synchronize()
        return atten_output.to(data_type).reshape(-1)

    def execute(self, batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale,
                data_type, target_group_size):
        q, k, v, seq_offset, bias, mask, max_seq_len, num_context, num_target = jagged_data_gen(
            batch_size, max_seq_len, head_num, head_dim, data_type, mask_type
        )

        output = self.custom_op_exec(q, k, v, seq_offset, bias, mask, max_seq_len, enable_bias, mask_type, silu_scale,
                                     data_type, num_context, num_target, target_group_size)
        gloden_res = self.gloden(q, k, v, seq_offset, bias, mask, max_seq_len, enable_bias, mask_type, silu_scale,
                                 data_type)

        if data_type == torch.bfloat16:
            res = allclose(output, gloden_res, 1e-2, 1e-2)
        elif data_type == torch.float16:
            res = allclose(output, gloden_res, 1e-3, 1e-3)
        else:
            res = allclose(output, gloden_res, 1e-4, 1e-4)
        assert res

    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("max_seq_len", [15, 1024])
    @pytest.mark.parametrize("head_dim", [16, 128])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [mask_tril, mask_none, mask_custom])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("target_group_size", [1, 3])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_dens_forward(self, batch_size, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                               data_type, target_group_size):
        self.execute(batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale,
                     data_type, target_group_size)

    @pytest.mark.parametrize("head_num", [2])
    @pytest.mark.parametrize("max_seq_len", [2570])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [mask_tril, mask_none, mask_custom])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("target_group_size", [1, 3])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_dens_forward_128bs(self, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                                     data_type, target_group_size):
        self.execute(128, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale,
                     data_type, target_group_size)

    @pytest.mark.parametrize("head_num", [2])
    @pytest.mark.parametrize("max_seq_len", [16])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [mask_tril, mask_none, mask_custom])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("target_group_size", [1, 3])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_dens_forward_2048bs(self, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                                      data_type, target_group_size):
        self.execute(2048, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale, data_type,
                     target_group_size)
