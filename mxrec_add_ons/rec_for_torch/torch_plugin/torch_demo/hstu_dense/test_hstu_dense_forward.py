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
import pytest
import torch
import torch.nn.functional as F

from test_common_utils import init, get_chip, allclose, mask_tril, mask_none, mask_custom

init()


def generate_tensor(batch_size, max_seq_len, num_heads, attention_dim, data_type, mask_type):
    total_num = batch_size * max_seq_len * num_heads * attention_dim

    q = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim)
    k = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim)
    v = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim)
    rel_attn_bias = torch.rand(batch_size, num_heads, max_seq_len, max_seq_len)
    if get_chip():
        invalid_attn_mask = torch.randint(0, 2, (max_seq_len, max_seq_len))
        invalid_attn_mask = torch.tril(invalid_attn_mask)
        invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    elif mask_type == mask_tril:
        invalid_attn_mask = 1 - torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len), diagonal=1)
    else:
        invalid_attn_mask = torch.randint(0, 2, size=(batch_size, num_heads, max_seq_len, max_seq_len))
    return q.to(data_type).to("npu"), k.to(data_type).to("npu"), v.to(data_type).to(
        "npu"), rel_attn_bias.to(data_type).to("npu"), invalid_attn_mask.to(data_type).to(
        "npu")


class TestHstuDenseDemo:
    @staticmethod
    def gloden(q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type):
        b, n, num_heads, linear_dim = q.shape
        silu_scale = 1 / max_seq_len if silu_scale == 0 else silu_scale
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        qk_attn = torch.matmul(q, k)

        qk_attn = qk_attn.to(torch.float32)
        bias = bias.to(torch.float32)
        mask = mask.to(torch.float32)
        if enable_bias:
            qk_attn = qk_attn + bias

        qk_attn = F.silu(qk_attn) * silu_scale

        if get_chip():
            mask = mask.repeat(1, num_heads, 1, 1)
            qk_attn = qk_attn * mask
        elif mask_type != mask_none:
            qk_attn = qk_attn * mask

        v = v.permute(0, 2, 1, 3)

        qk_attn = qk_attn.to(data_type)
        atten_output = torch.matmul(qk_attn, v)
        atten_output = atten_output.permute(0, 2, 1, 3)
        torch.npu.synchronize()
        return atten_output.cpu().to(data_type).reshape(-1)

    @staticmethod
    def custom_op_exec(q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type):
        if enable_bias:
            output = torch.ops.mxrec.hstu_dense(
                q, k, v, mask, bias, mask_type, max_seq_len, silu_scale
            )
        else:
            output = torch.ops.mxrec.hstu_dense(
                q, k, v, mask, None, mask_type, max_seq_len, silu_scale
            )

        torch.npu.synchronize()
        return output.cpu().to(data_type).reshape(-1)

    def execute(self, batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale, data_type):
        q, k, v, bias, mask = generate_tensor(batch_size, max_seq_len, head_num, head_dim, data_type, mask_type)

        torch.npu.synchronize()

        output = self.custom_op_exec(q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type)
        gloden_res = self.gloden(q, k, v, bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, data_type)

        torch.npu.synchronize()

        if data_type == torch.bfloat16:
            res = allclose(output, gloden_res, 1e-2, 1e-2)
        elif data_type == torch.float16:
            res = allclose(output, gloden_res, 1e-3, 1e-3)
        else:
            res = allclose(output, gloden_res, 1e-4, 1e-4)
        assert res

    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("max_seq_len", [1, 15, 31, 256, 768, 1023, 4095])
    @pytest.mark.parametrize("head_dim", [32, 64])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [mask_tril, mask_none, mask_custom])
    @pytest.mark.parametrize("silu_scale", [1 / 256])
    @pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
    def test_hstu_dens_normal(self, batch_size, head_num, max_seq_len, head_dim, enable_bias, mask_type, silu_scale,
                              data_type):
        self.execute(batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale, data_type)
