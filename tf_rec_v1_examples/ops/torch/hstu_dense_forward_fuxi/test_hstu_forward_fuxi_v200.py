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

from pathlib import Path
import os
import shutil
import unittest
import sysconfig
import subprocess

import pytest
import torch
import numpy as np
import torch.nn.functional as F
import torch_npu

torch.npu.config.allow_internal_format = False
CURR_DIR = Path(__file__).resolve().parent
torch.ops.load_library(str(CURR_DIR.parent.parent.parent.parent / 
                           "cust_op/framework/torch_plugin/torch_library/2.6.0/hstu_dense_forward_fuxi/build/libhstu_dense_fuxi_ops.so"))

device_id: int = 0
mask_tril: int = 0
mask_triu: int = 1
mask_none: int = 2
mask_custom: int = 3

torch.npu.set_device(device_id)


def generate_tensor(batch_size, max_seq_len, num_heads, attention_dim, data_type, mask_type):
    total_num = batch_size * max_seq_len * num_heads * attention_dim

    q = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim)
    k = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim)
    v = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim)
    ts_bias = torch.rand(batch_size, max_seq_len, max_seq_len)
    pos_bias = torch.rand(batch_size, max_seq_len, max_seq_len)

    invalid_attn_mask = torch.randint(0, 2, (max_seq_len, max_seq_len))
    invalid_attn_mask = torch.tril(invalid_attn_mask)
    invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)

    return q.to(data_type).to(f"npu:{device_id}"), k.to(data_type).to(f"npu:{device_id}"), \
        v.to(data_type).to(f"npu:{device_id}"), ts_bias.to(data_type).to(f"npu:{device_id}"), \
        pos_bias.to(data_type).to(f"npu:{device_id}"), invalid_attn_mask.to(data_type).to(f"npu:{device_id}")


class TestHstuNormalFuxiDemo:
    def gloden_op_exec(self, q, k, v, ts_bias, pos_bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, \
        data_type):

        batch, seq_len, num_head, dim = q.shape

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        qk_attn = torch.matmul(q, k)

        qk_attn = qk_attn.to(torch.float32)
        mask = mask.to(torch.float32)

        real_silu_scale = 1 / max_seq_len if silu_scale == 0 else silu_scale
        qk_attn = F.silu(qk_attn) * real_silu_scale

        mask = mask.repeat(1, num_head, 1, 1)
        qk_attn = qk_attn * mask

        v = v.permute(0, 2, 1, 3)

        qk_attn = qk_attn.to(data_type)
        atten_output = torch.matmul(qk_attn, v)
        atten_output = atten_output.permute(0, 2, 1, 3)
        atten_output = atten_output.reshape(batch, seq_len, -1)
        torch.npu.synchronize()

        if enable_bias:
            ts_bias = ts_bias.to(torch.float32)
            ts_bias = ts_bias.unsqueeze(1)
            ts_bias = ts_bias.repeat(1, num_head, 1, 1)
            ts_tmp = ts_bias * mask
            ts_tmp = ts_tmp.to(data_type)
            ts_out = torch.matmul(ts_tmp, v)
            ts_out = ts_out.permute(0, 2, 1, 3)
            ts_out = ts_out.reshape(batch, seq_len, -1)

            pos_bias = pos_bias.to(torch.float32)
            pos_bias = pos_bias.unsqueeze(1)
            pos_bias = pos_bias.repeat(1, num_head, 1, 1)
            pos_tmp = pos_bias * mask
            pos_tmp = pos_tmp.to(data_type)
            pos_out = torch.matmul(pos_tmp, v)
            pos_out = pos_out.permute(0, 2, 1, 3)
            pos_out = pos_out.reshape(batch, seq_len, -1)

            atten_output = torch.cat([atten_output, ts_out, pos_out], -1)

        return atten_output.cpu().to(data_type).reshape(-1)
    
    def custom_op_exec(self, q, k, v, ts_bias, pos_bias, mask, mask_type, max_seq_len, silu_scale, enable_bias, \
        data_type):
        if enable_bias:
            output = torch.ops.mxrec.hstu_fuxi(
                q, k, v, ts_bias, pos_bias, mask, mask_type, max_seq_len, silu_scale, "normal"
            )
        else:
            output = torch.ops.mxrec.hstu_fuxi(
                q, k, v, None, None, mask, mask_type, max_seq_len, silu_scale, "normal"
            )

        torch.npu.synchronize()
        return output.cpu().to(data_type).reshape(-1)
    
    def execute(self, batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale, data_type):
        q, k, v, ts_bias, pos_bias, mask = generate_tensor(batch_size, max_seq_len, head_num, head_dim, data_type, \
            mask_type)

        output = self.custom_op_exec(q, k, v, ts_bias, pos_bias, mask, mask_type, max_seq_len, silu_scale, \
            enable_bias, data_type)
        gloden = self.gloden_op_exec(q, k, v, ts_bias, pos_bias, mask, mask_type, max_seq_len, silu_scale, \
            enable_bias, data_type)

        torch.npu.synchronize()

        assert torch.allclose(output, gloden, 1e-3, 1e-3)


    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("max_seq_len", [768, 1024, 1536])
    @pytest.mark.parametrize("head_dim", [64])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [mask_tril])
    @pytest.mark.parametrize("silu_scale", [1 / 256])
    @pytest.mark.parametrize("data_type", [torch.float16])
    def test_hstu_dens_normal(self, batch_size, head_num, max_seq_len, head_dim, enable_bias, mask_type, \
        silu_scale, data_type):
        self.execute(batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale, data_type)
