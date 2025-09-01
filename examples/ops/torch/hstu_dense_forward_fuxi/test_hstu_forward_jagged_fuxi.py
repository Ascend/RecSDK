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

import os
import sys
import sysconfig
import pytest
import torch
import torch_npu
import torch.nn.functional as F
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
common_dir = os.path.abspath(os.path.join(current_dir, "..", "common"))
sys.path.append(common_dir)
from utils import allclose

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

device_id: int = 0

mask_tril: int = 0
mask_triu: int = 1
mask_none: int = 2
mask_custom: int = 3

bfloat16_pre: float = 5e-3
float16_pre: float = 1e-3
float32_pre: float = 1e-4

torch.npu.set_device(device_id)


def jagged_data_gen(batch_size, max_seq_len, num_heads, attention_dim, mask_type):
    seq_lens = np.random.randint(1, max_seq_len + 1, (batch_size))

    seq_offset = torch.concat((torch.zeros((1, ), dtype=torch.int64), \
        torch.cumsum(torch.from_numpy(seq_lens), axis=0))).to(torch.int64).numpy()
    
    total_seqs = np.sum(seq_lens)

    q = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32).uniform_(-1, 1)
    k = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32).uniform_(-1, 1)
    v = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32).uniform_(-1, 1)

    ts_bias = torch.zeros(batch_size, max_seq_len, max_seq_len).to(torch.float32).uniform_(-1, 1)
    pos_bias = torch.zeros(1, max_seq_len, max_seq_len).to(torch.float32).uniform_(-1, 1)
    for batch_id in range(batch_size):
        seq_len = seq_lens[batch_id]
        ts_bias[batch_id, 0:seq_len, 0:seq_len] = torch.rand(seq_len, seq_len).to(torch.float32)
        pos_bias[0, 0:seq_len, 0:seq_len] = torch.rand(seq_len, seq_len).to(torch.float32)

    if mask_type == mask_tril:
        mask = 1 - torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len), diagonal=1)
    else:
        mask = torch.randint(0, 2, size=(batch_size, num_heads, max_seq_len, max_seq_len))
    mask = mask.cpu().to(torch.float32)

    return q, k, v, seq_offset, ts_bias, pos_bias, mask


class TestHstuJaggedFuxi:
    @staticmethod
    def jagged_to_dense(jagged_tensor, seq_lens, max_seq_len, head_nums, atten_dim):
        need_pad_seq = []
        offset = 0
        for seq_len in seq_lens:
            src_tensor = torch.rand(max_seq_len, head_nums, atten_dim)
            src_tensor = torch.zeros((max_seq_len, head_nums, atten_dim))
            src_tensor[0:seq_len, :, :] = jagged_tensor[offset: offset + seq_len, :, :]
            need_pad_seq.append(src_tensor)
            offset = offset + seq_len
        
        dense_tensor = torch.nn.utils.rnn.pad_sequence(need_pad_seq, batch_first=True)
        return dense_tensor
    
    @staticmethod
    def dense_to_jagged(q, dense_tensor, seq_lens):
        tensor = torch.zeros_like(q).cpu()

        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            tensor[offset: offset + seq_len, :, :] = dense_tensor[batch_id, 0:seq_len, :, :]
            offset = offset + seq_len

        return tensor
    
    @staticmethod
    def custom_op_exec(q, k, v, seq_offset, ts_bias, pos_bias, mask, mask_type, max_seq_len, silu_scale, \
        enable_bias, data_type):
        q_npu = q.to(f"npu:{device_id}").to(data_type)
        k_npu = k.to(f"npu:{device_id}").to(data_type)
        v_npu = v.to(f"npu:{device_id}").to(data_type)
        ts_bias_npu = ts_bias.to(f"npu:{device_id}").to(data_type)
        pos_bias_npu = pos_bias.to(f"npu:{device_id}").to(data_type)
        mask_npu = mask.to(f"npu:{device_id}").to(data_type)

        if enable_bias:
            output = torch.ops.mxrec.hstu_fuxi(
                q_npu, k_npu, v_npu, ts_bias_npu, pos_bias_npu, mask_npu, mask_type, max_seq_len, silu_scale, \
                    "jagged", seq_offset
            )
        else:
            output = torch.ops.mxrec.hstu_fuxi(
                q_npu, k_npu, v_npu, None, None, mask_npu, mask_type, max_seq_len, silu_scale, "jagged", seq_offset
            )
        torch.npu.synchronize()
        return output.cpu().to(data_type).reshape(-1)

    def gloden_op_exec(self, q, k, v, seq_offset, ts_bias, pos_bias, mask, mask_type, max_seq_len, silu_scale, \
        enable_bias, data_type):
        head_nums = q.shape[1]
        head_dim = q.shape[2]
        batch_size = ts_bias.shape[0]

        seq_lens = np.zeros((batch_size, )).astype(np.int64)
        for batch_id in range(batch_size):
            seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]

        q_dens = self.jagged_to_dense(
            q, seq_lens, max_seq_len, head_nums, head_dim).to(data_type).to(f"npu:{device_id}")
        k_dens = self.jagged_to_dense(
            k, seq_lens, max_seq_len, head_nums, head_dim).to(data_type).to(f"npu:{device_id}")
        v_dens = self.jagged_to_dense(
            v, seq_lens, max_seq_len, head_nums, head_dim).to(data_type).to(f"npu:{device_id}")
        mask = mask.reshape(batch_size, head_nums, max_seq_len, max_seq_len).to(data_type).to(f"npu:{device_id}")
        ts_bias = ts_bias.reshape(batch_size, max_seq_len, max_seq_len).to(data_type).to(f"npu:{device_id}")
        pos_bias = pos_bias.reshape(1, max_seq_len, max_seq_len).to(data_type).to(f"npu:{device_id}")

        q_dens = q_dens.permute(0, 2, 1, 3)
        k_dens = k_dens.permute(0, 2, 3, 1)
        qk_attn = torch.matmul(q_dens, k_dens)

        qk_attn = qk_attn.to(torch.float32)
        silu_scale = 1 / max_seq_len if silu_scale == 0 else silu_scale
        qk_attn = F.silu(qk_attn) * silu_scale

        mask = mask.to(torch.float32)
        if mask_type != mask_none:
            qk_attn = qk_attn * mask

        v_dens = v_dens.permute(0, 2, 1, 3)

        qk_attn = qk_attn.to(data_type)
        atten_output = torch.matmul(qk_attn, v_dens)
        atten_output = atten_output.permute(0, 2, 1, 3).cpu()
        atten_output = self.dense_to_jagged(q, atten_output, seq_lens)
        atten_output = atten_output.reshape(-1, head_nums * head_dim)
        torch.npu.synchronize()

        if enable_bias:
            # timestampBias
            ts_bias = ts_bias.unsqueeze(1).repeat(1, head_nums, 1, 1)
            ts_bias = ts_bias.to(torch.float32)
            if mask_type != mask_none:
                ts_bias = ts_bias * mask

            ts_tmp = ts_bias.to(data_type)
            ts_out = torch.matmul(ts_tmp, v_dens)
            ts_out = ts_out.permute(0, 2, 1, 3).cpu()
            ts_out = self.dense_to_jagged(q, ts_out, seq_lens)
            ts_out = ts_out.reshape(-1, head_nums * head_dim)

            # positionBias
            pos_bias = pos_bias.unsqueeze(0).repeat(batch_size, head_nums, 1, 1)
            pos_bias = pos_bias.to(torch.float32)
            if mask_type != mask_none:
                pos_bias = pos_bias * mask

            pos_tmp = pos_bias.to(data_type)
            pos_out = torch.matmul(pos_tmp, v_dens)
            pos_out = pos_out.permute(0, 2, 1, 3).cpu()
            pos_out = self.dense_to_jagged(q, pos_out, seq_lens)
            pos_out = pos_out.reshape(-1, head_nums * head_dim)

            atten_output = torch.cat((atten_output, ts_out, pos_out), -1)

        return atten_output.to(data_type).reshape(-1)

    def execute(self, batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale, data_type):
        q, k, v, seq_offset, ts_bias, pos_bias, mask = jagged_data_gen(batch_size, max_seq_len, head_num, head_dim, \
            mask_type)

        gloden = self.gloden_op_exec(q, k, v, seq_offset, ts_bias, pos_bias, mask, mask_type, max_seq_len, silu_scale, \
            enable_bias, data_type)
        output = self.custom_op_exec(q, k, v, seq_offset, ts_bias, pos_bias, mask, mask_type, max_seq_len, silu_scale, \
            enable_bias, data_type)
        

        if data_type == torch.bfloat16:
            res = allclose(output, gloden, bfloat16_pre, bfloat16_pre)
        elif data_type == torch.float16:
            res = allclose(output, gloden, float16_pre, float16_pre)
        else:
            res = allclose(output, gloden, float32_pre, float32_pre)
        assert res

    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("max_seq_len", [15, 1024])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("head_dim", [16, 128])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [mask_tril, mask_none, mask_custom])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    def test_hstu_forward_jagged_fuxi(self, batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, \
        silu_scale, data_type):
        self.execute(batch_size, max_seq_len, head_num, head_dim, enable_bias, mask_type, silu_scale, data_type)