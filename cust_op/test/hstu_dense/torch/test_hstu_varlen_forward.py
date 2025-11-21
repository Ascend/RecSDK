#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch_npu

from test_common_utils import get_chip, allclose, MaskType, jagged_to_dense, dense_to_jagged


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True   # 确保CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False      # 关闭CuDNN自动优化


def valen_data_gen(batch_size, max_seq_len, head_num_q, head_num_k, attention_dim, data_type, mask_type):
    seq_lens = np.random.randint(1, max_seq_len + 1, (batch_size))
    seq_lens_k = seq_lens + np.random.randint(0, max_seq_len - seq_lens + 1)

    seq_offset = torch.concat((torch.zeros((1,), dtype=torch.int64), \
                               torch.cumsum(torch.from_numpy(seq_lens), axis=0))).to(torch.int64)
    seq_offset_k = torch.concat((torch.zeros((1,), dtype=torch.int64), \
                                 torch.cumsum(torch.from_numpy(seq_lens_k), axis=0))).to(torch.int64)
    max_seq_len = np.max(seq_lens)
    max_seq_len_k = np.max(seq_lens_k)
    total_seqs_q = np.sum(seq_lens)
    total_seqs_k = np.sum(seq_lens_k)

    q = torch.rand(total_seqs_q, head_num_q, attention_dim).to(torch.float32)
    q = q.uniform_(-1, 1)
    k = torch.rand(total_seqs_k, head_num_k, attention_dim).to(torch.float32)
    k = k.uniform_(-1, 1)
    v = torch.rand(total_seqs_k, head_num_k, attention_dim).to(torch.float32)
    v = v.uniform_(-1, 1)

    rel_attn_bias = torch.zeros(batch_size, head_num_q, max_seq_len, max_seq_len_k).to(torch.float32)
    for batch_id in range(batch_size):
        seq_len = seq_lens[batch_id]
        seq_len_k = seq_lens_k[batch_id]
        rel_attn_bias[batch_id, :, 0:seq_len, 0:seq_len_k] = torch.rand(seq_len, seq_len_k).to(torch.float32)

    if mask_type == MaskType.TRIL:
        invalid_attn_mask = torch.zeros(size=(batch_size, head_num_q, max_seq_len, max_seq_len_k))
        for i, (seq_len, seq_len_k) in enumerate(zip(seq_lens, seq_lens_k)):
            delta_qk = seq_len_k - seq_len
            invalid_attn_mask[i, :, :, :] = torch.tril(torch.ones(1, head_num_q, max_seq_len, max_seq_len_k),
                                                       diagonal=delta_qk)
    else:
        invalid_attn_mask = torch.randint(0, 2, size=(batch_size, head_num_q, max_seq_len, max_seq_len_k))
    invalid_attn_mask = invalid_attn_mask.cpu().to(torch.float32)

    return q, k, v, seq_offset, seq_offset_k, rel_attn_bias, invalid_attn_mask, max_seq_len, max_seq_len_k


class TestHstuVarlenDemo:
    @staticmethod
    def custom_op_exec(q, k, v, seq_offset, seq_offset_k, bias, mask, max_seq_len, max_seq_len_k, enable_bias,
                       mask_type, silu_scale, data_type):
        q_npu = q.to("npu").to(data_type)
        k_npu = k.to("npu").to(data_type)
        v_npu = v.to("npu").to(data_type)
        bias_npu = bias.to("npu").to(data_type)
        mask_npu = mask.to("npu").to(data_type)
        seq_offset = seq_offset.to("npu")
        seq_offset_k = seq_offset_k.to("npu")

        if enable_bias:
            # 函数重载：hstu_jagged -> hstu_jagged.delta
            output = torch.ops.mxrec.hstu_jagged(
                q_npu, k_npu, v_npu, mask_npu, bias_npu, mask_type, max_seq_len, max_seq_len_k, silu_scale,
                seq_offset, seq_offset_k
            )
        else:
            output = torch.ops.mxrec.hstu_jagged(
                q_npu, k_npu, v_npu, mask_npu, None, mask_type, max_seq_len, max_seq_len_k, silu_scale,
                seq_offset, seq_offset_k
            )
        torch.npu.synchronize()
        return output.cpu().to(data_type).reshape(-1)

    @staticmethod
    def golden_op_exec(q, k, v, seq_offset, seq_offset_k, bias, mask, max_seq_len, enable_bias, mask_type, silu_scale,
                       data_type):
        head_nums_q = q.shape[1]
        head_nums_k = k.shape[1]
        if head_nums_q != head_nums_k:
            assert head_nums_q % head_nums_k == 0, (f"head_num_q ({head_nums_q}) must be divisible by "
                                                    f"head_num_k({head_nums_k}) ")
        h_qk_ratio = head_nums_q // head_nums_k

        head_dim = q.shape[2]
        batch_size = bias.shape[0]

        seq_lens = np.zeros((batch_size,)).astype(np.int64)
        seq_lens_k = np.zeros((batch_size,)).astype(np.int64)
        for batch_id in range(batch_size):
            seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]
            seq_lens_k[batch_id] = seq_offset_k[batch_id + 1] - seq_offset_k[batch_id]

        silu_scale = 1 / max_seq_len if silu_scale == 0 else silu_scale

        q_dens = jagged_to_dense(q, seq_lens, head_nums_q, head_dim).to(data_type).to("npu")
        k_dens = jagged_to_dense(k, seq_lens_k, head_nums_k, head_dim).to(data_type).to("npu")
        v_dens = jagged_to_dense(v, seq_lens_k, head_nums_k, head_dim).to(data_type).to("npu")

        k_dens_expanded = k_dens.repeat_interleave(h_qk_ratio, dim=2)
        v_dens_expanded = v_dens.repeat_interleave(h_qk_ratio, dim=2)

        mask = mask.to(data_type).to("npu")
        attn_bias = bias.to(data_type).to("npu")

        q_dens = q_dens.permute(0, 2, 1, 3)
        k_dens = k_dens_expanded.permute(0, 2, 3, 1)

        qk_attn = torch.matmul(q_dens, k_dens)
        qk_attn = qk_attn.to(torch.float32)
        attn_bias = attn_bias.to(torch.float32)
        mask = mask.to(torch.float32)
        if enable_bias:
            qk_attn = qk_attn + attn_bias

        qk_attn = F.silu(qk_attn) * silu_scale

        if mask_type != MaskType.NONE:
            qk_attn = qk_attn * mask

        v_dens = v_dens_expanded.permute(0, 2, 1, 3)

        qk_attn = qk_attn.to(data_type)
        attn_output = torch.matmul(qk_attn, v_dens)
        attn_output = attn_output.permute(0, 2, 1, 3).cpu()

        attn_output = dense_to_jagged(q, attn_output, seq_lens)

        torch.npu.synchronize()
        return attn_output.to(data_type).reshape(-1)


    def execute(self, batch_size, max_seq_len, head_num_q, head_num_k, head_dim, enable_bias, mask_type, silu_scale,
                data_type):
        q, k, v, seq_offset, seq_offset_k, bias, mask, max_seq_len, max_seq_len_k = valen_data_gen(
            batch_size, max_seq_len, head_num_q, head_num_k, head_dim, data_type, mask_type
        )
        output = self.custom_op_exec(q, k, v, seq_offset, seq_offset_k, bias, mask, max_seq_len, max_seq_len_k,
                                     enable_bias, mask_type, silu_scale, data_type)
        golden = self.golden_op_exec(q, k, v, seq_offset, seq_offset_k, bias, mask, max_seq_len, enable_bias,
                                     mask_type, silu_scale, data_type)

        if data_type == torch.bfloat16:
            res = allclose(output, golden, 1e-2, 1e-2)
        elif data_type == torch.float16:
            res = allclose(output, golden, 1e-3, 1e-3)
        else:
            res = allclose(output, golden, 1e-4, 1e-4)
        assert res

    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("head_num_q, head_num_k", [
        (2, 2),
        (4, 4),
    ])
    @pytest.mark.parametrize("max_seq_len", [15, 1024])
    @pytest.mark.parametrize("head_dim", [16, 128])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_varlen_forward(self, batch_size, head_num_q, head_num_k, max_seq_len, head_dim, enable_bias,
                                 mask_type, silu_scale, data_type):
        self.execute(batch_size, max_seq_len, head_num_q, head_num_k, head_dim, enable_bias, mask_type, silu_scale,
                     data_type)

    @pytest.mark.parametrize("head_num_q, head_num_k", [(2, 2)])
    @pytest.mark.parametrize("max_seq_len", [2570])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_varlen_forward_128bs(self, head_num_q, head_num_k, max_seq_len, head_dim, enable_bias, mask_type,
                                       silu_scale, data_type):
        self.execute(128, max_seq_len, head_num_q, head_num_k, head_dim, enable_bias, mask_type, silu_scale,
                     data_type)

    @pytest.mark.parametrize("head_num_q, head_num_k", [(2, 2)])
    @pytest.mark.parametrize("max_seq_len", [16])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_varlen_forward_2048bs(self, head_num_q, head_num_k, max_seq_len, head_dim, enable_bias, mask_type,
                                        silu_scale, data_type):
        self.execute(2048, max_seq_len, head_num_q, head_num_k, head_dim, enable_bias, mask_type, silu_scale,
                     data_type)

    ## GQA
    @pytest.mark.parametrize("batch_size", [4, 16])
    @pytest.mark.parametrize("head_num_q", [8])
    @pytest.mark.parametrize("head_num_k", [8, 4, 2, 1])
    @pytest.mark.parametrize("max_seq_len", [16, 1024])
    @pytest.mark.parametrize("head_dim", [32, 64])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_varlen_forward_GQA(self, batch_size, head_num_q, head_num_k, max_seq_len, head_dim, enable_bias,
                                     mask_type, silu_scale, data_type):
        set_seed(1234)
        self.execute(batch_size, max_seq_len, head_num_q, head_num_k, head_dim, enable_bias, mask_type, silu_scale,
                     data_type)

    ## GQA
    @pytest.mark.parametrize("batch_size", [128])
    @pytest.mark.parametrize("head_num_q", [4])
    @pytest.mark.parametrize("head_num_k", [4, 2, 1])
    @pytest.mark.parametrize("max_seq_len", [16])
    @pytest.mark.parametrize("head_dim", [32, 64])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [0])
    @pytest.mark.parametrize("data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_varlen_forward_128bs_GQA(self, batch_size, head_num_q, head_num_k, max_seq_len, head_dim, enable_bias,
                                           mask_type, silu_scale, data_type):
        set_seed(1234)
        self.execute(batch_size, max_seq_len, head_num_q, head_num_k, head_dim, enable_bias, mask_type, silu_scale,
                     data_type)