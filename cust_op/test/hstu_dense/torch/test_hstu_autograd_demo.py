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
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from test_common_utils import MaskType, allclose

bfloat16_pre: float = 5e-3
float16_pre: float = 1e-3
float32_pre: float = 1e-4


def generate_tensor(batch_size, max_seq_len, num_heads, attention_dim, data_type, mask_type):
    total_num = batch_size * max_seq_len * num_heads * attention_dim

    q = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim).to(data_type).uniform_(-1, 1)
    k = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim).to(data_type).uniform_(-1, 1)
    v = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim).to(data_type).uniform_(-1, 1)
    rel_attn_bias = torch.rand(batch_size, num_heads, max_seq_len, max_seq_len).to(data_type).uniform_(-1, 1)

    if mask_type == MaskType.TRIL:
        mask = 1 - torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len), \
                              diagonal=1).to(data_type)
    else:
        mask = torch.randint(0, 2, size=(batch_size, num_heads, max_seq_len, max_seq_len)).to(data_type)

    return q, k, v, rel_attn_bias, mask


def jagged_data_gen(batch_size, max_seq_len, num_heads, attention_dim, mask_type):
    seq_lens = np.random.randint(1, max_seq_len + 1, (batch_size))

    seq_offset = torch.concat((torch.zeros((1,), dtype=torch.int32), \
                               torch.cumsum(torch.from_numpy(seq_lens), axis=0))).to(torch.int32).numpy()

    total_seqs = np.sum(seq_lens)

    q = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32).uniform_(-1, 1)
    k = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32).uniform_(-1, 1)
    v = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32).uniform_(-1, 1)

    rel_attn_bias = torch.zeros(batch_size, num_heads, max_seq_len, max_seq_len).to(torch.float32).uniform_(-1, 1)
    for batch_id in range(batch_size):
        seq_len = seq_lens[batch_id]
        rel_attn_bias[batch_id, :, 0:seq_len, 0:seq_len] = torch.rand(seq_len, seq_len).to(torch.float32)

    if mask_type == MaskType.TRIL:
        mask = 1 - torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len), diagonal=1)
    else:
        mask = torch.randint(0, 2, size=(batch_size, num_heads, max_seq_len, max_seq_len))
    mask = mask.cpu().to(torch.float32)

    return q, k, v, seq_offset, rel_attn_bias, mask, total_seqs


class TestHstuAutogradNormal:
    @staticmethod
    def golden_op_exec(q, k, v, bias, mask, batch_size, max_seq_len, num_heads, attention_dim, enable_bias, \
                       mask_type, silu_scale, data_type):
        q = torch.nn.Parameter(torch.Tensor(q).reshape(batch_size, max_seq_len, num_heads, \
                                                       attention_dim).to(torch.float32), requires_grad=True)
        k = torch.nn.Parameter(torch.Tensor(k).reshape(batch_size, max_seq_len, num_heads, \
                                                       attention_dim).to(torch.float32), requires_grad=True)
        v = torch.nn.Parameter(torch.Tensor(v).reshape(batch_size, max_seq_len, num_heads, \
                                                       attention_dim).to(torch.float32), requires_grad=True)
        bias = torch.nn.Parameter(torch.Tensor(bias).to(torch.float32), requires_grad=True)
        mask = torch.Tensor(mask).to(torch.float32)

        real_silu_scale = 1 / max_seq_len if silu_scale == 0 else silu_scale
        qk_attn = torch.einsum("bnhd,bmhd->bhnm", q, k)

        if enable_bias:
            qk_attn = qk_attn + bias

        qk_attn = F.silu(qk_attn) * real_silu_scale

        if mask_type != MaskType.NONE:
            qk_attn = qk_attn * mask

        attn_output = torch.einsum(
            "bhnm,bmhd->bnhd",
            qk_attn,
            v
        ).reshape(batch_size, max_seq_len, num_heads * attention_dim)

        loss = torch.mean(attn_output)
        loss.backward()

        q_grad = q.grad.detach().cpu().clone()
        k_grad = k.grad.detach().cpu().clone()
        v_grad = v.grad.detach().cpu().clone()
        bias_grad = bias.grad.detach().cpu().clone() if enable_bias else None

        return attn_output.cpu().to(data_type).to(torch.float32).reshape(-1), q_grad.to(torch.float32), \
            k_grad.to(torch.float32), v_grad.to(torch.float32), bias_grad

    @staticmethod
    def custom_op_exec(q, k, v, bias, mask, batch_size, max_seq_len, num_heads, attention_dim, enable_bias, \
                       mask_type, silu_scale, data_type):
        q = torch.nn.Parameter(torch.Tensor(q).reshape(batch_size, max_seq_len, num_heads, attention_dim), \
                               requires_grad=True).to("npu")
        k = torch.nn.Parameter(torch.Tensor(k).reshape(batch_size, max_seq_len, num_heads, attention_dim), \
                               requires_grad=True).to("npu")
        v = torch.nn.Parameter(torch.Tensor(v).reshape(batch_size, max_seq_len, num_heads, attention_dim), \
                               requires_grad=True).to("npu")
        bias = torch.nn.Parameter(torch.Tensor(bias), requires_grad=True).to("npu")
        mask = torch.Tensor(mask).to("npu")

        q.retain_grad()
        k.retain_grad()
        v.retain_grad()
        bias.retain_grad()

        if enable_bias:
            output = torch.ops.mxrec.hstu_dense(q, k, v, mask, bias, mask_type, max_seq_len, silu_scale)
        else:
            output = torch.ops.mxrec.hstu_dense(q, k, v, mask, None, mask_type, max_seq_len, silu_scale)

        torch.npu.synchronize()

        loss = torch.mean(output)
        loss.backward()

        q_grad = q.grad.cpu().clone()
        k_grad = k.grad.cpu().clone()
        v_grad = v.grad.cpu().clone()
        bias_grad = bias.grad.cpu().clone() if enable_bias else None

        return output.cpu().to(data_type).to(torch.float32).reshape(-1), q_grad.to(torch.float32), \
            k_grad.to(torch.float32), v_grad.to(torch.float32), bias_grad

    def execute(self, batch_size, max_seq_len, num_heads, attention_dim, enable_bias, mask_type, silu_scale, data_type):
        q, k, v, bias, mask = generate_tensor(batch_size, max_seq_len, num_heads, attention_dim, data_type, mask_type)

        output, q_grad_op, k_grad_op, v_grad_op, bias_grad_op = self.custom_op_exec(q, k, v, bias, mask, batch_size, \
                                                                                    max_seq_len, num_heads,
                                                                                    attention_dim, enable_bias,
                                                                                    mask_type, silu_scale, data_type)
        golden, q_grad, k_grad, v_grad, bias_grad = self.golden_op_exec(q, k, v, bias, mask, batch_size, max_seq_len, \
                                                                        num_heads, attention_dim, enable_bias,
                                                                        mask_type, silu_scale, data_type)

        if data_type == torch.bfloat16:
            res = allclose(output, golden, bfloat16_pre, bfloat16_pre)
        elif data_type == torch.float16:
            res = allclose(output, golden, float16_pre, float16_pre)
        else:
            res = allclose(output, golden, float32_pre, float32_pre)
        assert res
        assert allclose(q_grad, q_grad_op, float32_pre, float32_pre)
        assert allclose(k_grad, k_grad_op, float32_pre, float32_pre)
        assert allclose(v_grad, v_grad_op, float32_pre, float32_pre)
        if enable_bias:
            assert allclose(bias_grad.to(torch.float32), bias_grad_op.to(torch.float32), float32_pre, float32_pre)
        else:
            assert bias_grad is None
            assert bias_grad_op is None

    @pytest.mark.parametrize("batch_size", [2, 16])
    @pytest.mark.parametrize("max_seq_len", [256])
    @pytest.mark.parametrize("num_heads", [2])
    @pytest.mark.parametrize("attention_dim", [32])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [1 / 256])
    @pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
    def test_hstu_autograd_normal(self, batch_size, max_seq_len, num_heads, attention_dim, enable_bias, mask_type, \
                                  silu_scale, data_type):
        self.execute(batch_size, max_seq_len, num_heads, attention_dim, enable_bias, mask_type, silu_scale, data_type)


class TestHstuAutogradJagged:
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
            tensor[offset: offset + seq_len, :, :] = dense_tensor[batch_id, 0: seq_len, :, :]
            offset = offset + seq_len

        return tensor

    @staticmethod
    def compare_jagged_bias(bias_grad, bias_grad_golden, seq_offset, loss):
        seq_lens = torch.zeros(bias_grad.shape[0], dtype=torch.int32)
        for i in range(seq_lens.shape[0]):
            seq_lens[i] = seq_offset[i + 1] - seq_offset[i]

        for batch, seq_len in enumerate(seq_lens):
            equal = allclose(bias_grad[batch, :, :seq_len, :seq_len],
                             bias_grad_golden[batch, :, :seq_len, :seq_len],
                             loss, loss)
            if not equal:
                return False

        return True

    @staticmethod
    def custom_op_exec(q, k, v, seq_offset, bias, mask, total_seqs, max_seq_len, num_heads, attention_dim, \
                       enable_bias, mask_type, silu_scale, data_type):
        q = torch.nn.Parameter(torch.Tensor(q).reshape(total_seqs, num_heads, attention_dim), \
                               requires_grad=True).to("npu").to(data_type)
        k = torch.nn.Parameter(torch.Tensor(k).reshape(total_seqs, num_heads, attention_dim), \
                               requires_grad=True).to("npu").to(data_type)
        v = torch.nn.Parameter(torch.Tensor(v).reshape(total_seqs, num_heads, attention_dim), \
                               requires_grad=True).to("npu").to(data_type)
        seq_offset = torch.Tensor(seq_offset).to("npu").to(torch.int32)
        bias = torch.nn.Parameter(torch.Tensor(bias), requires_grad=True).to("npu").to(data_type)
        mask = torch.Tensor(mask).to("npu").to(data_type)

        q.retain_grad()
        k.retain_grad()
        v.retain_grad()
        bias.retain_grad()

        if enable_bias:
            output = torch.ops.mxrec.hstu_jagged(q, k, v, mask, bias, mask_type, max_seq_len, silu_scale, seq_offset)
        else:
            output = torch.ops.mxrec.hstu_jagged(q, k, v, mask, None, mask_type, max_seq_len, silu_scale, seq_offset)

        torch.npu.synchronize()

        loss = torch.mean(output)
        loss.backward()

        q_grad = q.grad.cpu().clone()
        k_grad = k.grad.cpu().clone()
        v_grad = v.grad.cpu().clone()
        bias_grad = bias.grad.cpu().clone() if enable_bias else None

        return output.cpu().to(data_type).to(torch.float32).reshape(-1), q_grad.to(torch.float32), \
            k_grad.to(torch.float32), v_grad.to(torch.float32), bias_grad

    def golden_op_exec(self, q, k, v, seq_offset, bias, mask, batch_size, max_seq_len, num_heads, attention_dim, \
                       enable_bias, mask_type, silu_scale, data_type):
        seq_lens = np.zeros((batch_size,)).astype(np.int32)
        for batch_id in range(batch_size):
            seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]

        q_nn = torch.nn.Parameter(torch.Tensor(q).to(torch.float32), requires_grad=True)
        k_nn = torch.nn.Parameter(torch.Tensor(k).to(torch.float32), requires_grad=True)
        v_nn = torch.nn.Parameter(torch.Tensor(v).to(torch.float32), requires_grad=True)
        bias = torch.nn.Parameter(torch.Tensor(bias).to(torch.float32), requires_grad=True)
        mask = torch.Tensor(mask).reshape(batch_size, num_heads, max_seq_len, max_seq_len).to(torch.float32)

        q_dens = self.jagged_to_dense(q_nn, seq_lens, max_seq_len, num_heads, attention_dim).to(torch.float32)
        k_dens = self.jagged_to_dense(k_nn, seq_lens, max_seq_len, num_heads, attention_dim).to(torch.float32)
        v_dens = self.jagged_to_dense(v_nn, seq_lens, max_seq_len, num_heads, attention_dim).to(torch.float32)

        real_silu_scale = 1 / max_seq_len if silu_scale == 0 else silu_scale
        qk_attn = torch.einsum("bnhd,bmhd->bhnm", q_dens, k_dens)

        if enable_bias:
            qk_attn = qk_attn + bias

        qk_attn = F.silu(qk_attn) * real_silu_scale

        if mask_type != MaskType.NONE:
            qk_attn = qk_attn * mask

        attn_output = torch.einsum(
            "bhnm,bmhd->bnhd",
            qk_attn,
            v_dens
        )

        attn_output = self.dense_to_jagged(q, attn_output, seq_lens)

        loss = torch.mean(attn_output)
        loss.backward()

        q_grad = q_nn.grad.detach().cpu().clone()
        k_grad = k_nn.grad.detach().cpu().clone()
        v_grad = v_nn.grad.detach().cpu().clone()
        bias_grad = bias.grad.detach().cpu().clone() if enable_bias else None

        return attn_output.cpu().to(data_type).to(torch.float32).reshape(-1), q_grad.to(torch.float32), \
            k_grad.to(torch.float32), v_grad.to(torch.float32), bias_grad

    def execute(self, batch_size, max_seq_len, num_heads, attention_dim, enable_bias, mask_type, silu_scale, data_type):
        q, k, v, seq_offset, bias, mask, total_seqs = jagged_data_gen(batch_size, max_seq_len, num_heads, \
                                                                      attention_dim, mask_type)

        output, q_grad_op, k_grad_op, v_grad_op, bias_grad_op = self.custom_op_exec(q, k, v, seq_offset, bias, mask, \
                                                                                    total_seqs, max_seq_len, num_heads,
                                                                                    attention_dim, enable_bias,
                                                                                    mask_type, silu_scale, data_type)
        golden, q_grad, k_grad, v_grad, bias_grad = self.golden_op_exec(q, k, v, seq_offset, bias, mask, batch_size, \
                                                                        max_seq_len, num_heads, attention_dim,
                                                                        enable_bias, mask_type, silu_scale, data_type)

        loss = float32_pre
        if data_type == torch.bfloat16:
            loss = bfloat16_pre
        elif data_type == torch.float16:
            loss = float16_pre
        output_res = allclose(output, golden, loss, loss)
        q_grad_res = allclose(q_grad_op, q_grad, loss, loss)
        k_grad_res = allclose(k_grad_op, k_grad, loss, loss)
        v_grad_res = allclose(v_grad_op, v_grad, loss, loss)

        bias_grad_res = False
        if enable_bias:
            bias_grad_res = self.compare_jagged_bias(bias_grad.to(torch.float32), bias_grad_op.to(torch.float32),
                                                     seq_offset, loss)
        else:
            bias_grad_res = bias_grad is None and bias_grad_op is None

        return output_res and q_grad_res and k_grad_res and v_grad_res and bias_grad_res

    @pytest.mark.parametrize("batch_size", [2, 16])
    @pytest.mark.parametrize("max_seq_len", [256])
    @pytest.mark.parametrize("num_heads", [2])
    @pytest.mark.parametrize("attention_dim", [32])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [1 / 256])
    @pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
    def test_hstu_autograd_jagged(self, batch_size, max_seq_len, num_heads, attention_dim, enable_bias, mask_type, \
                                  silu_scale, data_type):
        self.execute(batch_size, max_seq_len, num_heads, attention_dim, enable_bias, mask_type, silu_scale, data_type)
