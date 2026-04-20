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
import pytest
import torch
import torch_npu
import torch.nn.functional as F

from test_target_mask import create_causal_mask
from test_common_utils import allclose, MaskType, hstu_close_double
from typing import Tuple


def jagged_data_gen(
    batch_size,
    max_seq_len,
    head_num_q,
    head_num_k,
    head_dim_qk,
    head_dim_v,
    mask_type,
    data_type,
    num_context=None,
    num_target=None,
    target_group_size=None,
):
    min_seq_len = 1
    if num_context is not None:
        min_seq_len += num_context
    if num_target is not None:
        min_seq_len += num_target
    seq_lens_q = torch.randint(min_seq_len, max_seq_len + 1, (batch_size,), dtype=torch.int32)
    seq_lens_k = torch.randint(min_seq_len, max_seq_len + 1, (batch_size,), dtype=torch.int32)
    seq_lens_q = torch.where(seq_lens_k < seq_lens_q, seq_lens_k, seq_lens_q)

    seq_offset_q = torch.concat((torch.zeros((1,), dtype=torch.int32), torch.cumsum(seq_lens_q, axis=0))).numpy()
    seq_offset_k = torch.concat((torch.zeros((1,), dtype=torch.int32), torch.cumsum(seq_lens_k, axis=0))).numpy()

    total_len_q = torch.sum(seq_lens_q).item()
    total_len_k = torch.sum(seq_lens_k).item()

    grad = torch.rand(total_len_q, head_num_q, head_dim_v, dtype=data_type).uniform_(-1, 1)
    q = torch.rand(total_len_q, head_num_q, head_dim_qk, dtype=data_type).uniform_(-1, 1)
    k = torch.rand(total_len_k, head_num_k, head_dim_qk, dtype=data_type).uniform_(-1, 1)
    v = torch.rand(total_len_k, head_num_k, head_dim_v, dtype=data_type).uniform_(-1, 1)

    bias = torch.rand(batch_size, head_num_q, max_seq_len, max_seq_len, dtype=data_type).uniform_(-1, 1)

    if mask_type == MaskType.TRIL:
        mask = torch.zeros(batch_size, head_num_q, max_seq_len, max_seq_len)
        for sample_id, (seq_len_q, seq_len_k) in enumerate(zip(seq_lens_q, seq_lens_k)):
            mask_tensor = create_causal_mask(seq_len_q, seq_len_k, num_context, num_target, target_group_size)
            mask[sample_id, :, :seq_len_q, :seq_len_k] = mask_tensor
        mask = mask.to(data_type)
    elif mask_type == MaskType.TRIU:
        mask = torch.triu(torch.ones(batch_size, head_num_q, max_seq_len, max_seq_len, dtype=data_type))
    elif mask_type == MaskType.NONE:
        mask = None
    else:
        mask = torch.randint(0, 2, (batch_size, head_num_q, max_seq_len, max_seq_len), dtype=data_type)

    return grad, q, k, v, bias, mask, max_seq_len, seq_offset_q, seq_offset_k


class TestHstuJaggedDemo:
    @staticmethod
    def jagged_to_dense(jagged_tensor, seq_lens, max_seq_len, head_num, head_dim):
        batch_size = len(seq_lens)
        dense_tensor = torch.zeros(batch_size, max_seq_len, head_num, head_dim, dtype=jagged_tensor.dtype)

        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            dense_tensor[batch_id, :seq_len, :, :] = jagged_tensor[offset: offset + seq_len, :, :]
            offset = offset + seq_len

        return dense_tensor

    @staticmethod
    def dense_to_jagged(jagged_tensor, dense_tensor, seq_lens):
        dense_dim = dense_tensor.shape[3]
        # tensor: [b_s, n, d]
        tensor = torch.zeros(jagged_tensor.shape[0], jagged_tensor.shape[1], dense_dim)

        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            tensor[offset: offset + seq_len, :, :] = dense_tensor[batch_id, 0:seq_len, :, :]
            offset = offset + seq_len

        return tensor


    @staticmethod
    def _pad_qkv(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            seq_lens_q: list[int],
            seq_lens_k: list[int],
            max_seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L, H_q, D = q.shape
        L, H_v, V = v.shape
        padded_q = (
            TestHstuJaggedDemo.jagged_to_dense(q, seq_lens_q, max_seq_len, H_q, D)
            .view(-1, max_seq_len, H_q, D)
            .transpose(1, 2)
        )  # [B, H, N, A]
        padded_k = (
            TestHstuJaggedDemo.jagged_to_dense(k, seq_lens_k, max_seq_len, H_v, D)
            .view(-1, max_seq_len, H_v, D)
            .transpose(1, 2)
        )  # [B, H, N, A]
        padded_v = (
            TestHstuJaggedDemo.jagged_to_dense(v, seq_lens_k, max_seq_len, H_v, V)
            .view(-1, max_seq_len, H_v, V)
            .transpose(1, 2)
        )  # [B, H, N, D]
        return padded_q, padded_k, padded_v

    @staticmethod
    def pytorch_hstu_mha(
            max_seq_len: int,
            alpha: float,
            grad,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            seq_offsets_q: torch.Tensor,
            seq_offsets_k: torch.Tensor,
            valid_attn_mask,
            bias,
            has_bias,
    ) -> torch.Tensor:

        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        bias.requires_grad_(True) if has_bias else None

        scaling_seqlen = max_seq_len
        seq_lens_q = seq_offsets_q[1:] - seq_offsets_q[:-1]
        seq_lens_k = seq_offsets_k[1:] - seq_offsets_k[:-1]

        L, H_q, _ = q.shape
        _, H_k, V = v.shape
        q_d, k_d, v_d = TestHstuJaggedDemo._pad_qkv(
            q, k, v, seq_lens_q, seq_lens_k, max_seq_len
        )  # [B, H, N, D) and [B, H, N, V]

        if H_q != H_k:
            assert H_q % H_k == 0, (f"head_num_q ({H_q}) must be divisible by "
                                    f"head_num_k({H_k}) ")
        h_qk_ratio = H_q // H_k
        k_d_expend = k_d.repeat_interleave(h_qk_ratio, dim=1)
        v_d_expend = v_d.repeat_interleave(h_qk_ratio, dim=1)

        qk_attn = torch.einsum("bhxa,bhya->bhxy", q_d, k_d_expend)
        if has_bias:
            qk_attn += bias
        qk_attn *= alpha
        qk_attn = F.silu(qk_attn) / scaling_seqlen
        if valid_attn_mask is not None:
            qk_attn = qk_attn * valid_attn_mask

        attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v_d_expend)  # [B, H, N, V]

        tensor = TestHstuJaggedDemo.dense_to_jagged(
            q,
            attn_dense.transpose(1, 2),  # [B, N, H * V]
            seq_lens_q  # 已转换为序列长度列表
        )

        ops_forward_output = tensor.view(L, H_q, V)
        dbias_hstu = None
        if has_bias:
            dq_hstu, dk_hstu, dv_hstu, dbias_hstu = torch.autograd.grad(outputs=ops_forward_output,
                                                                        inputs=(q, k, v, bias), grad_outputs=grad)
        else:
            dq_hstu, dk_hstu, dv_hstu = torch.autograd.grad(outputs=ops_forward_output, inputs=(q, k, v),
                                                            grad_outputs=grad)
        return dq_hstu, dk_hstu, dv_hstu, dbias_hstu

    @staticmethod
    def custom_op_exec(
        grad,
        q,
        k,
        v,
        bias,
        mask,
        seq_offset_q,
        seq_offset_k,
        mask_type,
        max_seq_len,
        silu_scale,
        enable_bias,
        data_type,
        num_context,
        num_target,
        target_group_size,
        alpha,
    ):
        batch_size = len(seq_offset_q) - 1
        grad_npu = grad.to("npu")
        q_npu = q.to("npu")
        k_npu = k.to("npu")
        v_npu = v.to("npu")
        seq_offset_q = torch.Tensor(seq_offset_q).to("npu").to(torch.int32)
        seq_offset_k = torch.Tensor(seq_offset_k).to("npu").to(torch.int32)
        if num_context is not None:
            num_context = torch.Tensor([num_context for _ in range(batch_size)]).to("npu").to(torch.int32)
        if num_target is not None:
            num_target = torch.Tensor([num_target for _ in range(batch_size)]).to("npu").to(torch.int32)
        bias_npu = bias.to("npu") if enable_bias else None

        mask_npu = None
        if mask_type == 3:
            mask_npu = mask.to("npu")

        q_grad, k_grad, v_grad, bias_grad = torch.ops.mxrec.hstu_jagged_backward(
            grad_npu,
            q_npu,
            k_npu,
            v_npu,
            mask_npu,
            bias_npu,
            mask_type,
            max_seq_len,
            max_seq_len,
            silu_scale,
            seq_offset_q,
            seq_offset_k,
            num_context,
            num_target,
            target_group_size,
            alpha,
        )

        torch.npu.synchronize()
        return q_grad.cpu(), k_grad.cpu(), v_grad.cpu(), bias_grad.cpu() if enable_bias else None

    def execute(
        self,
        batch_size,
        max_seq_len,
        head_num_q,
        head_num_k,
        head_dim_qk,
        head_dim_v,
        mask_type,
        silu_scale,
        enable_bias,
        data_type,
        num_context=None,
        num_target=None,
        target_group_size=None,
        alpha=1.0,
    ):
        grad, q, k, v, bias, mask, max_seq_len, seq_offset_q, seq_offset_k = jagged_data_gen(
            batch_size,
            max_seq_len,
            head_num_q,
            head_num_k,
            head_dim_qk,
            head_dim_v,
            mask_type,
            data_type,
            num_context,
            num_target,
            target_group_size,
        )

        q_grad, k_grad, v_grad, attn_bias_grad = self.custom_op_exec(
            grad,
            q,
            k,
            v,
            bias,
            mask,
            seq_offset_q,
            seq_offset_k,
            mask_type,
            max_seq_len,
            silu_scale,
            enable_bias,
            data_type,
            num_context,
            num_target,
            target_group_size,
            alpha,
        )

        q_grad_golden, k_grad_golden, v_grad_golden, attn_bias_grad_golden = self.pytorch_hstu_mha(
            max_seq_len,
            alpha,
            grad,
            q,
            k,
            v,
            seq_offset_q,
            seq_offset_k,
            mask,
            bias,
            enable_bias
        )

        mask = mask.to(torch.float32) if mask is not None else None
        q_grad_golden_fp32, k_grad_golden_fp32, v_grad_golden_fp32, attn_bias_grad_golden_fp32 = self.pytorch_hstu_mha(
            max_seq_len,
            alpha,
            grad,
            q.to(torch.float32),
            k.to(torch.float32),
            v.to(torch.float32),
            seq_offset_q,
            seq_offset_k,
            mask,
            bias.to(torch.float32),
            enable_bias
        )

        q_res = hstu_close_double(q_grad, q_grad_golden, q_grad_golden_fp32, try_allclose=True, multiplier=5)
        k_res = hstu_close_double(k_grad, k_grad_golden, k_grad_golden_fp32, try_allclose=True, multiplier=5)
        v_res = hstu_close_double(v_grad, v_grad_golden, v_grad_golden_fp32, try_allclose=True, multiplier=5)
        bias_res = True if attn_bias_grad is None \
            else hstu_close_double(attn_bias_grad, attn_bias_grad_golden, attn_bias_grad_golden_fp32,
                                   try_allclose=True, multiplier=5)

        assert all((q_res, k_res, v_res, bias_res))

    @pytest.mark.parametrize("batch_size", [1, 4])  # 范围: [1, 2048]
    @pytest.mark.parametrize("head_num_q", [8])  # 范围: [1, 16]
    @pytest.mark.parametrize("head_num_k", [8, 4, 2, 1])
    @pytest.mark.parametrize("head_dim_qk", [1, 16, 32, 72])  # 范围: [1, 512]
    @pytest.mark.parametrize("head_dim_v", [16, 32])  # 范围: [16, 512]，必须是16的倍数
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL, MaskType.NONE, MaskType.CUSTOM])
    @pytest.mark.parametrize("silu_scale", [0.0])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("data_type", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("max_seq_len,num_context,num_target,target_group_size", [
        (1, None, None, None),
        (257, 1, 1, 1),
        (512, 128, 0, 1),
        (1234, 0, 512, 3),
        (1234, 128, 512, 3)
    ])
    @pytest.mark.parametrize("alpha", [0.5])
    def test_hstu_dens_jagged(
        self,
        batch_size,
        head_num_q,
        head_num_k,
        head_dim_qk,
        head_dim_v,
        mask_type,
        silu_scale,
        enable_bias,
        data_type,
        max_seq_len,
        num_context,
        num_target,
        target_group_size,
        alpha,
    ):
        self.execute(
            batch_size,
            max_seq_len,
            head_num_q,
            head_num_k,
            head_dim_qk,
            head_dim_v,
            mask_type,
            silu_scale,
            enable_bias,
            data_type,
            num_context,
            num_target,
            target_group_size,
            alpha
        )

    # batch_size泛化测试
    @pytest.mark.parametrize("batch_size", [1, 32, 128, 512, 1024])  # 范围: [1, 2048]
    @pytest.mark.parametrize("head_num_q", [4])  # 范围: [1, 16]
    @pytest.mark.parametrize("head_num_k", [4])
    @pytest.mark.parametrize("head_dim_qk", [128])  # 范围: [1, 512]
    @pytest.mark.parametrize("head_dim_v", [128])  # 范围: [16, 512]，必须是16的倍数
    @pytest.mark.parametrize("mask_type", [MaskType.TRIL])
    @pytest.mark.parametrize("silu_scale", [0.0])
    @pytest.mark.parametrize("enable_bias", [False])
    @pytest.mark.parametrize("data_type", [torch.float16])
    @pytest.mark.parametrize("max_seq_len,num_context,num_target,target_group_size", [
        (512, None, None, None),
    ])
    @pytest.mark.parametrize("alpha", [0.5])
    def test_hstu_dens_jagged_bs(
            self,
            batch_size,
            head_num_q,
            head_num_k,
            head_dim_qk,
            head_dim_v,
            mask_type,
            silu_scale,
            enable_bias,
            data_type,
            max_seq_len,
            num_context,
            num_target,
            target_group_size,
            alpha,
    ):
        self.execute(
            batch_size,
            max_seq_len,
            head_num_q,
            head_num_k,
            head_dim_qk,
            head_dim_v,
            mask_type,
            silu_scale,
            enable_bias,
            data_type,
            num_context,
            num_target,
            target_group_size,
            alpha
        )