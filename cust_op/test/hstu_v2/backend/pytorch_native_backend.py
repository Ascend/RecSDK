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
from typing import Tuple

import torch
import torch.nn.functional as F

from .common import jagged_to_dense, dense_to_jagged


class Kernel:
    def __init__(self, alpha, scale, has_rab, max_seqlen_q, max_seqlen_k,
                 seq_offset_q, seq_offset_k):
        self.alpha = alpha
        self.scale = scale
        self.has_rab = has_rab
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k

        self.seqlen_q = seq_offset_q[1:] - seq_offset_q[:-1]
        self.seqlen_k = seq_offset_k[1:] - seq_offset_k[:-1]


    def forward(self, q, k, v, rab, mask):
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        rab.requires_grad_(True) if self.has_rab else None

        real_silu_scale = 1 / self.max_seqlen_q if self.scale == 0.0 else self.scale

        seq_q, head_q, _ = q.shape
        _, head_k, dim_v = v.shape
        q_d, k_d, v_d = self._pad_qkv(q, k, v)  # [B, H, N, dim_q) and [B, H, N, dim_v]

        if head_q != head_k:
            if head_q % head_k != 0:
                raise ValueError(f"head_num_q ({head_q}) must be divisible by head_num_k({head_k}) ")
        
        h_qk_ratio = head_q // head_k
        k_d_expend = k_d.repeat_interleave(h_qk_ratio, dim=1)
        v_d_expend = v_d.repeat_interleave(h_qk_ratio, dim=1)
        qk_attn = torch.einsum("bhxa,bhya->bhxy", q_d, k_d_expend)
        if self.has_rab:
            qk_attn += rab
        qk_attn *= self.alpha
        qk_attn = F.silu(qk_attn) * real_silu_scale

        if mask is not None:
            qk_attn = qk_attn * mask

        attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v_d_expend)  # [B, H, N, dim_v]
        tensor = dense_to_jagged(
            q, attn_dense.transpose(1, 2), self.seqlen_q  # 已转换为序列长度列表
        )

        output = tensor.view(seq_q, head_q, dim_v)
        return output


    def backward(self, grad, q, k, v, rab, mask):
        q_grad, k_grad, v_grad, rab_grad = self.__backward_impl(grad, q, k, v, rab, mask)

        q_grad_fp32, k_grad_fp32, v_grad_fp32, rab_grad_fp32 = self.__backward_impl(
            grad.to(torch.float32),
            q.to(torch.float32),
            k.to(torch.float32),
            v.to(torch.float32),
            rab.to(torch.float32) if self.has_rab else None,
            mask)

        return q_grad, k_grad, v_grad, rab_grad, q_grad_fp32, k_grad_fp32, v_grad_fp32, rab_grad_fp32


    def __backward_impl(self, grad, q, k, v, rab, mask):
        forward_output = self.forward(
            q, k, v, rab, mask)
        rab_grad = None
        if self.has_rab:
            q_grad, k_grad, v_grad, rab_grad = torch.autograd.grad(outputs=forward_output,
                                                                   inputs=(q, k, v, rab), grad_outputs=grad)
        else:
            q_grad, k_grad, v_grad = torch.autograd.grad(outputs=forward_output, inputs=(q, k, v),
                                                         grad_outputs=grad)
        return q_grad, k_grad, v_grad, rab_grad


    def _pad_qkv(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_q, head_q, dim_q = q.shape
        seq_q, head_v, dim_v = v.shape
        padded_q = (
            jagged_to_dense(q, self.seqlen_q, self.max_seqlen_q, head_q, dim_q)
            .view(-1, self.max_seqlen_q, head_q, dim_q)
            .transpose(1, 2)
        )  # [B, H, N, A]
        padded_k = (
            jagged_to_dense(k, self.seqlen_k, self.max_seqlen_k, head_v, dim_q)
            .view(-1, self.max_seqlen_k, head_v, dim_q)
            .transpose(1, 2)
        )  # [B, H, N, A]
        padded_v = (
            jagged_to_dense(v, self.seqlen_k, self.max_seqlen_k, head_v, dim_v)
            .view(-1, self.max_seqlen_k, head_v, dim_v)
            .transpose(1, 2)
        )  # [B, H, N, dim_q]
        return padded_q, padded_k, padded_v


class Validator:
    @staticmethod
    def forward_verify(actual, ref):
        pass


    @staticmethod
    def backward_verify(actual, ref):
        """返回验证结果和详细精度数据"""
        q_grad, k_grad, v_grad, rab_grad = actual
        q_grad_ref, k_grad_ref, v_grad_ref, rab_grad_ref, \
            q_grad_ref_fp32, k_grad_ref_fp32, v_grad_ref_fp32, rab_grad_ref_fp32 = ref

        q_res, q_detail = Validator.__hstu_close_double(
            q_grad, q_grad_ref, q_grad_ref_fp32, try_allclose=True, multiplier=5)
        k_res, k_detail = Validator.__hstu_close_double(
            k_grad, k_grad_ref, k_grad_ref_fp32, try_allclose=True, multiplier=5)
        v_res, v_detail = Validator.__hstu_close_double(
            v_grad, v_grad_ref, v_grad_ref_fp32, try_allclose=True, multiplier=5)
        rab_res, rab_detail = (True, None) if rab_grad is None else Validator.__hstu_close_double(
            rab_grad, rab_grad_ref, rab_grad_ref_fp32, try_allclose=True, multiplier=5)

        # 汇总验证结果
        passed = q_res and k_res and v_res and rab_res

        # 返回 (通过标志, 详细精度数据)
        detail = {
            "DQ": q_detail,
            "DK": k_detail,
            "DV": v_detail,
            "DRAB": rab_detail
        }
        return passed, detail


    @staticmethod
    def __hstu_close_double(actual, ref, fp32_ref, try_allclose: bool = False, multiplier: int = 2):
        """返回验证结果和详细精度数据 (actual-fp32_out_ref, fp32_out_ref, actual-out_ref)"""
        # 算子输出
        actual = actual.reshape(-1)
        # 原生golden
        out_ref = ref.reshape(-1)
        # 高精度原生golden
        fp32_ref = fp32_ref.reshape(-1)

        if fp32_ref.dtype != torch.float32:
            raise ValueError("fp32_ref should be float32")

        original_try_allclose = try_allclose
        if try_allclose:
            try_allclose = torch.allclose(actual, out_ref)

        actual_fp32_out_ref = (actual - fp32_ref).abs().max().item()
        fp32_out_ref = (out_ref - fp32_ref).abs().max().item()
        actual_out_ref = (out_ref - actual).abs().max().item()

        passed = (actual_fp32_out_ref <= multiplier * fp32_out_ref) or try_allclose

        # 返回验证结果和详细精度数据
        detail = {
            "passed": passed,
            "actual-fp32_out_ref": actual_fp32_out_ref,
            "fp32_out_ref": fp32_out_ref,
            "actual-out_ref": actual_out_ref,
            "try_allclose": original_try_allclose and try_allclose
        }
        return passed, detail


class PytorchNative:
    def __init__(self):
        pass

    @staticmethod
    def kernel(alpha, scale, has_rab, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k):
        return Kernel(alpha, scale, has_rab, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k)

    @staticmethod
    def validator():
        return Validator()