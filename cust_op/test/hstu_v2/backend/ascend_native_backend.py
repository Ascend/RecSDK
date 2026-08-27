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
import torch
import torch.nn.functional as F

from .common import jagged_to_dense, dense_to_jagged


class Kernel:
    def __init__(self, alpha_, scale_, has_rab_, max_seqlen_q_, max_seqlen_k_, seq_offset_q, seq_offset_k):
        self.alpha = alpha_
        self.scale = scale_
        self.has_rab = has_rab_
        self.max_seqlen_q = max_seqlen_q_
        self.max_seqlen_k = max_seqlen_k_

        self.seqlen_q = seq_offset_q[1:] - seq_offset_q[:-1]
        self.seqlen_k = seq_offset_k[1:] - seq_offset_k[:-1]

    def forward(self, q, k, v, rab, mask):
        seq_k, head_k, _ = k.shape
        seq_q, head_q, dim_q = q.shape
        _, _, dim_v = v.shape
        data_type = q.dtype

        if head_q != head_k:
            if head_q % head_k != 0:
                raise ValueError(f"head_num_q ({head_q}) must be divisible by head_num_k({head_k}) ")

        q_dens = jagged_to_dense(q, self.seqlen_q, self.max_seqlen_q, head_q, dim_q).to("npu")
        k_dens = jagged_to_dense(k, self.seqlen_k, self.max_seqlen_k, head_k, dim_q).to("npu")
        v_dens = jagged_to_dense(v, self.seqlen_k, self.max_seqlen_k, head_k, dim_v).to("npu")

        if head_q != head_k:
            h_qk_ratio = head_q // head_k
            k_dens = k_dens.repeat_interleave(h_qk_ratio, dim=2)
            v_dens = v_dens.repeat_interleave(h_qk_ratio, dim=2)

        qk_attn = torch.matmul(q_dens.permute(0, 2, 1, 3), k_dens.permute(0, 2, 3, 1))

        qk_attn = qk_attn.float()

        if mask is not None:
            mask = mask.to("npu")
            mask = mask.float()

        if rab is not None:
            rab = rab.to("npu")
            rab = rab.float()
            qk_attn = qk_attn + rab

        real_silu_scale = 1 / self.max_seqlen_q if self.scale == 0.0 else self.scale
        qk_attn = qk_attn * self.alpha

        if mask is not None:
            qk_attn = F.silu(qk_attn) * real_silu_scale * mask
        else:
            qk_attn = F.silu(qk_attn) * real_silu_scale

        qk_attn = qk_attn.to(data_type)
        attn_dense = torch.matmul(qk_attn, v_dens.permute(0, 2, 1, 3))

        attn_dense = attn_dense.cpu()
        output = dense_to_jagged(q, attn_dense.permute(0, 2, 1, 3), self.seqlen_q)

        output = output.view(seq_q, head_q, dim_v)

        torch.npu.synchronize()

        return output

    def backward(self, grad, q, k, v, rab, mask):
        seq_k, head_k, _ = k.shape
        seq_q, head_q, dim_q = q.shape
        _, _, dim_v = v.shape
        data_type = q.dtype

        if head_q != head_k:
            if head_q % head_k != 0:
                raise ValueError(f"head_num_q ({head_q}) must be divisible by head_num_k({head_k}) ")

        grad_dens = jagged_to_dense(grad, self.seqlen_q, self.max_seqlen_q, head_q, dim_v).to("npu")
        q_dens = jagged_to_dense(q, self.seqlen_q, self.max_seqlen_q, head_q, dim_q).to("npu")
        k_dens = jagged_to_dense(k, self.seqlen_k, self.max_seqlen_k, head_k, dim_q).to("npu")
        v_dens = jagged_to_dense(v, self.seqlen_k, self.max_seqlen_k, head_k, dim_v).to("npu")

        qk = torch.matmul(q_dens.permute(0, 2, 1, 3), k_dens.permute(0, 2, 3, 1))
        gv = torch.matmul(grad_dens.permute(0, 2, 1, 3), v_dens.permute(0, 2, 3, 1))

        qk = qk.float()
        gv = gv.float()

        if mask is not None:
            mask = mask.to("npu")
            mask = mask.float()

        if rab is not None:
            rab = rab.to("npu")
            rab = rab.float()
            qkb = qk + rab
        else:
            qkb = qk

        qkb = qkb * self.alpha
        real_silu_scale = 1 / self.max_seqlen_q if self.scale == 0.0 else self.scale

        if mask is not None:
            score = F.silu(qkb) * real_silu_scale * mask
        else:
            score = F.silu(qkb) * real_silu_scale

        score = score.to(data_type)
        v_grad_dens = torch.matmul(score.permute(0, 1, 3, 2), grad_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        if mask is not None:
            rab_grad = gv * real_silu_scale * mask * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
        else:
            rab_grad = gv * real_silu_scale * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
        rab_grad = rab_grad * self.alpha
        rab_grad = rab_grad.to(data_type)
        k_grad_dens = torch.matmul(rab_grad.permute(0, 1, 3, 2), q_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        q_grad_dens = torch.matmul(rab_grad, k_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        rab_grad = rab_grad.cpu()
        q_grad_dens = q_grad_dens.cpu()
        q_grad = dense_to_jagged(q, q_grad_dens, self.seqlen_q)
        k_grad_dens = k_grad_dens.cpu()
        k_grad = dense_to_jagged(k, k_grad_dens, self.seqlen_k)
        v_grad_dens = v_grad_dens.cpu()
        v_grad = dense_to_jagged(v, v_grad_dens, self.seqlen_k)

        torch.npu.synchronize()

        return q_grad, k_grad, v_grad, rab_grad if rab is not None else None


class Validator:
    @staticmethod
    def forward_verify(actual, ref):
        """验证前向传播结果，返回验证结果和详细精度数据

        Args:
            actual: ascend 算子输出 (output tensor)
            ref: 参考输出，可为 tensor 或 (output, output_fp32) 元组；元组时取第一个元素作为比较目标
        """
        # ref 可能为 (output, output_fp32) 元组（pytorch 后端双精度标杆返回），兼容解包
        if isinstance(ref, tuple):
            ref = ref[0]

        data_type = actual.dtype

        # 根据数据类型设置误差容限
        if data_type == torch.float16:
            rtol = 1e-3
            atol = 1e-3
        elif data_type == torch.bfloat16:
            rtol = 5e-3
            atol = 5e-3
        else:
            raise ValueError(f"dtype {data_type} not support")

        # 比较实际输出和参考输出
        passed = torch.allclose(actual, ref, rtol=rtol, atol=atol)

        # 构建详细精度数据
        detail = {
            "OUT": {
                "passed": passed,
                "max_diff": torch.max(torch.abs(actual - ref)).item(),
                "mean_diff": torch.mean(torch.abs(actual - ref)).item(),
            }
        }

        return passed, detail

    @staticmethod
    def backward_verify(actual, ref):
        """返回验证结果和详细精度数据（只有通过状态，无详细精度数据）"""
        q_grad, k_grad, v_grad, rab_grad = actual
        q_grad_ref, k_grad_ref, v_grad_ref, rab_grad_ref = ref
        data_type = q_grad.dtype

        if data_type == torch.float16:
            loss = 1e-3
        elif data_type == torch.bfloat16:
            loss = 5e-3
        else:
            raise ValueError("dtype not support")

        q_res = torch.allclose(q_grad, q_grad_ref, loss, loss)
        k_res = torch.allclose(k_grad, k_grad_ref, loss, loss)
        v_res = torch.allclose(v_grad, v_grad_ref, loss, loss)
        if rab_grad is not None:
            drab_res = torch.allclose(rab_grad, rab_grad_ref, loss, loss)
        else:
            drab_res = True

        # 汇总验证结果
        passed = q_res and k_res and v_res and drab_res

        # 返回 (通过标志, 详细精度数据 - 只有通过状态)
        detail = {
            "DQ": {"passed": q_res},
            "DK": {"passed": k_res},
            "DV": {"passed": v_res},
            "DRAB": {"passed": drab_res},
        }
        return passed, detail


class AscendNative:
    def __init__(self):
        pass

    @staticmethod
    def kernel(alpha, scale, has_rab, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k):
        return Kernel(alpha, scale, has_rab, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k)

    @staticmethod
    def validator():
        return Validator()
