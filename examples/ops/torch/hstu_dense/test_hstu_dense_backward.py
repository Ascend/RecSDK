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
import sysconfig
import os
from copy import deepcopy

import pytest
import torch
import torch_npu
import torch.nn.functional as F
import numpy as np

from test_target_mask import ScoreShapeParam, compute_target_mask_each_block_concat
from test_common_utils import allclose

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

device_id: int = 0


def generate_tensor(batch_size, max_seq_len, num_heads, attention_dim, mask_type, data_type):
    grad = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)
    q = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)
    k = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)
    v = torch.empty(batch_size, max_seq_len, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)

    bias = torch.empty(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type).uniform_(-1, 1)

    if mask_type == 0:
        mask = torch.tril(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type))
    elif mask_type == 1:
        mask = torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type))
    elif mask_type == 2:
        mask = None
    else:
        mask = torch.randint(0, 2, size=(batch_size, num_heads, max_seq_len, max_seq_len)).to(data_type)

    return grad, q, k, v, bias, mask


torch.npu.set_device(device_id)


class TestHstuNormalDemo:
    @staticmethod
    def golden_op_exec(
        grad,
        q,
        k,
        v,
        bias,
        mask,
        mask_type,
        max_seq_len,
        silu_scale,
        enable_bias,
        data_type,
    ):
        batch_size, seq_len, head_num, head_dim = grad.shape

        qk = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))
        gv = torch.matmul(grad.permute(0, 2, 1, 3), v.permute(0, 2, 3, 1))

        qk = qk.float()
        gv = gv.float()

        if mask_type == 0 or mask_type == 3:
            mask = mask.float()

        if enable_bias:
            bias = bias.float()
            qkb = qk + bias
        else:
            qkb = qk

        real_silu_scale = 1 / max_seq_len if silu_scale == 0.0 else silu_scale

        if mask_type == 0 or mask_type == 3:
            score = F.silu(qkb) * real_silu_scale * mask
        else:
            score = F.silu(qkb) * real_silu_scale

        score = score.to(data_type)
        v_grad = torch.matmul(score.permute(0, 1, 3, 2), grad.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        if mask_type == 0 or mask_type == 3:
            attn_bias_grad = gv * real_silu_scale * mask * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
        else:
            attn_bias_grad = gv * real_silu_scale * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))

        attn_bias_grad = attn_bias_grad.to(data_type)
        k_grad = torch.matmul(attn_bias_grad.permute(0, 1, 3, 2), q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        q_grad = torch.matmul(attn_bias_grad, k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        torch.npu.synchronize()
        return q_grad.cpu(), k_grad.cpu(), v_grad.cpu(), attn_bias_grad.cpu()

    @staticmethod
    def custom_op_exec(
        grad,
        q,
        k,
        v,
        bias,
        mask,
        mask_type,
        max_seq_len,
        silu_scale,
        enable_bias,
        data_type,
    ):
        if enable_bias:
            q_grad, k_grad, v_grad, attn_bias_grad = torch.ops.mxrec.hstu_dense_backward(
                grad, q, k, v, mask, bias, mask_type, max_seq_len, silu_scale
            )
        else:
            q_grad, k_grad, v_grad, attn_bias_grad = torch.ops.mxrec.hstu_dense_backward(
                grad, q, k, v, mask, None, mask_type, max_seq_len, silu_scale
            )

        torch.npu.synchronize()
        return q_grad.cpu(), k_grad.cpu(), v_grad.cpu(), attn_bias_grad.cpu()

    def execute(
        self,
        batch_size,
        max_seq_len,
        head_num,
        head_dim,
        mask_type,
        silu_scale,
        enable_bias,
        data_type,
    ):
        grad, q, k, v, bias, mask = generate_tensor(batch_size, max_seq_len, head_num, head_dim, mask_type, data_type)

        grad_npu = grad.to(f"npu:{device_id}")
        q_npu = q.to(f"npu:{device_id}")
        k_npu = k.to(f"npu:{device_id}")
        v_npu = v.to(f"npu:{device_id}")
        bias_npu = None
        if enable_bias:
            bias_npu = bias.to(f"npu:{device_id}")
        mask_npu = None
        if mask_type == 0 or mask_type == 3:
            mask_npu = mask.to(f"npu:{device_id}")

        q_grad, k_grad, v_grad, attn_bias_grad = self.custom_op_exec(
            grad_npu,
            q_npu,
            k_npu,
            v_npu,
            bias_npu,
            mask_npu,
            mask_type,
            max_seq_len,
            silu_scale,
            enable_bias,
            data_type,
        )

        q_grad_golden, k_grad_golden, v_grad_golden, attn_bias_grad_golden = self.golden_op_exec(
            grad_npu,
            q_npu,
            k_npu,
            v_npu,
            bias_npu,
            mask_npu,
            mask_type,
            max_seq_len,
            silu_scale,
            enable_bias,
            data_type,
        )

        torch.npu.synchronize()

        loss = 1e-4
        if data_type == torch.float16:
            loss = 1e-3
        elif data_type == torch.bfloat16:
            loss = 1e-2

        q_res = allclose(q_grad, q_grad_golden, loss, loss)
        k_res = allclose(k_grad, k_grad_golden, loss, loss)
        v_res = allclose(v_grad, v_grad_golden, loss, loss)
        bias_res = not enable_bias or allclose(attn_bias_grad, attn_bias_grad_golden, loss, loss)

        assert q_res and k_res and v_res and bias_res

    @pytest.mark.parametrize("batch_size", [1, 4])  # 范围: [1, 2048]
    @pytest.mark.parametrize("max_seq_len", [1, 257, 512, 1234])  # 范围: [1, 20480]
    @pytest.mark.parametrize("head_num", [1, 16])  # 范围: [1, 16]
    @pytest.mark.parametrize("head_dim", [16, 32])  # 范围: [16, 512]，必须是16的倍数
    @pytest.mark.parametrize("mask_type", [0, 2, 3])
    @pytest.mark.parametrize("silu_scale", [1.0 / 256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
    def test_hstu_normal_case(
        self,
        batch_size,
        max_seq_len,
        head_num,
        head_dim,
        mask_type,
        silu_scale,
        enable_bias,
        data_type,
    ):
        self.execute(
            batch_size,
            max_seq_len,
            head_num,
            head_dim,
            mask_type,
            silu_scale,
            enable_bias,
            data_type,
        )