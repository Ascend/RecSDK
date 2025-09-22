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
import sysconfig
import os
from copy import deepcopy

import pytest
import torch
import torch_npu
import torch.nn.functional as F
import numpy as np

from test_target_mask import ScoreShapeParam, compute_target_mask_each_block_concat

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

MAX_NUM_TARGET = 512
device_id: int = 0


def cached_create_causal_mask(param: ScoreShapeParam) -> torch.Tensor:
    cached_file = f"cached_target_mask{param.target_group_size}.pt"
    if param.num_target > MAX_NUM_TARGET:
        raise ValueError("param.num_target should be < 512")
    if os.path.exists(cached_file):
        mask = torch.tril(torch.ones(param.seq_len, param.seq_len))
        mask[: param.num_context, :param.seq_len - param.num_target] = 1
        if param.num_target > 0:
            target_mask = torch.load(cached_file)
            mask[-param.num_target:, -param.num_target:] = target_mask[: param.num_target, : param.num_target]
        return mask
    else:
        _param = deepcopy(param)
        _param.seq_len += MAX_NUM_TARGET - _param.num_target
        _param.num_target = MAX_NUM_TARGET
        mask = compute_target_mask_each_block_concat(_param, use_npu=False)
        torch.save(mask[-MAX_NUM_TARGET:, -MAX_NUM_TARGET:], cached_file)
        return mask[: param.seq_len, :param.seq_len]


def jagged_data_gen(
    batch_size,
    max_seq_len,
    num_heads,
    attention_dim,
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
    seq_lens = torch.randint(min_seq_len, max_seq_len + 1, (batch_size,), dtype=torch.int64)
    seq_offset = torch.concat((torch.zeros((1,), dtype=torch.int64), torch.cumsum(seq_lens, axis=0))).numpy()

    total_seqs = torch.sum(seq_lens)

    grad = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)
    q = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)
    k = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)
    v = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(-1, 1)

    bias = torch.empty(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type).uniform_(-1, 1)

    if mask_type == 0:
        if num_context is None and num_target is None:
            mask = torch.tril(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len), diagonal=1)
        else:
            mask = torch.zeros(batch_size, num_heads, max_seq_len, max_seq_len)
            for sample_id, seq_len in enumerate(seq_lens):
                parm = ScoreShapeParam(
                    seq_len=seq_len,
                    num_target=num_target,
                    num_context=num_context,
                    num_history=None if num_target is None else seq_len - num_target,
                    target_group_size=target_group_size,
                    block_h=seq_len,
                    block_w=seq_len,
                )
                mask_tensor = cached_create_causal_mask(parm)
                mask[sample_id, :, :seq_len, :seq_len] = mask_tensor
            mask = mask.to(data_type)
    elif mask_type == 1:
        mask = torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type))
    elif mask_type == 2:
        mask = None
    else:
        mask = torch.empty(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type).uniform_(-1, 1)

    return grad, q, k, v, bias, mask, max_seq_len, seq_offset


torch.npu.set_device(device_id)


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
        tensor = torch.zeros_like(jagged_tensor)

        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            tensor[offset: offset + seq_len, :, :] = dense_tensor[batch_id, 0:seq_len, :, :]
            offset = offset + seq_len

        return tensor

    @staticmethod
    def compare_jagged_bias(bias_grad, bias_grad_golden, seq_offset, loss):
        seq_lens = torch.zeros(bias_grad.shape[0], dtype=torch.int64)
        for i in range(seq_lens.shape[0]):
            seq_lens[i] = seq_offset[i + 1] - seq_offset[i]

        for batch, seq_len in enumerate(seq_lens):
            equal = torch.allclose(
                bias_grad[batch, :, :seq_len, :seq_len],
                bias_grad_golden[batch, :, :seq_len, :seq_len],
                loss,
                loss,
            )
            if not equal:
                return False

        return True

    @staticmethod
    def custom_op_exec(
        grad,
        q,
        k,
        v,
        bias,
        mask,
        seq_offset,
        mask_type,
        max_seq_len,
        silu_scale,
        enable_bias,
        data_type,
        num_context,
        num_target,
        target_group_size,
    ):
        grad_npu = grad.to(f"npu:{device_id}")
        q_npu = q.to(f"npu:{device_id}")
        k_npu = k.to(f"npu:{device_id}")
        v_npu = v.to(f"npu:{device_id}")
        seq_offset = torch.LongTensor(seq_offset).to(f"npu:{device_id}")
        bias_npu = bias.to(f"npu:{device_id}")

        mask_npu = None
        if mask_type == 3:
            mask_npu = mask.to(f"npu:{device_id}")

        if enable_bias:
            q_grad, k_grad, v_grad, bias_grad = torch.ops.mxrec.hstu_jagged_backward(
                grad_npu,
                q_npu,
                k_npu,
                v_npu,
                mask_npu,
                bias_npu,
                mask_type,
                max_seq_len,
                silu_scale,
                seq_offset,
                None,
                None,
                target_group_size,
            )
        else:
            q_grad, k_grad, v_grad, bias_grad = torch.ops.mxrec.hstu_jagged_backward(
                grad_npu,
                q_npu,
                k_npu,
                v_npu,
                mask_npu,
                None,
                mask_type,
                max_seq_len,
                silu_scale,
                seq_offset,
                None,
                None,
                target_group_size,
            )

        torch.npu.synchronize()
        if enable_bias:
            return q_grad.cpu(), k_grad.cpu(), v_grad.cpu(), bias_grad.cpu()
        return q_grad.cpu(), k_grad.cpu(), v_grad.cpu(), None

    def golden_op_exec(
        self,
        grad,
        q,
        k,
        v,
        bias,
        mask,
        max_seq_len,
        seq_offset,
        mask_type,
        silu_scale,
        enable_bias,
        data_type,
    ):
        head_nums = grad.shape[1]
        head_dim = grad.shape[2]
        batch_size = bias.shape[0]

        seq_lens = np.zeros((batch_size,)).astype(np.int64)
        for batch_id in range(batch_size):
            seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]

        grad_dens = self.jagged_to_dense(grad, seq_lens, max_seq_len, head_nums, head_dim).to(f"npu:{device_id}")
        q_dens = self.jagged_to_dense(q, seq_lens, max_seq_len, head_nums, head_dim).to(f"npu:{device_id}")
        k_dens = self.jagged_to_dense(k, seq_lens, max_seq_len, head_nums, head_dim).to(f"npu:{device_id}")
        v_dens = self.jagged_to_dense(v, seq_lens, max_seq_len, head_nums, head_dim).to(f"npu:{device_id}")
        actual_seq_lens = torch.from_numpy(seq_lens).reshape(batch_size, 1, 1, 1).to(f"npu:{device_id}")
        actual_seq_lens = torch.broadcast_to(actual_seq_lens, bias.shape)

        qk = torch.matmul(q_dens.permute(0, 2, 1, 3), k_dens.permute(0, 2, 3, 1))
        gv = torch.matmul(grad_dens.permute(0, 2, 1, 3), v_dens.permute(0, 2, 3, 1))

        qk = qk.float()
        gv = gv.float()
        bias = bias.float()

        if mask_type == 0 or mask_type == 3:
            mask = mask.to(f"npu:{device_id}")
            mask = mask.float()

        if enable_bias:
            bias = bias.to(f"npu:{device_id}")
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
        v_grad_dens = torch.matmul(score.permute(0, 1, 3, 2), grad_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        if mask_type == 0 or mask_type == 3:
            bias_grad = gv * real_silu_scale * mask * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
        else:
            bias_grad = gv * real_silu_scale * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))

        bias_grad = bias_grad.to(data_type)
        k_grad_dens = torch.matmul(bias_grad.permute(0, 1, 3, 2), q_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        q_grad_dens = torch.matmul(bias_grad, k_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        bias_grad = bias_grad.cpu()
        q_grad_dens = q_grad_dens.cpu()
        q_grad = self.dense_to_jagged(q, q_grad_dens, seq_lens)
        k_grad_dens = k_grad_dens.cpu()
        k_grad = self.dense_to_jagged(k, k_grad_dens, seq_lens)
        v_grad_dens = v_grad_dens.cpu()
        v_grad = self.dense_to_jagged(v, v_grad_dens, seq_lens)

        torch.npu.synchronize()

        return q_grad, k_grad, v_grad, bias_grad

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
        num_context=None,
        num_target=None,
        target_group_size=None,
    ):
        grad, q, k, v, bias, mask, max_seq_len, seq_offset = jagged_data_gen(
            batch_size,
            max_seq_len,
            head_num,
            head_dim,
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
            seq_offset,
            mask_type,
            max_seq_len,
            silu_scale,
            enable_bias,
            data_type,
            num_context,
            num_target,
            target_group_size,
        )

        q_grad_golden, k_grad_golden, v_grad_golden, attn_bias_grad_golden = self.golden_op_exec(
            grad,
            q,
            k,
            v,
            bias,
            mask,
            max_seq_len,
            seq_offset,
            mask_type,
            silu_scale,
            enable_bias,
            data_type,
        )

        loss = 1e-4
        if data_type == torch.float16:
            loss = 1e-3
        elif data_type == torch.bfloat16:
            loss = 1e-2

        q_res = torch.allclose(q_grad, q_grad_golden, loss, loss)
        k_res = torch.allclose(k_grad, k_grad_golden, loss, loss)
        v_res = torch.allclose(v_grad, v_grad_golden, loss, loss)
        bias_res = not enable_bias or self.compare_jagged_bias(attn_bias_grad, attn_bias_grad_golden, seq_offset, loss)

        assert q_res and k_res and v_res and bias_res

    @pytest.mark.parametrize("batch_size", [2, 4])
    @pytest.mark.parametrize("max_seq_len", [256, 257, 1024, 1234])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("head_dim", [32, 128])
    @pytest.mark.parametrize("mask_type", [0, 2, 3])
    @pytest.mark.parametrize("silu_scale", [0.0, 1.0 / 256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("num_context", [None])
    @pytest.mark.parametrize("num_target", [None])
    @pytest.mark.parametrize("target_group_size", [None])
    def test_hstu_dens_jagged(
        self,
        batch_size,
        max_seq_len,
        head_num,
        head_dim,
        mask_type,
        silu_scale,
        enable_bias,
        data_type,
        num_context,
        num_target,
        target_group_size,
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
            num_context,
            num_target,
            target_group_size,
        )
