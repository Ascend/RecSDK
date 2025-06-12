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

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch_npu

torch.npu.config.allow_internal_format = False
CURR_DIR = Path(__file__).resolve().parent
torch.ops.load_library(str(CURR_DIR.parent.parent / 
                           "torch_library/2.6.0/hstu_dense_backward_fuxi/build/libhstu_dense_fuxi_backward_ops.so"))

device_id: int = 0
mask_tril: int = 0
mask_triu: int = 1
mask_none: int = 2
mask_custom: int = 3

torch.manual_seed(3)


def jagged_data_gen(batch_size, max_seq_len, num_heads, attention_dim, mask_type, data_type):
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int64)
    seq_offset = torch.concat((torch.zeros((1,), dtype=torch.int64), torch.cumsum(seq_lens, axis=0))).numpy()
    
    total_seqs = torch.sum(seq_lens)

    start = -1
    end = 1
    grad = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(start, end)
    q = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(start, end)
    k = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(start, end)
    v = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(start, end)
    bpos = torch.empty(1, max_seq_len, max_seq_len, dtype=data_type).uniform_(start, end)
    bts = torch.empty(batch_size, max_seq_len, max_seq_len, dtype=data_type).uniform_(start, end)
    grad_pos = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(start, end)
    grad_ts = torch.empty(total_seqs, num_heads, attention_dim, dtype=data_type).uniform_(start, end)

    if mask_type == 0:
        mask = torch.tril(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type))
    elif mask_type == 1:
        mask = torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type))
    elif mask_type == 2:
        mask = None
    else:
        mask = torch.empty(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type).uniform_(start, end)

    return grad, q, k, v, bpos, bts, grad_pos, grad_ts, mask, max_seq_len, seq_offset


torch.npu.set_device(device_id)


class TestHstuJaggedDemo:
    def jagged_to_dense(self, jagged_tensor, seq_lens, max_seq_len, head_num, head_dim):
        """
        Convert jagged tensor to dense tensor.
        Args:
            jagged_tensor: Jagged tensor. (bs, n, d) 
            seq_lens: Sequence lengths. batch_size = len(seq_lens)
            max_seq_len: Maximum sequence length.
            head_num: Number of heads.
            head_dim: Dimension of each head.
        Returns:
            Dense tensor. (b, Smax, n, d)
        """
        batch_size = len(seq_lens)
        dense_tensor = torch.zeros(batch_size, max_seq_len, head_num, head_dim, dtype=jagged_tensor.dtype)

        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            dense_tensor[batch_id, :seq_len, :, :] = jagged_tensor[offset: offset + seq_len, :, :]
            offset = offset + seq_len

        return dense_tensor

    def dense_to_jagged(self, jagged_tensor, dense_tensor, seq_lens):
        """
        Convert dense tensor to jagged tensor.
        Args:
            jagged_tensor: Jagged tensor. (total_seqs, num_heads, attention_dim)
            dense_tensor: Dense tensor. (batch_size, max_seq_len, num_heads, attention_dim)
            seq_lens: Sequence lengths. batch_size = len(seq_lens)
        Returns:
            Jagged tensor. (total_seqs, num_heads, attention_dim)
        """
        tensor = torch.zeros_like(jagged_tensor)

        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            tensor[offset: offset + seq_len, :, :] = dense_tensor[batch_id, 0: seq_len, :, :]
            offset = offset + seq_len

        return tensor

    def compare_jagged_bias(self, bias_grad, bias_grad_golden, seq_offset, loss):
        seq_lens = torch.zeros(bias_grad.shape[0], dtype=torch.int64)
        for i in range(seq_lens.shape[0]):
            seq_lens[i] = seq_offset[i + 1] - seq_offset[i]
        
        for batch, seq_len in enumerate(seq_lens):
            equal = torch.allclose(bias_grad[batch, :, :seq_len, :seq_len],
                                   bias_grad_golden[batch, :, :seq_len, :seq_len],
                                   loss, loss)
            if not equal:
                return False

        return True

    def golden_op_exec(self, grad, q, k, v, bpos, bts, grad_pos, grad_ts, mask, max_seq_len, seq_offset, 
                       mask_type, silu_scale, enable_bias, data_type):
        device = f"npu:{device_id}"
        head_nums = grad.shape[1]
        head_dim = grad.shape[2]
        batch_size = bts.shape[0] # maybe get from mask

        seq_lens = np.zeros((batch_size,)).astype(np.int64)
        for batch_id in range(batch_size):
            seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]

        grad_dens = self.jagged_to_dense(grad, seq_lens, max_seq_len, head_nums, head_dim).to(f"npu:{device_id}")
        q_dens = self.jagged_to_dense(q, seq_lens, max_seq_len, head_nums, head_dim).to(f"npu:{device_id}")
        k_dens = self.jagged_to_dense(k, seq_lens, max_seq_len, head_nums, head_dim).to(f"npu:{device_id}")
        v_dens = self.jagged_to_dense(v, seq_lens, max_seq_len, head_nums, head_dim).to(f"npu:{device_id}")

        qk = torch.matmul(q_dens.permute(0, 2, 1, 3), k_dens.permute(0, 2, 3, 1))
        gv = torch.matmul(grad_dens.permute(0, 2, 1, 3), v_dens.permute(0, 2, 3, 1))

        qk = qk.float()
        gv = gv.float()

        if mask_type == 0 or mask_type == 3:
            mask = mask.to(f"npu:{device_id}")
            mask = mask.float()

        bts_grad = None
        bpos_grad = None
        if enable_bias:
            bts = bts.to(device).float()
            bpos = bpos.to(device).float()

            bts_b = bts.reshape(batch_size, 1, max_seq_len, max_seq_len)\
                .expand(batch_size, head_nums, max_seq_len, max_seq_len)
            bpos_b = bpos.reshape(1, 1, max_seq_len, max_seq_len)\
                .expand(batch_size, head_nums, max_seq_len, max_seq_len)
            if mask_type == mask_tril or mask_type == mask_custom:
                bts_b = bts_b * mask
                bpos_b = bpos_b * mask

            # b Smax n d
            grad_pos_dens = self.jagged_to_dense(grad_pos, seq_lens, max_seq_len, head_nums, head_dim).to(device)
            grad_ts_dens = self.jagged_to_dense(grad_ts, seq_lens, max_seq_len, head_nums, head_dim).to(device)

            # b n Smax d x b n d Smax -> b n Smax Smax
            gpos_v = torch.matmul(grad_pos_dens.permute(0, 2, 1, 3), v_dens.permute(0, 2, 3, 1))
            gts_v = torch.matmul(grad_ts_dens.permute(0, 2, 1, 3), v_dens.permute(0, 2, 3, 1))
            if mask_type == mask_tril or mask_type == mask_custom:
                gpos_v = gpos_v * mask.to(data_type)
                gts_v = gts_v * mask.to(data_type)
            bpos_grad = gpos_v.sum(dim=1).sum(dim=0, keepdim=True)
            bts_grad = gts_v.sum(dim=1)


            # qkb = qk + bias
            qkb = qk

        else:
            qkb = qk

        real_silu_scale = 1 / max_seq_len if silu_scale == 0.0 else silu_scale
        
        if mask_type == mask_tril or mask_type == mask_custom:
            score = F.silu(qkb) * real_silu_scale * mask
        else:
            score = F.silu(qkb) * real_silu_scale

        score = score.to(data_type)
        v_grad_dens = torch.matmul(score.permute(0, 1, 3, 2), grad_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        if enable_bias:
            bts_m = bts_b.to(data_type)
            bpos_m = bpos_b.to(data_type)
            bts_gts = torch.matmul(bts_m.permute(0, 1, 3, 2), grad_ts_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            bpos_gpos = torch.matmul(bpos_m.permute(0, 1, 3, 2), grad_pos_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            v_grad_dens = v_grad_dens + bpos_gpos + bts_gts

        if mask_type == mask_tril or mask_type == mask_custom:
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

        if enable_bias:
            bpos_grad = bpos_grad.to(data_type)
            bts_grad = bts_grad.to(data_type)
            bpos_grad = bpos_grad.cpu() if bpos_grad is not None else None
            bts_grad = bts_grad.cpu() if bts_grad is not None else None

        torch.npu.synchronize()

        return q_grad, k_grad, v_grad, bpos_grad, bts_grad
    
    def custom_op_exec(self, grad, q, k, v, bpos, bts, mask, seq_offset, 
                       mask_type, max_seq_len, silu_scale, enable_bias, data_type):
        grad_npu = grad.to(f"npu:{device_id}")
        q_npu = q.to(f"npu:{device_id}")
        k_npu = k.to(f"npu:{device_id}")
        v_npu = v.to(f"npu:{device_id}")
        bpos_npu = bpos.to(f"npu:{device_id}")
        bts_npu = bts.to(f"npu:{device_id}")

        mask_npu = None
        if mask_type == 3:
            mask_npu = mask.to(f"npu:{device_id}")

        if enable_bias:
            q_grad, k_grad, v_grad, bpos_grad, bts_grad = torch.ops.mxrec.hstu_dense_backward_fuxi(
                grad_npu, q_npu, k_npu, v_npu, mask_npu, bpos_npu, bts_npu, "jagged", 
                mask_type, max_seq_len, silu_scale, seq_offset
            )
        else:
            q_grad, k_grad, v_grad, bpos_grad, bts_grad = torch.ops.mxrec.hstu_dense_backward_fuxi(
                grad_npu, q_npu, k_npu, v_npu, mask_npu, None, None, "jagged",
                mask_type, max_seq_len, silu_scale, seq_offset
            )

        torch.npu.synchronize()
        return (q_grad.cpu(), k_grad.cpu(), v_grad.cpu(),
                (enable_bias and bpos_grad.cpu()), (enable_bias and bts_grad.cpu()))
    
    def execute(self, batch_size, max_seq_len, head_num, head_dim, mask_type, silu_scale, enable_bias, data_type):
        def allclose(a, b, loss1, loss2):
            diff = torch.abs(a - b) >= loss1
            diff_count = torch.sum(diff)
            diff_ratio = diff_count / a.numel()
            return diff_ratio < loss2

        grad, q, k, v, bpos, bts, grad_pos, grad_ts, mask, max_seq_len, seq_offset = \
            jagged_data_gen(batch_size, max_seq_len, head_num, head_dim, mask_type, data_type)

        grads = torch.cat((grad, grad_ts, grad_pos), -1) if enable_bias else grad
        q_grad, k_grad, v_grad, bpos_grad, bts_grad = self.custom_op_exec(
            grads, q, k, v, bpos, bts, mask, seq_offset, 
            mask_type, max_seq_len, silu_scale, enable_bias, data_type
        )

        q_grad_golden, k_grad_golden, v_grad_golden, bpos_grad_golden, bts_grad_golden = self.golden_op_exec(
            grad, q, k, v, bpos, bts, grad_pos, grad_ts, mask, max_seq_len, seq_offset, 
            mask_type, silu_scale, enable_bias, data_type
        )

        loss = 1e-4
        if data_type == torch.float16:
            loss = 1e-3
        elif data_type == torch.bfloat16:
            loss = 1e-2

        q_res = allclose(q_grad, q_grad_golden, loss, loss)
        k_res = allclose(k_grad, k_grad_golden, loss, loss)
        v_res = allclose(v_grad, v_grad_golden, loss, loss)
        bpos_res = not enable_bias or allclose(bpos_grad, bpos_grad_golden, loss, loss)
        bts_res = not enable_bias or allclose(bts_grad, bts_grad_golden, loss, loss)

        def analysis(a, b):
            diff = torch.abs(a - b) > loss
            diff_count = torch.sum(diff)
            diff_ratio = diff_count / a.numel()
            print(f"diff_count: {diff_count}, diff_ratio: {diff_ratio}")
            if diff_count == 0:
                return
            print(diff)
            print(f"index: {torch.nonzero(diff)}")
            print(f"a: {a[diff]}")
            print(f"b: {b[diff]}")
            print("")
        

        assert q_res, analysis(q_grad, q_grad_golden)
        assert k_res, analysis(k_grad, k_grad_golden)
        assert bpos_res, analysis(bpos_grad, bpos_grad_golden)
        assert bts_res, analysis(bts_grad, bts_grad_golden)
        assert v_res, analysis(v_grad, v_grad_golden)

    @pytest.mark.parametrize("batch_size", [2, 32, 64])
    @pytest.mark.parametrize("max_seq_len", [256, 1024, 1234, 2048])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("head_dim", [32, 128])
    @pytest.mark.parametrize("mask_type", [0, 2, 3])
    @pytest.mark.parametrize("silu_scale", [1.0 / 256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
    def test_hstu_dens_jagged(self, batch_size, max_seq_len, head_num, head_dim, mask_type, silu_scale,
                              enable_bias, data_type):
        self.execute(batch_size, max_seq_len, head_num, head_dim, mask_type, silu_scale, enable_bias, data_type)

    @pytest.mark.parametrize("max_seq_len", [16])
    @pytest.mark.parametrize("head_num", [2, 6])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("mask_type", [0, 2, 3])
    @pytest.mark.parametrize("silu_scale", [0.0])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
    def test_hstu_dens_jagged_128bs(self, max_seq_len, head_num, head_dim, mask_type, silu_scale,
                                    enable_bias, data_type):
        self.execute(128, max_seq_len, head_num, head_dim, mask_type, silu_scale, enable_bias, data_type)

    @pytest.mark.parametrize("max_seq_len", [16])
    @pytest.mark.parametrize("head_num", [2])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("mask_type", [0, 2, 3])
    @pytest.mark.parametrize("silu_scale", [0.0, 1.0 / 256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("data_type", [torch.float16, torch.float32, torch.bfloat16])
    def test_hstu_dens_jagged_512bs(self, max_seq_len, head_num, head_dim, mask_type, silu_scale,
                                    enable_bias, data_type):
        self.execute(512, max_seq_len, head_num, head_dim, mask_type, silu_scale, enable_bias, data_type)


class TestHstuJaggedInvalidParams:
    @pytest.mark.parametrize("batch_size", [0, -1, 513])
    def test_invalid_batch_size(self, batch_size):
        test = TestHstuJaggedDemo()
        with pytest.raises(Exception):
            test.execute(batch_size, 16, 2, 64, 0, 0.0, True, torch.float16)

    @pytest.mark.parametrize("max_seq_len", [0, -1, 20481])
    def test_invalid_max_seq_len(self, max_seq_len):
        test = TestHstuJaggedDemo()
        with pytest.raises(Exception):
            test.execute(4, max_seq_len, 2, 64, 0, 0.0, True, torch.float16)

    @pytest.mark.parametrize("head_num", [0, 1, 3, 5, 7, 9, 10])
    def test_invalid_head_num(self, head_num):
        test = TestHstuJaggedDemo()
        with pytest.raises(Exception):
            test.execute(4, 16, head_num, 64, 0, 0.0, True, torch.float16)

    @pytest.mark.parametrize("head_dim", [0, 8, 15, 33, 65, 129, 257, 513, 1024])
    def test_invalid_head_dim(self, head_dim):
        test = TestHstuJaggedDemo()
        with pytest.raises(Exception):
            test.execute(4, 16, 2, head_dim, 0, 0.0, True, torch.float16)

    @pytest.mark.parametrize("mask_type", [-1, 1, 4, 10])
    def test_invalid_mask_type(self, mask_type):
        test = TestHstuJaggedDemo()
        with pytest.raises(Exception):
            test.execute(4, 16, 2, 64, mask_type, 0.0, True, torch.float16)

    @pytest.mark.parametrize("data_type", [torch.int32, torch.int64, torch.uint8, torch.float64])
    def test_invalid_data_type(self, data_type):
        test = TestHstuJaggedDemo()
        with pytest.raises(Exception):
            test.execute(4, 16, 2, 64, 0, 0.0, True, data_type)

    def test_combined_invalid_params(self):
        """Test multiple invalid parameters at once"""
        test = TestHstuJaggedDemo()
        with pytest.raises(Exception):
            test.execute(0, 0, 9, 1024, 5, 0.0, True, torch.int32)
