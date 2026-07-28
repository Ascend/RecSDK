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

torch.npu.config.allow_internal_format = False


class Kernel:
    def __init__(self, alpha, scale, has_rab, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k):
        self.alpha = alpha
        self.scale = scale
        self.has_rab = has_rab
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k

        self.seq_offset_q = seq_offset_q
        self.seq_offset_k = seq_offset_k

    def forward(self, q, k, v, rab, mask):
        q_npu = q.to("npu")
        k_npu = k.to("npu")
        v_npu = v.to("npu")
        rab_npu = rab.to("npu") if isinstance(rab, torch.Tensor) else None

        seq_offset_q = torch.Tensor(self.seq_offset_q).to("npu").to(torch.int32)
        seq_offset_k = torch.Tensor(self.seq_offset_k).to("npu").to(torch.int32)

        num_context = None
        num_target = None
        target_group_size = None
        mask_npu = None
        mask_type = 0

        output = torch.ops.mxrec.hstu_forward_v2(
            q_npu,
            k_npu,
            v_npu,
            mask_npu,
            rab_npu,
            mask_type,
            self.max_seqlen_q,
            self.max_seqlen_k,
            self.scale,
            seq_offset_q,
            seq_offset_k,
            num_context,
            num_target,
            target_group_size,
            self.alpha,
        )
        torch.npu.synchronize()
        return output.cpu()

    def create_backward_metadata(self, q, v):
        """调用 hstu_attn_metadata，为 HSTU backward 的 K 行调度生成 metadata。"""
        seq_offset_q = torch.as_tensor(self.seq_offset_q, dtype=torch.int32, device="npu")
        seq_offset_k = torch.as_tensor(self.seq_offset_k, dtype=torch.int32, device="npu")
        batch_size = seq_offset_q.numel() - 1
        head_num = q.shape[1]
        head_dim = max(q.shape[-1], v.shape[-1])

        metadata = torch.ops.mxrec.hstu_attn_metadata(
            seq_offset_q,
            seq_offset_k,
            None,
            None,
            batch_size,
            self.max_seqlen_q,
            self.max_seqlen_k,
            head_num,
            head_num,
            head_dim,
            0,
            -1,
            -1,
            "TND",
            "TND",
            "TND",
        )
        torch.npu.synchronize()
        return metadata

    def backward(
        self, grad, q, k, v, rab, mask, window_size, num_context, num_target, target_group_size, metadata=None
    ):
        grad_npu = grad.to("npu")
        q_npu = q.to("npu")
        k_npu = k.to("npu")
        v_npu = v.to("npu")

        if rab is not None:
            rab_npu = rab.to("npu")
        else:
            rab_npu = None

        seq_offset_q = torch.Tensor(self.seq_offset_q).to("npu").to(torch.int32)
        seq_offset_k = torch.Tensor(self.seq_offset_k).to("npu").to(torch.int32)

        batch_size = len(seq_offset_q) - 1
        if num_context is not None:
            num_context = torch.Tensor([num_context for _ in range(batch_size)]).to("npu").to(torch.int32)
        if num_target is not None:
            num_target = torch.Tensor([num_target for _ in range(batch_size)]).to("npu").to(torch.int32)

        q_grad, k_grad, v_grad, rab_grad = torch.ops.mxrec.hstu_backward_v2(
            grad_npu,
            q_npu,
            k_npu,
            v_npu,
            self.max_seqlen_q,
            self.max_seqlen_k,
            seq_offset_q,
            seq_offset_k,
            rab_npu,
            num_context,
            num_target,
            self.scale,
            target_group_size,
            self.alpha,
            window_size[0],
            window_size[1],
            metadata,
        )

        torch.npu.synchronize()
        return q_grad.cpu(), k_grad.cpu(), v_grad.cpu(), rab_grad.cpu() if rab is not None else None


class AscendFuse:
    def __init__(self, device, ops_library_dir):
        torch.npu.set_device(device)
        torch.ops.load_library(ops_library_dir)

    @staticmethod
    def kernel(alpha, scale, has_rab, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k):
        return Kernel(alpha, scale, has_rab, max_seqlen_q, max_seqlen_k, seq_offset_q, seq_offset_k)

    @staticmethod
    def validator():
        raise RuntimeError("backend ascend fuse no validator")
