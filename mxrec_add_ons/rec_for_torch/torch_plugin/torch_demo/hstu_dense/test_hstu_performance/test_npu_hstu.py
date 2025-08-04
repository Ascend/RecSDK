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

import argparse
import os
import sysconfig

import torch

import config
from test_read_benchmark import (
    logger,
    DATASETS,
    load_params,
    read_and_validate_parameters,
)

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    silu_value: float,
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
    seq_offset: torch.Tensor,
    data_type: torch.dtype,
    device: str,
):
    n: int = invalid_attn_mask.size(-1)
    torch.npu.set_device(device)

    q_ = q.reshape(-1, num_heads, attention_dim).to(device=device).to(data_type)
    k_ = k.reshape(-1, num_heads, attention_dim).to(device=device).to(data_type)
    v_ = v.reshape(-1, num_heads, attention_dim).to(device=device).to(data_type)
    grad = grad.to(device=device).to(data_type)

    seq_offset = seq_offset.to(device=device).tolist()

    if len(invalid_attn_mask.shape) == 2:
        invalid_attn_mask = invalid_attn_mask.repeat(
            len(seq_offset) - 1, num_heads, 1, 1
        )
    if len(invalid_attn_mask.shape) == 4 and invalid_attn_mask.shape[1] == 1:
        invalid_attn_mask = invalid_attn_mask.repeat(1, num_heads, 1, 1)

    logger.info(f"invalid_attn_mask shape: {invalid_attn_mask.shape}")

    invalid_attn_mask = invalid_attn_mask.to(device=device).to(data_type)
    mask_type = 3
    silu_value = silu_value / n
    local_cycle_nums = 100
    for _ in range(local_cycle_nums):
        grad_output = torch.ops.mxrec.hstu_dense(
            q_,
            k_,
            v_,
            invalid_attn_mask,
            None,
            mask_type,
            n,
            silu_value,
            "jagged",
            seq_offset,
        )
        q_grad, k_grad, v_grad, _ = torch.ops.mxrec.hstu_dense_backward(
            grad,
            q_,
            k_,
            v_,
            invalid_attn_mask,
            None,
            "jagged",
            mask_type,
            n,
            silu_value,
            seq_offset,
        )

        torch.npu.synchronize()
        grad_output = grad_output.reshape(-1, num_heads * linear_dim)

    save_dir = DATASETS

    torch.save(grad_output, os.path.join(save_dir, "npu_out.pth"))
    torch.save(q_grad, os.path.join(save_dir, "npu_q.pth"))
    torch.save(k_grad, os.path.join(save_dir, "npu_k.pth"))
    torch.save(v_grad, os.path.join(save_dir, "npu_v.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read CSV file and run a specific index benchmark"
    )
    parser.add_argument(
        "--index", type=int, required=True, help="index of the benchmark to run"
    )
    args = parser.parse_args()

    devicex = 0
    deviceg = f"npu:{devicex}"
    logger.info(f"device: {deviceg}")
    read_dir = os.path.join(os.path.realpath(config.NFS_DIR), DATASETS)

    _, param_ben = read_and_validate_parameters(args.index)
    param = load_params(DATASETS)
    try:
        grad_data = param["grad"]
        q_data = param["q"]
        k_data = param["k"]
        v_data = param["v"]
        bias_data = param["rab"]
        mask_data = param["attn_mask"]
        max_seq_len_data = param["num_contexts"]
        seq_offset_data = param["seq_offsets_q_wt"]
        num_heads_data = v_data.shape[1]
        data_type_data = param_ben["dtype"]
        alpha_data = param["alpha"]
        attention_dim_data = q_data.shape[2]
        linear_dim_data = v_data.shape[2]
    except KeyError as e:
        logger.error(e)
        exit(1)

    _hstu_attention_maybe_from_cache(
        num_heads=num_heads_data,
        attention_dim=attention_dim_data,
        linear_dim=linear_dim_data,
        silu_value=alpha_data,
        grad=grad_data,
        q=q_data,
        k=k_data,
        v=v_data,
        invalid_attn_mask=mask_data,
        seq_offset=seq_offset_data,
        data_type=data_type_data,
        device=deviceg,
    )
