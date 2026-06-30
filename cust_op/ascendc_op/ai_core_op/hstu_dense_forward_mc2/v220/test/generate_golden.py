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

import os

import torch
import torch.nn.functional as F
import subprocess
import itertools
import argparse


global_string_ref = None
torch.npu.config.allow_internal_format = False

local_rank = int(os.environ["LOCAL_RANK"])
device_id: int = local_rank
print("device_id :", device_id)
mask_tril: int = 0
mask_triu: int = 1
mask_none: int = 2
mask_custom: int = 3

DEFAULT_RANK_SIZE = 8


def get_chip():
    result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=False)
    for line in result.stdout.splitlines():
        if "310P" in line:
            return True
    return False


def set_ascend_env(rank, rank_size, local_rank_size, file=None, dev_id=-1, dev_index=1):
    rank = str(rank)
    rank_size_str = str(rank_size)
    local_rank_size = int(local_rank_size)

    os.environ["MOX_USE_NPU"] = "1"
    os.environ["FUSION_TENSOR_SIZE"] = "2000000000"
    os.environ["MOX_USE_TF_ESTIMATOR"] = "0"
    os.environ["MOX_USE_TDT"] = "1"
    os.environ["HEARTBEAT"] = "1"
    os.environ["CONTINUE_TRAIN"] = "true"

    os.environ["RANK_ID"] = rank
    local_rank_id = int(rank) % int(local_rank_size)
    if dev_id != -1:
        os.environ["DEVICE_ID"] = str(dev_id)
        os.environ["ASCEND_DEVICE_ID"] = str(dev_id)
    else:
        os.environ["DEVICE_ID"] = str(local_rank_id)
        os.environ["ASCEND_DEVICE_ID"] = str(local_rank_id)
    if dev_index != -1:
        os.environ["DEVICE_INDEX"] = str(dev_index)
    else:
        os.environ["DEVICE_INDEX"] = str(local_rank_id)

    os.environ["RANK_SIZE"] = rank_size_str
    if file:
        os.environ["RANK_TABLE_FILE"] = file

    os.environ["HCCL_CONNECT_TIMEOUT"] = "600"

    os.environ["JOB_ID"] = "10086"
    os.environ["SOC_VERSION"] = "Ascend910"
    os.environ["GE_AICPU_FLAG"] = "1"
    os.environ["NEW_GE_FE_ID"] = "1"
    os.environ["EXPERIMENTAL_DYNAMIC_PARTITION"] = "1"
    os.environ["ENABLE_FORCE_V2_CONTROL"] = "1"


def skip_seq_len(seq_len):
    block_len = 128
    if get_chip() and seq_len % block_len:
        return True
    return False


def generate_tensor(batch_size, max_seq_len, num_heads, attention_dim, dataType, maskType):
    total_num = batch_size * max_seq_len * num_heads * attention_dim

    q = torch.ones(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim) * (local_rank)
    k = torch.ones(total_num * 8).reshape(batch_size, max_seq_len * 8, num_heads, attention_dim)  # * local_rank  + 10
    v = torch.ones(total_num * 8).reshape(batch_size, max_seq_len * 8, num_heads, attention_dim)  # * local_rank  + 20

    for i in range(max_seq_len * 8):
        k[:, i, :, :] = i // max_seq_len + 1
        v[:, i, :, :] = i // max_seq_len + 2

    # Defer creation of bias/mask to avoid OOM for large seq_len
    # bias shape (B, heads, seq, seq) = 32 GiB for seq=16384!
    # Keep them on CPU as small dummies if not needed, create only when required
    rel_attn_bias = torch.zeros(1)  # dummy, will be replaced if enableBias=True
    invalid_attn_mask = torch.zeros(1)  # dummy, will be replaced if maskType!=none
    # Move to NPU as-is (small dummies), actual data created lazily in gloden_op_exec
    return (
        q.to(dataType).to(f"npu:{device_id}"),
        k.to(dataType).to(f"npu:{device_id}"),
        v.to(dataType).to(f"npu:{device_id}"),
        rel_attn_bias.to(f"npu:{device_id}"),
        invalid_attn_mask.to(f"npu:{device_id}"),
    )


class TestHstuNormalDemo:
    def gloden_op_exec(self, q, k, v, bias, mask, maskType, max_seq_len, siluScale, enableBias, dataType):
        B, n, num_heads, linear_dim = q.shape

        siluScale = 1 / max_seq_len if siluScale == 0 else siluScale

        # transpose: q -> (B, heads, q_seq, dim), k -> (B, heads, dim, k_seq), v -> (B, heads, v_seq, dim)
        q = q.permute(0, 2, 1, 3).to(torch.float32)
        k = k.permute(0, 2, 3, 1).to(torch.float32)
        v = v.permute(0, 2, 1, 3).to(torch.float32)
        k_seq_len = k.shape[3]  # = max_seq_len * 8
        q_seq_len = q.shape[2]

        # Delay conversion of bias/mask to avoid OOM for large seq_len
        # bias shape: (B, heads, q_seq_len, q_seq_len) -> 32 GiB for seq=16384!
        # Only convert when actually needed
        need_bias = enableBias
        need_mask = maskType != mask_none

        # Chunked attention to avoid OOM: qk_attn full size = B*heads*q_seq*k_seq
        # With bs=8,seq=16384 this is (8,4,16384,131072)=256GiB, too large for NPU
        # Instead, chunk k along seq_len dimension and accumulate output
        chunk_size = 2048
        atten_output = torch.zeros(B, num_heads, q_seq_len, linear_dim, device=q.device, dtype=torch.float32)

        for chunk_start in range(0, k_seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, k_seq_len)
            k_chunk = k[:, :, :, chunk_start:chunk_end]  # (B, heads, dim, chunk)
            v_chunk = v[:, :, chunk_start:chunk_end, :]  # (B, heads, chunk, dim)

            # partial attention: (B, heads, q_seq, chunk)
            qk_attn_chunk = torch.matmul(q, k_chunk)

            if need_bias:
                # bias shape is (B, heads, q_seq_len, q_seq_len)
                # For the 8-card scenario, k_seq = q_seq * 8
                # bias is only defined for local q_seq positions
                # Slice bias to match current k chunk
                bias_chunk_start = chunk_start % q_seq_len
                bias_chunk_end = min(bias_chunk_start + (chunk_end - chunk_start), q_seq_len)
                bias_chunk = bias[:, :, :, bias_chunk_start:bias_chunk_end].to(torch.float32)
                qk_attn_chunk = qk_attn_chunk + bias_chunk

            qk_attn_chunk = F.silu(qk_attn_chunk) * siluScale

            if need_mask:
                # mask shape: (B, heads, q_seq_len, q_seq_len)
                # For the 8-card scenario, apply local mask slice
                mask_chunk_start = chunk_start % q_seq_len
                mask_chunk_end = min(mask_chunk_start + (chunk_end - chunk_start), q_seq_len)
                mask_chunk = mask[:, :, :, mask_chunk_start:mask_chunk_end].to(torch.float32)
                qk_attn_chunk = qk_attn_chunk * mask_chunk

            # partial output: (B, heads, q_seq, dim)
            atten_output += torch.matmul(qk_attn_chunk, v_chunk)

            # Free chunk tensors
            del qk_attn_chunk, k_chunk, v_chunk

        atten_output = atten_output.permute(0, 2, 1, 3)
        torch.npu.synchronize()
        return atten_output.cpu().to(dataType).reshape(-1)

    def execute(self, batch_size, max_seq_len, head_num, head_dim, enableBias, maskType, siluScale, dataType):
        q, k, v, bias, mask = generate_tensor(batch_size, max_seq_len, head_num, head_dim, dataType, maskType)
        print("[DEBUG]generate_tensor down")
        torch.npu.synchronize()

        q_cpu = q.cpu().to(dataType)
        _ = k.cpu().to(dataType)
        _ = v.cpu().to(dataType)

        # 关键修改4：在保存前，再次检查数据类型和值（用于调试）
        print(f"q_cpu dtype: {q_cpu.dtype}")  # 应该是torch.float32
        print(f"q_cpu first few values: {q_cpu.flatten()[:10]}")  # 应该接近8.0

        gloden = self.gloden_op_exec(q, k, v, bias, mask, maskType, max_seq_len, siluScale, enableBias, dataType)

        gloden_cpu = gloden.cpu().to(dataType)
        with open("bin_file/output_" + str(local_rank) + "_tensor.bin", "wb") as f:
            f.write(gloden_cpu.numpy().tobytes())
        print("rankid: ", local_rank, "save q k v output done.\n")

        torch.npu.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=31, help="batch size")
    parser.add_argument("--seq", type=int, default=1024, help="seq length")
    args = parser.parse_args()

    hstuDeme = TestHstuNormalDemo()

    max_seq_len = [args.seq]
    paramsSeqlen = [i for i in max_seq_len if not skip_seq_len(i)]
    batch_size = [args.bs]
    head_num = [4]
    max_seq_len = paramsSeqlen
    head_dim = [64]
    # enableBias = [True] if get_chip() else [True, False]
    enableBias = [False]
    maskType = [mask_none]
    siluScale = [1 / 256]
    dataType = [torch.float32]

    print("test case: batch_size, max_seq_len, head_num, head_dim, enableBias, maskType, siluScale, dataType")
    for combo in itertools.product(
        batch_size, max_seq_len, head_num, head_dim, enableBias, maskType, siluScale, dataType
    ):
        print(f"test case: {combo}")
        hstuDeme.execute(*combo)


if __name__ == "__main__":
    torch.npu.set_device(device_id)
    set_ascend_env(device_id, DEFAULT_RANK_SIZE, DEFAULT_RANK_SIZE)
    main()
