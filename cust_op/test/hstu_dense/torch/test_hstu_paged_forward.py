#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
# Copyright c) 2025 NVIDIA CORPORATION AFFILIATES. All rights reserved.
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

from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from test_common_utils import allclose, MaskType


def pad_input(unpadded_input, cu_seqlen, batch, seqlen):
    indices = []
    for i in range(batch):
        indices.append(
            torch.arange(seqlen * i, seqlen * i + cu_seqlen[i + 1] - cu_seqlen[i])
        )
    indices = torch.cat(indices)
    output = torch.zeros(
        (batch * seqlen),
        *unpadded_input.shape[1:],
        device=unpadded_input.device,
        dtype=unpadded_input.dtype
    )
    output[indices] = unpadded_input
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def pad_input_delta_q(unpadded_input, cu_seqlen_q, cu_seqlen_k, batch, seqlen):
    indices = []
    for i in range(batch):
        act_seqlen_q = (cu_seqlen_q[i + 1] - cu_seqlen_q[i]).item()
        act_seqlen_k = (cu_seqlen_k[i + 1] - cu_seqlen_k[i]).item()
        indices.append(
            torch.arange(
                seqlen * i + act_seqlen_k - act_seqlen_q, seqlen * i + act_seqlen_k
            )
        )
    indices = torch.cat(indices)
    output = torch.zeros(
        (batch * seqlen),
        *unpadded_input.shape[1:],
        device=unpadded_input.device,
        dtype=unpadded_input.dtype
    )
    output[indices] = unpadded_input
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def unpad_input(padded_input, cu_seqlen):
    padded_input.reshape(padded_input.size(0), padded_input.size(1), -1)
    output = []
    for i in range(len(cu_seqlen) - 1):
        output.append(padded_input[i, : (cu_seqlen[i + 1] - cu_seqlen[i]), :])
    return torch.cat(output, dim=0)


def unpad_input_delta_q(padded_input, cu_seqlen_q, cu_seqlen_k, batch, seqlen):
    padded_input.reshape(padded_input.size(0), padded_input.size(1), -1)
    output = []
    for i in range(batch):
        act_seqlen_q = (cu_seqlen_q[i + 1] - cu_seqlen_q[i]).item()
        act_seqlen_k = (cu_seqlen_k[i + 1] - cu_seqlen_k[i]).item()
        output.append(padded_input[i, act_seqlen_k - act_seqlen_q: act_seqlen_k, :])
    return torch.cat(output, dim=0)


def generate_attn_mask(batch_size, num_heads, max_seq_len, max_seq_len_k, dtype,
                       seqlen, kv_raw_metadata, kvcache_page_size, num_candidates):
    mask = torch.zeros(
            (batch_size, num_heads, max_seq_len, max_seq_len_k),
            dtype=dtype,
            device=torch.npu.current_device(),
        )
    for seq_idx in range(batch_size):
        qlen = seqlen[seq_idx].item()
        cachelen = (
            kv_raw_metadata[1][seq_idx + 1] - kv_raw_metadata[1][seq_idx] - 1
        ) * kvcache_page_size + kv_raw_metadata[2][seq_idx] # (pageNum - 1) * pageSize + lastPageLen
        num_cand = num_candidates[seq_idx].item()
        seq_mask = torch.cat(
            [
                torch.tril(
                    torch.ones((qlen, cachelen), dtype=torch.int32),
                    diagonal=cachelen + num_cand - qlen,
                ),
                torch.cat(
                    [
                        torch.zeros((qlen - num_cand, num_cand), dtype=torch.int32),
                        torch.eye(num_cand, dtype=torch.int32),
                    ],
                    dim=0,
                ),
            ],
            dim=1,
        )
        mask[seq_idx, :, :qlen, : cachelen + num_cand].copy_(seq_mask.type(dtype))
    return mask


def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    seqlen_q: int,
    seqlen_k: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    rab: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,
    upcast: bool = True,
    is_delta_q: bool = False,
    alpha: float = 1.0,
):
    B: int = q_offsets.size(0) - 1
    if is_delta_q:
        padded_q = pad_input_delta_q(q, q_offsets, k_offsets, B, seqlen_k)
    else:
        padded_q = pad_input(q, q_offsets, B, seqlen_q)
    padded_k = pad_input(k, k_offsets, B, seqlen_k)
    padded_v = pad_input(v, k_offsets, B, seqlen_k)

    padded_q = padded_q.view(B, seqlen_q, num_heads, attention_dim)
    padded_k = padded_k.view(B, seqlen_k, num_heads, attention_dim)
    padded_v = padded_v.view(B, seqlen_k, num_heads, linear_dim)
    if upcast:
        padded_q, padded_k, padded_v = (
            padded_q.float(),
            padded_k.float(),
            padded_v.float(),
        )
        if rab is not None:
            rab = rab.float()
    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        padded_q,
        padded_k,
    )

    if rab is not None:
        padding = (
            0,
            qk_attn.shape[-1] - rab.shape[-1],
            0,
            qk_attn.shape[-2] - rab.shape[-2],
        )
        rab = F.pad(rab, padding, value=0)
        masked_qk_attn = qk_attn + rab
    else:
        masked_qk_attn = qk_attn
    masked_qk_attn = masked_qk_attn * alpha
    masked_qk_attn = F.silu(masked_qk_attn)
    masked_qk_attn = masked_qk_attn / seqlen_q
    
    if invalid_attn_mask is not None:
        if invalid_attn_mask.ndim == 2:
            invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(0)
        ext_invalid_attn_mask = torch.zeros_like(masked_qk_attn)
        d0, d1, d2, d3 = invalid_attn_mask.shape
        d1 = masked_qk_attn.shape[1]
        ext_invalid_attn_mask[:d0, :d1, :d2, :d3].copy_(
            invalid_attn_mask.type(masked_qk_attn.dtype)[:, :, :, :]
        )
        masked_qk_attn = masked_qk_attn * ext_invalid_attn_mask[:, :, :, :]

    attn_output = torch.einsum(
        "bhnm,bmhd->bnhd",
        masked_qk_attn,
        padded_v,
    )

    attn_output = attn_output.reshape(B, seqlen_q, num_heads * linear_dim)
    if is_delta_q:
        attn_output = unpad_input_delta_q(
            attn_output, q_offsets, k_offsets, B, seqlen_k
        )
    else:
        attn_output = unpad_input(attn_output, q_offsets)
    attn_output = attn_output.reshape(-1, num_heads * linear_dim)

    return attn_output


def _hstu_paged_kv_attention(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    seqlen_q: int,
    seqlen_k: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    num_targets: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
    upcast: bool = True,
    kv_cache: torch.Tensor = None,
    page_offsets: torch.Tensor = None,
    page_ids: torch.Tensor = None,
    last_page_lens: torch.Tensor = None,
    alpha: float = 1.0,
):
    k_con = torch.empty((0, num_heads, attention_dim), device=k.device, dtype=k.dtype)
    v_con = torch.empty((0, num_heads, attention_dim), device=v.device, dtype=v.dtype)

    for i, last_page_len in enumerate(last_page_lens):
        page_num = page_offsets[i + 1] - page_offsets[i]
        new_history_len = q_offsets[i + 1] - q_offsets[i] - num_targets[i]
        for j in range(page_num - 1):
            k_con = torch.cat(
                (k_con, kv_cache[page_ids[page_offsets[i] + j], 0, :, :, :]), dim=0
            )
            v_con = torch.cat(
                (v_con, kv_cache[page_ids[page_offsets[i] + j], 1, :, :, :]), dim=0
            )
        k_con = torch.cat(
            (
                k_con,
                kv_cache[
                    page_ids[page_offsets[i + 1] - 1], 0, : last_page_len, :, :
                ],
            ),
            dim=0,
        )
        k_con = torch.cat(
            (k_con, k[(q_offsets[i] + new_history_len): q_offsets[i + 1], :, :]), dim=0
        )
        v_con = torch.cat(
            (
                v_con,
                kv_cache[
                    page_ids[page_offsets[i + 1] - 1], 1, : last_page_len, :, :
                ],
            ),
            dim=0,
        )
        v_con = torch.cat(
            (v_con, v[(q_offsets[i] + new_history_len): q_offsets[i + 1], :, :]), dim=0
        )

    return _hstu_attention_maybe_from_cache(
        num_heads=num_heads,
        attention_dim=attention_dim,
        linear_dim=linear_dim,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        q=q,
        k=k_con,
        v=v_con,
        q_offsets=q_offsets,
        k_offsets=k_offsets,
        rab=None,
        invalid_attn_mask=invalid_attn_mask,
        upcast=upcast,
        is_delta_q=False,
        alpha=alpha,
    )


def get_offsets_from_lengths(lengths):
    offsets = torch.zeros(
        (lengths.shape[0] + 1,), dtype=lengths.dtype, device=lengths.device
    )
    torch.cumsum(lengths, 0, out=offsets[1:])
    return offsets


def generate_kvdata_testcase(
    max_seq_len: int, batch_size: int, num_heads: int, head_dim: int, is_prev_hist: bool = False
):
    min_seq_len = min(1, max_seq_len // 2)
    new_hist_lens = torch.randint(
        min_seq_len, max_seq_len + 1, (batch_size,), dtype=torch.int32
    )
    prev_lengths = torch.randint(
            1, max_seq_len + 1, (batch_size,), dtype=torch.int32
        ) * int(is_prev_hist)

    lengths = new_hist_lens + prev_lengths
    num_tokens = torch.sum(lengths).item()
    values = torch.randn(
        (2, num_tokens, num_heads, head_dim),
        dtype=torch.bfloat16,
        device=torch.npu.current_device(),
    ).uniform_(-1, 1)
    return (values, new_hist_lens, lengths)


def setup_kvcache_testcase(
    page_size: int,
    kvcache_num_heads: int,
    kvcache_head_dim: int,
    kvcache_dtype: torch.dtype,
    new_history_kv_data: torch.Tensor,  # [num_layers, 2, num_tokens, num_heads, head_dim]
    new_history_kv_lengths: torch.Tensor,
    kvcache_table: Optional[torch.Tensor] = None,
):
    batch_size = new_history_kv_lengths.shape[0]

    new_history_kv_length_offsets = get_offsets_from_lengths(new_history_kv_lengths)

    kv_page_indices = list()
    kv_page_indptr = list([0])
    kv_last_page_len = list()
    total_page_size = new_history_kv_lengths.sum() // page_size + batch_size
    if kvcache_table is None:
        kvcache_table = torch.zeros(
            (total_page_size, 2, page_size, kvcache_num_heads, kvcache_head_dim),
            dtype=kvcache_dtype,
            device=torch.npu.current_device(),
        )

    acc_num_pages = 0
    for seq_idx in range(batch_size):
        new_history_length = new_history_kv_lengths[seq_idx].item()

        user_kv_data = new_history_kv_data[
            :,
            new_history_kv_length_offsets[seq_idx]: new_history_kv_length_offsets[
                seq_idx + 1
            ],
            ...,
        ]

        # Allocation
        num_pages = (new_history_length + page_size - 1) // page_size
        page_ids = list(range(acc_num_pages, acc_num_pages + num_pages))
        acc_num_pages += num_pages

        # Copy data
        last_page_size = new_history_length % page_size
        for page_idx in range(0, (new_history_length - last_page_size) // page_size):
            page_id = page_ids[page_idx]
            token_begin = page_idx * page_size
            token_end = (page_idx + 1) * page_size
            kvcache_table[page_id, ...].copy_(
                user_kv_data[:, token_begin:token_end, ...], non_blocking=True
            )
        if last_page_size > 0:
            page_idx = (new_history_length - last_page_size) // page_size
            page_id = page_ids[page_idx]
            token_begin = page_idx * page_size
            token_end = token_begin + last_page_size
            kvcache_table[page_id, ...] *= 0.0
            kvcache_table[page_id, :, :last_page_size, ...].copy_(
                user_kv_data[:, token_begin:token_end, ...], non_blocking=True
            )

        kv_page_indices.extend(page_ids)
        kv_page_indptr.append(acc_num_pages)
        kv_last_page_len.append(last_page_size if last_page_size > 0 else page_size)

    torch.npu.synchronize()

    return kvcache_table, (kv_page_indices, kv_page_indptr, kv_last_page_len)


@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize(
    "max_new_history_seqlen,max_num_candidates",
    [
        (100, 0),
        (200, 25),
        (750, 80),
        (2048, 300),
    ],
)
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [32, 128])
@pytest.mark.parametrize("page_size", [32, 128, 256])
@pytest.mark.parametrize("is_prev_hist", [True, False])
def test_paged_hstu_attn_kernel(
    batch_size,
    dtype,
    max_new_history_seqlen,
    max_num_candidates,
    num_heads,
    head_dim,
    page_size,
    is_prev_hist
):
    device = torch.npu.current_device()

    max_seq_len = max_new_history_seqlen + max_num_candidates
    max_kvcache_history_seqlen = max_new_history_seqlen

    # (values, new_hist_lens, lengths)
    kv_data, new_hist_lens, kvdata_seqlen = generate_kvdata_testcase(
        max_kvcache_history_seqlen, batch_size, num_heads, head_dim, is_prev_hist=is_prev_hist
    )
    # kvcache_table[total_page_num, 2, page_size, num_head, head_dims]
    # kv_raw_metadata tuple(kv_page_indices, kv_page_num, kv_last_page_len)
    kvcache_table, kv_raw_metadata = setup_kvcache_testcase(
        page_size, num_heads, head_dim, dtype, kv_data, kvdata_seqlen, None
    )

    if max_num_candidates > 0:
        num_candidates = torch.randint(
            1, max_num_candidates + 1, (batch_size,), dtype=torch.int32
        )
    else:
        num_candidates = torch.zeros((batch_size,), dtype=torch.int32)
    num_candidates_offsets = get_offsets_from_lengths(num_candidates)

    kvdata_seqlen_offsets = get_offsets_from_lengths(kvdata_seqlen)
    seqlen = new_hist_lens + num_candidates
    seqlen_offsets = get_offsets_from_lengths(seqlen)
    num_tokens = seqlen_offsets[-1].item()

    query = torch.randn(
        (num_tokens, num_heads, head_dim), dtype=dtype, device=device
    ).uniform_(-1, 1)
    key = torch.randn(
        (num_tokens, num_heads, head_dim), dtype=dtype, device=device
    ).uniform_(-1, 1)
    value = torch.randn(
        (num_tokens, num_heads, head_dim), dtype=dtype, device=device
    ).uniform_(-1, 1)

    max_kvdata_len = kvdata_seqlen.max().item()
    max_target_len = num_candidates.max().item()
    torch.npu.synchronize()
    seqlen_k = num_candidates.npu() + kvdata_seqlen.npu()
    seqlen_offsets_k = get_offsets_from_lengths(seqlen_k)
    max_seq_len_q = seqlen.max()
    max_seq_len_k = seqlen_k.max()
    # generate attn mask
    mask = generate_attn_mask(batch_size, num_heads, max_seq_len_q, max_seq_len_k, dtype,
                              seqlen, kv_raw_metadata, page_size, num_candidates)

    attn_out_golden = _hstu_paged_kv_attention(
        num_heads=num_heads,
        attention_dim=head_dim,
        linear_dim=head_dim,
        seqlen_q=max_seq_len_q,
        seqlen_k=max_seq_len_k,
        q=query,
        k=key,
        v=value,
        q_offsets=seqlen_offsets.npu(),
        k_offsets=seqlen_offsets_k.npu(),
        num_targets=num_candidates.npu(),
        invalid_attn_mask=mask.npu(),
        upcast=False,
        kv_cache=kvcache_table,
        page_offsets=torch.tensor(
            kv_raw_metadata[1], dtype=torch.int32, device=device
        ),
        page_ids=torch.tensor(kv_raw_metadata[0], dtype=torch.int32, device=device),
        last_page_lens=torch.tensor(
            kv_raw_metadata[2], dtype=torch.int32, device=device
        ),
        alpha=1.0 / (head_dim ** 0.5),
    )
    torch.npu.synchronize()
    attn_out_golden = attn_out_golden.view(-1, num_heads, head_dim)
    attn_out_custom = torch.ops.mxrec.hstu_paged(q=query,
                                                 k=key,
                                                 v=value,
                                                 kv_cache=kvcache_table.npu(),
                                                 mask=mask.npu(),
                                                 attn_bias=None,
                                                 mask_type=MaskType.CUSTOM,
                                                 max_seq_len=max_seq_len_q,
                                                 max_seq_len_k=max_seq_len_k,
                                                 silu_scale=1.0 / max_seq_len_q,
                                                 alpha=1.0 / (head_dim ** 0.5),
                                                 target_group_size=1,
                                                 seq_offset=seqlen_offsets.npu().to(torch.int64),
                                                 seq_offset_k=seqlen_offsets_k.npu().to(torch.int64),
                                                 seq_offset_t=num_candidates_offsets.npu().to(torch.int64),
                                                 page_offsets=torch.tensor(kv_raw_metadata[1]).npu().to(torch.int64),
                                                 page_ids=torch.tensor(kv_raw_metadata[0]).npu().to(torch.int64),
                                                 last_page_len=torch.tensor(kv_raw_metadata[2]).npu().to(torch.int64),
                                                 num_target=num_candidates.npu().to(torch.int64))
    torch.npu.synchronize()
    output = attn_out_custom.cpu()
    golden = attn_out_golden.cpu()
    if dtype == torch.bfloat16:
        res = allclose(output, golden, 1e-2, 1e-2)
    elif dtype == torch.float16:
        res = allclose(output, golden, 1e-3, 1e-3)
    else:
        res = allclose(output, golden, 1e-4, 1e-4)
    assert res


if __name__ == "__main__":
    test_paged_hstu_attn_kernel(batch_size=1,
                                dtype=torch.float16,
                                max_new_history_seqlen=32,
                                max_num_candidates=1,
                                num_heads=1,
                                head_dim=16,
                                page_size=32,
                                is_prev_hist=True)
