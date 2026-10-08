#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""KV cache metadata (NPU mirror of GPU package).

Mirrors `corelib/recsys_kvcache_manager/recsys_kvcache_manager/kvcache_metadata.py`
so the base class ABC can reference `KVCacheMetadata`. Phase 1 native_host does
not exercise every field; they exist for parity with the GPU package so
future FlexKV / top-level KVCacheManager can subclass cleanly.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class KVCacheMetadata:
    """KV cache metadata for a batch (mirrors GPU package)."""

    page_ids_npu_buffer: torch.Tensor
    metadata_npu_buffer: torch.Tensor

    kv_indices: Optional[torch.Tensor] = None
    kv_indptr: Optional[torch.Tensor] = None
    kv_last_page_len: Optional[torch.Tensor] = None
    total_history_lengths: Optional[torch.Tensor] = None
    total_history_offsets: Optional[torch.Tensor] = None
    new_history_offsets: Optional[torch.Tensor] = None

    batch_indices: Optional[torch.Tensor] = None
    position: Optional[torch.Tensor] = None
    new_history_nnz: int = 0
    new_history_nnz_npu: Optional[torch.Tensor] = None

    kv_seqlens: Optional[torch.Tensor] = None
    kv_seqlen_offsets: Optional[torch.Tensor] = None

    kv_cache_table: Optional[List[torch.Tensor]] = None

    kv_onload_handle: Optional[object] = None

    max_seqlen: Optional[int] = 0

    onboard_slot_mappings: Optional[List[torch.Tensor]] = None
    onboard_task_ids: Optional[torch.Tensor] = None


def get_kvcache_metadata_buffer(
    batch_size: int,
    num_new_tokens: int,
    num_pages: int,
    page_ids_npu_buffer: Optional[torch.Tensor] = None,
    metadata_npu_buffer: Optional[torch.Tensor] = None,
    device: Optional[int] = None,
):
    if device is None:
        device = torch.npu.current_device()

    page_ids_npu_buffer = (
        torch.empty(
            (num_pages,),
            dtype=torch.int32,
            device=f"npu:{device}",
        )
        if page_ids_npu_buffer is None
        else page_ids_npu_buffer
    )

    metadata_npu_buffer = (
        torch.empty(
            (5 * batch_size + 4 + num_new_tokens * 2,),
            dtype=torch.int32,
            device=f"npu:{device}",
        )
        if metadata_npu_buffer is None
        else metadata_npu_buffer
    )

    page_indptr_buffer = metadata_npu_buffer.narrow(0, 0, batch_size + 1)
    last_page_lens_buffer = metadata_npu_buffer.narrow(0, batch_size + 1, batch_size)
    total_history_lengths = metadata_npu_buffer.narrow(0, batch_size * 2 + 1, batch_size)
    total_history_offsets = metadata_npu_buffer.narrow(0, batch_size * 3 + 1, batch_size + 1)
    new_history_nnz_npu = metadata_npu_buffer.narrow(0, batch_size * 4 + 2, 1)
    new_history_offsets = metadata_npu_buffer.narrow(0, batch_size * 4 + 3, batch_size + 1)
    batch_indices_buffer = metadata_npu_buffer.narrow(0, batch_size * 5 + 4, num_new_tokens)
    position_buffer = metadata_npu_buffer.narrow(0, batch_size * 5 + 4 + num_new_tokens, num_new_tokens)

    kv_seqlens = torch.empty_like(total_history_lengths)
    kv_seqlen_offsets = torch.empty_like(total_history_offsets)

    return KVCacheMetadata(
        page_ids_npu_buffer=page_ids_npu_buffer,
        metadata_npu_buffer=metadata_npu_buffer,
        kv_indices=page_ids_npu_buffer,
        kv_indptr=page_indptr_buffer,
        kv_last_page_len=last_page_lens_buffer,
        total_history_lengths=total_history_lengths,
        total_history_offsets=total_history_offsets,
        new_history_offsets=new_history_offsets,
        batch_indices=batch_indices_buffer,
        position=position_buffer,
        new_history_nnz=num_new_tokens,
        new_history_nnz_npu=new_history_nnz_npu,
        kv_seqlens=kv_seqlens,
        kv_seqlen_offsets=kv_seqlen_offsets,
        kv_onload_handle=None,
    )
