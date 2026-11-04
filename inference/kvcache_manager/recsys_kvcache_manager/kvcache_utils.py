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

"""KV cache utilities (NPU mirror of GPU package).

Mirrors `corelib/recsys_kvcache_manager/recsys_kvcache_manager/kvcache_utils.py`
so the base classes in `host_kvstorage_manager.py` can reference the same
types (`KVIndexMeta`, `KVLookupResult`) without touching the native subclass.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import torch


class KVCacheOffloadMode(Enum):
    LAZY = "lazy"
    EAGER = "eager"


@dataclass
class KVLookupResult:
    user_ids: torch.Tensor
    cached_start_indices: Optional[torch.Tensor] = None
    cached_lengths: Optional[torch.Tensor] = None
    npu_cached_start_indices: Optional[torch.Tensor] = None
    npu_cached_lengths: Optional[torch.Tensor] = None
    host_cached_start_indices: Optional[torch.Tensor] = None
    host_cached_lengths: Optional[torch.Tensor] = None
    token_ids: Optional[torch.Tensor] = None
    token_mask: Optional[torch.Tensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def merge(cls, lookup_res1, lookup_res2):
        """Merge NPU and host lookup results.

        Assumes one result has npu_cached_* filled and the other has host_cached_* filled.
        """
        if not torch.equal(lookup_res1.user_ids, lookup_res2.user_ids):
            raise ValueError("Lookup results must contain identical user_ids.")
        if lookup_res1.npu_cached_start_indices is not None and lookup_res1.npu_cached_lengths is not None:
            if lookup_res2.host_cached_start_indices is None or lookup_res2.host_cached_lengths is None:
                raise ValueError("Host lookup result must contain cached start indices and lengths.")

        if lookup_res1.host_cached_start_indices is not None and lookup_res1.host_cached_lengths is not None:
            if lookup_res2.npu_cached_start_indices is None or lookup_res2.npu_cached_lengths is None:
                raise ValueError("NPU lookup result must contain cached start indices and lengths.")
            lookup_res1, lookup_res2 = lookup_res2, lookup_res1

        # assume lookup_res1 is npu lookup result, lookup_res2 is host lookup result
        batch_size = lookup_res1.user_ids.size(0)
        cached_start_indices = torch.empty_like(lookup_res1.npu_cached_start_indices)
        cached_lengths = torch.empty_like(lookup_res1.npu_cached_lengths)
        for i in range(batch_size):
            cached_start_ind = 0
            cached_len = 0
            if lookup_res2.host_cached_lengths[i] == 0:
                cached_start_ind = lookup_res1.npu_cached_start_indices[i]
                cached_len = lookup_res1.npu_cached_lengths[i]
            elif lookup_res1.npu_cached_lengths[i] == 0:
                if lookup_res2.host_cached_start_indices[i] != 0:
                    raise ValueError("Host caching must start from the beginning of the sequence.")
                cached_start_ind = lookup_res2.host_cached_start_indices[i]
                cached_len = lookup_res2.host_cached_lengths[i]
            else:
                if lookup_res2.host_cached_start_indices[i] != 0:
                    raise ValueError("Host caching must start from the beginning of the sequence.")
                if lookup_res1.npu_cached_start_indices[i] < 0:
                    raise ValueError("Invalid NPU cache start index.")

                if lookup_res1.npu_cached_start_indices[i] > lookup_res2.host_cached_lengths[i]:
                    raise ValueError(
                        "No gaps allowed: NPU cache start index should be smaller than or equal to host cached length."
                    )
                cached_len = max(
                    lookup_res2.host_cached_lengths[i],
                    lookup_res1.npu_cached_start_indices[i] + lookup_res1.npu_cached_lengths[i],
                ).item()

            cached_start_indices[i] = cached_start_ind
            cached_lengths[i] = cached_len

        if (getattr(lookup_res1, "extra", {}) or {}) != {}:
            raise ValueError("NPU lookup results should not have extra fields.")
        merged_extra = getattr(lookup_res2, "extra", {}) or {}

        return cls(
            user_ids=lookup_res1.user_ids,
            cached_start_indices=cached_start_indices,
            cached_lengths=cached_lengths,
            npu_cached_start_indices=lookup_res1.npu_cached_start_indices,
            npu_cached_lengths=lookup_res1.npu_cached_lengths,
            host_cached_start_indices=lookup_res2.host_cached_start_indices,
            host_cached_lengths=lookup_res2.host_cached_lengths,
            extra=merged_extra,
        )


@dataclass
class KVIndexMeta:
    user_ids: torch.Tensor
    seq_lengths: torch.Tensor
