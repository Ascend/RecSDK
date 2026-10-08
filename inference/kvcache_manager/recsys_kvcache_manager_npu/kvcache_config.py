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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import torch

from .kvcache_utils import KVCacheOffloadMode


class HostKVStorageBackend(str, Enum):
    NATIVE_HOST = "native_host"


class HostKVStorageFailPolicy(str, Enum):
    FAIL_OPEN = "fail_open"
    FAIL_CLOSE = "fail_close"


@dataclass
class KVCacheConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    page_size: int
    offload_chunksize: int

    num_primary_cache_pages: int
    num_buffer_pages: int
    host_capacity_per_layer: int

    max_batch_size: int
    max_seq_len: int

    dtype: torch.dtype
    device: int

    host_kvstorage_backend: str = HostKVStorageBackend.NATIVE_HOST.value

    onload_timeout_ms: float = 0.0
    offload_timeout_ms: float = 1000.0
    offload_mode: str = KVCacheOffloadMode.LAZY.value
    host_kvstorage_fail_policy: str = HostKVStorageFailPolicy.FAIL_OPEN.value

    extra_configs: Dict[str, Any] = field(default_factory=dict)


def get_kvcache_config(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    page_size: int,
    offload_chunksize: int,
    num_primary_cache_pages: int,
    num_buffer_pages: int,
    host_capacity_per_layer: int,
    max_batch_size: int,
    max_seq_len: int,
    dtype: torch.dtype,
    device: int,
    host_kvstorage_backend: str = HostKVStorageBackend.NATIVE_HOST.value,
    onload_timeout_ms: float = 0.0,
    offload_timeout_ms: float = 0.0,
    offload_mode: str = KVCacheOffloadMode.LAZY.value,
    host_kvstorage_fail_policy: str = HostKVStorageFailPolicy.FAIL_OPEN.value,
    extra_configs: Optional[Dict[str, Any]] = None,
) -> KVCacheConfig:
    return KVCacheConfig(  # type: ignore
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        offload_chunksize=offload_chunksize,
        num_primary_cache_pages=num_primary_cache_pages,
        num_buffer_pages=num_buffer_pages,
        host_capacity_per_layer=host_capacity_per_layer,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
        host_kvstorage_backend=host_kvstorage_backend,
        onload_timeout_ms=onload_timeout_ms,
        offload_timeout_ms=offload_timeout_ms,
        offload_mode=offload_mode,
        host_kvstorage_fail_policy=host_kvstorage_fail_policy,
        extra_configs=dict(extra_configs or {}),
    )
