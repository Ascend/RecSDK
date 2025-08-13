#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, List
from enum import Enum

import torch


class InitializerType(Enum):
    LINEAR = 0
    TRUNCATED_NORMAL = 1
    UNIFORM = 2


class AdmitAndEvictConfig:
    admit_threshold: Optional[int]
    not_admitted_default_value: Optional[float]
    evict_threshold: Optional[int]
    evict_step_interval: Optional[int]


class EmbConfig:
    table_name: str
    initializer_type: InitializerType  # LINEAR TRUNCATED_NORMAL UNIFORM
    emb_dim: int
    optim_num: int
    cache_size: int
    weight_init_min: float = 0.0
    weight_init_max: float = 1.0
    weight_init_mean: float = 0.5
    weight_init_stddev: float = 0.5
    admit_and_evict_config: AdmitAndEvictConfig


class SwapInfo:
    swapout_keys: List[List[int]]
    swapout_offs: torch.Tensor
    swapin_keys: List[List[int]]
    swapin_offs: torch.Tensor
    batch_offs: torch.Tensor


class SwapinTensor:
    swapout_keys: List[List[int]]
    swapin_optims: List[torch.Tensor]
    jagged_offs: torch.Tensor


class SwapoutTensor:
    swapout_keys: List[List[int]]
    swapin_optims: List[torch.Tensor]
    jagged_offs: torch.Tensor


class AsyncSwapInfo:
    def get() -> SwapInfo:
        pass


class AsyncSwapinTensor:
    def get() -> "AsyncSwapinTensor":
        pass


class EmbcacheManager:
    def __init__(self, configs: List[EmbConfig]):
        pass
