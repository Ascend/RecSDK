#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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


from typing import Optional
import abc

import torch

from dynamic_emb.distributed.dynamicemb_config import DynamicEmbInitializerArgs
from dynamic_emb_extensions import (
    CurandStateContext,
    const_init,
    debug_init,
    normal_init,
    truncated_normal_init,
    uniform_init,
)


class BaseDynamicEmbInitializer(abc.ABC):
    def __init__(self, args: DynamicEmbInitializerArgs):
        self._args = args
        if self._args.lower is None:
            self._args.lower = 0.0
        if self._args.upper is None:
            self._args.upper = 1.0

    @abc.abstractmethod
    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],
    ) -> None: ...


class NormalInitializer(BaseDynamicEmbInitializer):
    def __init__(self, args: DynamicEmbInitializerArgs):
        super().__init__(args)
        self._curand_state = CurandStateContext()

    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],  # remove it when debug mode is removed
    ) -> None:
        normal_init(buffer, indices, self._curand_state, self._args.mean, self._args.std_dev)


class ConstantInitializer(BaseDynamicEmbInitializer):
    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],  # remove it when debug mode is removed
    ) -> None:
        const_init(buffer, indices, self._args.value)


class UniformInitializer(BaseDynamicEmbInitializer):
    def __init__(self, args: DynamicEmbInitializerArgs):
        super().__init__(args)
        self._curand_state = CurandStateContext()

    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],  # remove it when debug mode is removed
    ) -> None:
        uniform_init(buffer, indices, self._curand_state, self._args.lower, self._args.upper)


class TruncatedNormalInitializer(BaseDynamicEmbInitializer):
    def __init__(self, args: DynamicEmbInitializerArgs):
        super().__init__(args)
        self._curand_state = CurandStateContext()

    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],  # remove it when debug mode is removed
    ) -> None:
        truncated_normal_init(
            buffer,
            indices,
            self._curand_state,
            self._args.mean,
            self._args.std_dev,
            self._args.lower,
            self._args.upper,
        )


class DebugInitializer(BaseDynamicEmbInitializer):
    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],
    ) -> None:
        if keys is None:
            raise ValueError("DebugInitializer requires keys, but got None.")
        debug_init(buffer, indices, keys)
