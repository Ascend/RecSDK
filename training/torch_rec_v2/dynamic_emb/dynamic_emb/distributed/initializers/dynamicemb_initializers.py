#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import abc
from typing import Optional

import torch

from dynamic_emb.distributed.dynamicemb_config import DynamicEmbInitializerArgs


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
    ) -> None:
        ...


class NormalInitializer(BaseDynamicEmbInitializer):
    def __init__(self, args: DynamicEmbInitializerArgs):
        super().__init__(args)

    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],  # remove it when debug mode is removed
    ) -> None:
        init_shape = (len(indices),) + buffer.shape[1:]
        normal_tensor = torch.normal(
            mean=self._args.mean,
            std=self._args.std_dev,
            size=init_shape,
            device=buffer.device,
            dtype=buffer.dtype
        )

        buffer[indices] = normal_tensor


class ConstantInitializer(BaseDynamicEmbInitializer):
    def __init__(self, args: DynamicEmbInitializerArgs):
        super().__init__(args)

    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],  # remove it when debug mode is removed
    ) -> None:
        init_shape = (len(indices),) + buffer.shape[1:]
        constant_tensor = torch.full(
            fill_value=self._args.value,
            size=init_shape,
            device=buffer.device,
            dtype=buffer.dtype
        )
        buffer[indices] = constant_tensor


class UniformInitializer(BaseDynamicEmbInitializer):
    def __init__(self, args: DynamicEmbInitializerArgs):
        super().__init__(args)

    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],  # remove it when debug mode is removed
    ) -> None:
        init_shape = (len(indices),) + buffer.shape[1:]
        low, high = self._args.lower, self._args.upper
        uniform_tensor = (high - low) * torch.rand(init_shape, dtype=buffer.dtype, device=buffer.device) + low
        buffer[indices] = uniform_tensor