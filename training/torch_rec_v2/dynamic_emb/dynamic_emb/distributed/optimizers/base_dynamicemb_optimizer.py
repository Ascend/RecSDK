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

import abc
import copy
import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from dynamic_emb_extensions import OptimizerType, DynamicEmbDataType


@enum.unique
class EmbOptimType(enum.Enum):
    ADAM = "adam"
    ADAMW = "adamW"
    NONE = "none"

    def __str__(self) -> str:
        return self.value


def string_to_opt_type(optimizer_str: str) -> EmbOptimType:
    try:
        return EmbOptimType(optimizer_str)
    except ValueError as e:
        raise ValueError(f"'{optimizer_str}' is not a valid EmbOptimType.") from e


def convert_optimizer_type(optimizer: EmbOptimType) -> OptimizerType:
    if optimizer == EmbOptimType.ADAM:
        return OptimizerType.Adam
    elif optimizer == EmbOptimType.ADAMW:
        return OptimizerType.AdamW
    else:
        raise ValueError(
            f"Not supported optimizer type, optimizer type = {optimizer} {type(optimizer)} {optimizer.value}."
        )


@dataclass
class OptimizerArgs:
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    eps: float = 1e-8
    initial_accumulator_value: float = 0.0
    momentum: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    other: dict = field(default_factory=dict)


def get_required_arg(args: Dict[str, Any], key: str) -> Any:
    if key not in args:
        raise ValueError(f"Input args does not contain required optimizer argument: {key}")
    return args[key]


class BaseDynamicEmbeddingOptimizerV2(abc.ABC):
    def __init__(
        self,
        opt_args: OptimizerArgs,
    ) -> None:
        self._opt_args: OptimizerArgs = copy.deepcopy(opt_args)

    @abc.abstractmethod
    def update(
        self,
        grads: torch.Tensor,
        embs: torch.Tensor,
        states: Optional[torch.Tensor],
    ) -> None:
        ...

    @abc.abstractmethod
    def fused_update(
        self,
        grads: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        ...

    @abc.abstractmethod
    def fused_update_with_pointer(
        self,
        grads: torch.Tensor,
        value_ptr: torch.Tensor,  # pointers to embeddng + optimizer states
        value_type: Optional[DynamicEmbDataType] = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def get_opt_args(self) -> Dict[str, Any]:
        ...

    @abc.abstractmethod
    def set_opt_args(self, args: Dict[str, Any]) -> None:
        ...

    @abc.abstractmethod
    def get_state_dim(self, emb_dim: int) -> int:
        """
        Get the state dim.
        """

    def set_learning_rate(self, new_lr: float) -> None:
        self._opt_args.learning_rate = new_lr

    def get_initial_optim_states(self) -> float:
        return self._opt_args.initial_accumulator_value

    def set_initial_optim_states(self, value: float) -> None:
        self._opt_args.initial_accumulator_value = value

    def step(self) -> None:
        pass
