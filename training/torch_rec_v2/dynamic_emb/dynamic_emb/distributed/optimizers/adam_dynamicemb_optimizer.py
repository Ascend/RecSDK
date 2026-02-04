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

from typing import Any, Dict, Optional

import torch

from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import (
    BaseDynamicEmbeddingOptimizerV2,
    OptimizerArgs,
    get_required_arg,
)
from dynamic_emb_extensions import dynamic_emb_Adam_with_pointer, DynamicEmbDataType


class AdamDynamicEmbeddingOptimizerV2(BaseDynamicEmbeddingOptimizerV2):
    def __init__(
        self,
        opt_args: OptimizerArgs,
    ) -> None:
        super().__init__(opt_args)
        self._iterations: int = 0

    def step(self):
        self._iterations += 1

    def update(
        self,
        grads: torch.Tensor,
        embs: torch.Tensor,
        states: Optional[torch.Tensor],
    ) -> None:
        pass

    def fused_update(
        self,
        grads: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        pass

    def fused_update_with_pointer(
        self,
        grads: torch.Tensor,
        value_ptr: torch.Tensor,  # pointers to embeddng + optimizer states
        value_type: Optional[DynamicEmbDataType] = None,
    ) -> None:
        lr = self._opt_args.learning_rate
        beta1 = self._opt_args.beta1
        beta2 = self._opt_args.beta2
        weight_decay = self._opt_args.weight_decay
        eps = self._opt_args.eps

        emb_dim = grads.size(1)
        state_dim = self.get_state_dim(emb_dim)

        dynamic_emb_Adam_with_pointer(
            grads,
            value_ptr,
            value_type,
            state_dim,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            self._iterations,
        )

    def get_opt_args(self):
        ret_args = {
            "opt_type": "adam",
            "lr": self._opt_args.learning_rate,
            "iters": self._iterations,
            "beta1": self._opt_args.beta1,
            "beta2": self._opt_args.beta2,
            "eps": self._opt_args.eps,
            "weight_decay": self._opt_args.weight_decay,
        }
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")
        self._iterations = get_required_arg(args, "iters")
        self._opt_args.beta1 = get_required_arg(args, "beta1")
        self._opt_args.beta2 = get_required_arg(args, "beta2")
        self._opt_args.eps = get_required_arg(args, "eps")
        self._opt_args.weight_decay = get_required_arg(args, "weight_decay")
        return

    def get_state_dim(self, emb_dim: int) -> int:
        """
        Get the state dim.
        """
        return emb_dim * 2
