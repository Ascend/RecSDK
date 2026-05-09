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

from typing import Any, Dict, Optional

import torch

from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import (
    BaseDynamicEmbeddingOptimizerV2,
    OptimizerArgs,
    get_required_arg,
)
from dynamic_emb_extensions import dynamic_emb_adagrad_fused, dynamic_emb_adagrad_with_pointer, DynamicEmbDataType


class AdagradDynamicEmbeddingOptimizerV2(BaseDynamicEmbeddingOptimizerV2):
    def __init__(
        self,
        opt_args: OptimizerArgs,
    ) -> None:
        super().__init__(opt_args)

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
        lr = self._opt_args.learning_rate
        eps = self._opt_args.eps

        dynamic_emb_adagrad_fused(
            grads,
            values,
            lr,
            eps,
        )

    def fused_update_with_pointer(
        self,
        grads: torch.Tensor,
        value_ptr: torch.Tensor,  # pointers to embeddng + optimizer states
        value_type: Optional[DynamicEmbDataType] = None,
    ) -> None:
        lr = self._opt_args.learning_rate
        eps = self._opt_args.eps

        emb_dim = grads.size(1)
        state_dim = self.get_state_dim(emb_dim)

        dynamic_emb_adagrad_with_pointer(
            grads,
            value_ptr,
            value_type,
            state_dim,
            lr,
            eps,
        )

    def get_opt_args(self):
        ret_args = {
            "opt_type": "exact_adagrad",
            "lr": self._opt_args.learning_rate,
            "eps": self._opt_args.eps,
            "initial_accumulator_value": self._opt_args.initial_accumulator_value,
        }
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")
        self._opt_args.eps = get_required_arg(args, "eps")
        self._opt_args.initial_accumulator_value = get_required_arg(args, "initial_accumulator_value")
        return

    def get_state_dim(self, emb_dim: int) -> int:
        """
        Get the state dim.
        """
        return emb_dim
