#!/usr/bin/env python3
# pylint: disable=duplicate-code
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

from typing import Any, Dict, List, Optional

import torch

from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import (
    BaseDynamicEmbeddingOptimizer,
    BaseDynamicEmbeddingOptimizerV2,
    get_required_arg,
)
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbTable,
    torch_to_dyn_emb,
)
from dynamic_emb_extensions import (
    dynamic_emb_sgd_with_table,
    dynamic_emb_sgd_fused,
    dynamic_emb_sgd_with_pointer,
    dynamic_emb_sgd_with_pointer_hybrid,
    dynamic_emb_sgd_fused_hybrid,
    DynamicEmbDataType,
)


class SGDDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def update(
        self,
        hashtables: List[DynamicEmbTable],
        indices: List[torch.Tensor],
        grads: List[torch.Tensor],
    ) -> None:
        for ht in hashtables:
            if ht not in self._hashtables:
                raise ValueError(
                    f"DynamicEmb ERROR: Hashtable {ht} not found in hashtables in class {self.__class__.__name__}."
                )

        lr = self._opt_args.learning_rate
        for i, ht in enumerate(hashtables):
            state_idx = self._table_state_map[ht]
            table_option = self._table_options[state_idx]

            grad = grads[i]
            indice = indices[i]
            num_indice = indice.shape[0]
            weight_dtype = torch_to_dyn_emb(table_option.embedding_dtype)

            dynamic_emb_sgd_with_table(
                ht,
                num_indice,
                indice,
                grad,
                lr,
                weight_dtype,
            )

    def get_opt_args(self):
        ret_args = {
            "lr": self._opt_args.learning_rate,
            "opt_type": "exact_sgd",
        }
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")


class SGDDynamicEmbeddingOptimizerV2(BaseDynamicEmbeddingOptimizerV2):
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
        dynamic_emb_sgd_fused(
            grads,
            values,
            lr,
        )

    def fused_update_hybrid(
        self,
        grads: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        lr = self._opt_args.learning_rate
        dynamic_emb_sgd_fused_hybrid(
            grads,
            values,
            lr,
        )

    def fused_update_with_pointer(
        self,
        grads: torch.Tensor,
        value_ptr: torch.Tensor,  # pointers to embeddng + optimizer states
        value_type: Optional[DynamicEmbDataType] = None,
    ) -> None:
        lr = self._opt_args.learning_rate
        dynamic_emb_sgd_with_pointer(
            grads,
            value_ptr,
            value_type,
            lr,
        )

    def fused_update_with_pointer_hybrid(
        self,
        grads: torch.Tensor,
        value_ptr: torch.Tensor,  # pointers to embeddng + optimizer states
        value_type: Optional[DynamicEmbDataType] = None,
    ) -> None:
        lr = self._opt_args.learning_rate
        dynamic_emb_sgd_with_pointer_hybrid(
            grads,
            value_ptr,
            value_type,
            lr,
        )

    def get_opt_args(self):
        ret_args = {
            "opt_type": "sgd",
            "lr": self._opt_args.learning_rate,
        }
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")

    def get_state_dim(self, emb_dim: int) -> int:
        """
        Get the state dim.
        """
        return 0
