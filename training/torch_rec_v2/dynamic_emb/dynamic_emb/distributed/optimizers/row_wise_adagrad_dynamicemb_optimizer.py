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

from typing import Any, Dict, List, Optional

import torch

from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import (
    BaseDynamicEmbeddingOptimizer,
    BaseDynamicEmbeddingOptimizerV2,
    OptimizerArgs,
    get_required_arg,
)
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbTableOptions,
    DynamicEmbTable,
    torch_to_dyn_emb,
)
from dynamic_emb_extensions import (
    dynamic_emb_rowwise_adagrad_fused,
    dynamic_emb_rowwise_adagrad_with_pointer,
    dynamic_emb_rowwise_adagrad_with_table,
    DynamicEmbDataType,
)


class RowWiseAdagradDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def __init__(
        self,
        opt_args: OptimizerArgs,
        table_options: List[DynamicEmbTableOptions],
        hashtables: List[DynamicEmbTable],
    ) -> None:
        super().__init__(opt_args, table_options, hashtables)

        self._state_dict["Gt"] = hashtables

        for table in hashtables:
            table.set_initial_optstate(self._opt_args.initial_accumulator_value)

    def update(
        self,
        hashtables: List[DynamicEmbTable],
        indices: List[torch.Tensor],
        grads: List[torch.Tensor],
    ) -> None:
        for ht in hashtables:
            if ht not in self._table_state_map.keys():
                raise ValueError(
                    f"DynamicEmb ERROR: Hashtable {ht} not found in _table_state_map in class {self.__class__.__name__}."
                )
        lr = self._opt_args.learning_rate
        eps = self._opt_args.eps
        for i, ht in enumerate(hashtables):
            state_idx = self._table_state_map[ht]
            table_option = self._table_options[state_idx]

            indice = indices[i]
            grad = grads[i]
            num_indice = indice.shape[0]

            weight_dtype = torch_to_dyn_emb(table_option.embedding_dtype)

            dynamic_emb_rowwise_adagrad_with_table(
                ht, num_indice, indice, grad, lr, eps, weight_dtype
            )

    def get_opt_args(self):
        ret_args = {
            "opt_type": "exact_row_wise_adagrad",
            "lr": self._opt_args.learning_rate,
            "eps": self._opt_args.eps,
            "initial_accumulator_value": self._opt_args.initial_accumulator_value,
        }
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")
        self._opt_args.eps = get_required_arg(args, "eps")
        initial_value = get_required_arg(args, "initial_accumulator_value")
        self._opt_args.initial_accumulator_value = initial_value
        for table in self._state_dict["Gt"]:
            table.set_initial_optstate(initial_value)
        return


class RowWiseAdagradDynamicEmbeddingOptimizerV2(BaseDynamicEmbeddingOptimizerV2):
    def __init__(
        self,
        opt_args: OptimizerArgs,
        emb_dtype: torch.dtype,
    ) -> None:
        super().__init__(opt_args)
        
        DTYPE_NUM_BYTES: Dict[torch.dtype, int] = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }
        self._optim_state_dim = 16 // DTYPE_NUM_BYTES[emb_dtype]

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

        dynamic_emb_rowwise_adagrad_fused(
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

        dynamic_emb_rowwise_adagrad_with_pointer(
            grads,
            value_ptr,
            value_type,
            state_dim,
            lr,
            eps,
        )

    def get_opt_args(self):
        ret_args = {
            "opt_type": "exact_row_wise_adagrad",
            "lr": self._opt_args.learning_rate,
            "eps": self._opt_args.eps,
            "initial_accumulator_value": self._opt_args.initial_accumulator_value,
        }
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")
        self._opt_args.eps = get_required_arg(args, "eps")
        initial_value = get_required_arg(args, "initial_accumulator_value")
        self._opt_args.initial_accumulator_value = initial_value
        return

    def get_state_dim(self, emb_dim: int) -> int:
        """
        Get the state dim.
        """
        return self._optim_state_dim
