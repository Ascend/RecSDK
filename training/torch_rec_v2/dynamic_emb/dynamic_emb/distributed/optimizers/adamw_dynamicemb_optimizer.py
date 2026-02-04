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

import torch

from dynamic_emb.distributed.optimizers.adam_dynamicemb_optimizer import AdamDynamicEmbeddingOptimizerV2
from dynamic_emb_extensions import dynamic_emb_AdamW_with_pointer, DynamicEmbDataType


class AdamWDynamicEmbeddingOptimizerV2(AdamDynamicEmbeddingOptimizerV2):
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

        dynamic_emb_AdamW_with_pointer(
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
