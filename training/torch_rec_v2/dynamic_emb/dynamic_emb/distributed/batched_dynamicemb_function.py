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

from typing import List, Optional
from dataclasses import dataclass
import torch
from dynamic_emb.distributed.initializers.dynamicemb_initializers import BaseDynamicEmbInitializer
from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import BaseDynamicEmbeddingOptimizerV2
from dynamic_emb.distributed.key_value_table import (
    KeyValueTableFunction,
)
from dynamic_emb_extensions import (
    get_table_range_op,
    segmented_unique_op,
    gather_embedding,
    reduce_grads,
)
from dynamic_emb.distributed.types import Cache, Storage


@dataclass
class DynamicEmbeddingFunctionV2Config:
    indices: torch.Tensor
    offsets: torch.Tensor
    caches: List[Optional[Cache]]
    storages: List[Storage]
    feature_offsets: torch.Tensor
    output_dtype: torch.dtype
    initializers: List[BaseDynamicEmbInitializer]
    optimizer: BaseDynamicEmbeddingOptimizerV2
    enable_prefetch: bool = False
    input_dist_dedup: bool = False
    training: bool = True


class DynamicEmbeddingFunctionV2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        config: DynamicEmbeddingFunctionV2Config,
        *args,
    ):
        indices = config.indices
        offsets = config.offsets
        caches = config.caches
        storages = config.storages
        feature_offsets = config.feature_offsets
        output_dtype = config.output_dtype
        initializers = config.initializers
        optimizer = config.optimizer
        enable_prefetch = config.enable_prefetch
        input_dist_dedup = config.input_dist_dedup
        training = config.training

        table_num = len(storages)
        if table_num == 0:
            raise ValueError("table_num must be greater than 0")
        emb_dtype = storages[0].embedding_dtype()
        emb_dim = storages[0].embedding_dim()
        caching = caches[0] is not None
        indices_table_range = get_table_range_op(offsets, feature_offsets)
        inverse = torch.tensor([], dtype=torch.int64, device=indices.device)
        if training or caching:
            (
                unique_indices,
                inverse,
                unique_indices_table_range,
                h_unique_indices_table_range,
            ) = segmented_unique_op(indices, indices_table_range)
            # h_unique_indices_table_range = unique_indices_table_range.cpu()
        else:
            h_unique_indices_table_range = indices_table_range.cpu()
            unique_indices = indices

        unique_embs = torch.empty(
            unique_indices.shape[0], emb_dim, dtype=emb_dtype, device=indices.device
        )

        for i in range(table_num):
            begin = h_unique_indices_table_range[i]
            end = h_unique_indices_table_range[i + 1]
            unique_indices_per_table = unique_indices[begin:end]
            unique_embs_per_table = unique_embs[begin:end, :]
            KeyValueTableFunction.lookup(
                storages[i],
                unique_indices_per_table,
                unique_embs_per_table,
                initializers[i],
                training,
            )

        if training or caching:
            output_embs = gather_embedding(unique_embs, inverse)
        else:
            output_embs = unique_embs

        if training:
            # save context
            backward_tensors = [
                indices,
            ]
            ctx.save_for_backward(*backward_tensors)
            ctx.input_dist_dedup = input_dist_dedup

            ctx.unique_indices = unique_indices
            ctx.unique_embs = unique_embs
            ctx.inverse = inverse
            ctx.indices_table_range = indices_table_range
            ctx.h_indices_table_range = indices_table_range.cpu()
            ctx.h_unique_indices_table_range = h_unique_indices_table_range
            ctx.unique_indices_table_range = unique_indices_table_range
            ctx.caches = caches
            ctx.storages = storages
            ctx.optimizer = optimizer
            ctx.enable_prefetch = enable_prefetch
        return output_embs

    @staticmethod
    def backward(ctx, grads):
        # parse context
        (indices,) = ctx.saved_tensors
        indices_table_range = ctx.indices_table_range
        h_indices_table_range = ctx.h_indices_table_range
        h_unique_indices_table_range = ctx.h_unique_indices_table_range
        caches = ctx.caches
        storages = ctx.storages
        optimizer = ctx.optimizer
        caching = caches[0] is not None

        unique_grads_indices, unique_grads = reduce_grads(grads, ctx.unique_indices, ctx.inverse)

        optimizer.step()
        table_num = len(storages)
        for i in range(table_num):
            begin = h_unique_indices_table_range[i]
            end = h_unique_indices_table_range[i + 1]
            unique_grads_indices_per_table = unique_grads_indices[begin:end]
            unique_grads_per_table = unique_grads[begin:end, :]

            KeyValueTableFunction.update(
                storages[i],
                unique_grads_indices_per_table,
                unique_grads_per_table,
                optimizer,
            )
        return (None,) * 2
