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
from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import (
    BaseDynamicEmbeddingOptimizer,
    BaseDynamicEmbeddingOptimizerV2,
)
from dynamic_emb.distributed.key_value_table import (
    KeyValueTableCachingFunction,
    KeyValueTableFunction,
)
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbInitializerArgs,
    DynamicEmbPoolingMode,
    dyn_emb_to_torch,
)
from dynamic_emb.distributed.types import Cache, Storage
from dynamic_emb_extensions import (
    DynamicEmbTable,
    find_and_initialize,
    find_or_insert,
    get_table_range_op,
    segmented_unique_op,
    gather_embedding,
    reduce_grads,
    lookup_forward,
    lookup_backward,
)


class DynamicEmbeddingBagFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        offsets: torch.Tensor,  # [feature_num * batch_size]
        use_index_dedup: bool,
        table_offsets_in_feature: List[int],
        tables: List[DynamicEmbTable],
        scores: List[int],
        total_D: int,
        dims: List[int],
        feature_table_map: List[int],
        embedding_dtype: torch.dtype,
        output_dtype: torch.dtype,
        pooling_mode: DynamicEmbPoolingMode,
        unique_op,
        device: torch.device,
        optimizer: BaseDynamicEmbeddingOptimizer,
        training: bool,
        eval_initializers: List[DynamicEmbInitializerArgs],
        *args,
    ):
        table_num = len(tables)
        if table_num != len(table_offsets_in_feature) - 1:
            raise ValueError(
                f"table_num ({table_num}) must be equal to len(table_offsets_in_feature) - 1 ({len(table_offsets_in_feature) - 1})."
            )

        # split indices, offsets by table.
        indices_list: List[torch.Tensor] = []
        biased_offsets_list: List[torch.Tensor] = []

        feature_num = table_offsets_in_feature[-1]
        if feature_num != len(feature_table_map):
            raise ValueError(
                f"feature_num ({feature_num}) must be equal to len(feature_table_map) ({len(feature_table_map)})."
            )
        feature_batch_size = offsets.shape[0] - 1
        batch_size = feature_batch_size // feature_num
        if feature_batch_size % feature_num != 0:
            raise ValueError(
                f"feature_batch_size ({feature_batch_size}) must be divisible by feature_num ({feature_num})."
            )
        # The offsets is on device in torchrec, however, the unique_op and lookup_op are done table by table.
        # So we need to know one index belong to which table, to let op know the boundary.
        # Therefore, copy offsets to cpu is necessary, otherwise, many things will be coupled together.
        # For example, UniqueOp have to accept (indices, offsets, table_offsets_in_feature, table_id) as inputs,
        #   and we have to copy table_offsets_in_feature from cpu to gpu.

        h_offsets = offsets.to("cpu")

        for i in range(table_num):
            feature_id_begin, feature_id_end = (
                table_offsets_in_feature[i],
                table_offsets_in_feature[i + 1],
            )
            offset_begin, offset_end = (
                feature_id_begin * batch_size,
                feature_id_end * batch_size,
            )
            # include offset_end to know the boundary of the last feature.
            biased_offsets_list.append(offsets[offset_begin : offset_end + 1])

            indices_begin, indices_end = h_offsets[offset_begin], h_offsets[offset_end]
            indices_list.append(indices[indices_begin:indices_end])

        unique_indices_list = []
        inverse_indices_list = []
        unique_count_list = []
        for i in range(table_num):
            unique_indices, inverse_indices = torch.unique(indices_list[i], sorted=False, return_inverse=True)
            unique_indices_list.append(unique_indices)
            inverse_indices_list.append(inverse_indices.to(biased_offsets_list[i].dtype))
            unique_count_list.append(inverse_indices.shape[0])

        unique_embedding_list = []
        for i in range(table_num):
            unique_indices = unique_indices_list[i]
            num_unique_indices = unique_indices.shape[0]
            tmp_value_type_torch = dyn_emb_to_torch(tables[i].get_value_type())
            tmp_unique_embs = torch.empty(num_unique_indices, dims[i], dtype=tmp_value_type_torch, device=device)

            if training:
                find_or_insert(
                    tables[i],
                    num_unique_indices,
                    unique_indices,
                    tmp_unique_embs,
                    scores[i],
                )
            else:
                value_ptrs = torch.empty(num_unique_indices, dtype=torch.int64, device=device)
                founds = torch.empty(num_unique_indices, dtype=torch.bool, device=device)
                find_and_initialize(
                    tables[i],
                    num_unique_indices,
                    unique_indices,
                    value_ptrs,
                    tmp_unique_embs,
                    founds,
                    eval_initializers[i].as_ctype(),
                )

            unique_embedding_list.append(tmp_unique_embs)

        if pooling_mode == DynamicEmbPoolingMode.NONE:
            combiner = -1
            total_embs_num = indices.numel()
            # All tables have the same dim.
            embs = torch.empty(total_embs_num, dims[0], dtype=output_dtype, device=device)
        else:
            if pooling_mode == DynamicEmbPoolingMode.SUM:
                combiner = 0
            elif pooling_mode == DynamicEmbPoolingMode.MEAN:
                combiner = 1
            else:
                raise ValueError("Not support pooling mode.")
            total_embs_num = offsets.shape[0] - 1
            embs = torch.empty(batch_size, total_D, dtype=output_dtype, device=device)

        accum_D = 0
        for i in range(table_num):
            num_embeddings = biased_offsets_list[i].shape[0] - 1
            lookup_forward(
                unique_embedding_list[i],
                embs,
                biased_offsets_list[i],
                inverse_indices_list[i],
                combiner,
                total_D,
                accum_D,
                dims[i],
                num_embeddings,
                batch_size,
            )
            accum_D += dims[i] * (num_embeddings // batch_size)
            if num_embeddings % batch_size != 0:
                raise ValueError(f"num_embeddings ({num_embeddings}) must be divisible by batch_size ({batch_size}).")

        if training:
            backward_tensors = [indices, offsets]
            ctx.save_for_backward(*backward_tensors)
            ctx.tables = tables
            ctx.unique_indices_list = unique_indices_list
            ctx.inverse_indices_list = inverse_indices_list
            ctx.biased_offsets_list = biased_offsets_list
            ctx.dims = dims
            ctx.batch_size = batch_size
            ctx.feature_table_map = feature_table_map
            ctx.device = device
            ctx.optimizer = optimizer
            ctx.scores = scores
            ctx.combiner = combiner

        return embs

    @staticmethod
    def backward(ctx, grad):
        # if we want to do the value check, we shouldn't to update the embeddings.
        tables = ctx.tables
        unique_indices_list = ctx.unique_indices_list
        inverse_indices_list = ctx.inverse_indices_list
        biased_offsets_list = ctx.biased_offsets_list
        dims = ctx.dims
        batch_size = ctx.batch_size
        feature_table_map_list = ctx.feature_table_map
        indices, offsets = ctx.saved_tensors
        device = ctx.device
        optimizer = ctx.optimizer
        table_num = len(tables)
        combiner = ctx.combiner

        offsets_list_per_table = []
        for i in range(table_num):
            offsets_list_per_table.append(biased_offsets_list[i] - biased_offsets_list[i][0])

        feature_num_per_table = [0] * table_num
        for i in range(len(feature_table_map_list)):
            feature_num_per_table[feature_table_map_list[i]] += 1

        dim_offset_per_table = [0]
        for i in range(table_num):
            dim_offset_per_table.append(feature_num_per_table[i] * dims[i] + dim_offset_per_table[i])

        dyn_emb_to_torch(tables[0].get_value_type())
        dyn_emb_to_torch(tables[0].get_key_type())

        unique_count_list = []
        for i in range(table_num):
            unique_count_list.append(unique_indices_list[i].shape[0])

        unique_backward_grads_per_table = []
        for i in range(table_num):
            unique_backward_grads_per_table.append(
                torch.zeros(unique_count_list[i] * dims[i], dtype=grad.dtype, device=device)
            )

        for i in range(table_num):
            grad_for_table = grad[:, dim_offset_per_table[i] : dim_offset_per_table[i + 1]]

            splits = torch.split(grad_for_table, dims[i], dim=-1)
            result = torch.cat(splits, dim=0)
            grad_for_table = result.reshape(-1, dims[i]).contiguous()
            lookup_backward(
                grad_for_table,
                unique_backward_grads_per_table[i],
                unique_indices_list[i],
                inverse_indices_list[i],
                offsets_list_per_table[i],
                dims[i],
                table_num,
                batch_size,
                feature_num_per_table[i],
                offsets_list_per_table[i][-1].item(),
                combiner,
            )

        unique_grads_per_table = []
        for i, unique_grad in enumerate(unique_backward_grads_per_table):
            unique_grads_per_table.append(unique_grad.reshape(-1, dims[i]))

        optimizer.update(tables, unique_indices_list, unique_grads_per_table)

        return (None,) * 18


def dynamicemb_prefetch(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    caches: List[Optional[Cache]],
    storages: List[Storage],
    feature_offsets: torch.Tensor,
    initializers: List[BaseDynamicEmbInitializer],
    unique_op=None,
    training: bool = True,
    forward_stream: Optional[torch.npu.Stream] = None,
):
    table_num = len(storages)
    if table_num == 0:
        raise ValueError("table_num must be greater than 0.")
    caching = caches[0] is not None

    indices_table_range = get_table_range_op(offsets, feature_offsets)
    if training or caching:
        (
            unique_indices,
            inverse,
            unique_indices_table_range,
            h_unique_indices_table_range,
        ) = segmented_unique_op(indices, indices_table_range)
    else:
        h_unique_indices_table_range = indices_table_range.cpu()
        unique_indices = indices

    for i in range(table_num):
        begin = h_unique_indices_table_range[i]
        end = h_unique_indices_table_range[i + 1]
        unique_indices_per_table = unique_indices[begin:end]

        KeyValueTableCachingFunction.prefetch(
            caches[i],
            storages[i],
            unique_indices_per_table,
            initializers[i],
            training,
            forward_stream,
        )


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
        else:
            h_unique_indices_table_range = indices_table_range.cpu()
            unique_indices = indices

        unique_embs = torch.empty(unique_indices.shape[0], emb_dim, dtype=emb_dtype, device=indices.device)

        for i in range(table_num):
            begin = h_unique_indices_table_range[i]
            end = h_unique_indices_table_range[i + 1]
            unique_indices_per_table = unique_indices[begin:end]
            unique_embs_per_table = unique_embs[begin:end, :]

            if caching:
                KeyValueTableCachingFunction.lookup(
                    caches[i],
                    storages[i],
                    unique_indices_per_table,
                    unique_embs_per_table,
                    initializers[i],
                    enable_prefetch,
                    training,
                )
            else:
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

        if output_embs.dtype != output_dtype:
            output_embs = output_embs.to(dtype=output_dtype)

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
            ctx.h_unique_indices_table_range = h_unique_indices_table_range
            ctx.caches = caches
            ctx.storages = storages
            ctx.optimizer = optimizer
            ctx.enable_prefetch = enable_prefetch
        return output_embs

    @staticmethod
    def backward(ctx, grads):
        # parse context
        (indices,) = ctx.saved_tensors
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

            if caching:
                KeyValueTableCachingFunction.update(
                    caches[i],
                    storages[i],
                    unique_grads_indices_per_table,
                    unique_grads_per_table,
                    optimizer,
                )
            else:
                KeyValueTableFunction.update(
                    storages[i],
                    unique_grads_indices_per_table,
                    unique_grads_per_table,
                    optimizer,
                )

        return (None,) * 2
