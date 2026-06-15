#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Functions in this file are usually used in the tests internally.
"""

# pylint: disable=unexpected-keyword-arg
# pylint: disable=redefined-outer-name
import logging
import copy
import sysconfig
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torchrec
from torch import nn
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    QCommsConfig,
    get_qcomm_codecs_registry,
)
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import BaseEmbeddingConfig
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType

from dynamic_emb.distributed.dynamicemb_config import DynamicEmbTableOptions
from dynamic_emb.distributed.dump_load import find_sharded_modules, get_dynamic_emb_module
from dynamic_emb.distributed.key_value_table import KeyValueTable
from dynamic_emb.distributed.planner.enumerators import DynamicEmbeddingEnumerator
from dynamic_emb.distributed.planner.planners import DynamicEmbeddingShardingPlanner
from dynamic_emb.distributed.planner.types import DynamicEmbParameterConstraints
from dynamic_emb.distributed.embedding import DynamicEmbeddingCollectionSharder


logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)

# 加载torchrec_npu执行所依赖的fbgemm_npu_api.so库
try:
    torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
except FileNotFoundError:
    logging.warning("libfbgemm_npu_api.so is not exist")
except Exception as e:
    logging.warning("libfbgemm_npu_api.so failed to load: %s", e)


def generate_sparse_feature(feature_names, num_embeddings_list, local_batch_size, lookup_iter=0):
    """
    Generate a KeyedJaggedTensor for one lookup iteration across all embedding tables.

    Parameters:
    - feature_names: List of N embedding table names
    - num_embeddings_list: List of N embedding table sizes
    - local_batch_size: Batch size per rank
    - lookup_iter: Current lookup iteration (0, 1, 2, ...)

    Returns:
    - A KeyedJaggedTensor for the current lookup iteration
    """
    indices = []
    lengths = []

    # Start index for this lookup iteration
    start_idx = lookup_iter * local_batch_size

    # For each embedding table
    for f_idx, table_size in enumerate(num_embeddings_list):
        # For each sample in local batch
        for b_idx in range(local_batch_size):
            # Calculate global index for this lookup
            global_idx = start_idx + b_idx

            # If index exceeds table size, use 1 as fallback
            if global_idx >= table_size:
                global_idx = 1

            indices.append(global_idx)
            lengths.append(1)  # Single lookup

    return (
        torchrec.KeyedJaggedTensor(
            keys=feature_names,
            values=torch.tensor(indices, dtype=torch.int64).npu(),
            lengths=torch.tensor(lengths, dtype=torch.int64).npu(),
        ),
        torch.tensor(indices, dtype=torch.int64).npu(),
    )


def get_dynamic_planner(
    eb_configs: List[BaseEmbeddingConfig],
    batch_size: int,
    device,
):
    dict_const = {}
    compute_kernel_type = ["fused"]
    for config in eb_configs:
        const = DynamicEmbParameterConstraints(
            sharding_types=[
                ShardingType.ROW_WISE.value,
            ],
            compute_kernels=compute_kernel_type,
            use_dynamicemb=True,
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=1024**3,
            ),
        )

        dict_const[config.name] = const

    topology = Topology(
        world_size=dist.get_world_size(),
        compute_device=device.type,
    )
    enumerator = DynamicEmbeddingEnumerator(
        topology=topology,
        constraints=dict_const,
    )

    return DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=topology,
        constraints=dict_const,
        batch_size=batch_size,
        enumerator=enumerator,
        debug=True,
    )


def get_torchrec_planner(
    eb_configs: List[BaseEmbeddingConfig],
    batch_size: int,
    device,
):
    dict_const = {}
    compute_kernel_type = ["fused"]
    for config in eb_configs:
        const = ParameterConstraints(
            sharding_types=[
                ShardingType.ROW_WISE.value,
            ],
            compute_kernels=compute_kernel_type,
        )
        dict_const[config.name] = const

    topology = Topology(
        world_size=dist.get_world_size(),
        compute_device=device.type,
    )

    return EmbeddingShardingPlanner(
        topology=topology,
        batch_size=batch_size,
        constraints=dict_const,
        debug=True,
    )


class ConstructTwinModule:
    def __init__(
        self,
        table_num: int,
        dims: List[int],
        num_embeddings: List[int],
        multi_hot_sizes: List[int],
        batch_size: int = 65536,
        output_dtype: SparseType = SparseType.FP32,
        fwd_a2a_precision: CommType = CommType.FP32,
        bwd_a2a_precision: CommType = CommType.FP32,
        optimizer_kwargs: Dict[str, Any] = None,
        init_fn: Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]] = None,
        rank: int = 0,
        world_size: int = 1,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()

        self._table_num = table_num
        self._dims = dims
        self._num_embeddings = num_embeddings

        self._batch_size = batch_size
        self._multi_hot_sizes = multi_hot_sizes
        self._output_dtype = output_dtype
        self._fwd_a2a_precision = fwd_a2a_precision
        self._bwd_a2a_precision = bwd_a2a_precision
        self._optimizer_kwargs = optimizer_kwargs
        self._rank = rank
        self._world_size = world_size
        self._scale_factor = scale_factor
        self._init_fn = init_fn

        self._table_names = [f"t_{t}" for t in range(self._table_num)]
        self._feature_names = [f"f_{t}" for t in range(self._table_num)]

        self._dynamicemb_model = None
        self._torchrec_model = None

    def _construct_dynamicemb_module(self) -> None:
        assert self._rank == torch.npu.current_device()

        configs = [
            torchrec.EmbeddingConfig(
                name=self._table_names[feature_idx],
                embedding_dim=self._dims[feature_idx],
                num_embeddings=self._num_embeddings[feature_idx] * self._scale_factor,
                feature_names=[self._feature_names[feature_idx]],
            )
            for feature_idx in range(self._table_num)
        ]
        collection = torchrec.EmbeddingCollection(
            device=torch.device("meta"),
            tables=configs,
        )

        optimizer_kwargs_copy = copy.deepcopy(self._optimizer_kwargs)
        fused_params = {"output_dtype": self._output_dtype}

        optimizer_type = optimizer_kwargs_copy.pop("optimizer")

        if optimizer_type == "sgd":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_SGD
        elif optimizer_type == "exact_sgd":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_SGD
        elif optimizer_type == "adam":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.ADAM
        elif optimizer_type == "exact_adagrad":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_ADAGRAD
        elif optimizer_type == "exact_row_wise_adagrad":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_ROWWISE_ADAGRAD
        else:
            raise ValueError(f"unknown optimizer type {optimizer_type} type = {type(optimizer_type)}")
        fused_params.update(optimizer_kwargs_copy)

        qcomm_codecs_registry = get_qcomm_codecs_registry(
            qcomms_config=QCommsConfig(
                forward_precision=self._fwd_a2a_precision,
                backward_precision=self._bwd_a2a_precision,
            )
        )

        sharder = DynamicEmbeddingCollectionSharder(
            qcomm_codecs_registry=qcomm_codecs_registry,
            fused_params=fused_params,
            use_index_dedup=False,
        )

        device = torch.device(f"npu:{self._rank}")
        planner = get_dynamic_planner(
            configs,
            self._batch_size,
            device,
        )
        plan = planner.collective_plan(collection, [sharder], dist.GroupMember.WORLD)
        self._dynamicemb_model = DistributedModelParallel(
            module=collection,
            device=device,
            sharders=[sharder],
            plan=plan,
        )

    def _construct_torchrec_module(self) -> None:
        assert self._rank == torch.npu.current_device()

        configs = [
            torchrec.EmbeddingConfig(
                name=self._table_names[feature_idx],
                embedding_dim=self._dims[feature_idx],
                num_embeddings=self._num_embeddings[feature_idx],
                feature_names=[self._feature_names[feature_idx]],
                init_fn=self._init_fn,
            )
            for feature_idx in range(self._table_num)
        ]
        collection = torchrec.EmbeddingCollection(
            device=torch.device("meta"),
            tables=configs,
        )

        optimizer_kwargs_copy = copy.deepcopy(self._optimizer_kwargs)
        fused_params = {"output_dtype": self._output_dtype}

        optimizer_type = optimizer_kwargs_copy.pop("optimizer")

        if optimizer_type == "sgd":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_SGD
        elif optimizer_type == "exact_sgd":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_SGD
        elif optimizer_type == "adam":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.ADAM
        elif optimizer_type == "exact_adagrad":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_ADAGRAD
        elif optimizer_type == "exact_row_wise_adagrad":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_ROWWISE_ADAGRAD
        else:
            raise ValueError(f"unknown optimizer type {optimizer_type} type = {type(optimizer_type)}")
        fused_params.update(optimizer_kwargs_copy)

        qcomm_codecs_registry = get_qcomm_codecs_registry(
            qcomms_config=QCommsConfig(
                forward_precision=self._fwd_a2a_precision,
                backward_precision=self._bwd_a2a_precision,
            )
        )

        sharder = EmbeddingCollectionSharder(
            qcomm_codecs_registry=qcomm_codecs_registry,
            fused_params=fused_params,
            use_index_dedup=False,
        )

        device = torch.device(f"npu:{self._rank}")
        planner = get_torchrec_planner(
            configs,
            self._batch_size,
            device,
        )
        plan = planner.collective_plan(collection, [sharder], dist.GroupMember.WORLD)
        self._torchrec_model = DistributedModelParallel(
            module=collection,
            device=device,
            sharders=[sharder],
            plan=plan,
        )

    def init_twin_embedding_model(self):
        self._construct_dynamicemb_module()
        self._construct_torchrec_module()
        max_table_size = max(self._num_embeddings)
        total_iterations = (max_table_size + self._batch_size - 1) // self._batch_size

        all_indices = {feature: [] for feature in self._feature_names}
        all_values = {feature: [] for feature in self._feature_names}

        # now it can do only one collection
        collections_list: List[Tuple[str, str, nn.Module]] = find_sharded_modules(self._dynamicemb_model, "")
        if not collections_list:
            raise RuntimeError("Failed to find any sharded modules in dynamicemb_model")
        _, _, tmp_collection_module = collections_list[0]

        tmp_dynamic_emb_module_list = get_dynamic_emb_module(tmp_collection_module)
        table_name_map_hkv_table = {}
        for dynamic_emb_module in tmp_dynamic_emb_module_list:
            tmp_table_names = dynamic_emb_module.table_names
            tmp_tables = dynamic_emb_module.tables

            for i, tmp_table_name in enumerate(tmp_table_names):
                table_name_map_hkv_table[tmp_table_name] = tmp_tables[i]

        # Perform all lookup iterations
        for iter_idx in range(total_iterations):
            # Generate sparse feature for current iteration
            sparse_feature, sparse_indices = generate_sparse_feature(
                self._feature_names, self._num_embeddings, self._batch_size, iter_idx
            )

            # Forward pass through model
            output = self._torchrec_model(sparse_feature)

            # Process output
            # For non-pooled case, output contains indices and values
            feature_idx = 0
            for feature, sparse_data in output.items():
                tmp_feature_name = self._feature_names[feature_idx]
                tmp_indices = sparse_indices[feature_idx * self._batch_size : (feature_idx + 1) * self._batch_size]
                all_indices[tmp_feature_name].append(tmp_indices)
                all_values[tmp_feature_name].append(sparse_data.values())
                feature_idx += 1

        final_results = {}
        for feature in self._feature_names:
            indices = torch.cat(all_indices[feature], dim=0)
            values = torch.cat(all_values[feature], dim=0)

            # Create a sparse tensor representation
            final_results[feature] = {"indices": indices, "values": values}

        for feature, result in final_results.items():
            indices = result["indices"]
            values = result["values"]
            # Use mask for filtering to ensure correct correspondence
            mask = indices % self._world_size == self._rank
            filtered_indices = indices[mask]

            # Get the dimension for this feature
            dim = self._dims[self._feature_names.index(feature)]

            # Reshape values to [num_indices, dim]
            values = values.reshape(-1, dim)
            # Select values based on the SAME mask as filtered_indices
            filtered_values = values[mask, :]

            # Remove duplicates from filtered indices and values
            unique_indices, unique_inverse_indices = torch.unique(filtered_indices, return_inverse=True)
            unique_values = filtered_values[unique_inverse_indices, :]

            tmp_table_name = feature.replace("f_", "t_")
            cur_hkv_table: KeyValueTable = table_name_map_hkv_table[tmp_table_name]
            optstate_dim = cur_hkv_table.optim_state_dim()
            initial_accumulator = cur_hkv_table.init_optimizer_state()
            optstate = (
                torch.ones(
                    unique_values.size(0),
                    optstate_dim,
                    dtype=unique_values.dtype,
                    device=unique_values.device,
                )
                * initial_accumulator
            )
            unique_values = torch.cat((unique_values, optstate), dim=1).contiguous()
            unique_values = unique_values.reshape(-1)

            cur_hkv_table.insert(unique_indices, unique_values)
        # In TorchREC, once a forward lookup occurs, the iteration in the module gets updated(even you don't do backward).
        # This makes it difficult to accurately test ADAM since the iteration count affects the optimizer's behavior.
        # So we reset the optimizer_step in here
        self._torchrec_model.fused_optimizer.set_optimizer_step(0)

    @property
    def dynamicemb_model(self):
        return self._dynamicemb_model

    @property
    def torchrec_model(self):
        return self._torchrec_model
