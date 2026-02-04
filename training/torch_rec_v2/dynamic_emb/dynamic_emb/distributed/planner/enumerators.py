#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.constants import POOLING_FACTOR
from torchrec.distributed.planner.enumerators import (
    EmbeddingEnumerator,
    get_partition_by_type,
    _get_tower_index,
)
from torchrec.distributed.planner.types import (
    Shard,
    ShardEstimator,
    ShardingOption,
    Topology,
)
from torchrec.distributed.planner.utils import sharder_name
from torchrec.distributed.sharding_plan import calculate_shard_sizes_and_offsets
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheParams,
    KeyValueParams,
    ModuleSharder,
    ShardingType,
)
from torchrec.modules.embedding_configs import DataType
from torchrec.modules.embedding_tower import EmbeddingTower, EmbeddingTowerCollection
from torchrec.tensor_types import check

from dynamic_emb.distributed.planner.types import DynamicEmbParameterConstraints
from dynamic_emb.distributed.dynamicemb_config import (
    next_power_of_2,
    MIN_BATCH_SIZE,
    MAX_BATCH_SIZE,
    DEFAULT_BATCH_SIZE,
)
from rec_sdk_common.validator.safe_checker import class_safe_check, int_safe_check


@dataclass
class ExtractConstraintsParam:
    use_dynamicemb: bool = True
    input_lengths: List[float] = field(default_factory=lambda: [POOLING_FACTOR])
    col_wise_shard_dim: Optional[int] = None
    cache_params: Optional[CacheParams] = None
    enforce_hbm: Optional[bool] = None
    stochastic_rounding: Optional[bool] = None
    bounds_check_mode: Optional[BoundsCheckMode] = None
    feature_names: Optional[List[str]] = None
    output_dtype: Optional[DataType] = None
    device_group: Optional[str] = None
    key_value_params: Optional[KeyValueParams] = None


@dataclass
class CalculateShardParam:
    tensor: torch.Tensor
    world_size: int
    local_world_size: int
    sharding_type: str
    use_dynamicemb: bool
    col_wise_shard_dim: Optional[int] = None
    device_memory_sizes: Optional[List[int]] = None


class DynamicEmbeddingEnumerator(EmbeddingEnumerator):
    def __init__(
        self,
        topology: Topology,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
        constraints: Optional[Dict[str, DynamicEmbParameterConstraints]] = None,
        estimator: Optional[Union[ShardEstimator, List[ShardEstimator]]] = None,
        use_exact_enumerate_order: Optional[bool] = False,
    ) -> None:
        """
        DynamicEmbeddingEnumerator extends the EmbeddingEnumerator to handle dynamic embedding tables.

        Args:
            topology: Topology
                The topology of the NPU and Host memory.
            batch_size: Optional[int], optional
                The batch size for training. Defaults to BATCH_SIZE.
                The creation and usage are consistent with the same types in TorchREC.
            constraints: Optional[Dict[str, DynamicEmbParameterConstraints]], optional
                A dictionary of constraints for the parameters. Defaults to None.
            estimator: Optional[Union[ShardEstimator, List[ShardEstimator]]], optional
                An estimator or a list of estimators for estimating shard sizes. Defaults to None.
                The creation and usage are consistent with the same types in TorchREC.
            use_exact_enumerate_order: Optional[bool], optional
                Whether to enumerate shardable parameters in the exact name_children enumeration order.
        """
        class_safe_check("topology", topology, (Topology,))
        int_safe_check("batch_size", batch_size, min_value=MIN_BATCH_SIZE, max_value=MAX_BATCH_SIZE)
        class_safe_check("constraints", constraints, (dict, type(None)))
        if constraints is not None:
            for k, v in constraints.items():
                class_safe_check("key of constraints", k, (str,))
                class_safe_check("value of constraints", v, (DynamicEmbParameterConstraints,))
        check(estimator is None, "estimator should be None")
        check(not use_exact_enumerate_order, "use_exact_enumerate_order should be False")

        super().__init__(topology, batch_size, constraints, estimator, use_exact_enumerate_order)
        self._constraints = constraints

    def enumerate(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> List[ShardingOption]:
        """
        Generates relevant sharding options given module and sharders.

        Args:
            module (nn.Module): module to be sharded.
            sharders (List[ModuleSharder[nn.Module]]): provided sharders for module.

        Returns:
            List[ShardingOption]: valid sharding options with values populated.
        """

        def _process_sharding_options():
            for sharding_type in self._filter_sharding_types(
                name, sharder.sharding_types(self._compute_device), constraints_param.use_dynamicemb
            ):
                for compute_kernel in self._filter_compute_kernels(
                    name,
                    sharder.compute_kernels(sharding_type, self._compute_device),
                    sharding_type,
                    constraints_param.use_dynamicemb,
                ):
                    calculate_shard_param = CalculateShardParam(
                        tensor=param,
                        world_size=self._world_size,
                        local_world_size=self._local_world_size,
                        sharding_type=sharding_type,
                        use_dynamicemb=constraints_param.use_dynamicemb,
                        col_wise_shard_dim=constraints_param.col_wise_shard_dim,
                        device_memory_sizes=self._device_memory_sizes,
                    )
                    shard_sizes, shard_offsets = _calculate_shard_sizes_and_offsets(calculate_shard_param)
                    dependency = None
                    if isinstance(child_module, EmbeddingTower):
                        dependency = child_path
                    elif isinstance(child_module, EmbeddingTowerCollection):
                        tower_index = _get_tower_index(name, child_module)
                        dependency = child_path + ".tower_" + str(tower_index)
                    sharding_options_per_table.append(
                        ShardingOption(
                            name=name,
                            tensor=param,
                            module=(child_path, child_module),
                            input_lengths=constraints_param.input_lengths,
                            batch_size=self._batch_size,
                            compute_kernel=compute_kernel,
                            sharding_type=sharding_type,
                            partition_by=get_partition_by_type(sharding_type),
                            shards=[
                                Shard(size=size, offset=offset) for size, offset in zip(shard_sizes, shard_offsets)
                            ],
                            cache_params=constraints_param.cache_params,
                            enforce_hbm=constraints_param.enforce_hbm,
                            stochastic_rounding=constraints_param.stochastic_rounding,
                            bounds_check_mode=constraints_param.bounds_check_mode,
                            dependency=dependency,
                            is_pooled=is_pooled,
                            feature_names=constraints_param.feature_names,
                            output_dtype=constraints_param.output_dtype,
                            key_value_params=constraints_param.key_value_params,
                        )
                    )

        self._sharder_map = {sharder_name(sharder.module_type): sharder for sharder in sharders}
        sharding_options: List[ShardingOption] = []

        named_modules_queue = [("", module)]
        while named_modules_queue:
            if not self._use_exact_enumerate_order:
                child_path, child_module = named_modules_queue.pop()
            else:
                child_path, child_module = named_modules_queue.pop(0)
            sharder_key = sharder_name(type(child_module))
            sharder = self._sharder_map.get(sharder_key, None)
            if not sharder:
                for n, m in child_module.named_children():
                    if child_path != "":
                        named_modules_queue.append((child_path + "." + n, m))
                    else:
                        named_modules_queue.append((n, m))
                continue

            is_pooled = ShardingOption.module_pooled(child_module, child_path)

            for name, param in sharder.shardable_parameters(child_module).items():
                constraints_param = _extract_constraints_for_param(self._constraints, name)

                # skip for other device groups
                if constraints_param.device_group and constraints_param.device_group != self._compute_device:
                    continue

                sharding_options_per_table: List[ShardingOption] = []
                _process_sharding_options()
                if not sharding_options_per_table:
                    raise RuntimeError(
                        f"No available sharding type and compute kernel combination after applying user provided "
                        f"constraints for {name}. Module: {sharder_key}, sharder: {sharder.__class__.__name__}, "
                        f"compute device: {self._compute_device}. To debug, search above for warning logs about "
                        f"no available sharding types/compute kernels for table: {name}"
                    )

                sharding_options.extend(sharding_options_per_table)

        self.populate_estimates(sharding_options)

        return sharding_options

    def _filter_sharding_types(self, name: str, allowed_sharding_types: List[str], use_dynamicemb: bool) -> List[str]:
        if use_dynamicemb:
            return [ShardingType.ROW_WISE.value]

        return super()._filter_sharding_types(name, allowed_sharding_types)

    def _filter_compute_kernels(
        self,
        name: str,
        allowed_compute_kernels: List[str],
        sharding_type: str,
        use_dynamicemb: bool,
    ) -> List[str]:
        # Unlike the definition in planners, the "customized_kernel" here is not within the internal enumeration scope
        # and should be defined as "fused".
        if use_dynamicemb:
            return [EmbeddingComputeKernel.FUSED.value]

        return super()._filter_compute_kernels(name, allowed_compute_kernels, sharding_type)


def _extract_constraints_for_param(
    constraints: Optional[Dict[str, DynamicEmbParameterConstraints]],
    name: str,
) -> ExtractConstraintsParam:
    extract_constraints_param = ExtractConstraintsParam()

    if constraints and constraints.get(name):
        extract_constraints_param.use_dynamicemb = constraints[name].use_dynamicemb
        extract_constraints_param.input_lengths = constraints[name].pooling_factors
        extract_constraints_param.col_wise_shard_dim = constraints[name].min_partition
        extract_constraints_param.cache_params = constraints[name].cache_params
        extract_constraints_param.enforce_hbm = constraints[name].enforce_hbm
        extract_constraints_param.stochastic_rounding = constraints[name].stochastic_rounding
        extract_constraints_param.bounds_check_mode = constraints[name].bounds_check_mode
        extract_constraints_param.feature_names = constraints[name].feature_names
        extract_constraints_param.output_dtype = constraints[name].output_dtype
        extract_constraints_param.device_group = constraints[name].device_group
        extract_constraints_param.key_value_params = constraints[name].key_value_params

    return extract_constraints_param


def _calculate_shard_sizes_and_offsets(
    calculate_shard_param: CalculateShardParam,
) -> Tuple[List[List[int]], List[List[int]]]:
    # The bucket size of the underlying HKV table must be aligned to a power of 2.
    if calculate_shard_param.use_dynamicemb:
        (rows, columns) = calculate_shard_param.tensor.shape
        num_aligned_embedding_per_rank = int(next_power_of_2(math.ceil(rows / calculate_shard_param.world_size)))
        embedding_dim = columns
        sizes = [[num_aligned_embedding_per_rank, embedding_dim]] * calculate_shard_param.world_size
        # A value of 0 indicates that column-wise sharding is not yet supported.
        offsets = [[num_aligned_embedding_per_rank * i, 0] for i in range(calculate_shard_param.world_size)]
        return sizes, offsets

    return calculate_shard_sizes_and_offsets(
        calculate_shard_param.tensor,
        calculate_shard_param.world_size,
        calculate_shard_param.local_world_size,
        calculate_shard_param.sharding_type,
        calculate_shard_param.col_wise_shard_dim,
        calculate_shard_param.device_memory_sizes,
    )
