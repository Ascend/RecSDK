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
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Union

import torch_npu
from torch import distributed as dist
from torch import nn
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import EmbeddingShardingPlanner
from torchrec.distributed.planner.types import (
    Enumerator,
    Partitioner,
    PerfModel,
    Proposer,
    Stats,
    StorageReservation,
    Topology,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ParameterSharding,
    ShardingPlan,
)
from torchrec.modules.embedding_configs import BaseEmbeddingConfig, EmbeddingConfig, DATA_TYPE_NUM_BITS

from dynamic_emb.distributed.planner.types import DynamicEmbParameterConstraints
from dynamic_emb.distributed.planner.enumerators import DynamicEmbeddingEnumerator
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbTableOptions,
    DynamicEmbKernel,
    DistType,
    next_power_of_2,
    get_optimizer_state_dim,
)
from rec_sdk_common.validator.safe_checker import class_safe_check


@dataclass
class DynamicEmbParameterSharding(ParameterSharding):
    """
    DynamicEmb-specific parameter constraints that extend ParameterSharding.
    """

    compute_kernel: str = EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value
    customized_compute_kernel: Optional[str] = DynamicEmbKernel
    dist_type: str = DistType.ROUNDROBIN.value
    dynamicemb_options: Optional[DynamicEmbTableOptions] = field(default_factory=DynamicEmbTableOptions)

    def get_additional_fused_params(self):
        all_fields = {f.name: getattr(self, f.name) for f in fields(DynamicEmbParameterSharding)}
        parameter_sharding_fields = {f.name: getattr(self, f.name) for f in fields(ParameterSharding)}
        return {k: v for k, v in all_fields.items() if k not in parameter_sharding_fields}


class DynamicEmbeddingShardingPlanner:
    def __init__(
        self,
        eb_configs: List[BaseEmbeddingConfig],
        constraints: Dict[str, DynamicEmbParameterConstraints],
        topology: Optional[Topology] = None,
        batch_size: Optional[int] = None,
        enumerator: Optional[Enumerator] = None,
        storage_reservation: Optional[StorageReservation] = None,
        proposer: Optional[Union[Proposer, List[Proposer]]] = None,
        partitioner: Optional[Partitioner] = None,
        performance_model: Optional[PerfModel] = None,
        stats: Optional[Union[Stats, List[Stats]]] = None,
        debug: bool = True,
    ):
        """
        DynamicEmbeddingShardingPlanner wraps the API of EmbeddingShardingPlanner from the Torchrec repo,
        giving it the ability to plan dynamic embedding tables. The only difference from EmbeddingShardingPlanner
        is that DynamicEmbeddingShardingPlanner has an additional parameter `eb_configs`, which is a list of
        TorchREC BaseEmbeddingConfig. This is because the dynamic embedding table needs to re-plan the number of
        embedding vectors on each rank to align with the power of 2.

        Parameters
        ----------
        eb_configs : List[BaseEmbeddingConfig]
            A list of TorchREC BaseEmbeddingConfig in the TorchREC model
        constraints : Dict[str, DynamicEmbParameterConstraints]
            A dictionary of constraints for every TorchREC embedding table and Dynamic embedding table.
        topology : Optional[Topology], optional
            The topology of NPU and Host memory. If None, a default topology will be created. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
            Note: The memory budget does not include the consumption of dynamicemb.
        batch_size : Optional[int], optional
            The batch size for training. Defaults to None, will set 512 in Planner.
        enumerator : Optional[Enumerator], optional
            An enumerator for sharding. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        storage_reservation : Optional[StorageReservation], optional
            Storage reservation details. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        proposer : Optional[Union[Proposer, List[Proposer]]], optional
            A proposer or a list of proposers for proposing sharding plans. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        partitioner : Optional[Partitioner], optional
            A partitioner for partitioning the embedding tables. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        performance_model : Optional[PerfModel], optional
            A performance model for evaluating sharding plans. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        stats : Optional[Union[Stats, List[Stats]]], optional
            Statistics or a list of statistics for the sharding process. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        debug : bool, optional
            A flag indicating whether to enable debug mode. Defaults to True.
        """

        super(DynamicEmbeddingShardingPlanner, self).__init__()

        _validate_configs(constraints, eb_configs)
        class_safe_check("topology", topology, (Topology, type(None)))
        class_safe_check("batch_size", batch_size, (int, type(None)))
        class_safe_check("enumerator", enumerator, (DynamicEmbeddingEnumerator, type(None)))

        self._dyn_emb_table_consts = constraints
        self._topology = topology
        if self._topology is None:
            self._topology = Topology(
                local_world_size=get_local_size(),
                world_size=dist.get_world_size(),
                compute_device="npu" if torch_npu.npu.is_available() else "cpu",
            )
        self._enumerator = enumerator
        if self._enumerator is None:
            self._enumerator = DynamicEmbeddingEnumerator(
                topology=self._topology,
                constraints=constraints,
            )
        _reserve_storage_for_dyn_emb(self._dyn_emb_table_consts, eb_configs)

        self._torchrec_planner = EmbeddingShardingPlanner(
            topology=self._topology,
            constraints=self._dyn_emb_table_consts,
            batch_size=batch_size,
            enumerator=self._enumerator,
            storage_reservation=storage_reservation,
            proposer=proposer,
            partitioner=partitioner,
            performance_model=performance_model,
            stats=stats,
            debug=debug,
        )

    def collective_plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        pg: Optional[dist.ProcessGroup] = dist.GroupMember.WORLD,
    ) -> ShardingPlan:
        """
        Generate a collective sharding plan.

        Parameters
        ----------
        module : nn.Module
            The PyTorch module to be sharded.
        sharders : List[ModuleSharder[nn.Module]]
            A list of module sharders.
        pg : Optional[dist.ProcessGroup], optional
            The process group for distributed training. Defaults to dist.GroupMember.WORLD.

        Returns
        -------
        ShardingPlan
            The generated sharding plan.
        """

        torchrec_plan = self._torchrec_planner.collective_plan(module, sharders, pg)
        dyn_emb_names = self._dyn_emb_table_consts.keys()
        for dyn_emb_name in dyn_emb_names:
            for _, torchrec_module_plan in torchrec_plan.plan.items():
                torchrec_module_plan: Dict[str, ParameterSharding]
                for table_name, table_plan in torchrec_module_plan.items():
                    if dyn_emb_name == table_name:
                        dynamic_emb_param_sharding = DynamicEmbParameterSharding(
                            sharding_spec=table_plan.sharding_spec,
                            sharding_type=table_plan.sharding_type,
                            ranks=table_plan.ranks,
                            cache_params=table_plan.cache_params,
                            stochastic_rounding=table_plan.stochastic_rounding,
                            bounds_check_mode=table_plan.bounds_check_mode,
                            output_dtype=table_plan.output_dtype,
                            key_value_params=table_plan.key_value_params,
                            compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
                            customized_compute_kernel=DynamicEmbKernel,
                            dist_type=DistType.ROUNDROBIN.value,
                            dynamicemb_options=self._dyn_emb_table_consts[table_name].dynamicemb_options,
                        )
                        torchrec_module_plan[table_name] = dynamic_emb_param_sharding
        return torchrec_plan


def _validate_configs(
    constraints: Dict[str, DynamicEmbParameterConstraints],
    eb_configs: List[BaseEmbeddingConfig],
):
    if constraints is None or eb_configs is None:
        raise ValueError("Constraints and eb_configs must not be None")
    class_safe_check("eb_configs", eb_configs, (list,))
    for eb_config in eb_configs:
        class_safe_check("eb_config", eb_config, (EmbeddingConfig,))
    class_safe_check("constraints", constraints, (dict,))
    for k, v in constraints.items():
        class_safe_check("key of constraints", k, (str,))
        class_safe_check("value of constraints", v, (DynamicEmbParameterConstraints,))

    # Extract names from eb_configs
    config_names = [config.name for config in eb_configs]

    # Check if each BaseEmbeddingConfig's name matches the keys in the constraints dictionary
    for config_name in config_names:
        if config_name not in constraints:
            raise ValueError(f"Config name '{config_name}' does not match any key in constraints")

    # Verify that each BaseEmbeddingConfig name is unique
    if len(set(config_names)) != len(config_names):
        raise ValueError("Config names must be unique")

    # Ensure that all constraints keys have corresponding BaseEmbeddingConfig with matching name
    if set(config_names) != set(constraints.keys()):
        raise ValueError("Not all constraint keys have matching BaseEmbeddingConfig names")

    world_size = dist.get_world_size()
    for i, config_name in enumerate(config_names):
        if not constraints[config_name].use_dynamicemb:
            continue
        tmp_config = eb_configs[i]
        # modify num_embeddings per rank to power of 2
        num_aligned_embedding_per_rank = int(next_power_of_2(math.ceil(tmp_config.num_embeddings / world_size)))
        if num_aligned_embedding_per_rank < constraints[config_name].dynamicemb_options.bucket_capacity:
            num_aligned_embedding_per_rank = constraints[config_name].dynamicemb_options.bucket_capacity

        if tmp_config.num_embeddings != int(num_aligned_embedding_per_rank * world_size):
            constraints[config_name].dynamicemb_options.num_aligned_embedding_per_rank = num_aligned_embedding_per_rank


def _reserve_storage_for_dyn_emb(
    dyn_emb_table_const: Dict[str, DynamicEmbParameterConstraints], eb_configs: List[BaseEmbeddingConfig]
) -> None:
    world_size = dist.get_world_size()
    for eb_config in eb_configs:
        for name, constraint in dyn_emb_table_const.items():
            # 默认值：值类型的字节数 * 表的行数(对齐到2的幂次) * 表的列数
            if constraint.dynamicemb_options.global_hbm_for_values == 0:
                embedding_type_bytes = DATA_TYPE_NUM_BITS[eb_config.data_type] / 8  # bytes
                eb_num_embeddings_next_power_of_2 = next_power_of_2(eb_config.num_embeddings)
                total_dim = eb_config.embedding_dim + get_optimizer_state_dim(
                    constraint.dynamicemb_options.optimizer_type,
                    eb_config.embedding_dim,
                    constraint.dynamicemb_options.embedding_dtype,
                )
                total_hbm_need = embedding_type_bytes * total_dim * eb_num_embeddings_next_power_of_2
                constraint.dynamicemb_options.global_hbm_for_values = int(total_hbm_need)
            device_memory_in_bytes_per_rank = math.ceil(
                constraint.dynamicemb_options.global_hbm_for_values / world_size
            )
            constraint.dynamicemb_options.local_hbm_for_values = device_memory_in_bytes_per_rank
