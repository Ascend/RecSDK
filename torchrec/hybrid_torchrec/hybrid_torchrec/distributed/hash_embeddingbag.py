#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional, Type, TypeVar

import torch
from torch import nn

from hybrid_torchrec.distributed.embeddingbag import HybridShardedEmbeddingBagCollection
from hybrid_torchrec.distributed.sharding.hybrid_rw_sharding import (
    HybridHashRwPooledEmbeddingSharding,
)
from hybrid_torchrec.distributed.sharding.hybrid_tw_sharding import (
    HybridHashTwPooledEmbeddingSharding,
)
from hybrid_torchrec.modules.hash_embeddingbag import HashEmbeddingBagCollection
from hybrid_torchrec.modules.ids_process import HashMapBase
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingContext,
    EmbeddingShardingInfo,
)
from torchrec.distributed.embedding_types import BaseEmbeddingSharder
from torchrec.distributed.embeddingbag import replace_placement_with_meta_device
from torchrec.distributed.types import (
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

Out = TypeVar("Out")


class HybridShardedHashEmbeddingBagCollection(HybridShardedEmbeddingBagCollection):
    """
    Sharded implementation of EmbeddingBagCollection.
    This is part of the public API to allow for manual data dist pipelining.
    """

    def __init__(
        self,
        module: HashEmbeddingBagCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        host_env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        module_fqn: Optional[str] = None,
    ) -> None:
        self._module_fqn = module_fqn
        self.table2hashmap: Dict[str, HashMapBase] = self.create_table2hashmap(module)
        super().__init__(
            module,
            table_name_to_parameter_sharding,
            env,
            host_env,
            fused_params,
            device,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )

    def create_hybrid_embedding_bag_sharding(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        host_env: ShardingEnv,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> EmbeddingSharding[
        EmbeddingShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]:
        sharding_type = sharding_infos[0].param_sharding.sharding_type
        if device is not None and device.type == "meta":
            replace_placement_with_meta_device(sharding_infos)
        if sharding_type == ShardingType.TABLE_WISE.value:
            return HybridHashTwPooledEmbeddingSharding(
                sharding_infos,
                self.table2hashmap,
                env,
                host_env,
                device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        elif sharding_type == ShardingType.ROW_WISE.value:
            return HybridHashRwPooledEmbeddingSharding(
                sharding_infos,
                self.table2hashmap,
                env,
                self._host_env,
                device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        else:
            raise ValueError(
                f"Sharding type not supported {sharding_type} for hybrid mode"
            )

    def create_table2hashmap(self, module):
        table2hashmap = {}
        for name in module.embedding_bags.keys():
            hashmap = module.embedding_bags[name].ids2slot_dict
            table2hashmap[name] = hashmap
        return table2hashmap


class HybridHashEmbeddingBagCollectionSharder(
    BaseEmbeddingSharder[EmbeddingBagCollection]
):
    """
    This implementation uses non-fused `EmbeddingBagCollection`
    """

    def __init__(
        self,
        host_env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(
            fused_params=fused_params, qcomm_codecs_registry=qcomm_codecs_registry
        )
        self._host_env = host_env

    @property
    def module_type(self) -> Type[HashEmbeddingBagCollection]:
        return HashEmbeddingBagCollection

    def shard(
        self,
        module: EmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> HybridShardedHashEmbeddingBagCollection:
        return HybridShardedHashEmbeddingBagCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            host_env=self._host_env,
            fused_params=self.fused_params,
            device=device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            module_fqn=module_fqn,
        )

    def shardable_parameters(
        self, module: EmbeddingBagCollection
    ) -> Dict[str, nn.Parameter]:
        return {
            name.split(".")[0]: param
            for name, param in module.embedding_bags.named_parameters()
        }
