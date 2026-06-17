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
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingContext,
    EmbeddingShardingInfo,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, ShardingType
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.types import (
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from rec_sdk_common.validator.safe_checker import class_safe_check
from dynamic_emb.distributed.dynamicemb_config import DynamicEmbKernel
from dynamic_emb.distributed.sharding.rw_sequence_sharding import RwPooledDynamicEmbeddingSharding


class ShardedDynamicEmbeddingBagCollection(ShardedEmbeddingBagCollection):
    supported_compute_kernels: List[str] = [kernel.value for kernel in EmbeddingComputeKernel] + [DynamicEmbKernel]

    @classmethod
    def create_embedding_bag_sharding(
        cls,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        permute_embeddings: bool = False,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> EmbeddingSharding[EmbeddingShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor]:
        """
        override this function to provide customized EmbeddingSharding
        """
        sharding_type = sharding_infos[0].param_sharding.sharding_type

        if sharding_type == ShardingType.ROW_WISE.value:
            return RwPooledDynamicEmbeddingSharding(
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        else:
            return super().create_embedding_bag_sharding(
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                permute_embeddings=permute_embeddings,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )


class DynamicEmbeddingBagCollectionSharder(EmbeddingBagCollectionSharder):
    """
    DynamicEmbeddingBagCollectionSharder extends the EmbeddingBagCollectionSharder class from the TorchREC repo.
    The usage is completely consistent with TorchREC's EmbeddingBagCollectionSharder.
    """

    def __init__(
        self,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        class_safe_check("fused_params", fused_params, (dict, type(None)))
        if fused_params is not None:
            for k, v in fused_params.items():
                class_safe_check("key of fused_params", k, (str,))
        class_safe_check("qcomm_codecs_registry", qcomm_codecs_registry, (dict, type(None)))
        if qcomm_codecs_registry is not None:
            for k, v in qcomm_codecs_registry.items():
                class_safe_check("key of qcomm_codecs_registry", k, (str,))
                class_safe_check("value of qcomm_codecs_registry", v, (QuantizedCommCodecs,))

        super().__init__(fused_params, qcomm_codecs_registry)

    def shard(
        self,
        module: EmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedEmbeddingBagCollection:
        return ShardedDynamicEmbeddingBagCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            fused_params=self.fused_params,
            device=device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            module_fqn=module_fqn,
        )
