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

import argparse
from typing import List, Dict, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import SparseType
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.planner.types import ShardingPlan
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from dynamic_emb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
    DynamicEmbeddingEnumerator,
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
    DynamicEmbeddingCollectionSharder,
    EmbOptimType,
)


class MovieLensModel(nn.Module):
    def __init__(
        self,
        embedding_module: EmbeddingCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
    ):
        super().__init__()
        self.embedding_module = embedding_module

        embedding_dim = embedding_module.embedding_configs()[0].embedding_dim
        for config in embedding_module.embedding_configs():
            if embedding_dim != config.embedding_dim:
                raise ValueError("The embedding dimensions must be of the same size.")

        over_arch_layers = []
        if dense_in_features == 0:
            input_dim = embedding_dim
        else:
            input_dim = dense_arch_layer_sizes[-1] + embedding_dim

        over_arch_layers.append(
            nn.Linear(
                input_dim,
                over_arch_layer_sizes[0],
            )
        )
        over_arch_layers.append(nn.ReLU())
        for i in range(len(over_arch_layer_sizes) - 1):
            over_arch_layers.append(
                nn.Linear(over_arch_layer_sizes[i], over_arch_layer_sizes[i + 1])
            )
            over_arch_layers.append(nn.ReLU())
        over_arch_layers.append(nn.Linear(over_arch_layer_sizes[-1], 1))
        self.over_arch = nn.Sequential(*over_arch_layers)

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        embeddings = self.embedding_module(kjt)

        sparse_features = torch.cat(
            [embeddings[k].values() for k in embeddings.keys()], dim=0
        )

        prediction = self.over_arch(sparse_features)
        num_features = len(kjt.keys())
        batch = len(kjt.lengths()) // num_features
        hotness = 1
        # batch_size x hotness(1) x num_feature
        x = prediction.view(hotness * num_features, batch)
        return torch.sum(x.t(), dim=-1)


def get_sharder(
    args: argparse.Namespace, optimizer_type: EmbOptimType
) -> DynamicEmbeddingCollectionSharder:
    # set optimizer args
    learning_rate = args.lr
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0
    eps = 0.001

    # Put args into a optimizer kwargs , which is same usage of torchrec
    optimizer_kwargs = {
        "optimizer": optimizer_type,
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": weight_decay,
        "eps": eps,
    }

    fused_params: Dict[str, Any] = {}
    fused_params["output_dtype"] = (
        SparseType.FP32
    )  # data type of the output after lookup, and can differ from the stored.
    fused_params.update(optimizer_kwargs)

    return DynamicEmbeddingCollectionSharder(
        fused_params=fused_params,
        use_index_dedup=True,
    )


# use a function wrap all the planner code
def get_planner(
    device: torch.device,
    eb_configs: List[EmbeddingConfig],
    batch_size: int,
    training: bool,
) -> DynamicEmbeddingShardingPlanner:
    dict_const = {}
    for eb_config in eb_configs:
        const = DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            compute_kernels=["fused"],
            dynamicemb_options=DynamicEmbTableOptions(
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.NORMAL
                ),
                training=training,
            ),
        )
        dict_const[eb_config.name] = const

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
    )


def get_optimizer_type(opt: str) -> EmbOptimType:
    if opt == "adam":
        return EmbOptimType.ADAM
    else:
        raise ValueError(f"Unknown optimizer type: {opt}")


def apply_dmp(
    model: nn.Module, args: argparse.Namespace, training: bool, device: torch.device
) -> nn.Module:
    eb_configs = model.embedding_module.embedding_configs()
    optimizer_type = get_optimizer_type(args.optimizer)

    sharder = get_sharder(args, optimizer_type)

    planner = get_planner(
        device,
        eb_configs,
        args.batch_size,
        training=training,
    )
    plan: ShardingPlan = planner.collective_plan(
        model, [sharder], dist.GroupMember.WORLD
    )

    return DistributedModelParallel(
        module=model,
        device=device,
        sharders=[sharder],
        plan=plan,
    )


def create_model(
    args: argparse.Namespace, device: torch.device, training: bool = True
) -> nn.Module:
    # Define the configuration parameters for the embedding table,
    # including its name, embedding dimension, total number of embeddings, and feature name.
    eb_configs = [
        EmbeddingConfig(
            name="user_id",
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings,
            feature_names=["user_id"],
        ),
        EmbeddingConfig(
            name="movie_id",
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings,
            feature_names=["movie_id"],
        ),
        EmbeddingConfig(
            name="gender",
            embedding_dim=args.embedding_dim,
            num_embeddings=2,
            feature_names=["gender"],
        ),
        EmbeddingConfig(
            name="age",
            embedding_dim=args.embedding_dim,
            num_embeddings=100,
            feature_names=["age"],
        ),
        EmbeddingConfig(
            name="occupation",
            embedding_dim=args.embedding_dim,
            num_embeddings=50,
            feature_names=["occupation"],
        ),
        EmbeddingConfig(
            name="year",
            embedding_dim=args.embedding_dim,
            num_embeddings=2050,
            feature_names=["year"],
        ),
    ]

    ec = EmbeddingCollection(
        tables=eb_configs,
        device=device,
    )

    mlp_dims = [int(dim) for dim in args.mlp_dims.split(",")]

    model = MovieLensModel(
        embedding_module=ec,
        dense_in_features=0,
        dense_arch_layer_sizes=[1, 1],  # placeholder
        over_arch_layer_sizes=mlp_dims,
    )

    model = apply_dmp(model, args, training, device)

    return model
