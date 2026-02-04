#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from typing import Dict

import pytest
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from torchrec import EmbeddingCollection, DataType, EmbeddingConfig
from torchrec.distributed.planner import Topology
from torchrec.distributed.types import ShardingType, ShardingPlan

from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbTableOptions,
    DynamicEmbInitializerMode,
    DynamicEmbInitializerArgs,
    DistType,
)
from dynamic_emb.distributed.embedding import DynamicEmbeddingCollectionSharder
from dynamic_emb.distributed.planner.enumerators import DynamicEmbeddingEnumerator
from dynamic_emb.distributed.planner.planners import (
    DynamicEmbParameterSharding,
    DynamicEmbeddingShardingPlanner,
)
from dynamic_emb.distributed.planner.types import DynamicEmbParameterConstraints


class TestDynamicEmbParameterSharding:
    @staticmethod
    def test_ok():
        de_param_sharding = DynamicEmbParameterSharding(
            sharding_type="row_wise",
            compute_kernel="fused",
        )
        assert de_param_sharding.sharding_type == "row_wise"
        assert de_param_sharding.compute_kernel == "fused"
        assert de_param_sharding.dist_type == DistType.ROUNDROBIN.value


class TestDynamicEmbeddingShardingPlanner:
    def __init__(self):
        self.eb_configs = None
        self.eb_sharder = None
        self.dict_const = None
        self.topology = None

    def setup_method(self):
        self.eb_configs = [
            EmbeddingConfig(
                name="clothes_id",
                embedding_dim=8,
                num_embeddings=8,
                feature_names=["clothes_id"],
                data_type=DataType.FP32,
            ),
            EmbeddingConfig(
                name="shoes_id",
                embedding_dim=8,
                num_embeddings=16,
                feature_names=["shoes_id"],
                data_type=DataType.FP32,
            ),
        ]

        fused_params = {
            "output_dtype": SparseType.FP32,
            "prefetch_pipeline": False,
            "optimizer": "adam",
            "learning_rate": 0.001,
        }
        self.eb_sharder = DynamicEmbeddingCollectionSharder(
            fused_params=fused_params,
            use_index_dedup=False,
        )

        self.dict_const = {}
        for eb_config in self.eb_configs:
            const = DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                compute_kernels=["fused"],
                dynamicemb_options=DynamicEmbTableOptions(),
            )
            self.dict_const[eb_config.name] = const

        device = torch.device("npu:0")
        self.topology = Topology(
            world_size=2,
            compute_device=device.type,
        )

    def test_rw_planner(self, monkeypatch):
        def _mock_get_world_size():
            return 2

        monkeypatch.setattr("dynamic_emb.distributed.planner.planners.dist.get_world_size", _mock_get_world_size)

        expected_ranks = [0, 1, 0, 1]
        ranks = []
        eb_planner = DynamicEmbeddingShardingPlanner(
            eb_configs=self.eb_configs,
            topology=self.topology,
            constraints=self.dict_const,
            batch_size=32,
            enumerator=DynamicEmbeddingEnumerator(
                topology=self.topology,
                constraints=self.dict_const,
            ),
        )

        ec = EmbeddingCollection(
            tables=self.eb_configs,
            device=torch.device("npu"),
        )

        plan = eb_planner._torchrec_planner.plan(ec, [self.eb_sharder])

        def _mock_collective_plan(self, module, sharders, pg):
            return plan

        monkeypatch.setattr(
            "torchrec.distributed.planner.EmbeddingShardingPlanner.collective_plan", _mock_collective_plan
        )
        sharding_plan: ShardingPlan = eb_planner.collective_plan(ec, [self.eb_sharder], None)
        for _, plan_ in sharding_plan.plan.items():
            plan_: Dict[str, DynamicEmbParameterSharding]
            for _, param_sharding in plan_.items():
                ranks.extend(param_sharding.ranks)
        assert sorted(expected_ranks) == sorted(ranks)

    @pytest.mark.parametrize("invalid_value", [["xxx"], None])
    def test_eb_configs_value_err(self, invalid_value):
        with pytest.raises(ValueError):
            DynamicEmbeddingShardingPlanner(
                eb_configs=invalid_value,
                constraints=self.dict_const,
            )

    def test_constraints_value_err_case1(self):
        with pytest.raises(ValueError):
            DynamicEmbeddingShardingPlanner(
                eb_configs=self.eb_configs,
                constraints=None,
            )

    def test_constraints_value_err_case2(self):
        with pytest.raises(ValueError):
            DynamicEmbeddingShardingPlanner(
                eb_configs=self.eb_configs,
                constraints={123: self.dict_const["clothes_id"]},
            )

    def test_constraints_value_err_case3(self):
        with pytest.raises(ValueError):
            DynamicEmbeddingShardingPlanner(
                eb_configs=self.eb_configs,
                constraints={"xxx": 123},
            )

    def test_config_name_not_in_constraints(self):
        with pytest.raises(ValueError):
            tmp_eb_config = self.eb_configs
            tmp_eb_config.append(
                EmbeddingConfig(
                    name="xxx_id",
                    embedding_dim=8,
                    num_embeddings=8,
                    feature_names=["xxx_id"],
                    data_type=DataType.FP32,
                ),
            )
            DynamicEmbeddingShardingPlanner(
                eb_configs=tmp_eb_config,
                constraints=self.dict_const,
            )

    def test_config_name_not_unique_err(self):
        with pytest.raises(ValueError):
            tmp_eb_config = self.eb_configs
            tmp_eb_config.append(
                EmbeddingConfig(
                    name="clothes_id",
                    embedding_dim=8,
                    num_embeddings=8,
                    feature_names=["xxx_id"],
                    data_type=DataType.FP32,
                ),
            )
            DynamicEmbeddingShardingPlanner(
                eb_configs=tmp_eb_config,
                constraints=self.dict_const,
            )
