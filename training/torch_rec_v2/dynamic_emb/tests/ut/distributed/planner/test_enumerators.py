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

import pytest
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import BoundsCheckMode
from torchrec import DataType, EmbeddingCollection
from torchrec.distributed.fbgemm_qcomm_codec import get_qcomm_codecs_registry, QCommsConfig, CommType
from torchrec.distributed.planner import Topology
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.modules.embedding_configs import ShardingType, EmbeddingConfig

from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbTableOptions,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
)
from dynamic_emb.distributed.embedding import DynamicEmbeddingCollectionSharder
from dynamic_emb.distributed.planner.enumerators import DynamicEmbeddingEnumerator
from dynamic_emb.distributed.planner.types import DynamicEmbParameterConstraints


class TestDynamicEmbeddingEnumerator:
    def __init__(self):
        self.sharder = None
        self.ec = None
        self.const = None
        self.topology = None
        self.enumerator = None

    def setup_method(self) -> None:
        optimizer_kwargs = {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0,
            "eps": 0.001,
        }
        fused_params = {"output_dtype": SparseType.FP32}
        fused_params.update(optimizer_kwargs)
        fused_params["prefetch_pipeline"] = False

        qcomm_codecs_registry = get_qcomm_codecs_registry(
            qcomms_config=QCommsConfig(
                forward_precision=CommType.FP32,
                backward_precision=CommType.FP32,
            )
        )

        self.sharder = DynamicEmbeddingCollectionSharder(
            qcomm_codecs_registry=qcomm_codecs_registry,
            fused_params=fused_params,
            use_index_dedup=False,
        )
        eb_configs = [
            EmbeddingConfig(
                name="user_id",
                embedding_dim=8,
                num_embeddings=8,
                feature_names=["user_id"],
                data_type=DataType.FP32,
            ),
            EmbeddingConfig(
                name="item_id",
                embedding_dim=8,
                num_embeddings=16,
                feature_names=["item_id"],
                data_type=DataType.FP32,
            ),
        ]
        self.ec = EmbeddingCollection(
            tables=eb_configs,
            device=torch.device("npu"),
        )

        self.const = DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            compute_kernels=["fused"],
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=1024,
                initializer_args=DynamicEmbInitializerArgs(mode=DynamicEmbInitializerMode.NORMAL),
            ),
        )
        self.topology = Topology(
            world_size=2,
            compute_device="npu",
        )
        self.enumerator = DynamicEmbeddingEnumerator(
            topology=self.topology,
            constraints={"user_id": self.const, "item_id": self.const},
        )

    @staticmethod
    def test_topology_type_err():
        with pytest.raises(ValueError):
            DynamicEmbeddingEnumerator(
                topology="xxx",
            )

    def test_init_ok(self):
        assert isinstance(self.enumerator, EmbeddingEnumerator)

    def test_rw_sharding_ok(self):
        sharding_options = self.enumerator.enumerate(self.ec, [self.sharder])
        # The rank size is 2.
        expected_rw_shard_sizes = [[[4, 8], [4, 8]], [[8, 8], [8, 8]]]
        expected_rw_shard_offsets = [[[0, 0], [4, 0]], [[0, 0], [8, 0]]]
        for i, sharding_option in enumerate(sharding_options):
            assert sharding_option.sharding_type == ShardingType.ROW_WISE.value
            assert [shard.size for shard in sharding_option.shards] == expected_rw_shard_sizes[i]
            assert [shard.offset for shard in sharding_option.shards] == expected_rw_shard_offsets[i]

    def test_batch_size_value_err(self):
        with pytest.raises(ValueError):
            DynamicEmbeddingEnumerator(
                topology=self.topology,
                batch_size=-1,
            )

    def test_constraints_value_err_case1(self):
        with pytest.raises(ValueError):
            DynamicEmbeddingEnumerator(
                topology=self.topology,
                constraints=["xxx"],
            )

    def test_constraints_value_err_case2(self):
        with pytest.raises(ValueError):
            DynamicEmbeddingEnumerator(
                topology=self.topology,
                constraints={123: self.const},
            )

    def test_constraints_value_err_case3(self):
        with pytest.raises(ValueError):
            DynamicEmbeddingEnumerator(
                topology=self.topology,
                constraints={"item_id": "xxx"},
            )

    def test_estimator_should_be_default_value(self):
        with pytest.raises(ValueError):
            DynamicEmbeddingEnumerator(
                topology=self.topology,
                estimator="xxx",
            )

    def test_use_exact_enumerate_order_should_be_default_value(self):
        with pytest.raises(ValueError):
            DynamicEmbeddingEnumerator(
                topology=self.topology,
                use_exact_enumerate_order=True,
            )
