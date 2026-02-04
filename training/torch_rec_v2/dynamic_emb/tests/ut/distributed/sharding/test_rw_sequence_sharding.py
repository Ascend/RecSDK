#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import unittest
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharding_spec import EnumerableShardingSpec
from torchrec import DataType, PoolingType
from torchrec.distributed import ParameterSharding
from torchrec.distributed.embedding_sharding import EmbeddingShardingInfo
from torchrec.distributed.embedding_types import GroupedEmbeddingConfig, EmbeddingComputeKernel, ShardedEmbeddingTable
from torchrec.distributed.types import ShardingType, ShardingEnv
from torchrec.modules.embedding_configs import EmbeddingTableConfig

from dynamic_emb import DynamicEmbTableOptions, DynamicEmbInitializerArgs
from dynamic_emb.distributed.dynamicemb_config import DynamicEmbEvictStrategy
from dynamic_emb.distributed.sharding.rw_sharding import DynamicEmbRwSparseFeaturesDist
from dynamic_emb.distributed.batched_dynamicemb_compute_kernel import BatchedDynamicEmbedding
from dynamic_emb.distributed.sharding.rw_sequence_sharding import (
    GroupedEmbeddingsLookup,
    RwSequenceDynamicEmbeddingSharding,
)
from dynamic_emb_extensions import SafeCheckMode, OptimizerType


def _create_table_options() -> DynamicEmbTableOptions:
    table_options = DynamicEmbTableOptions(training=False)
    # mock inferred from the context
    table_options.index_type = torch.int64
    table_options.embedding_dtype = torch.float32
    table_options.evict_strategy = DynamicEmbEvictStrategy.LRU
    table_options.dim = 8
    table_options.init_capacity = 128
    table_options.max_capacity = 256
    table_options.local_hbm_for_values = 1024
    table_options.bucket_capacity = 128
    table_options.max_load_factor = 0.8
    table_options.block_size = 128
    table_options.io_block_size = 1024
    table_options.device_id = 0
    table_options.io_by_cpu = False
    table_options.use_constant_memory = False
    table_options.reserved_key_start_bit = 0
    table_options.num_of_buckets_per_alloc = 1
    table_options.initializer_args = DynamicEmbInitializerArgs()
    table_options.safe_check_mode = SafeCheckMode.IGNORE
    table_options.optimizer_type = OptimizerType.Null
    return table_options


class TestGroupedEmbeddingsLookup(unittest.TestCase):
    @mock.patch.multiple(
        "dynamic_emb.distributed.batched_dynamicemb_table",
        device_timestamp=MagicMock(return_value=123),
    )
    def test_create_embedding_kernel(self):
        pg = MagicMock()
        pg.size.return_value = 1
        table_options = _create_table_options()
        embedding_table = ShardedEmbeddingTable(
            num_embeddings=8,
            embedding_dim=16,
            name="user_table",
            feature_names=["user_ids"],
            data_type=DataType.FP32,
            pooling=PoolingType.NONE,
            compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL,
            local_rows=8,
            local_cols=16,
            fused_params={"dynamicemb_options": table_options},
        )
        config = GroupedEmbeddingConfig(
            data_type=embedding_table.data_type,
            pooling=embedding_table.pooling,
            is_weighted=False,
            has_feature_processor=False,
            compute_kernel=embedding_table.compute_kernel,
            embedding_tables=[embedding_table],
            fused_params={"dynamicemb_options": table_options},
        )
        device = torch.device("cpu")

        grouped_embeddings_lookup = GroupedEmbeddingsLookup(
            grouped_configs=[config],
            pg=pg,
            device=device,
        )
        embedding_kernel = grouped_embeddings_lookup._emb_modules[0]
        assert isinstance(embedding_kernel, BatchedDynamicEmbedding)

    @mock.patch.multiple(
        "dynamic_emb.distributed.batched_dynamicemb_table",
        device_timestamp=MagicMock(return_value=123),
    )
    def test_create_embedding_kernel_err(self):
        pg = MagicMock()
        pg.size.return_value = 1
        table_options = _create_table_options()
        embedding_table = ShardedEmbeddingTable(
            num_embeddings=8,
            embedding_dim=16,
            name="user_table",
            feature_names=["user_ids"],
            data_type=DataType.FP32,
            pooling=PoolingType.NONE,
            compute_kernel="xxx",
            local_rows=8,
            local_cols=16,
            fused_params={"dynamicemb_options": table_options},
        )
        config = GroupedEmbeddingConfig(
            data_type=embedding_table.data_type,
            pooling=embedding_table.pooling,
            is_weighted=False,
            has_feature_processor=False,
            compute_kernel=embedding_table.compute_kernel,
            embedding_tables=[embedding_table],
            fused_params={"dynamicemb_options": table_options},
        )
        device = torch.device("cpu")

        with pytest.raises(ValueError):
            GroupedEmbeddingsLookup(
                grouped_configs=[config],
                pg=pg,
                device=device,
            )


class _MockDevice:
    def __init__(self, *args, **kwargs):
        pass

    @property
    def type(self):
        return "cpu"


class TestRwSequenceDynamicEmbeddingSharding:
    @staticmethod
    @mock.patch.multiple(
        "torchrec.distributed.types",
        init_device_mesh=MagicMock(return_value=None),
        _get_pg_default_device=MagicMock(return_value=_MockDevice()),
    )
    def test_init():
        sharding_info = EmbeddingShardingInfo(
            embedding_config=EmbeddingTableConfig(
                num_embeddings=8,
                embedding_dim=16,
                name="item_table",
                feature_names=["item_ids"],
            ),
            param_sharding=ParameterSharding(
                sharding_type=ShardingType.ROW_WISE.value,
                compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
                sharding_spec=(EnumerableShardingSpec([ShardMetadata(shard_sizes=[8, 16], shard_offsets=[1, 0])])),
            ),
            param=torch.empty(size=(1, 8)),
            fused_params={},
        )
        pg = MagicMock(spec=dist.ProcessGroup)
        pg.size.return_value = 1
        pg._get_backend_name.return_value = "gloo"
        env = ShardingEnv(
            world_size=1,
            rank=0,
            pg=pg,
        )
        device = torch.device("cpu")

        rw_seq_dynamic_emb_sharding = RwSequenceDynamicEmbeddingSharding(
            sharding_infos=[sharding_info],
            env=env,
            device=device,
        )
        assert rw_seq_dynamic_emb_sharding._dist_type_per_feature["item_ids"] == "continuous"

    @staticmethod
    @mock.patch.multiple(
        "torchrec.distributed.types",
        init_device_mesh=MagicMock(return_value=None),
        _get_pg_default_device=MagicMock(return_value=_MockDevice()),
    )
    def test_dist_type_is_roundrobin():
        sharding_info = EmbeddingShardingInfo(
            embedding_config=EmbeddingTableConfig(
                num_embeddings=8,
                embedding_dim=16,
                name="item_table",
                feature_names=["item_ids"],
            ),
            param_sharding=ParameterSharding(
                sharding_type=ShardingType.ROW_WISE.value,
                compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
                sharding_spec=(EnumerableShardingSpec([ShardMetadata(shard_sizes=[8, 16], shard_offsets=[1, 0])])),
            ),
            param=torch.empty(size=(1, 8)),
            fused_params={"dist_type": "roundrobin"},
        )
        pg = MagicMock(spec=dist.ProcessGroup)
        pg.size.return_value = 1
        pg._get_backend_name.return_value = "gloo"
        env = ShardingEnv(
            world_size=1,
            rank=0,
            pg=pg,
        )
        device = torch.device("cpu")

        rw_seq_dynamic_emb_sharding = RwSequenceDynamicEmbeddingSharding(
            sharding_infos=[sharding_info],
            env=env,
            device=device,
        )
        assert rw_seq_dynamic_emb_sharding._dist_type_per_feature["item_ids"] == "roundrobin"

    @staticmethod
    @mock.patch.multiple(
        "torchrec.distributed.types",
        init_device_mesh=MagicMock(return_value=None),
        _get_pg_default_device=MagicMock(return_value=_MockDevice()),
    )
    def test_dist_type_must_be_same():
        sharding_info1 = EmbeddingShardingInfo(
            embedding_config=EmbeddingTableConfig(
                num_embeddings=8,
                embedding_dim=16,
                name="item_table",
                feature_names=["item_ids"],
            ),
            param_sharding=ParameterSharding(
                sharding_type=ShardingType.ROW_WISE.value,
                compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
                sharding_spec=(EnumerableShardingSpec([ShardMetadata(shard_sizes=[8, 16], shard_offsets=[1, 0])])),
            ),
            param=torch.empty(size=(1, 8)),
            fused_params={"dist_type": "roundrobin"},
        )
        sharding_info2 = EmbeddingShardingInfo(
            embedding_config=EmbeddingTableConfig(
                num_embeddings=8,
                embedding_dim=16,
                name="item_table",
                feature_names=["item_ids"],
            ),
            param_sharding=ParameterSharding(
                sharding_type=ShardingType.ROW_WISE.value,
                compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
                sharding_spec=(EnumerableShardingSpec([ShardMetadata(shard_sizes=[8, 16], shard_offsets=[1, 0])])),
            ),
            param=torch.empty(size=(1, 8)),
            fused_params={"dist_type": "continuous"},
        )
        pg = MagicMock(spec=dist.ProcessGroup)
        pg.size.return_value = 1
        pg._get_backend_name.return_value = "gloo"
        env = ShardingEnv(
            world_size=1,
            rank=0,
            pg=pg,
        )
        device = torch.device("cpu")

        with pytest.raises(ValueError):
            RwSequenceDynamicEmbeddingSharding(
                sharding_infos=[sharding_info1, sharding_info2],
                env=env,
                device=device,
            )

    @staticmethod
    @mock.patch.multiple(
        "torchrec.distributed.types",
        init_device_mesh=MagicMock(return_value=None),
        _get_pg_default_device=MagicMock(return_value=_MockDevice()),
    )
    def test_create_input_dist():
        sharding_info = EmbeddingShardingInfo(
            embedding_config=EmbeddingTableConfig(
                num_embeddings=8,
                embedding_dim=16,
                name="item_table",
                feature_names=["item_ids"],
            ),
            param_sharding=ParameterSharding(
                sharding_type=ShardingType.ROW_WISE.value,
                compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
                sharding_spec=(EnumerableShardingSpec([ShardMetadata(shard_sizes=[8, 16], shard_offsets=[1, 0])])),
            ),
            param=torch.empty(size=(1, 8)),
            fused_params={"dist_type": "roundrobin"},
        )
        pg = MagicMock(spec=dist.ProcessGroup)
        pg.size.return_value = 1
        pg._get_backend_name.return_value = "gloo"
        env = ShardingEnv(
            world_size=1,
            rank=0,
            pg=pg,
        )
        device = torch.device("cpu")

        rw_seq_dynamic_emb_sharding = RwSequenceDynamicEmbeddingSharding(
            sharding_infos=[sharding_info],
            env=env,
            device=device,
        )
        input_dist = rw_seq_dynamic_emb_sharding.create_input_dist(device=device)
        assert isinstance(input_dist, DynamicEmbRwSparseFeaturesDist)

    @staticmethod
    @mock.patch.multiple(
        "torchrec.distributed.types",
        init_device_mesh=MagicMock(return_value=None),
        _get_pg_default_device=MagicMock(return_value=_MockDevice()),
    )
    def test_create_lookup():
        table_options = _create_table_options()
        sharding_info = EmbeddingShardingInfo(
            embedding_config=EmbeddingTableConfig(
                num_embeddings=8,
                embedding_dim=16,
                name="item_table",
                feature_names=["item_ids"],
            ),
            param_sharding=ParameterSharding(
                sharding_type=ShardingType.ROW_WISE.value,
                compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
                sharding_spec=(EnumerableShardingSpec([ShardMetadata(shard_sizes=[8, 16], shard_offsets=[1, 0])])),
            ),
            param=torch.empty(size=(1, 8)),
            fused_params={
                "dist_type": "roundrobin",
                "dynamicemb_options": table_options,
            },
        )
        pg = MagicMock(spec=dist.ProcessGroup)
        pg.size.return_value = 1
        pg._get_backend_name.return_value = "gloo"
        env = ShardingEnv(
            world_size=1,
            rank=0,
            pg=pg,
        )
        device = torch.device("cpu")

        rw_seq_dynamic_emb_sharding = RwSequenceDynamicEmbeddingSharding(
            sharding_infos=[sharding_info],
            env=env,
            device=device,
        )
        lookup = rw_seq_dynamic_emb_sharding.create_lookup(device=device)
        assert isinstance(lookup, GroupedEmbeddingsLookup)
