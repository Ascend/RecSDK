#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Union
from unittest.mock import patch, MagicMock

import pytest
import torch
from hybrid_torchrec.distributed import get_default_hybrid_sharders
from hybrid_torchrec.distributed.embeddingbag import (
    HybridEmbeddingBagCollectionSharder,
    HybridShardedEmbeddingBagCollection,
    KJTList,
    device_is_in,
    _pin_and_move
)
from torch.distributed._tensor import DTensor
from torchrec import KeyedJaggedTensor, KeyedTensor
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology, ParameterConstraints
from torchrec.distributed.types import ShardingEnv, ShardedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection

DEVICE = torch.device("cpu")


def set_env() -> (ShardingEnv, ShardingEnv):
    if not torch.distributed.is_initialized():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "6001"
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        os.environ["LOCAL_RANK"] = "0"
    pg = torch.distributed.group.WORLD
    env = ShardingEnv(world_size=1, rank=0, pg=pg)
    host_env = ShardingEnv(world_size=1, rank=0, pg=pg)
    return env, host_env


def create_ebc() -> EmbeddingBagCollection:
    embedding_bag_configs = [
        EmbeddingBagConfig(
            name="table1",
            embedding_dim=128,
            num_embeddings=1000,
            feature_names=["feature1"],
            pooling=PoolingType.SUM,
        ),
        EmbeddingBagConfig(
            name="table2",
            embedding_dim=64,
            num_embeddings=500,
            feature_names=["feature2"],
            pooling=PoolingType.MEAN,
        ),
    ]
    return EmbeddingBagCollection(tables=embedding_bag_configs, device=DEVICE), embedding_bag_configs


def create_planner() -> EmbeddingShardingPlanner:
    constraints = {
        "table1": ParameterConstraints(
            sharding_types=["row_wise"], compute_kernels=["fused"],
        ),
        "table2": ParameterConstraints(
            sharding_types=["row_wise"], compute_kernels=["fused"],
        ),
    }

    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size=1, compute_device=DEVICE.type),
        constraints=constraints,
    )
    return planner


def hybrid_sharded_embedding_bag_collection() -> HybridShardedEmbeddingBagCollection:
    env, host_env = set_env()
    module, embedding_bag_configs = create_ebc()
    planner = create_planner()
    hybrid_sharder = get_default_hybrid_sharders(host_env=host_env)

    plan = planner.collective_plan(module, hybrid_sharder, torch.distributed.GroupMember.WORLD)
    sharded_params = plan.get_plan_for_module(list(plan.plan.keys())[0])

    hybrid_sharded_ebc = HybridShardedEmbeddingBagCollection(
        module=module,
        table_name_to_parameter_sharding=sharded_params,
        env=env,
        host_env=host_env,
        fused_params=None,
        device=DEVICE,
        qcomm_codecs_registry=None,
        module_fqn=None,
    )

    # Assertions to verify initialization
    assert hybrid_sharded_ebc._env == env
    assert hybrid_sharded_ebc._host_env == host_env
    assert hybrid_sharded_ebc._device == DEVICE
    assert hybrid_sharded_ebc._embedding_bag_configs == embedding_bag_configs
    assert hybrid_sharded_ebc._table_names == ["table1", "table2"]
    assert hybrid_sharded_ebc._table_name_to_config["table1"].name == "table1"
    assert hybrid_sharded_ebc._table_name_to_config["table2"].name == "table2"

    return hybrid_sharded_ebc


class TestHybridShardedEmbeddingBagCollection:
    @staticmethod
    @patch("torchrec.distributed.planner.ParameterConstraints.__post_init__", return_value=None)
    @patch("torchrec.tensor_types.check", return_value=None)
    @patch("torchrec.distributed.model_parallel.check", return_value=None)
    @patch("torchrec.distributed.planner.types.check", return_value=None)
    def test_init(*mock):
        result = hybrid_sharded_embedding_bag_collection()
        assert result is not None

    @staticmethod
    @patch("torchrec.distributed.planner.ParameterConstraints.__post_init__", return_value=None)
    @patch("torchrec.tensor_types.check", return_value=None)
    @patch("torchrec.distributed.model_parallel.check", return_value=None)
    @patch("torchrec.distributed.planner.types.check", return_value=None)
    def test_input_dist(*mock):
        ebc = hybrid_sharded_embedding_bag_collection()
        # Create a mock KeyedJaggedTensor input
        features = KeyedJaggedTensor(
            keys=["feature1", "feature2"],
            values=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32),
            lengths=torch.tensor([3, 3], dtype=torch.int32),
        )

        # Call the input_dist method
        ctx = ebc.create_context()
        awaitable = ebc.input_dist(ctx, features)

        # Wait for the result
        result = awaitable.wait()

    @staticmethod
    @patch("torchrec.distributed.planner.ParameterConstraints.__post_init__", return_value=None)
    @patch("torchrec.tensor_types.check", return_value=None)
    @patch("torchrec.distributed.model_parallel.check", return_value=None)
    @patch("torchrec.distributed.planner.types.check", return_value=None)
    def test_post_input_dist(*mock):
        ebc = hybrid_sharded_embedding_bag_collection()
        features = KJTList([
            KeyedJaggedTensor(
                keys=["feature1"],
                values=torch.tensor([1, 2, 3], dtype=torch.int32),
                lengths=torch.tensor([3], dtype=torch.int32),
            ),
            KeyedJaggedTensor(
                keys=["feature2"],
                values=torch.tensor([4, 5, 6], dtype=torch.int32),
                lengths=torch.tensor([3], dtype=torch.int32),
            ),
        ])

        # Call the input_dist method
        ctx = ebc.create_context()
        awaitable = ebc.post_input_dist(ctx, features)

        # Wait for the result
        result = awaitable.wait()

    @staticmethod
    @patch("torchrec.distributed.planner.ParameterConstraints.__post_init__", return_value=None)
    @patch("torchrec.tensor_types.check", return_value=None)
    @patch("torchrec.distributed.model_parallel.check", return_value=None)
    @patch("torchrec.distributed.planner.types.check", return_value=None)
    @patch("torchrec.distributed.embeddingbag.construct_output_kt",
           return_value=KeyedTensor(["feature1", "feature2"], [2, 2], torch.tensor((1, 2, 3, 4))))
    def test_output_dist(*mock):
        ebc = hybrid_sharded_embedding_bag_collection()

        # Call the input_dist method
        ctx = ebc.create_context()
        ctx.divisor = 1

        # Mock output tensors
        output = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        ]

        # Call the output_dist method
        awaitable = ebc.output_dist(ctx, output)

        # Wait for the result
        result = awaitable.wait()

    @staticmethod
    @patch("torchrec.distributed.planner.ParameterConstraints.__post_init__", return_value=None)
    @patch("torchrec.tensor_types.check", return_value=None)
    @patch("torchrec.distributed.model_parallel.check", return_value=None)
    @patch("torchrec.distributed.planner.types.check", return_value=None)
    @patch("torchrec.distributed.embeddingbag.construct_output_kt",
           return_value=KeyedTensor(["feature1", "feature2"], [2, 2], torch.tensor((1, 2, 3, 4))))
    def test_compute_and_output_dist(*mock):
        ebc = hybrid_sharded_embedding_bag_collection()

        # Call the input_dist method
        ctx = ebc.create_context()
        ctx.divisor = 1

        features = KJTList([
            KeyedJaggedTensor(
                keys=["feature1"],
                values=torch.tensor([1, 2, 3], dtype=torch.int32),
                lengths=torch.tensor([3], dtype=torch.int32),
            ),
            KeyedJaggedTensor(
                keys=["feature2"],
                values=torch.tensor([4, 5, 6], dtype=torch.int32),
                lengths=torch.tensor([3], dtype=torch.int32),
            ),
        ])

        # Call the output_dist method
        awaitable = ebc.compute_and_output_dist(ctx, features)

        # Wait for the result
        result = awaitable.wait()

    @staticmethod
    @patch("torchrec.distributed.planner.ParameterConstraints.__post_init__", return_value=None)
    @patch("torchrec.tensor_types.check", return_value=None)
    @patch("torchrec.distributed.model_parallel.check", return_value=None)
    @patch("torchrec.distributed.planner.types.check", return_value=None)
    def test_load_state_dict_tensor(*mock):
        ebc = hybrid_sharded_embedding_bag_collection()
        module, _ = create_ebc()
        ebc.load_state_dict(module.state_dict(), strict=False)

    @staticmethod
    @patch("torchrec.distributed.planner.ParameterConstraints.__post_init__", return_value=None)
    @patch("torchrec.tensor_types.check", return_value=None)
    @patch("torchrec.distributed.model_parallel.check", return_value=None)
    @patch("torchrec.distributed.planner.types.check", return_value=None)
    def test_load_state_dict_tensor_(*mock):
        ebc = hybrid_sharded_embedding_bag_collection()
        tensor = torch.randn(10, 10)
        model_shards_dtensor = {"local_tensors": [tensor], "local_offsets": [(0, 0)]}
        state_dict = {"table1": tensor}
        for key in state_dict.keys():
            ebc._pre_load_state_dict_with_torch_tensor(key, model_shards_dtensor, None, state_dict)

    @staticmethod
    @patch("torchrec.distributed.planner.ParameterConstraints.__post_init__", return_value=None)
    @patch("torchrec.tensor_types.check", return_value=None)
    @patch("torchrec.distributed.model_parallel.check", return_value=None)
    @patch("torchrec.distributed.planner.types.check", return_value=None)
    @pytest.mark.parametrize("cnt_local_shards", [1, 2])
    def test_load_state_dict_sharded_tensor(mock1, mock2, mock3, mock4, cnt_local_shards):
        ebc = hybrid_sharded_embedding_bag_collection()
        module, _ = create_ebc()

        state_dict = dict()
        for key, val in module.state_dict().items():
            len_local_shards = len(val) // cnt_local_shards

            local_shards = [MagicMock() for _ in range(cnt_local_shards)]
            for i, local_shard in enumerate(local_shards):
                local_shard.tensor = val[i * len_local_shards:(i + 1) * len_local_shards]

            state_dict[key] = MagicMock(spec=ShardedTensor)
            state_dict[key].local_shards.return_value = local_shards

            metadata = MagicMock()
            shards_metadata = MagicMock()

            metadata.shards_metadata = [shards_metadata]
            shards_metadata.shard_sizes = list(val.shape)
            state_dict[key].metadata.return_value = metadata

        ebc.load_state_dict(state_dict, strict=False)

    @staticmethod
    @patch("torchrec.distributed.planner.ParameterConstraints.__post_init__", return_value=None)
    @patch("torchrec.tensor_types.check", return_value=None)
    @patch("torchrec.distributed.model_parallel.check", return_value=None)
    @patch("torchrec.distributed.planner.types.check", return_value=None)
    @pytest.mark.parametrize("cnt_local_shards", [1, 2])
    def test_load_state_dict_dtensor(mock1, mock2, mock3, mock4, cnt_local_shards):
        ebc = hybrid_sharded_embedding_bag_collection()
        module, _ = create_ebc()

        state_dict = dict()
        for key, val in module.state_dict().items():
            shards_wrapper = MagicMock()
            local_shards = []

            state_dict[key] = MagicMock(spec=DTensor)
            state_dict[key].to_local.return_value = shards_wrapper
            shards_wrapper.local_shards.return_value = local_shards
            shards_wrapper.local_sizes.return_value = [val.shape]

            len_local_shards = len(val) // cnt_local_shards
            for i in range(cnt_local_shards):
                local_shards.append(val[i * len_local_shards:(i + 1) * len_local_shards])

        ebc.load_state_dict(state_dict, strict=False)


@pytest.mark.parametrize("device", [torch.device("cuda:0"), torch.device("cpu"), "npu:0", "cpu"])
@pytest.mark.parametrize("check_device", [["meta", "cpu"]])
def test_device_check_func(device: Union[torch.device, str], check_device: list[str]):
    device_is_in(device, check_device)


def test_pin_and_move_cpu():
    device = torch.device("cpu")
    tensor = torch.tensor([1, 2, 3])

    result = _pin_and_move(tensor, device)

    assert result.device.type == "cpu"
    assert torch.equal(result, tensor)
    assert not result.is_pinned()  # CPU上不应被pin


@pytest.mark.parametrize("compute_device_type", ["cuda", "npu", "cpu"])
def test_sharding_types(compute_device_type):
    HybridEmbeddingBagCollectionSharder.sharding_types(None, compute_device_type)
