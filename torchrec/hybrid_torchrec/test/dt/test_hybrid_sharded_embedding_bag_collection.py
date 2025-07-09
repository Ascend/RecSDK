#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Union
from unittest.mock import patch

import pytest
import torch
from hybrid_torchrec.distributed import get_default_hybrid_sharders
from hybrid_torchrec.distributed.embeddingbag import (
    HybridShardedEmbeddingBagCollection,
    KJTList,
    device_is_in,
    _pin_and_move
)
from torchrec import KeyedJaggedTensor, KeyedTensor
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology, ParameterConstraints
from torchrec.distributed.types import ShardingEnv
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection


DEVICE = torch.device("cpu")


def set_env():
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


def create_ebc():
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


def create_planner():
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


def hybrid_sharded_embedding_bag_collection():
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


@patch("torchrec.distributed.planner.ParameterConstraints.__post_init__", return_value=None)
@patch("torchrec.tensor_types.check", return_value=None)
@patch("torchrec.distributed.model_parallel.check", return_value=None)
@patch("torchrec.distributed.planner.types.check", return_value=None)
def test_hybrid_sharded_embedding_bag_collection_init(*mock):
    ebc = hybrid_sharded_embedding_bag_collection()


@patch("torchrec.distributed.planner.ParameterConstraints.__post_init__", return_value=None)
@patch("torchrec.tensor_types.check", return_value=None)
@patch("torchrec.distributed.model_parallel.check", return_value=None)
@patch("torchrec.distributed.planner.types.check", return_value=None)
def test_hybrid_sharded_embedding_bag_collection_input_dist(*mock):
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


@patch("torchrec.distributed.planner.ParameterConstraints.__post_init__", return_value=None)
@patch("torchrec.tensor_types.check", return_value=None)
@patch("torchrec.distributed.model_parallel.check", return_value=None)
@patch("torchrec.distributed.planner.types.check", return_value=None)
def test_hybrid_sharded_embedding_bag_collection_post_input_dist(*mock):
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


@patch("torchrec.distributed.planner.ParameterConstraints.__post_init__", return_value=None)
@patch("torchrec.tensor_types.check", return_value=None)
@patch("torchrec.distributed.model_parallel.check", return_value=None)
@patch("torchrec.distributed.planner.types.check", return_value=None)
@patch("torchrec.distributed.embeddingbag.construct_output_kt",
       return_value=KeyedTensor(["feature1", "feature2"], [2, 2], torch.tensor((1, 2, 3, 4))))
def test_hybrid_sharded_embedding_bag_collection_output_dist(*mock):
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


@patch("torchrec.distributed.planner.ParameterConstraints.__post_init__", return_value=None)
@patch("torchrec.tensor_types.check", return_value=None)
@patch("torchrec.distributed.model_parallel.check", return_value=None)
@patch("torchrec.distributed.planner.types.check", return_value=None)
@patch("torchrec.distributed.embeddingbag.construct_output_kt",
       return_value=KeyedTensor(["feature1", "feature2"], [2, 2], torch.tensor((1, 2, 3, 4))))
def test_hybrid_sharded_embedding_bag_collection_compute_and_output_dist(*mock):
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
