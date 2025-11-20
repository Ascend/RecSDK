#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import random
import pytest

import hybrid_torchrec
import torch
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
import torchrec
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

from util import setup_logging

WORLD_SIZE = 2


def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "6000"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    os.environ["LOCAL_RANK"] = f"{rank}"


def lookup(rank, world_size, num_embeddings):
    torch.npu.set_device(rank)
    setup(rank, world_size)
    device = torch.device("npu")
    logging.info("world_size: %d, rank: %d", world_size, rank)

    ddp_ebc = _get_dmp_model(device, rank, num_embeddings, world_size)

    input_data = [4096, 666, 334, 9999, 4096, 666]
    random.shuffle(input_data)
    logging.info(f"shuffle rank:%d, data: %s: ", rank, input_data)
    kjt = KeyedJaggedTensor.from_jt_dict(
        {
            "product": JaggedTensor(
                values=torch.tensor(input_data, device=device),
                lengths=torch.tensor([1, 1, 1, 1, 1, 1], device=device)
            )
        }
    )
    ret = ddp_ebc(kjt)
    ret = ret.wait().values()[0].to("cpu")
    logging.info("Call ddp EmbeddingBagCollection Forward rank: %d, ret: %s", rank, ret)


def _get_dmp_model(device, rank, num_embeddings, world_size):
    host_gp = dist.new_group(backend="gloo")
    host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
    ebc = hybrid_torchrec.HashEmbeddingBagCollection(
        device=device,
        tables=[
            hybrid_torchrec.HashEmbeddingBagConfig(
                name="product_table",
                embedding_dim=8,
                num_embeddings=num_embeddings,
                feature_names=["product"],
                pooling=torchrec.PoolingType.SUM,
            )
        ],
    )
    apply_optimizer_in_backward(
        optimizer_class=torch.optim.Adam,
        params=ebc.parameters(),
        optimizer_kwargs=dict(lr=0.02),
    )
    constrains = {
        f"table{i}": ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"]) for i in range(1)
    }
    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size=world_size, compute_device="npu"),
        constraints=constrains,
    )
    plan = planner.collective_plan(ebc, get_default_hybrid_sharders(host_env), dist.GroupMember.WORLD)
    ddp_ebc = torchrec.distributed.DistributedModelParallel(
        ebc,
        sharders=get_default_hybrid_sharders(host_env),
        device=torch.device("npu"),
        plan=plan,
    )
    return ddp_ebc


def exe_slice_table(rank, num_embeddings, expect_error):
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))
    if expect_error:
        with pytest.raises(RuntimeError):
            lookup(rank, WORLD_SIZE, num_embeddings)
    else:
        lookup(rank, WORLD_SIZE, num_embeddings)


@pytest.mark.parametrize("num_embeddings, expect_error", [(4, True), (6, False)])
def test_hybrid_save_and_load(num_embeddings, expect_error):
    mp.spawn(
        exe_slice_table,
        args=(
            num_embeddings,
            expect_error
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )

