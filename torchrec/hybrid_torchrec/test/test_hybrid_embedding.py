#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import logging
import os
from dataclasses import dataclass
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from dataset import RandomRecDataset, Batch
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
from model import Model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from util import setup_logging

import torchrec
import torchrec.distributed
import torchrec.distributed.shard
from torchrec import (
    EmbeddingConfig,
    EmbeddingCollection,
)
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.keyed import CombinedOptimizer


LOOP_TIMES = 8
BATCH_NUM = 32
WORLD_SIZE = 2


def generate_base_config(embedding_dims: List[int],
                         num_embeddings: List[int]) -> List[EmbeddingConfig]:
    test_table_configs: List[EmbeddingConfig] = []
    for i, (table_dim, num_embedding) in enumerate(zip(embedding_dims, num_embeddings)):
        config = EmbeddingConfig(
            name=f"table{i}",
            embedding_dim=table_dim,
            num_embeddings=num_embedding,
            feature_names=[f"feat{i}"],
            init_fn=weight_init,
        )
        test_table_configs.append(config)
    return test_table_configs


@dataclass
class ExecuteConfig:
    world_size: int
    table_num: int
    embedding_dims: List[int]
    num_embeddings: List[int]
    sharding_type: str
    lookup_len: int
    device: str


def execute(rank: int, config: ExecuteConfig):
    world_size = config.world_size
    table_num = config.table_num
    embedding_dims = config.embedding_dims
    num_embeddings = config.num_embeddings
    sharding_type = config.sharding_type
    lookup_len = config.lookup_len
    device = config.device
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))
    embedding_config = generate_base_config(embedding_dims, num_embeddings)

    dataset = RandomRecDataset(
        batch_size=BATCH_NUM,
        lookup_len=lookup_len,
        num_lookups=[num_embedding // 2 for num_embedding in num_embeddings],
        num_tables=table_num,
    )
    gloden_dataset_loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )

    test_model = TestModel(rank, world_size, device)

    gloden_results = test_model.cpu_gloden_loss(
        embedding_config, gloden_dataset_loader, sharding_type
    )
    test_results = test_model.test_loss(embedding_config, data_loader, sharding_type)
    for gloden, result in zip(gloden_results, test_results):
        logging.debug("")
        logging.debug("===========================")
        logging.debug("result test %s", result)
        logging.debug("gloden test %s", gloden)
        assert torch.allclose(
            gloden, result, rtol=1e-04, atol=1e-04
        ), "gloden and result is not closed"


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.randn(1, param.shape[1]).repeat(param.shape[0], 1)
    param.data.copy_(result)


class TestModel:
    def __init__(self, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.pg_method = "hccl" if device == "npu" else "gloo"
        if device == "npu":
            torch_npu.npu.set_device(rank)
        self.setup(rank=rank, world_size=world_size)

    @staticmethod
    def cpu_gloden_loss(
        embedding_config: List[EmbeddingConfig],
        dataloader: DataLoader[Batch],
        sharding_type: str,
    ):
        pg = dist.new_group(backend="gloo")
        table_num = len(embedding_config)
        ec = EmbeddingCollection(device="cpu", tables=embedding_config)

        num_features = sum([c.num_features() for c in embedding_config])
        ec = Model(ec, num_features)
        model = DDP(ec, device_ids=None, process_group=pg)

        opt = torch.optim.Adagrad(ec.parameters(), lr=0.02, eps=1e-8)
        results = []
        batch: Batch
        iter_ = iter(dataloader)
        for _ in range(LOOP_TIMES):
            batch = next(iter_)
            opt.zero_grad()
            loss, output = model(batch)
            results.append(loss.detach().cpu())
            results.append(output.detach().cpu())
            loss.backward()
            opt.step()

        for i in range(table_num):
            logging.debug(
                "single table%d weight %s",
                i,
                ec.ec.embeddings[f"table{i}"].weight,
            )
        return results

    def setup(self, rank: int, world_size: int):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "6000"
        dist.init_process_group(self.pg_method, rank=rank, world_size=world_size)

    def test_loss(
        self,
        embedding_config: List[EmbeddingConfig],
        dataloader: DataLoader[Batch],
        sharding_type: str,
    ):
        num_features = sum([c.num_features() for c in embedding_config])
        rank, world_size = self.rank, self.world_size
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
        # Shard
        table_num = len(embedding_config)
        ec = EmbeddingCollection(device="meta", tables=embedding_config)
        ec = Model(ec, num_features)
        apply_optimizer_in_backward(
            optimizer_class=torch.optim.Adagrad,
            params=ec.parameters(),
            optimizer_kwargs={"lr": 0.02},
        )
        # Shard
        constrans = {
            f"table{i}": ParameterConstraints(sharding_types=[sharding_type])
            for i in range(table_num)
        }
        planner = EmbeddingShardingPlanner(
            topology=Topology(world_size=self.world_size, compute_device=self.device),
            constraints=constrans,
        )
        plan = planner.collective_plan(
            ec, get_default_hybrid_sharders(host_env), dist.GroupMember.WORLD
        )
        if self.rank == 0:
            logging.debug(plan)

        ddp_model = torchrec.distributed.DistributedModelParallel(
            ec,
            sharders=get_default_hybrid_sharders(host_env),
            device=torch.device(self.device),
            plan=plan,
        )
        logging.debug(ddp_model)
        # Optimizer
        optimizer = CombinedOptimizer([ddp_model.fused_optimizer])
        results = []
        batch: Batch
        iter_ = iter(dataloader)
        for _ in range(LOOP_TIMES):
            batch = next(iter_).to(self.device)
            optimizer.zero_grad()
            loss, output = ec(batch)
            results.append(loss.detach().cpu())
            results.append(output.detach().cpu())
            loss.backward()
            optimizer.step()

        for i in range(table_num):
            logging.debug(
                "shard table%d weight %s",
                i,
                ddp_model.module.ec.embeddings[f"table{i}"].weight,
            )
        return results


params = {
    "table_num": [3],
    "embedding_dims": [[32, 32, 32]],
    "num_embeddings": [[400, 4000, 400]],
    "sharding_type": ["table_wise", "row_wise"],
    "lookup_len": [1024],
    "device": ["npu"]
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_hybrid_embedding(config: ExecuteConfig):
    if config.device == "cpu" and config.sharding_type == "row_wise":
        return
    mp.spawn(
        execute,
        args=(config,),
        nprocs=WORLD_SIZE,
        join=True,
    )
