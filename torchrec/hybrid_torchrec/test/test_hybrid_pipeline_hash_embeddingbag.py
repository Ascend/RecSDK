#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import logging
import sysconfig
import os
from dataclasses import dataclass
from typing import List

import pytest
import torch_npu
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, Adagrad
from torch.utils.data import DataLoader
from dataset import RandomRecDataset, Batch
from hybrid_torchrec import HashEmbeddingBagCollection, HashEmbeddingBagConfig
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
from hybrid_torchrec.distributed.hybrid_train_pipeline import (
    HybridTrainPipelineSparseDist,
)
from model import Model
from util import setup_logging

import torchrec
from torchrec import EmbeddingBagConfig, EmbeddingBagCollection
import torchrec.distributed
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionAwaitable
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer


torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

OPTIMIZER_PARAM = {
    Adam: dict(lr=0.02),
    Adagrad: dict(lr=0.02, eps=1.0e-8),
}

WORLD_SIZE = 2
LOOP_TIMES = 8
BATCH_NUM = 32


@dataclass
class ExecuteConfig:
    world_size: int
    table_num: int
    embedding_dims: List[int]
    num_embeddings: List[int]
    sharding_type: str
    lookup_len: int
    device: str
    optim: type


def execute(rank: int, config: ExecuteConfig):
    world_size = config.world_size
    table_num = config.table_num
    embedding_dims = config.embedding_dims
    num_embeddings = config.num_embeddings
    sharding_type = config.sharding_type
    lookup_len = config.lookup_len
    device = config.device
    optim = config.optim
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))
    dataset_gloden = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num)
    dataset = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num)
    dataset_loader_gloden = DataLoader(
        dataset_gloden,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )
    embedding_config = []
    for i in range(table_num):
        ebc_config = HashEmbeddingBagConfig(
            name=f"table{i}",
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[f"feat{i}"],
            pooling=pool_type,
            init_fn=weight_init,
        )
        embedding_config.append(ebc_config)

    test_model = TestModel(rank, world_size, device)
    gloden_results = test_model.cpu_gloden_loss(embedding_config, dataset_loader_gloden, optim)
    test_results = test_model.test_loss(embedding_config, data_loader, sharding_type, optim)
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
    result = torch.randn((1, param.shape[1])).repeat(param.shape[0], 1)
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
        embedding_config: List[EmbeddingBagConfig], dataloader: DataLoader[Batch], optim
    ):
        pg = dist.new_group(backend="gloo")
        table_num = len(embedding_config)
        ebc = HashEmbeddingBagCollection(device="cpu", tables=embedding_config)

        num_features = sum([c.num_features() for c in embedding_config])
        ebc = Model(ebc, num_features)
        model = DDP(ebc, device_ids=None, process_group=pg)

        opt = optim(ebc.parameters(), **OPTIMIZER_PARAM[optim])
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
                ebc.ebc.embedding_bags[f"table{i}"].weight,
            )
        return results

    def setup(self, rank: int, world_size: int):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "6000"
        dist.init_process_group(self.pg_method, rank=rank, world_size=world_size)
        os.environ["LOCAL_RANK"] = f"{rank}"

    def test_loss(
        self,
        embedding_config: List[EmbeddingBagConfig],
        dataloader: DataLoader[Batch],
        sharding_type: str,
        optim,
    ):
        rank, world_size = self.rank, self.world_size
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)

        table_num = len(embedding_config)
        ebc = HashEmbeddingBagCollection(device=self.device, tables=embedding_config)
        num_features = sum([c.num_features() for c in embedding_config])
        ebc = Model(ebc, num_features)
        apply_optimizer_in_backward(
            optimizer_class=optim,
            params=ebc.parameters(),
            optimizer_kwargs=OPTIMIZER_PARAM[optim],
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
            ebc, get_default_hybrid_sharders(host_env), dist.GroupMember.WORLD
        )
        if self.rank == 0:
            logging.debug(plan)

        ddp_model = torchrec.distributed.DistributedModelParallel(
            ebc,
            sharders=get_default_hybrid_sharders(host_env),
            device=torch.device(self.device),
            plan=plan,
        )
        logging.debug(ddp_model)
        # Optimizer
        optimizer = CombinedOptimizer([ddp_model.fused_optimizer])
        results = []
        iter_ = iter(dataloader)
        ddp_model.train()
        pipe = HybridTrainPipelineSparseDist(
            ddp_model,
            optimizer=optimizer,
            device=torch.device(self.device),
            return_loss=True,
        )
        for _ in range(LOOP_TIMES):
            out, loss = pipe.progress(iter_)
            results.append(loss.detach().cpu())
            results.append(out.detach().cpu())

        for i in range(table_num):
            logging.debug(
                "shard table%d weight %s",
                i,
                ddp_model.module.ebc.embedding_bags[f"table{i}"].weight,
            )
        return results


params = {
    "table_num": [2],
    "embedding_dims": [[32, 64, 128]],
    "num_embeddings": [[400, 4000, 400]],
    "pool_type": [torchrec.PoolingType.MEAN],
    "sharding_type": ["table_wise", "row_wise"],
    "lookup_len": [1024],
    "device": ["cpu", "npu"],
    "optim": [Adagrad]
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_hybrid_pipeline_hash_embedding_bag(config: ExecuteConfig):
    if config.device == "cpu" and (config.sharding_type == "row_wise" or config.optim == Adam):
        return
    mp.spawn(
        execute,
        args=(config,),
        nprocs=WORLD_SIZE,
        join=True,
    )
