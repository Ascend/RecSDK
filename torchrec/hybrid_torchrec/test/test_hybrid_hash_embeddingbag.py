#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, Adagrad
from torch.utils.data import DataLoader

from dataset import RandomRecDataset, Batch
from hybrid_torchrec import HashEmbeddingBagCollection, HashEmbeddingBagConfig
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
from model import Model
from util import setup_logging

import torchrec
import torchrec.distributed
from torchrec import (
    EmbeddingBagConfig,
)
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer

torch.ops.load_library("/home/zengxiong/20250401_torchrec/depend/torch2.6.0/mindxsdk-mxrec-add-ons-poc/torch_library/2.5.1/common/build/libfbgemm_npu_api.so")

LOOP_TIMES = 8
BATCH_NUM = 32
WORLD_SIZE = 2

OPTIMIZER_PARAM = {
    Adam: dict(lr=0.02),
    Adagrad: dict(lr=0.02, eps=1.0e-8),
}


def generate_hash_config(
    embedding_dims, num_embeddings, pool_type
) -> List[HashEmbeddingBagConfig]:
    test_table_configs: List[HashEmbeddingBagCollection] = []
    for i, (table_dim, num_embedding) in enumerate(zip(embedding_dims, num_embeddings)):
        config = HashEmbeddingBagConfig(
            name=f"table{i}",
            embedding_dim=table_dim,
            num_embeddings=num_embedding,
            feature_names=[f"feat{i}"],
            pooling=pool_type,
            init_fn=weight_init,
        )
        test_table_configs.append(config)
    return test_table_configs


def execute(
    rank,
    world_size,
    table_num,
    embedding_dims,
    num_embeddings,
    pool_type,
    sharding_type,
    lockup_len,
    device,
    optim,
):
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))
    embeding_config = generate_hash_config(embedding_dims, num_embeddings, pool_type)

    dataset = RandomRecDataset(BATCH_NUM, lockup_len, num_embeddings, table_num)
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

    gloden_results = test_model.cpu_gloden_loss(embeding_config, gloden_dataset_loader, optim)
    test_results = test_model.test_loss(embeding_config, data_loader, sharding_type, optim)
    for gloden, result in zip(gloden_results, test_results):
        logging.debug("")
        logging.debug("===========================")
        logging.debug("result test %s", gloden)
        logging.debug("gloden test %s", result)
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
        embeding_config: List[EmbeddingBagConfig], dataloader: DataLoader[Batch], optim
    ):
        pg = dist.new_group(backend="gloo")
        table_num = len(embeding_config)
        ebc = HashEmbeddingBagCollection(device="cpu", tables=embeding_config)

        num_features = sum([c.num_features() for c in embeding_config])
        ebc = Model(ebc, num_features)
        model = DDP(ebc, device_ids=None, process_group=pg)
        opt = optim(ebc.parameters(), **OPTIMIZER_PARAM[optim])

        results = []
        batch: Batch
        iter_ = iter(dataloader)
        for _ in range(LOOP_TIMES):
            batch = next(iter_)
            opt.zero_grad()
            loss, out = model(batch)
            results.append(loss.detach().cpu())
            results.append(out.detach().cpu())
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
        embeding_config: List[EmbeddingBagConfig],
        dataloader: DataLoader[Batch],
        sharding_type: str,
        optim,
    ):
        num_features = sum([c.num_features() for c in embeding_config])
        rank, world_size = self.rank, self.world_size
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
        # Shard
        table_num = len(embeding_config)
        ebc = HashEmbeddingBagCollection(device=torch.device("meta"), tables=embeding_config)
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
        batch: Batch
        iter_ = iter(dataloader)
        for _ in range(LOOP_TIMES):
            batch = next(iter_).to(self.device)
            optimizer.zero_grad()
            loss, out = ebc(batch)
            results.append(loss.detach().cpu())
            results.append(out.detach().cpu())
            loss.backward()
            optimizer.step()

        for i in range(table_num):
            logging.debug(
                "shard table%d weight %s",
                i,
                ddp_model.module.ebc.embedding_bags[f"table{i}"].weight,
            )
        return results


@pytest.mark.parametrize("table_num", [3])
@pytest.mark.parametrize("embedding_dims", [[32, 64, 128]])
@pytest.mark.parametrize("num_embeddings", [[400, 4000, 400]])
@pytest.mark.parametrize("pool_type", [torchrec.PoolingType.MEAN])
@pytest.mark.parametrize("sharding_type", ["table_wise", "row_wise"])
@pytest.mark.parametrize("lockup_len", [1024])
@pytest.mark.parametrize("device", ["npu"])
@pytest.mark.parametrize("optim", [Adagrad])
def test_hstu_dens_normal(
    table_num,
    embedding_dims,
    num_embeddings,
    pool_type,
    sharding_type,
    lockup_len,
    device,
    optim,
):
    if device == "cpu" and (sharding_type == "row_wise" or optim == Adam):
        return
    mp.spawn(
        execute,
        args=(
            WORLD_SIZE,
            table_num,
            embedding_dims,
            num_embeddings,
            pool_type,
            sharding_type,
            lockup_len,
            device,
            optim,
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )
