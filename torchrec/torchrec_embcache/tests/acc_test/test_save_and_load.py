#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import logging
import multiprocessing
import os
from dataclasses import dataclass
from typing import List

import pytest
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset import RandomRecDataset, Batch
from model import Model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchrec_embcache.distributed.embedding_bag import (
    EmbCacheEmbeddingBagCollection,
)
from torchrec_embcache.distributed.sharding.embedding_sharder import (
    EmbCacheEmbeddingBagCollectionSharder,
)
from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist
from torchrec_embcache.saver import Saver
from util import setup_logging

import torchrec
from torchrec import EmbeddingBagConfig, EmbeddingBagCollection
import torchrec.distributed
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.keyed import CombinedOptimizer


_world_size_env = int(os.getenv('WORLD_SIZE', '2'))
WORLD_SIZE = _world_size_env
LOOP_TIMES = 500
BATCH_NUM = 1000


@dataclass
class ExecuteConfig:
    world_size: int
    table_num: int
    embedding_dims: List[int]
    num_embeddings: List[int]
    pool_type: torchrec.PoolingType
    sharding_type: str
    lookup_len: int
    device: str


def execute(rank: int, config: ExecuteConfig):
    world_size = config.world_size
    table_num = config.table_num
    embedding_dims = config.embedding_dims
    num_embeddings = config.num_embeddings
    pool_type = config.pool_type
    sharding_type = config.sharding_type
    lookup_len = config.lookup_len
    device = config.device
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))
    dataset_gloden = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num)
    dataset = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num)
    dataset_loader_gloden = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )
    embeding_config = []
    for i in range(table_num):
        ebc_config = EmbeddingBagConfig(
            name=f"table{i}",
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[f"feat{i}"],
            pooling=pool_type,
            init_fn=weight_init,
            weight_init_min=0.0,
            weight_init_max=1.0,
        )
        embeding_config.append(ebc_config)

    test_model = TestModel(rank, world_size, device)
    gloden_results = test_model.test_loss(
        embeding_config, dataset_loader_gloden, sharding_type, training=True
    )
    test_results = test_model.test_loss(
        embeding_config, data_loader, sharding_type, training=False
    )
    i = 0
    for gloden, result in zip(gloden_results, test_results):
        logging.debug("")
        logging.debug("==============batch %d================", i // 2)
        logging.debug("result test %s", result)
        logging.debug("gloden test %s", gloden)
        i += 1
        assert torch.allclose(
            gloden, result, rtol=1e-04, atol=1e-04
        ), "gloden and result is not closed"


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = (
        torch.linspace(0, 1, steps=param.shape[1])
        .unsqueeze(0)
        .repeat(param.shape[0], 1)
    )
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
        embeding_config: List[EmbeddingBagConfig], dataloader: DataLoader[Batch]
    ):
        pg = dist.new_group(backend="gloo")
        table_num = len(embeding_config)
        ebc = EmbeddingBagCollection(device=torch.device("cpu"), tables=embeding_config)

        num_features = sum([c.num_features() for c in embeding_config])
        ebc = Model(ebc, num_features)
        model = DDP(ebc, device_ids=None, process_group=pg)

        opt = torch.optim.Adagrad(model.parameters(), lr=0.02, eps=1e-8)
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
        embeding_config: List[EmbeddingBagConfig],
        dataloader: DataLoader[Batch],
        sharding_type: str,
        training: bool,
    ):
        rank, world_size = self.rank, self.world_size
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)

        table_num = len(embeding_config)
        ebc = EmbCacheEmbeddingBagCollection(
            device=torch.device("meta"),
            tables=embeding_config,
            batch_size=2,
            multi_hot_sizes=[1] * table_num,
            world_size=dist.get_world_size(),
        )
        num_features = sum([c.num_features() for c in embeding_config])
        ebc = Model(ebc, num_features)
        apply_optimizer_in_backward(
            optimizer_class=torch.optim.Adagrad,
            params=ebc.parameters(),
            optimizer_kwargs={"lr": 0.02},
        )
        # Shard
        constrans = {
            f"table{i}": ParameterConstraints(sharding_types=[sharding_type])
            for i in range(table_num)
        }
        rank = int(os.environ["LOCAL_RANK"])
        npu_device: torch.device = torch.device(f"npu:{rank}")
        cpu_device = torch.device("cpu")
        cpu_pg = dist.new_group(backend="gloo")
        cpu_env = ShardingEnv.from_process_group(cpu_pg)
        hash_shader = EmbCacheEmbeddingBagCollectionSharder(
            cpu_device=cpu_device,
            cpu_env=cpu_env,
            npu_device=npu_device,
            npu_env=ShardingEnv.from_process_group(dist.GroupMember.WORLD),
        )
        shaders = [hash_shader]
        planner = EmbeddingShardingPlanner(
            topology=Topology(world_size=self.world_size, compute_device=self.device),
            constraints=constrans,
        )
        plan = planner.collective_plan(ebc, shaders, dist.GroupMember.WORLD)
        if self.rank == 0:
            logging.debug(plan)

        ddp_model = torchrec.distributed.DistributedModelParallel(
            ebc,
            sharders=shaders,
            device=npu_device,
            plan=plan,
        )

        logging.debug(ddp_model)
        # Optimizer
        optimizer = CombinedOptimizer([ddp_model.fused_optimizer])
        results = []
        iter_ = iter(dataloader)
        ddp_model.train()
        pipe = EmbCacheTrainPipelineSparseDist(
            ddp_model,
            optimizer=optimizer,
            cpu_device=cpu_device,
            npu_device=npu_device,
            return_loss=True,
        )
        save_dir = os.path.abspath("save_dir")
        if training and os.path.exists(save_dir):
            os.system(f"rm -rf {save_dir}")
        if training:
            os.mkdir(save_dir)
        saver = Saver(rank=rank)
        if training:
            for _ in range(LOOP_TIMES):
                out, loss = pipe.progress(iter_)

            ddp_model.eval()
            for _ in range(LOOP_TIMES):
                out, loss = pipe.progress(iter_)
                results.append(loss.detach().cpu())
                results.append(out.detach().cpu())
            logging.info("ddp_model.state_dict %s", ddp_model.state_dict())
            # dense-保存
            torch.save(ddp_model.state_dict(), f"save_dir/model_{rank}.pt")
            torch.save(optimizer.state_dict(), f"save_dir/optimizer_{rank}.pt")

            saver.save(ddp_model, "save_dir/sparse")

        else:
            ddp_state_dict = torch.load(f"save_dir/model_{rank}.pt", weights_only=False)
            logging.info("ddp_state_dict %s", ddp_state_dict)
            ddp_model.load_state_dict(ddp_state_dict)
            optimizer.load_state_dict(
                torch.load(f"save_dir/optimizer_{rank}.pt", weights_only=False)
            )

            # sparse-加载
            saver.load(ddp_model, "save_dir/sparse")
            ddp_model.eval()
            for _ in range(LOOP_TIMES):
                out, loss = pipe.progress(iter_)
            for _ in range(LOOP_TIMES):
                out, loss = pipe.progress(iter_)
                results.append(loss.detach().cpu())
                results.append(out.detach().cpu())

        return results


params = {
    "world_size": [WORLD_SIZE],
    "table_num": [2],
    "embedding_dims": [[128, 128]],
    "num_embeddings": [[4000, 400]],
    "pool_type": [torchrec.PoolingType.SUM],
    "sharding_type": ["row_wise"],
    "lookup_len": [128],  # batchsize
    "device": ["npu"],
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_hstu_dens_normal(config: ExecuteConfig):
    mp.spawn(
        execute,
        args=(config,),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    test_hstu_dens_normal(ExecuteConfig(
        world_size=WORLD_SIZE,
        table_num=2,
        embedding_dims=[128, 128],
        num_embeddings=[4000, 400],
        pool_type=torchrec.PoolingType.SUM,
        sharding_type="row_wise",
        lookup_len=128,
        device="npu",
    ))
