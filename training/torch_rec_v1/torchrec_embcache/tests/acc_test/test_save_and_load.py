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
import shutil
from dataclasses import dataclass
from typing import List

import pytest
import torch
import torch_npu
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
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
from torchrec_embcache.utils import safe_makedirs
import torchrec
from torchrec import EmbeddingBagConfig, EmbeddingBagCollection
import torchrec.distributed
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FDSP,
    StateDictType
)
from torchrec.distributed.types import ShardingEnv
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer

from util import setup_logging

WORLD_SIZE = 2
LOOP_TIMES = 500
BATCH_NUM = LOOP_TIMES * 2  # will execute LOOP_TIMES*2 times lookup when save load


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
    dataset = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num)
    dataset_loader_golden = DataLoader(
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
    embedding_config = []
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
        embedding_config.append(ebc_config)

    test_model = TestModel(rank, world_size, device)
    golden_results, _ = test_model.test_loss(
        embedding_config, dataset_loader_golden, sharding_type, training=True
    )
    test_results, _ = test_model.test_loss(
        embedding_config, data_loader, sharding_type, training=False
    )
    i = 0
    for golden, result in zip(golden_results, test_results):
        logging.debug("==============batch %d================", i // 2)
        logging.debug("result test %s", result)
        logging.debug("golden test %s", golden)
        i += 1
        assert torch.allclose(
            golden, result, rtol=1e-04, atol=1e-04
        ), "golden and result is not closed"


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
        training: bool,
    ):
        rank, world_size = self.rank, self.world_size

        table_num = len(embedding_config)
        ebc = EmbCacheEmbeddingBagCollection(
            device=torch.device("meta"),
            tables=embedding_config,
            batch_size=2,
            multi_hot_sizes=[1] * table_num,
            world_size=dist.get_world_size(),
        )
        num_features = sum([c.num_features() for c in embedding_config])
        ebc = Model(ebc, num_features)
        apply_optimizer_in_backward(
            optimizer_class=torch.optim.Adagrad,
            params=ebc.parameters(),
            optimizer_kwargs={"lr": 0.02},
        )
        # Shard
        constrains = {
            f"table{i}": ParameterConstraints(sharding_types=[sharding_type], compute_kernels=["fused"])
            for i in range(table_num)
        }
        cpu_device = torch.device("cpu")
        npu_device: torch.device = torch.device("npu")
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
            constraints=constrains,
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
        iter_ = iter(dataloader.dataset)
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
            shutil.rmtree(save_dir, ignore_errors=True)
        if training:
            safe_makedirs(save_dir)

        saver = Saver(rank=rank)
        state_dict = {
            "model": ddp_model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        if training:
            for _ in range(LOOP_TIMES):
                _, _ = pipe.progress(iter_)

            ddp_model.eval()
            for _ in range(LOOP_TIMES):
                out, loss = pipe.progress(iter_)
                results.append(loss.detach().cpu())
                results.append(out.detach().cpu())
            logging.info("ddp_model.state_dict %s", ddp_model.state_dict())
            # save dense
            dcp.save(
                state_dict=state_dict,
                checkpoint_id="save_dir/dense"
            )
            # save sparse
            saver.save(ddp_model, "save_dir/sparse")

        else:
            # load dense
            dcp.load(
                state_dict=state_dict,
                checkpoint_id="save_dir/dense"
            )
            # load sparse
            saver.load(ddp_model, "save_dir/sparse")

            ddp_model.eval()
            for _ in range(LOOP_TIMES):
                _, _ = pipe.progress(iter_)
            for _ in range(LOOP_TIMES):
                out, loss = pipe.progress(iter_)
                results.append(loss.detach().cpu())
                results.append(out.detach().cpu())

        # Must return ddp_model, it is necessary to maintain the reference count about static ThreadPool in C++ code,
        # to facilitate the use of subsequent tasks.
        return results, ddp_model


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
