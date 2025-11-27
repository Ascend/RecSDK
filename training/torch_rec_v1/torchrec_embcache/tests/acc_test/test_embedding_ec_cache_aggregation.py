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
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.multiprocessing.spawn import ProcessRaisedException
from torch.optim import Adagrad, SGD, SparseAdam
from dataset import RandomRecDataset, Batch
from model import ModelEc
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchrec_embcache.distributed.embedding import EmbCacheEmbeddingCollection
from torchrec_embcache.distributed.configs import AdmitAndEvictConfig, EmbCacheEmbeddingConfig
from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist
from torchrec_embcache.distributed.sharding.embedding_sharder import EmbCacheEmbeddingCollectionSharder
from util import setup_logging

import torchrec
import torchrec.distributed
from torchrec import EmbeddingCollection
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer
from torchrec.optim import AccumulateAdagrad, AccumulateAdam, AccumulateSGD

WORLD_SIZE = 2
LOOP_TIMES = 20
ACCUMULATE_STEP = 5
BATCH_NUM = 1000

OPTIMIZER_PARAM = {
    # 注: Rec SDK Torch中Adam优化器融合算子使用的更新算法为sparse 更新，和SparseAdam算法对应,和torch原生Adam优化器更新算法有差异
    Adagrad: dict(lr=0.02, eps=1.0e-8),
    SGD: dict(lr=0.02),
    SparseAdam: dict(lr=0.02),
}


@dataclass
class ExecuteConfig:
    world_size: int
    table_num: int
    embedding_dims: List[int]
    num_embeddings: List[int]
    sharding_type: str
    optim: torch.optim
    lookup_len: int
    device: str
    admit_threshold: float
    err_pattern: str = ''


def execute(rank: int, config: ExecuteConfig):
    world_size = config.world_size
    embedding_dims = config.embedding_dims
    num_embeddings = config.num_embeddings
    sharding_type = config.sharding_type
    optim = config.optim
    lookup_len = config.lookup_len
    device = config.device
    table_num = config.table_num
    admit_threshold = config.admit_threshold
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))
    embedding_configs = []
    for i in range(table_num):
        admit_and_evict_config = AdmitAndEvictConfig(admit_threshold=admit_threshold, not_admitted_default_value=0.99)
        ec_config = EmbCacheEmbeddingConfig(
            name=f"table{i}",
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[f"feat{i}"],
            init_fn=weight_init,
            weight_init_min=0.0,
            weight_init_max=1.0,
            admit_and_evict_config=admit_and_evict_config
        )
        embedding_configs.append(ec_config)

    dataset_golden = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num)
    dataset = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num)
    dataset_loader_golden = DataLoader(
        dataset_golden,
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
    test_model = TestModel(rank, world_size, device)
    golden_results = test_model.cpu_golden_loss(embedding_configs, dataset_loader_golden, optim)
    test_results = test_model.test_loss(embedding_configs, data_loader, sharding_type, optim)
    i = 0
    for golden, result in zip(golden_results, test_results):
        logging.debug("")
        logging.debug("==============batch %d================", i // 2)
        logging.debug("result test %s", result)
        logging.debug("golden test %s", golden)
        i += 1
        enable_feature_admit = any(emb_config.admit_and_evict_config.is_feature_admit_enabled()
                                   for emb_config in embedding_configs)
        if enable_feature_admit:
            continue
        assert torch.allclose(
            golden, result, rtol=1e-04, atol=1e-04
        ), "golden and result is not closed"
        logging.debug("golden and result is closed")


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.linspace(0, 1, steps=param.shape[1]).unsqueeze(0).repeat(param.shape[0], 1)
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
    def cpu_golden_loss(
        embeding_config: List[EmbCacheEmbeddingConfig], dataloader: DataLoader[Batch], optim
    ):
        pg = dist.new_group(backend="gloo")
        table_num = len(embeding_config)
        ec = EmbeddingCollection(device=torch.device("cpu"), tables=embeding_config)

        if optim == SparseAdam:
            for config in embeding_config:
                ec.embeddings[config.name].sparse = True

        num_features = sum([c.num_features() for c in embeding_config])
        ec = ModelEc(ec, num_features)
        model = DDP(ec, device_ids=None, process_group=pg)

        opt = optim(model.parameters(), **OPTIMIZER_PARAM[optim])

        results = []
        batch: Batch
        iter_ = iter(dataloader)
        opt.zero_grad()
        for step in range(LOOP_TIMES):
            batch = next(iter_)
            loss, output = model(batch)
            results.append(loss.detach().cpu())
            results.append(output.detach().cpu())
            loss.backward()
            if (step + 1) % ACCUMULATE_STEP == 0:
                opt.step()
                opt.zero_grad()

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
        os.environ["LOCAL_RANK"] = f"{rank}"

    def test_loss(
        self,
        embedding_config: List[EmbCacheEmbeddingConfig],
        dataloader: DataLoader[Batch],
        sharding_type: str,
        optim,
    ):
        rank, world_size = self.rank, self.world_size
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)

        table_num = len(embedding_config)
        ec = EmbCacheEmbeddingCollection(device=torch.device("meta"), tables=embedding_config,
                                         batch_size=2, multi_hot_sizes=[1] * table_num,
                                         world_size=dist.get_world_size())
        num_features = sum([c.num_features() for c in embedding_config])
        ec = ModelEc(ec, num_features)

        # 优化器映射到梯度累积优化器
        if optim == Adagrad:
            accumulate_optim = AccumulateAdagrad
        elif optim == SGD:
            accumulate_optim = AccumulateSGD
        elif optim == SparseAdam:
            accumulate_optim = AccumulateAdam

        apply_optimizer_in_backward(
            optimizer_class=accumulate_optim,
            params=ec.parameters(),
            optimizer_kwargs={**OPTIMIZER_PARAM[optim], "use_accumulate": True, "accumulate_step": ACCUMULATE_STEP},
        )
        # Shard
        constrans = {
            f"table{i}": ParameterConstraints(sharding_types=[sharding_type], compute_kernels=["fused"])
            for i in range(table_num)
        }
        rank = int(os.environ["LOCAL_RANK"])
        npu_device: torch.device = torch.device(f"npu:{rank}")
        cpu_device = torch.device("cpu")
        cpu_pg = dist.new_group(backend="gloo")
        cpu_env = ShardingEnv.from_process_group(cpu_pg)
        hash_shader = EmbCacheEmbeddingCollectionSharder(
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
        plan = planner.collective_plan(
            ec, shaders, dist.GroupMember.WORLD
        )
        if self.rank == 0:
            logging.debug(plan)

        ddp_model = torchrec.distributed.DistributedModelParallel(
            ec,
            sharders=shaders,
            device=npu_device,
            plan=plan,
        )

        logging.debug(ddp_model)
        # Optimizer
        optimizer = CombinedOptimizer([ddp_model.fused_optimizer])
        results = []
        batch: Batch
        iter_ = iter(dataloader)
        ddp_model.train()

        optimizer.zero_grad()
        for step in range(LOOP_TIMES):
            batch = next(iter_).to(self.device)
            loss, out = ec(batch)
            results.append(loss.detach().cpu())
            results.append(out.detach().cpu())
            loss.backward()
            if (step + 1) % ACCUMULATE_STEP == 0:
                optimizer.step()
                optimizer.zero_grad()

        return results


params = {
    "world_size": [WORLD_SIZE],
    "table_num": [2],
    "embedding_dims": [[128, 128]],
    "num_embeddings": [[4000, 400]],
    "sharding_type": ["row_wise"],
    "optim": [Adagrad, SGD, SparseAdam],
    "lookup_len": [128],  # batchsize
    "device": ["npu"],
    "admit_threshold": [-1], 
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
