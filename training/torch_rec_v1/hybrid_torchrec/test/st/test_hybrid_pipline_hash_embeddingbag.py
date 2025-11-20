#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import List
import logging
import sysconfig
import pytest
from dataset import RandomRecDataset, Batch
from model import Model
from util import setup_logging
import torch
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import Adam, Adagrad, SGD

from hybrid_torchrec import HashEmbeddingBagCollection, HashEmbeddingBagConfig
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
from hybrid_torchrec.distributed.hybrid_train_pipeline import (
    HybridTrainPipelineSparseDist,
)

import torchrec
from torchrec import EmbeddingBagConfig
import torchrec.distributed
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.keyed import CombinedOptimizer

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

OPTIMIZER_PARAM = {
    Adam: dict(lr=0.02),
    Adagrad: dict(lr=0.02, eps=1.0e-8),
    SGD: dict(lr=0.02),
}

WORLD_SIZE = 2
LOOP_TIMES = 8
BATCH_NUM = 32


def execute(
    rank,
    world_size,
    table_num,
    embedding_dims,
    num_embeddings,
    pool_type,
    sharding_type,
    lookup_len,
    device,
    optim,
    keys_per_table,
):
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))
    # batch_num, lookup_lens, num_embeddings, table_num
    dataset_gloden = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num, keys_per_table=keys_per_table)
    dataset = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num, keys_per_table=keys_per_table)
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
    feature_names_all_table = []
    embedding_configs = []
    for i in range(table_num):
        feature_names = [f"feat{i}_key{j}" for j in range(keys_per_table[i])]
        ebc_config = HashEmbeddingBagConfig(
            name=f"table{i}",
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=feature_names,
            pooling=pool_type,
            init_fn=weight_init,
        )
        embedding_configs.append(ebc_config)
        feature_names_all_table.extend(feature_names)

    test_model = TestModel(rank, world_size, device, feature_names_all_table)
    golden_results = test_model.cpu_golden_loss(embedding_configs, dataset_loader_gloden, optim)
    test_results = test_model.test_loss(embedding_configs, data_loader, sharding_type, optim)
    count = 0
    for golden, result in zip(golden_results, test_results):
        batch_id = count // 2
        logging.debug("=============rank:%d, batch_id:%d==============", rank, batch_id)
        logging.debug("rank:%d, batch_id:%d, golden test %s", rank, batch_id, golden)
        logging.debug("rank:%d, batch_id:%d, result test %s", rank, batch_id, result)
        assert torch.allclose(
            golden, result, rtol=1e-04, atol=1e-04
        ), f"rank:{rank}, batch_id:{batch_id}, golden and result is not closed"
        count += 1


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.randn((1, param.shape[1])).repeat(param.shape[0], 1)
    param.data.copy_(result)


class TestModel:
    def __init__(self, rank, world_size, device, feature_names_all_table):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.pg_method = "hccl" if device == "npu" else "gloo"
        if device == "npu":
            torch_npu.npu.set_device(rank)
        self.setup(rank=rank, world_size=world_size)
        self.feature_names_all_table = feature_names_all_table

    def cpu_golden_loss(
        self, embedding_configs: List[EmbeddingBagConfig], dataloader: DataLoader[Batch], optim
    ):
        pg = dist.new_group(backend="gloo")
        table_num = len(embedding_configs)
        ebc = HashEmbeddingBagCollection(device="cpu", tables=embedding_configs)

        num_features = sum([c.num_features() for c in embedding_configs])
        ebc = Model(ebc, num_features, feature_names_all_table=self.feature_names_all_table)
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
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
        dist.init_process_group(self.pg_method, rank=rank, world_size=world_size)
        os.environ["LOCAL_RANK"] = f"{rank}"

    def test_loss(
        self,
        embedding_configs: List[EmbeddingBagConfig],
        dataloader: DataLoader[Batch],
        sharding_type: str,
        optim,
    ):
        rank, world_size = self.rank, self.world_size
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)

        table_num = len(embedding_configs)
        ebc = HashEmbeddingBagCollection(device=self.device, tables=embedding_configs)
        num_features = sum([c.num_features() for c in embedding_configs])
        ebc = Model(ebc, num_features, feature_names_all_table=self.feature_names_all_table)
        apply_optimizer_in_backward(
            optimizer_class=optim,
            params=ebc.parameters(),
            optimizer_kwargs=OPTIMIZER_PARAM[optim],
        )
        # Shard
        constrans = {
            f"table{i}": ParameterConstraints(
                sharding_types=[sharding_type], compute_kernels=["fused"]
            )
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


@pytest.mark.parametrize("table_num", [2])
@pytest.mark.parametrize("embedding_dims", [[32, 64]])
@pytest.mark.parametrize("num_embeddings", [[400, 4000]])
@pytest.mark.parametrize("pool_type", [torchrec.PoolingType.MEAN])
@pytest.mark.parametrize("sharding_type", ["row_wise"])
@pytest.mark.parametrize("lookup_len", [1024])
@pytest.mark.parametrize("device", ["npu"])
@pytest.mark.parametrize("optim", [Adagrad, SGD])
@pytest.mark.parametrize("keys_per_table", [[1, 1], [2, 1]])  # 每个table对应几个feat key
def test_hybrid_pipeline_hash_embedding_bag(
    table_num,
    embedding_dims,
    num_embeddings,
    pool_type,
    sharding_type,
    lookup_len,
    device,
    optim,
    keys_per_table,
):
    mp.spawn(
        execute,
        args=(
            WORLD_SIZE,
            table_num,
            embedding_dims,
            num_embeddings,
            pool_type,
            sharding_type,
            lookup_len,
            device,
            optim,
            keys_per_table,
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )
