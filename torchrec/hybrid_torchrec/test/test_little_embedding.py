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
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from dataset import RandomRecDatasetUnique, Batch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, Adagrad
from torch.utils.data import DataLoader
from hybrid_torchrec import HashEmbeddingBagCollection, HashEmbeddingBagConfig
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
from hybrid_torchrec.modules.little_embedding import (
    HashEmbeddingModuleCollection,
    EmbeddingConfig,
    OptimizerArgs,
    OptimType,
)
from model import Model
from util import setup_logging

import torchrec
import torchrec.distributed
from torchrec import (
    EmbeddingBagConfig,
    EmbeddingBagCollection,
    JaggedTensor,
    KeyedJaggedTensor,
)
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.keyed import CombinedOptimizer


torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

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
    optim: type


def execute(rank: int, config: ExecuteConfig):
    world_size = config.world_size
    table_num = config.table_num
    embedding_dims = config.embedding_dims
    num_embeddings = config.num_embeddings
    pool_type = config.pool_type
    sharding_type = config.sharding_type
    lookup_len = config.lookup_len
    device = config.device
    optim = config.optim
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))
    embedding_config = generate_hash_config(embedding_dims, num_embeddings, pool_type)

    dataset = RandomRecDatasetUnique(BATCH_NUM, lookup_len, num_embeddings, table_num)
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
        embedding_config, gloden_dataset_loader, optim
    )
    test_results = test_model.test_loss(
        embedding_config, data_loader, sharding_type, optim
    )
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


def split_ranks(batch: Batch, world_size):
    jt_dict_each_rank = [{} for _ in range(world_size)]
    jt_dict = batch.sparse_features.to_dict()
    for k in jt_dict.keys():
        jt: JaggedTensor = jt_dict[k]
        ids_for_eatch_rank = [[] for _ in range(world_size)]
        for a_id in jt.values():
            ids_for_eatch_rank[a_id % world_size].append(a_id)
        for rank in range(world_size):
            jt_dict_each_rank[rank][k] = JaggedTensor(
                values=torch.Tensor(ids_for_eatch_rank[rank]).long(),
                lengths=torch.ones(len(ids_for_eatch_rank[rank])).long(),
            )
    batch.sparse_features = [
        KeyedJaggedTensor.from_jt_dict(jt_dict) for jt_dict in jt_dict_each_rank
    ]


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
        embedding_config: List[EmbeddingBagConfig],
        dataloader: DataLoader[Batch],
        sharding_type: str,
        optim,
    ):
        num_features = sum([c.num_features() for c in embedding_config])
        # Shard
        table_num = len(embedding_config)
        opmizer_args = OptimizerArgs(
            learning_rate=OPTIMIZER_PARAM[optim]["lr"],
            eps=OPTIMIZER_PARAM[optim]["eps"],
        )
        new_config = []
        for config in embedding_config:
            new_config.append(
                EmbeddingConfig(
                    table_name=config.feature_names[0],
                    num_embedding=config.num_embeddings,
                    embedding_dim=config.embedding_dim,
                    optimizer=OptimType.EXACT_ADAGRAD,
                    optimizer_args=opmizer_args,
                    init_fn=weight_init,
                )
            )
        ebc = HashEmbeddingModuleCollection(configs=new_config)
        ebc = Model(ebc, num_features)
        # Optimizer
        results = []
        batch: Batch
        iter_ = iter(dataloader)
        for _ in range(LOOP_TIMES):
            batch = next(iter_).to(self.device)
            split_ranks(batch, self.world_size)
            loss, out = ebc(batch)
            results.append(loss.detach().cpu())
            results.append(out.detach().cpu())
            loss.backward()

        return results


params = {
    "world_size": [WORLD_SIZE],
    "table_num": [3],
    "embedding_dims": [[32, 32, 32]],
    "num_embeddings": [[4000, 40000, 40000]],
    "pool_type": [torchrec.PoolingType.MEAN],
    "sharding_type": ["table_wise"],
    "lookup_len": [1024],
    "device": ["npu"],
    "optim": [Adagrad],
}


@pytest.mark.parametrize(
    "config", [ExecuteConfig(*v) for v in itertools.product(*params.values())]
)
def test_hybrid_hash_embedding_bag(config: ExecuteConfig):
    if config.device == "cpu" and (
        config.sharding_type == "row_wise" or config.optim == Adam
    ):
        return
    mp.spawn(
        execute,
        args=(config,),
        nprocs=WORLD_SIZE,
        join=True,
    )
