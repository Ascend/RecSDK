#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Iterator
from unittest.mock import MagicMock
# 创建模拟的b模块注册到sys.modules
mock_npu = MagicMock()
mock_npu.npu = MagicMock()  # 显式定义npu子模块
sys.modules['torch_npu'] = mock_npu

import pytest
import pytz
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
from hybrid_torchrec import HashEmbeddingBagCollection, HashEmbeddingBagConfig

import torchrec.distributed.shard
from torchrec import (
    EmbeddingBagConfig,
    EmbeddingBagCollection,
)
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.streamable import Pipelineable
from torchrec import KeyedJaggedTensor, JaggedTensor




LOOP_TIMES = 8
BATCH_NUM = 32
WORLD_SIZE = 2


def generate_base_config(
        embedding_dims,
        num_embeddings,
        pool_type) -> List[EmbeddingBagConfig]:
    test_table_configs: List[EmbeddingBagConfig] = []
    for i, (table_dim, num_embedding) in enumerate(zip(embedding_dims, num_embeddings)):
        config = EmbeddingBagConfig(
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
class Batch(Pipelineable):
    sparse_features: KeyedJaggedTensor
    labels: torch.Tensor

    def __init__(self, sparse_features, labels) -> None:
        self.sparse_features = sparse_features
        self.labels = labels

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        return Batch(
            sparse_features=self.sparse_features.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
        )

    def record_stream(self, stream) -> None:
        self.sparse_features.record_stream(stream)
        self.labels.record_stream(stream)

    def pin_memory(self) -> "Batch":
        return Batch(
            sparse_features=self.sparse_features.pin_memory(),
            labels=self.labels.pin_memory(),
        )


class RandomRecDataset(IterableDataset[Batch]):
    def __init__(self, batch_num, lookup_lens, num_embeddings, table_num):
        super().__init__()
        self.index = 0
        self.lookup_lens = lookup_lens
        self.num_embeddings = num_embeddings
        self.table_num = table_num
        self.batch_num = batch_num
        torch.manual_seed(1)
        self.data = [self.generate_one_batch() for _ in range(batch_num)]

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def generate_one_batch(self) -> Batch:
        input_dict = {}
        feature_len = len(self.num_embeddings)
        for ind in range(feature_len - 1, -1, -1):
            name = f"feat{ind}"
            id_range = self.num_embeddings[ind]
            ids = torch.randint(0, id_range, (self.lookup_lens,))
            lengths = torch.ones(self.lookup_lens).long()
            input_dict[name] = JaggedTensor(values=ids, lengths=lengths)
        kjt_tensor = KeyedJaggedTensor.from_jt_dict(input_dict)
        label = torch.randint(0, 2, (self.lookup_lens,))
        return Batch(kjt_tensor, label)


def permute_values(kjt: KeyedJaggedTensor, feature_num) -> torch.Tensor:
    keys_nums = feature_num
    values = []
    jt_dict = kjt.to_dict()
    for k in range(keys_nums):
        k = f"feat{k}"
        jt = jt_dict[k]
        values.append(jt)
    values = torch.concat(values, dim=1)
    return values


class Model(torch.nn.Module):
    def __init__(self, ebc, feature_num):
        super().__init__()
        self._ebc = ebc
        self.feature_num = feature_num

    @property
    def ebc(self):
        return self._ebc

    def forward(self, batch: Batch):
        result = self._ebc(batch.sparse_features)
        result = permute_values(result, self.feature_num)
        loss = result.sum()
        return loss, result


def setup_logging(rank):
    from datetime import datetime

    this_time = str(
        datetime.now(tz=pytz.timezone("PRC")).strftime(
            "%m_%d_%H_%M_%S",
        )
    )
    log_format = logging.Formatter(
        fmt=f"[rank{rank}][%(levelname)s][%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    file_handler = logging.FileHandler(
        f"test_rank{rank}_{this_time}.log", encoding="utf-8"
    )
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.randn((1, param.shape[1])).repeat(param.shape[0], 1)
    param.data.copy_(result)


def execute(
        rank,
        world_size,
        table_num,
        embedding_dims,
        num_embeddings,
        pool_type,
        lockup_len,
):
    device = 'cpu'
    sharding_type = 'row_wise'
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))
    embeding_config = generate_base_config(embedding_dims, num_embeddings, pool_type)

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
        pin_memory_device="cpu",
        num_workers=1,
    )
    test_model = TestModel(rank, world_size, device)
    gloden_results = test_model.cpu_gloden_loss(embeding_config,
                                                gloden_dataset_loader)
    test_model.test_shard_plan(embeding_config, data_loader, sharding_type)
    for gloden in gloden_results:
        logging.debug("")
        logging.debug("===========================")
        logging.debug("result test %s", gloden)
        assert tuple(gloden.size()) == (10, 224)   # lockup_len, sum(embedding_dims)


class TestModel:
    def __init__(self, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.pg_method = "gloo"
        self.setup(rank=rank, world_size=world_size)

    @staticmethod
    def cpu_gloden_loss(
        embeding_config: List[EmbeddingBagConfig],
        dataloader: DataLoader[Batch]
    ):
        pg = dist.new_group(backend="gloo")
        table_num = len(embeding_config)
        ebc = HashEmbeddingBagCollection(device="cpu", tables=embeding_config)

        num_features = sum([c.num_features() for c in embeding_config])
        ebc = Model(ebc, num_features)
        model = DDP(ebc, device_ids=None, process_group=pg)

        opt = torch.optim.Adagrad(ebc.parameters(), lr=0.02, eps=1e-8)
        results = []
        batch: Batch
        iter_ = iter(dataloader)
        for _ in range(LOOP_TIMES):
            batch = next(iter_)
            opt.zero_grad()
            loss, output = model(batch)
            results.append(output.detach().cpu())

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

    def test_shard_plan(
        self,
        embeding_config: List[EmbeddingBagConfig],
        dataloader: DataLoader[Batch],
        sharding_type: str,
    ):
        num_features = sum([c.num_features() for c in embeding_config])
        rank, world_size = self.rank, self.world_size
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
        # Shard
        table_num = len(embeding_config)
        ebc = EmbeddingBagCollection(device="meta", tables=embeding_config)
        ebc = Model(ebc, num_features)
        apply_optimizer_in_backward(
            optimizer_class=torch.optim.Adagrad,
            params=ebc.parameters(),
            optimizer_kwargs={"lr": 0.02},
        )
        # sharding_type
        constrans = {
            f"table{i}": ParameterConstraints(
                sharding_types=[sharding_type], compute_kernels=["fused"]
            )
            for i in range(table_num)
        }
        

@pytest.mark.parametrize("table_num", [3])
@pytest.mark.parametrize("embedding_dims", [[32, 64, 128]])
@pytest.mark.parametrize("num_embeddings", [[400, 4000, 400]])
@pytest.mark.parametrize("pool_type", [torchrec.PoolingType.MEAN, torchrec.PoolingType.SUM])
@pytest.mark.parametrize("lockup_len", [10])
def test_embedding_bag_collection(
        table_num,
        embedding_dims,
        num_embeddings,
        pool_type,
        lockup_len,
):
    mp.spawn(
        execute,
        args=(
            WORLD_SIZE,
            table_num,
            embedding_dims,
            num_embeddings,
            pool_type,
            lockup_len,
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )
