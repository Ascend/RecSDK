#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import itertools
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Iterator, Optional
import logging
import pytz
import pytest

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import Optimizer, Adam, Adagrad, SGD, SparseAdam
import torchrec
from torchrec import EmbeddingBagConfig, EmbeddingBagCollection, KeyedJaggedTensor, Pipelineable, JaggedTensor
import torchrec.distributed
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.sharding_plan import get_default_sharders
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.optim.keyed import CombinedOptimizer
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward

from model_convert_2_gpu import ModelConverter

WORLD_SIZE = 1  # 参与训练卡数 使用的cpu table_wise，只能单卡对比
LOOP_TIMES = 100  # 执行lookup次数，不能大于Batch数
BATCH_NUM = LOOP_TIMES * 2  # 生成的Dataset中的Batch数. 保存加载时需执行2次，生成2倍的数据
_DIR_MODE = 0o750

OPTIMIZER_PARAM = {
    Adam: dict(lr=0.02),
    Adagrad: dict(lr=0.02, eps=1.0e-8),
    SGD: dict(lr=0.02),
    SparseAdam: dict(lr=0.02),
}


def makedirs(path):
    os.makedirs(path, mode=_DIR_MODE, exist_ok=True)


def setup_logging(rank, level=logging.INFO):
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
    logger.setLevel(level)


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

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
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
        for ind in range(feature_len):
            name = f"feat{ind}"
            id_range = self.num_embeddings[ind]
            ids = torch.randint(0, id_range, (self.lookup_lens,))
            lengths = torch.ones(self.lookup_lens).long()
            input_dict[name] = JaggedTensor(values=ids, lengths=lengths)
        kjt_tensor = KeyedJaggedTensor.from_jt_dict(input_dict)
        label = torch.randint(0, 2, (self.lookup_lens,))
        return Batch(kjt_tensor, label)


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
    optim: torch.optim.Optimizer
    load_paths: List[str]


def execute_lookup(
    rank,
    config: ExecuteConfig
):
    world_size = config.world_size
    table_num = config.table_num
    embedding_dims = config.embedding_dims
    num_embeddings = config.num_embeddings
    pool_type = config.pool_type
    sharding_type = config.sharding_type
    lookup_lens = config.lookup_len
    device = config.device
    optim = config.optim
    load_paths = config.load_paths
    setup_logging(rank, level=logging.INFO)
    logging.info("this test %s", os.path.basename(__file__))
    # generate dateset
    dataset = RandomRecDataset(BATCH_NUM, lookup_lens, num_embeddings, table_num)
    data_loader, dataset_loader_golden, embedding_configs = get_dataset_and_table_configs(
        dataset, embedding_dims, num_embeddings, pool_type, table_num
    )

    test_model = TestModel(rank, world_size, device)
    test_config = TestConfig(
        embedding_config=embedding_configs,
        sharding_type=sharding_type,
        optim=optim,
    )
    golden_results = test_model.test_save_load(
        test_config, dataset_loader_golden, training=True, load_paths=load_paths
    )
    test_results = test_model.test_save_load(
        test_config, data_loader, training=False, load_paths=load_paths
    )
    batch_id = 0
    for golden, result in zip(golden_results, test_results):
        logging.info(f"============rank:{rank}, batch_id:{batch_id}===============")
        logging.info(f"rank:{rank}, batch_id:{batch_id}, golden test %s", golden)
        logging.info(f"rank:{rank}, batch_id:{batch_id}, result test %s", result)
        compare_ret = torch.allclose(
            golden, result, rtol=1e-04, atol=1e-04
        )
        if not compare_ret:
            raise ValueError(f"rank:{rank}, batch_id:{batch_id}, golden and result is not closed")
        batch_id += 1


def _get_embedding_config_list(embedding_dims, num_embeddings, pool_type, table_num):
    embedding_config = []
    for i in range(table_num):
        ebc_config = EmbeddingBagConfig(
            name=f"table{i}",
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[f"feat{i}"],
            pooling=pool_type,
            init_fn=weight_init,
        )
        embedding_config.append(ebc_config)
    return embedding_config


def get_dataset_and_table_configs(dataset, embedding_dims, num_embeddings, pool_type, table_num):
    dataset_loader_golden = _get_cpu_data_loader(dataset)
    data_loader = _get_cpu_data_loader(dataset)
    embedding_config = _get_embedding_config_list(embedding_dims, num_embeddings, pool_type, table_num)
    return data_loader, dataset_loader_golden, embedding_config


def _get_cpu_data_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
    )


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.randn((1, param.shape[1])).repeat(param.shape[0], 1)
    param.data.copy_(result)


@dataclass
class TestConfig:
    embedding_config: List[EmbeddingBagConfig]
    sharding_type: str
    optim: Optimizer


class TestModel:
    def __init__(self, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.pg_method = "gloo"
        self.setup(rank=rank, world_size=world_size)

    def setup(self, rank: int, world_size: int):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "6000"
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
        dist.init_process_group(self.pg_method, rank=rank, world_size=world_size)
        os.environ["LOCAL_RANK"] = f"{rank}"

    def test_save_load(
        self,
        config: TestConfig,
        dataloader: DataLoader[Batch],
        training: bool = True,
        load_paths: Optional[List[str]] = None,
    ):
        if load_paths is None:
            load_paths = ["save_dir/sparse"]
        dmp_model = self.get_dmp_model(config)
        logging.debug(dmp_model)
        optimizer = CombinedOptimizer([dmp_model.fused_optimizer])
        results = []
        iter_ = iter(dataloader)
        dmp_model.train()
        pipe = TrainPipelineSparseDist(
            dmp_model,
            optimizer=optimizer,
            device=torch.device(self.device),
        )
        rank = self.rank
        if training:
            for _ in range(LOOP_TIMES):
                _ = pipe.progress(iter_)

            dmp_model.eval()
            for _ in range(LOOP_TIMES):
                out = pipe.progress(iter_)
                results.append(out.detach().cpu())
            logging.info("dmp_model.state_dict %s", dmp_model.state_dict())
        else:
            # 加载数据
            model_converter = ModelConverter(rank)
            # 调用load接口，会直接将NPU保存的稀疏表数据（embedding, optimizer）加载到dmp_model中
            for load_path in load_paths:
                model_converter.load(dmp_model, load_path, config.optim)

            # 重新查表
            dmp_model.eval()
            for _ in range(LOOP_TIMES):
                _ = pipe.progress(iter_)
            for _ in range(LOOP_TIMES):
                out = pipe.progress(iter_)
                results.append(out.detach().cpu())

        return results

    def get_dmp_model(self, config: TestConfig):
        embedding_configs = config.embedding_config
        sharding_type = config.sharding_type
        optim = config.optim
        table_num = len(embedding_configs)
        ebc = EmbeddingBagCollection(device="meta", tables=embedding_configs)
        num_features = sum([c.num_features() for c in embedding_configs])
        ebc = Model(ebc, num_features)
        # 该dict会在调用中修改，必须深拷贝。否则第二次有unexpected keyed param
        optimizer_args = copy.deepcopy(OPTIMIZER_PARAM[optim])
        apply_optimizer_in_backward(
            optimizer_class=optim,
            params=ebc.parameters(),
            optimizer_kwargs=optimizer_args,
        )
        # shard constraints
        constrains = {
            f"table{i}": ParameterConstraints(
                sharding_types=[sharding_type], compute_kernels=["fused"]
            )
            for i in range(table_num)
        }
        planner = EmbeddingShardingPlanner(
            topology=Topology(world_size=self.world_size, compute_device=self.device),
            constraints=constrains,
        )
        plan = planner.collective_plan(
            ebc, get_default_sharders(), dist.GroupMember.WORLD
        )
        if self.rank == 0:
            logging.debug(plan)
        dmp_model = torchrec.distributed.DistributedModelParallel(
            ebc,
            sharders=get_default_sharders(),
            device=torch.device(self.device),
            plan=plan,
        )
        return dmp_model


params = {
    "world_size": [WORLD_SIZE],
    "table_num": [3],
    "embedding_dims": [[16, 16, 16]],
    "num_embeddings": [[40000, 20000, 50000]],
    "pool_type": [torchrec.PoolingType.MEAN],
    "sharding_type": ["table_wise"],
    "lookup_len": [200],
    "device": ["cpu"],
    "optim": [Adagrad],
    "load_paths": [["save_dir/sparse"]],
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_hybrid_pipeline_lookup(config: ExecuteConfig):
    mp.spawn(
        execute_lookup,
        args=(
            config,
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == '__main__':
    config_list = [ExecuteConfig(*v) for v in itertools.product(*params.values())]
    test_hybrid_pipeline_lookup(config_list[0])
