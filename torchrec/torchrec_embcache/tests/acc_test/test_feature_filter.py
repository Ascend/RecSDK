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
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from dataset import RandomRecDataset, Batch
from model import ModelEc as Model
from torch.utils.data import DataLoader
from torchrec_embcache.distributed.embedding import EmbCacheEmbeddingCollection
from torchrec_embcache.distributed.modules.cache_embedding_configs import (EmbCacheEmbeddingConfig,
                                                                                     AdmitAndEvictConfig)
from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist
from torchrec_embcache.distributed.sharding.embedding_sharder import EmbCacheEmbeddingCollectionSharder
from torchrec_embcache.saver import Saver
from util import setup_logging

import torchrec
import torchrec.distributed
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.keyed import CombinedOptimizer

_SAVE_PATH = "save_dir/sparse"

lib_fbgemm_npu_api_so_path = os.getenv('LIB_FBGEMM_NPU_API_SO_PATH')
if lib_fbgemm_npu_api_so_path is None:
    raise RuntimeError("LIB_FBGEMM_NPU_API_SO_PATH environment variable is not set.")
torch.ops.load_library(lib_fbgemm_npu_api_so_path)


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "2"))
LOOP_TIMES = 500
EVICT_STEP_INTERVAL = LOOP_TIMES // 4
BATCH_NUM = LOOP_TIMES


def _check_admit_key_count(data_loader_golden, embedding_configs: List[EmbCacheEmbeddingConfig], rank):
    # 1 手动统计key count
    iter_ = iter(data_loader_golden)
    loop_time = 0
    table_key_count = [{} for _ in range(len(embedding_configs))]

    while loop_time < LOOP_TIMES:
        loop_time += 1
        batch: Batch = next(iter_, None)
        if batch is None:
            break
        kjt = batch.sparse_features
        if len(kjt.keys()) != len(embedding_configs):
            return
        values = kjt.values()
        offset_per_key = kjt.offset_per_key()
        for i in range(len(offset_per_key) - 1):
            values_per_table = values[offset_per_key[i]: offset_per_key[i + 1]]
            for ids in values_per_table:
                ids = ids.item()
                if ids % WORLD_SIZE != rank:
                    continue
                if ids in table_key_count[i]:
                    table_key_count[i][ids] = table_key_count[i][ids] + 1
                else:
                    table_key_count[i][ids] = 1

    # 2 读取保存目录下的key count
    key_file_saved = os.path.join(_SAVE_PATH, "table{}", "rank{}".format(rank), "key", "slice.data")
    count_file_saved = os.path.join(_SAVE_PATH, "table{}", "rank{}".format(rank), "admit_count", "slice.data")
    table_key_count_saved = [{} for _ in range(len(embedding_configs))]
    for i in range(len(embedding_configs)):
        if not os.path.exists(key_file_saved.format(i)):
            raise ValueError(f"file:{key_file_saved.format(i)} is not exist when check key count data.")
        if not os.path.exists(count_file_saved.format(i)):
            raise ValueError(f"file:{count_file_saved.format(i)} is not exist when check key count data.")
        key_data = np.fromfile(key_file_saved.format(i), dtype=np.int64).reshape(-1)
        count_data = np.fromfile(count_file_saved.format(i), dtype=np.int64).reshape(-1)
        for index in range(key_data.shape[0]):
            ids = key_data[index]
            count = count_data[index]
            table_key_count_saved[i][ids] = count

    logging.info("rankId:" + str(rank) + ", table_key_count:%s", table_key_count)
    logging.info("rankId:" + str(rank) + ", table_key_count_saved:%s", table_key_count_saved)

    # 3 对比数据
    length_equal = all(len(table_key_count[i]) == len(table_key_count_saved[i]) for i in range(len(embedding_configs)))
    assert length_equal, "key count length is not equal."
    for i in range(len(embedding_configs)):
        keys = table_key_count[i].keys()
        for key in keys:
            # 手动统计key count * WORLD_SIZE 即为所有卡all2all通信后key count
            count_equal = table_key_count[i][key] * WORLD_SIZE == table_key_count_saved[i][key]
            assert count_equal, "key count value is not equal."


@dataclass
class ExecuteConfig:
    world_size: int
    table_num: int
    embedding_dims: List[int]
    num_embeddings: List[int]
    sharding_type: str
    lookup_len: int
    device: str
    enable_admit: bool
    enable_evict: bool



def execute(config: ExecuteConfig, rank: int):
    world_size = config.world_size
    table_num = config.table_num
    embedding_dims = config.embedding_dims
    num_embeddings = config.num_embeddings
    sharding_type = config.sharding_type
    lookup_len = config.lookup_len
    device = config.device
    enable_admit = config.enable_admit
    enable_evict = config.enable_evict
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))

    dataset = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num, is_evict_enabled=True)
    dataset_golden = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num, is_evict_enabled=True)
    data_loader_golden = DataLoader(
        dataset_golden,
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
    embedding_configs = []
    default_config = AdmitAndEvictConfig()
    admit_threshold = 2 if enable_admit else default_config.admit_threshold
    evict_threshold = 2000_0000 if enable_evict else default_config.evict_threshold
    for i in range(table_num):
        admit_and_evict_config = AdmitAndEvictConfig(admit_threshold=admit_threshold, evict_threshold=evict_threshold)
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

    test_model = TestModel(rank, world_size, device)
    test_results = test_model.test_loss(embedding_configs, data_loader, sharding_type, enable_evict, training=True)
    is_enable_global_unique = os.environ.get("ENABLE_GLOBAL_UNIQUE", "False").lower() in ('true', '1', 'yes')
    if enable_admit and not enable_evict and is_enable_global_unique:
        _check_admit_key_count(data_loader_golden, embedding_configs, rank)

    for i, result in enumerate(test_results):
        logging.debug("")
        logging.debug("==============batch %d================", i // 2)
        logging.debug("result test %s", result)
    # load
    test_model.test_loss(embedding_configs, data_loader_golden, sharding_type, enable_evict, training=False)


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

    def setup(self, rank: int, world_size: int):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "6015"
        dist.init_process_group(self.pg_method, rank=rank, world_size=world_size)
        os.environ["LOCAL_RANK"] = f"{rank}"

    def test_loss(
        self,
        embedding_configs: List[EmbCacheEmbeddingConfig],
        dataloader: DataLoader[Batch],
        sharding_type: str,
        enable_evict: bool,
        training: True,
    ):
        rank, world_size = self.rank, self.world_size
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)

        table_num = len(embedding_configs)
        ec = EmbCacheEmbeddingCollection(device=torch.device("meta"), tables=embedding_configs,
                                         batch_size=2, multi_hot_sizes=[1] * table_num,
                                         world_size=dist.get_world_size())
        num_features = sum([c.num_features() for c in embedding_configs])
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
        if training:
            iter_ = iter(dataloader)
            ddp_model.train()
            evict_step_interval = EVICT_STEP_INTERVAL if enable_evict else None
            pipe = EmbCacheTrainPipelineSparseDist(
                ddp_model,
                optimizer=optimizer,
                cpu_device=cpu_device,
                npu_device=npu_device,
                return_loss=True,
                evict_step_interval=evict_step_interval
            )

            for _ in range(LOOP_TIMES):
                out, loss = pipe.progress(iter_)
                results.append(loss.detach().cpu())
                results.append(out.detach().cpu())

            save_dir = os.path.abspath("save_dir")
            if os.path.exists(save_dir):
                os.rmdir(save_dir)
            os.makedirs(save_dir, exist_ok=True) 

            saver = Saver(rank=rank)
            saver.save(ddp_model, _SAVE_PATH)
        else:
            saver = Saver(rank=rank)
            saver.load(ddp_model, _SAVE_PATH)
        return results


params = {
    "world_size": [WORLD_SIZE],
    "table_num": [2],
    "embedding_dims": [[128, 128]],
    "num_embeddings": [[4000, 400]],
    "sharding_type": ["row_wise"],
    "lookup_len": [128],  # batchsize
    "device": ["npu"],
    "enable_admit": [True],
    "enable_evict": [True],
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


params = {
    "world_size": [WORLD_SIZE],
    "table_num": [2],
    "embedding_dims": [[128, 128]],
    "num_embeddings": [[4000, 400]],
    "sharding_type": ["row_wise"],
    "lookup_len": [128],  # batchsize
    "device": ["npu"],
    "enable_admit": [True],
    "enable_evict": [False],
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_admit_count_correctness(config: ExecuteConfig):
    mp.spawn(
        execute,
        args=(config,),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == '__main__':
    test_admit_count_correctness(ExecuteConfig(
        world_size=2,
        table_num=2,
        embedding_dims=[128, 128],
        num_embeddings=[4000, 400],
        sharding_type="row_wise",
        lookup_len=128,
        device="npu",
        enable_admit=True,
        enable_evict=False
    ))
