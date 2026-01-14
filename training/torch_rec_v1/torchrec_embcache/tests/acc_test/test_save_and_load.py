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
from typing import List, Optional

import numpy as np
import pytest
import torch
import torch_npu
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from dataset import RandomRecDataset, Batch
from model import ModelEc as Model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchrec_embcache.distributed.embedding import (
    EmbCacheEmbeddingCollection,
)
from torchrec_embcache.distributed.sharding.embedding_sharder import (
    EmbCacheEmbeddingCollectionSharder,
)
from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist
from torchrec_embcache.saver import Saver
from torchrec_embcache.utils import safe_makedirs
import torchrec
from torchrec_embcache.distributed.configs import (EmbCacheEmbeddingConfig,
                                                   AdmitAndEvictConfig)
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

from util import setup_logging, setup_main_logging

WORLD_SIZE = 2
# 不同的WORLD_SIZE
DIFF_WORLD_SIZE = WORLD_SIZE + 1
SAVE_BASE_TIMES = 2
SAVE_DELTA_TIMES = 2
SAVE_TIMES = SAVE_BASE_TIMES * (SAVE_DELTA_TIMES + 1)
SAVE_STEPS = 100
LOOP_TIMES = SAVE_TIMES * SAVE_STEPS
BATCH_NUM = LOOP_TIMES * 2  # will execute LOOP_TIMES*2 times lookup when save load
RESULT_TMP_DIR = "result_tmp_dir"
EVICT_STEP_INTERVAL = LOOP_TIMES // 4


@dataclass
class ExecuteConfig:
    save_world_size: int
    load_world_size: int
    table_num: int
    embedding_dims: List[int]
    num_embeddings: List[int]
    sharding_type: str
    lookup_len: int
    device: str
    incremental: bool
    enable_admit: bool = False
    enable_evict: bool = False


def execute(rank: int, config: ExecuteConfig, save_load_mode: Optional[str] = None):
    save_world_size = config.save_world_size
    load_world_size = config.load_world_size
    table_num = config.table_num
    embedding_dims = config.embedding_dims
    num_embeddings = config.num_embeddings
    sharding_type = config.sharding_type
    lookup_len = config.lookup_len
    device = config.device
    incremental = config.incremental
    enable_admit = config.enable_admit
    enable_evict = config.enable_evict
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))
    dataset = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num, is_evict_enabled=enable_evict)
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
    default_admit_evict_config = AdmitAndEvictConfig()
    admit_threshold = 2 if enable_admit else default_admit_evict_config.admit_threshold
    evict_threshold = 2000_00000 if enable_evict else default_admit_evict_config.evict_threshold
    for i in range(table_num):
        admit_and_evict_config = AdmitAndEvictConfig(admit_threshold=admit_threshold,
                                                     not_admitted_default_value=0.999,
                                                     evict_threshold=evict_threshold,
                                                     evict_step_interval=EVICT_STEP_INTERVAL)
        ec_config = EmbCacheEmbeddingConfig(
            name=f"table{i}",
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[f"feat{i}"],
            init_fn=weight_init,
            weight_init_min=0.0,
            weight_init_max=1.0,
            is_incremental=incremental,
            admit_and_evict_config=admit_and_evict_config,
        )
        embedding_config.append(ec_config)
    if save_load_mode is None:
        world_size = save_world_size
        test_model = TestModel(rank, world_size, device)
        golden_results, _ = test_model.test_loss(embedding_config, dataset_loader_golden, sharding_type, training=True)
        test_results, _ = test_model.test_loss(embedding_config, data_loader, sharding_type, training=False)
        _compare_tensor(golden_results, test_results)
    elif save_load_mode == "save":
        world_size = save_world_size
        test_model = TestModel(rank, world_size, device)
        test_model.test_loss(embedding_config, data_loader, sharding_type, training=True, save_load_mode="save")
    elif save_load_mode == "load":
        world_size = load_world_size
        test_model = TestModel(rank, world_size, device)
        test_model.test_loss(embedding_config, data_loader, sharding_type, training=False, save_load_mode="load")


def _compare_tensor(golden_results, test_results):
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


def _read_admit_and_evict_data(path: str, config, incremental: bool = False):
    base_file_path = os.path.join(path, (f"base_{SAVE_BASE_TIMES}/" if incremental else "") + "table{}")
    delta_file_paths = [
        os.path.join(path, f"base_{SAVE_BASE_TIMES}_delta_{i+1}/" + "table{}") for i in range(SAVE_DELTA_TIMES)
    ] if incremental else []
    admit_key_saved_files = [os.path.join(base_file_path, "key", "slice.data")] + \
        [os.path.join(delta_file_path, "key", "slice.data") for delta_file_path in delta_file_paths]
    admit_count_saved_files = [os.path.join(base_file_path, "admit_count", "slice.data")] + \
        [os.path.join(delta_file_path, "admit_count", "slice.data") for delta_file_path in delta_file_paths]
    evict_key_saved_files = [os.path.join(base_file_path, "evict_timestamp", "slice_evict_key.data")] + \
        [os.path.join(delta_file_path, "evict_timestamp", "slice_evict_key.data") 
         for delta_file_path in delta_file_paths]
    evict_ts_saved_files = [os.path.join(base_file_path, "evict_timestamp", "slice_evict_ts.data")] + \
        [os.path.join(delta_file_path, "evict_timestamp", "slice_evict_ts.data") 
         for delta_file_path in delta_file_paths]
        
    table_key_count_saved = [{} for _ in range(config.table_num)]
    table_key_ts_saved = [{} for _ in range(config.table_num)]
    for i in range(config.table_num):
        if config.enable_admit:
            for admit_key_saved_file, admit_count_saved_file in zip(admit_key_saved_files, admit_count_saved_files):
                if not os.path.exists(admit_key_saved_file.format(i)):
                    raise ValueError(f"file:{admit_key_saved_file.format(i)} is not exist when check admit key data.")
                if not os.path.exists(admit_count_saved_file.format(i)):
                    raise ValueError(
                        f"file:{admit_count_saved_file.format(i)} is not exist when check admit count data.")
                key_data = np.fromfile(admit_key_saved_file.format(i), dtype=np.int64).reshape(-1)
                count_data = np.fromfile(admit_count_saved_file.format(i), dtype=np.int64).reshape(-1)
                for index in range(key_data.shape[0]):
                    ids = key_data[index]
                    count = count_data[index]
                    table_key_count_saved[i][ids] = count
        if config.enable_evict:
            for evict_key_saved_file, evict_ts_saved_file in zip(evict_key_saved_files, evict_ts_saved_files):
                if not os.path.exists(evict_key_saved_file.format(i)):
                    raise ValueError(f"file:{evict_key_saved_file.format(i)} is not exist when check evict key data.")
                if not os.path.exists(evict_ts_saved_file.format(i)):
                    raise ValueError(f"file:{evict_ts_saved_file.format(i)} is not exist when check evict ts data.")
                evict_key_data = np.fromfile(evict_key_saved_file.format(i), dtype=np.int64).reshape(-1)
                evict_ts_data = np.fromfile(evict_ts_saved_file.format(i), dtype=np.int64).reshape(-1)
                for index in range(evict_key_data.shape[0]):
                    ids = evict_key_data[index]
                    ts = evict_ts_data[index]
                    table_key_ts_saved[i][ids] = max(ts, table_key_ts_saved[i].get(ids, 0))

    return table_key_count_saved, table_key_ts_saved


def _compare_table_data(table1, table2):
    if len(table1) != len(table2):
        return False
    for key in table1:
        if key not in table2:
            return False
        if table1[key] != table2[key]:
            return False
    return True


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
        embedding_config: List[EmbCacheEmbeddingConfig],
        dataloader: DataLoader[Batch],
        sharding_type: str,
        training: bool,
        save_load_mode: Optional[str] = None,
    ):
        rank, world_size = self.rank, self.world_size

        table_num = len(embedding_config)
        ec = EmbCacheEmbeddingCollection(
            device=torch.device("meta"),
            tables=embedding_config,
            batch_size=2,
            multi_hot_sizes=[1] * table_num,
            world_size=dist.get_world_size(),
        )
        num_features = sum([c.num_features() for c in embedding_config])
        ec = Model(ec, num_features)
        apply_optimizer_in_backward(
            optimizer_class=torch.optim.Adagrad,
            params=ec.parameters(),
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
        hash_shader = EmbCacheEmbeddingCollectionSharder(
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
        plan = planner.collective_plan(ec, shaders, dist.GroupMember.WORLD)
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
        if training:
            if embedding_config[0].is_incremental:
                for base_time in range(SAVE_BASE_TIMES):
                    for _ in range(SAVE_STEPS):
                        _, _ = pipe.progress(iter_)
                    saver.save(ddp_model, f"save_dir/sparse/base_{base_time+1}")
                    for delta_time in range(SAVE_DELTA_TIMES):
                        for _ in range(SAVE_STEPS):
                            _, _ = pipe.progress(iter_)
                        saver.save(ddp_model, f"save_dir/sparse/base_{base_time+1}_delta_{delta_time+1}", 
                                   incremental=True)
            else:
                for _ in range(LOOP_TIMES):
                    _, _ = pipe.progress(iter_)

            ddp_model.eval()
            for _ in range(LOOP_TIMES):
                out, loss = pipe.progress(iter_)
                results.append(loss.detach().cpu())
                results.append(out.detach().cpu())
            if rank == 0 and save_load_mode == "save":
                if not os.path.exists(RESULT_TMP_DIR):
                    safe_makedirs(RESULT_TMP_DIR)
                torch.save(results, os.path.join(RESULT_TMP_DIR, "results_train.pt"))
            # save sparse
            if not embedding_config[0].is_incremental:
                saver.save(ddp_model, "save_dir/sparse")
            # 由于差异卡加载时torch保存的dense数据会校验报错，且该测试用例里的dense部分没有参数，所以不保存dense部分
        else:
            # sparse-加载
            if embedding_config[0].is_incremental:
                saver.load(ddp_model, f"save_dir/sparse/base_{SAVE_BASE_TIMES}")
                for delta_time in range(SAVE_DELTA_TIMES):
                    saver.load(ddp_model, f"save_dir/sparse/base_{SAVE_BASE_TIMES}_delta_{delta_time+1}",
                               incremental=True)
            else:
                saver.load(ddp_model, "save_dir/sparse")
            # 由于差异卡加载时torch保存的dense数据会校验报错，且该测试用例里的dense部分没有参数，所以不加载dense部分
            ddp_model.eval()
            for _ in range(LOOP_TIMES):
                _, _ = pipe.progress(iter_)
            for _ in range(LOOP_TIMES):
                out, loss = pipe.progress(iter_)
                results.append(loss.detach().cpu())
                results.append(out.detach().cpu())
            if rank == 0 and save_load_mode == "load":
                if not os.path.exists(RESULT_TMP_DIR):
                    raise RuntimeError("golden results dir not exist")
                torch.save(results, os.path.join(RESULT_TMP_DIR, "results_infer.pt"))
            # for admit and evict test, save again after load
            if embedding_config[0].admit_and_evict_config.is_feature_admit_enabled() or \
                embedding_config[0].admit_and_evict_config.is_feature_evict_enabled():
                saver.save(ddp_model, "save_dir/sparse_after_load")
        # Must return ddp_model, it is necessary to maintain the reference count about static ThreadPool in C++ code,
        # to facilitate the use of subsequent tasks.
        return results, ddp_model


params = {
    "save_world_size": [WORLD_SIZE],
    "load_world_size": [WORLD_SIZE, DIFF_WORLD_SIZE],
    "table_num": [2],
    "embedding_dims": [[128, 128]],
    "num_embeddings": [[4000, 400]],
    "sharding_type": ["row_wise"],
    "lookup_len": [128],  # batchsize
    "device": ["npu"],
    "incremental": [False, True],
    "enable_admit": [False], # 暂时不考虑开启admit的情况
    "enable_evict": [False], # 暂时不考虑开启evict的情况
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_hstu_dens_normal(config: ExecuteConfig):
    save_load_mode = None
    if config.save_world_size == config.load_world_size:
        mp.spawn(
            execute,
            args=(config, save_load_mode),
            nprocs=config.save_world_size,
            join=True,
        )
    else:
        save_load_mode = "save"
        mp.spawn(
            execute,
            args=(config, save_load_mode),
            nprocs=config.save_world_size,
            join=True,
        )
        save_load_mode = "load"
        mp.spawn(
            execute,
            args=(config, save_load_mode),
            nprocs=config.load_world_size,
            join=True,
        )
        setup_main_logging()
        golden_results = torch.load(os.path.join(RESULT_TMP_DIR, "results_train.pt"))
        test_results = torch.load(os.path.join(RESULT_TMP_DIR, "results_infer.pt"))
        _compare_tensor(golden_results, test_results)
        # clean temp dir
        shutil.rmtree(RESULT_TMP_DIR, ignore_errors=True)
    if config.enable_evict or config.enable_admit:
        save_path = "save_dir/sparse"
        after_load_path = "save_dir/sparse_after_load"
        save_table_key_count_saved, save_table_key_ts_saved = _read_admit_and_evict_data(save_path, config, 
                                                                                         config.incremental)
        after_load_table_key_count_saved, after_load_table_key_ts_saved = \
            _read_admit_and_evict_data(after_load_path, config, incremental=False)
        for idx in range(config.table_num):
            if config.enable_admit:
                assert _compare_table_data(save_table_key_count_saved[idx], after_load_table_key_count_saved[idx]), \
                    f"table {idx} admit count data not match after load"
            if config.enable_evict:
                assert _compare_table_data(save_table_key_ts_saved[idx], after_load_table_key_ts_saved[idx]), \
                    f"table {idx} evict timestamp data not match after load"


if __name__ == "__main__":
    multiprocessing.freeze_support()
    test_hstu_dens_normal(ExecuteConfig(
        save_world_size=WORLD_SIZE,
        load_world_size=WORLD_SIZE,
        table_num=2,
        embedding_dims=[128, 128],
        num_embeddings=[4000, 400],
        sharding_type="row_wise",
        lookup_len=128,
        device="npu",
        incremental=False,
        enable_admit=False,
        enable_evict=False,
    ))
