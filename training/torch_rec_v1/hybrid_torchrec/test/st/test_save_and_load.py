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
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import logging

import pytest

import torch
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import Adam, Adagrad, SGD, SparseAdam

from hybrid_torchrec import HashEmbeddingBagCollection, HashEmbeddingBagConfig
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
from hybrid_torchrec.distributed.hybrid_train_pipeline import (
    HybridTrainPipelineSparseDist,
)
from hybrid_torchrec.saver import Saver

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

from dataset import Batch, RandomRecDatasetV2
from hybrid_torchrec.utils import safe_makedirs
from model import Model
from util import setup_logging

OPTIMIZER_PARAM = {
    # 注: Rec SDK Torch中Adam优化器融合算子使用的更新算法为sparse 更新，和SparseAdam算法对应
    #   和torch原生Adam优化器更新算法有差异
    Adam: dict(lr=0.02),
    Adagrad: dict(lr=0.02, eps=1.0e-8),
    Adagrad: dict(lr=0.02, eps=1.0e-8),
    SGD: dict(lr=0.02),
    SparseAdam: dict(lr=0.02),
}

WORLD_SIZE = 2  # 参与训练卡数
LOOP_TIMES = 100  # 执行lookup次数，不能大于Batch数
BATCH_NUM = LOOP_TIMES * 2  # 生成的Dataset中的Batch数. 保存加载时需执行2次，生成2倍的数据

_BIG_TABLE_BS = 128
_BIG_TABLE_NUM_EMBEDDINGS = 500_0000
_ID_NUM_PER_SAMPLE = 20
_LOOKUP_LEN = _BIG_TABLE_BS * _ID_NUM_PER_SAMPLE  # 一次lookup中，KJT内每个表key(表)对应的id个数。多个表的id数需要相同

_ONLY_EXEC_NPU_LOOKUP = False  # True:仅执行npu查表，统计耗时. False:执行npu查表，并和cpu查表对比精度
_COLLECT_NPU_PROF = False  # 采集npu profiling性能数据


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
    ids_repeat_rate: Union[float, None]


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
    setup_logging(rank, level=logging.INFO)
    logging.info("this test %s", os.path.basename(__file__))
    # generate dateset
    dataset = RandomRecDatasetV2(BATCH_NUM, lookup_lens, num_embeddings, table_num, rank,
                                 id_repeat_rate=config.ids_repeat_rate, save_dataset=True)
    data_loader, dataset_loader_golden, embedding_configs = get_dataset_and_table_configs(
        dataset, embedding_dims, num_embeddings, pool_type, table_num
    )

    test_model = TestModel(rank, world_size, device)
    if _ONLY_EXEC_NPU_LOOKUP:
        _ = test_model.test_npu_loss(embedding_configs, data_loader, sharding_type, optim, return_data=False)
    else:
        golden_results = test_model.cpu_golden_loss(embedding_configs, dataset_loader_golden, optim)
        test_results = test_model.test_npu_loss(embedding_configs, data_loader, sharding_type, optim, return_data=True)
        data_compare(golden_results, rank, test_results)


def _get_embedding_config_list(embedding_dims, num_embeddings, pool_type, table_num):
    embedding_config = []
    for i in range(table_num):
        ebc_config = HashEmbeddingBagConfig(
            name=f"table{i}",
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[f"feat{i}"],
            pooling=pool_type,
            init_fn=weight_init,
        )
        embedding_config.append(ebc_config)
    return embedding_config


def execute_save_load(
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
    setup_logging(rank)
    logging.info("this test %s", os.path.basename(__file__))
    # generate dateset
    dataset = RandomRecDatasetV2(BATCH_NUM, lookup_lens, num_embeddings, table_num, rank,
                                 id_repeat_rate=config.ids_repeat_rate, save_dataset=True)
    data_loader_4_save, _, embedding_configs = get_dataset_and_table_configs(
        dataset, embedding_dims, num_embeddings, pool_type, table_num
    )
    data_loader_4_load = _get_npu_data_loader(dataset)
    test_model = TestModel(rank, world_size, device)
    golden_results = test_model.test_npu_save_load(embedding_configs, data_loader_4_save, sharding_type, optim,
                                                   training=True)
    test_results = test_model.test_npu_save_load(embedding_configs, data_loader_4_load, sharding_type, optim,
                                                 training=False)
    data_compare(golden_results, rank, test_results)


def data_compare(golden_results, rank, test_results):
    counter = 0
    for golden, result in zip(golden_results, test_results):
        batch_id = counter // 2
        counter += 1
        logging.info(f"============rank:{rank}, batch_id:{batch_id}===============")
        logging.info(f"rank:{rank}, batch_id:{batch_id}, result test %s", golden)
        logging.info(f"rank:{rank}, batch_id:{batch_id}, golden test %s", result)
        assert torch.allclose(
            golden, result, rtol=1e-04, atol=1e-04
        ), f"rank:{rank}, batch_id:{batch_id}, golden and result is not closed"


def get_dataset_and_table_configs(dataset, embedding_dims, num_embeddings, pool_type, table_num):
    dataset_loader_golden = _get_cpu_data_loader(dataset)
    data_loader = _get_npu_data_loader(dataset)
    embedding_config = _get_embedding_config_list(embedding_dims, num_embeddings, pool_type, table_num)
    return data_loader, dataset_loader_golden, embedding_config


def _get_cpu_data_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
    )


def _get_npu_data_loader(dataset):
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )
    return data_loader


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
    def cpu_golden_loss(
        embedding_config: List[EmbeddingBagConfig], dataloader: DataLoader[Batch], optim
    ):
        ebc = EmbeddingBagCollection(device="cpu", tables=embedding_config)
        num_features = sum([c.num_features() for c in embedding_config])

        if optim == Adam:
            optim = SparseAdam
            # 注: Rec SDK Torch中Adam优化器融合算子使用的更新算法和SparseAdam算法对应，和torch原生Adam更新算法有差异
            # 因此cpu侧需使用SparseAdam优化器进行更新,需将torch.nn.Embedding对象的sparse字段更新为True
            for config in embedding_config:
                ebc.embedding_bags[config.name].sparse = True

        ebc = Model(ebc, num_features)
        pg = dist.new_group(backend="gloo")
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
        return results

    def setup(self, rank: int, world_size: int):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "6000"
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
        dist.init_process_group(self.pg_method, rank=rank, world_size=world_size)
        os.environ["LOCAL_RANK"] = f"{rank}"

    def test_npu_loss(
        self,
        embedding_configs: List[EmbeddingBagConfig],
        dataloader: DataLoader[Batch],
        sharding_type: str,
        optim,
        return_data: bool = False
    ):
        rank = self.rank
        ddp_model = self.get_dmp_model(embedding_configs, optim, sharding_type)
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
        if _COLLECT_NPU_PROF:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=torch_npu.profiler.ExportType.Text,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.Memory,
            )
            prof = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=100, active=10, repeat=1, skip_first=0
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./profiler"),
                record_shapes=False,
                profile_memory=True,
                experimental_config=experimental_config,
            )
            prof.start()
        for step in range(LOOP_TIMES):
            start_time = time.time()
            out, loss = pipe.progress(iter_)
            if _COLLECT_NPU_PROF:
                prof.step()
            if return_data:
                results.append(loss.detach().cpu())
                results.append(out.detach().cpu())
            end_time = time.time()
            logging.info("=== rank:%d, step:%d, sparse(forward and backward) cost time: %.8f(s).",
                         rank, step, end_time - start_time)

        if _COLLECT_NPU_PROF:
            prof.stop()

        return results

    def test_npu_save_load(
        self,
        embedding_config: List[EmbeddingBagConfig],
        dataloader: DataLoader[Batch],
        sharding_type: str,
        optim,
        training: bool = True
    ):
        ddp_model = self.get_dmp_model(embedding_config, optim, sharding_type)
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
        rank = self.rank
        saver = Saver(rank)
        save_dir = os.path.abspath("save_dir")
        if training and os.path.exists(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)
        if training:
            safe_makedirs(save_dir)
            for _ in range(LOOP_TIMES):
                _, _ = pipe.progress(iter_)

            ddp_model.eval()
            for _ in range(LOOP_TIMES):
                out, loss = pipe.progress(iter_)
                results.append(loss.detach().cpu())
                results.append(out.detach().cpu())
            logging.info("ddp_model.state_dict %s", ddp_model.state_dict())
            # dense-保存
            torch.save(ddp_model.state_dict(), f"save_dir/model_{rank}.pt")
            torch.save(optimizer.state_dict(), f"save_dir/optimizer_{rank}.pt")
            # sparse-保存
            saver.save(ddp_model, "save_dir/sparse")
        else:
            # 加载数据
            ddp_state_dict = torch.load(f"save_dir/model_{rank}.pt", weights_only=False)
            logging.info("torch load ddp_state_dict ret: %s", ddp_state_dict)
            ddp_model.load_state_dict(ddp_state_dict)
            optimizer.load_state_dict(
                torch.load(f"save_dir/optimizer_{rank}.pt", weights_only=False)
            )
            # 加载sparse数据
            saver.load(ddp_model, "save_dir/sparse")

            # 重新查表
            ddp_model.eval()
            for _ in range(LOOP_TIMES):
                _, _ = pipe.progress(iter_)
            for _ in range(LOOP_TIMES):
                out, loss = pipe.progress(iter_)
                results.append(loss.detach().cpu())
                results.append(out.detach().cpu())

        return results

    def get_dmp_model(self, embedding_configs, optim, sharding_type):
        rank, world_size = self.rank, self.world_size
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
        table_num = len(embedding_configs)
        ebc = HashEmbeddingBagCollection(device="meta", tables=embedding_configs)
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
            ebc, get_default_hybrid_sharders(host_env), dist.GroupMember.WORLD
        )
        if self.rank == 0:
            logging.debug(plan)
        dmp_model = torchrec.distributed.DistributedModelParallel(
            ebc,
            sharders=get_default_hybrid_sharders(host_env),
            device=torch.device(self.device),
            plan=plan,
        )
        return dmp_model


params = {
    "world_size": [WORLD_SIZE],
    "table_num": [3],
    "embedding_dims": [[32, 64, 128]],
    "num_embeddings": [[40000, 20000, _BIG_TABLE_NUM_EMBEDDINGS]],
    "pool_type": [torchrec.PoolingType.MEAN],
    "sharding_type": ["row_wise"],
    "lookup_len": [_LOOKUP_LEN],
    "device": ["npu"],
    "optim": [Adam],
    "ids_repeat_rate": [0.2],
}


# 执行查表用例时，仅执行Adam优化器(CPU查表比较慢)，其他优化器在其他用例中已覆盖
@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_hybrid_pipeline_lookup(config: ExecuteConfig):
    mp.spawn(
        execute_lookup,
        args=(
            config,
        ),
        nprocs=config.world_size,
        join=True,
    )


params["optim"] = [Adam, Adagrad, SGD]


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_hybrid_save_and_load(config: ExecuteConfig):
    mp.spawn(
        execute_save_load,
        args=(
            config,
        ),
        nprocs=config.world_size,
        join=True,
    )


if __name__ == '__main__':
    # 修改参数值，使其和 model_convert_2_gpu.py 的测试脚本test.py参数一致
    config_list = [ExecuteConfig(*v) for v in itertools.product(*params.values())]
    config = config_list[0]
    config.world_size = 1
    config.embedding_dims = [16, 16, 16]
    config.num_embeddings = [40000, 20000, 50000]
    config.lookup_len = 200
    config.optim = Adagrad
    config.ids_repeat_rate = None
    Path("dataset.pt").unlink(missing_ok=True)
    test_hybrid_save_and_load(config)
