#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import math
from dataclasses import dataclass
import itertools
import logging
import os
from typing import List

import pytest
import torch
import torch_npu
import torch.multiprocessing as mp
import torch.distributed as dist
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchrec_embcache.distributed.embedding import EmbCacheEmbeddingCollection
from torchrec_embcache.distributed.configs import (EmbCacheEmbeddingConfig,
                                                   AdmitAndEvictConfig,
                                                   ShowClickParams,
                                                   AdmitAndEvictPolicyType)
from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist
from torchrec_embcache.distributed.sharding.embedding_sharder import EmbCacheEmbeddingCollectionSharder
import torchrec
import torchrec.distributed
from torchrec import EmbeddingCollection
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.keyed import CombinedOptimizer

from torchrec import KeyedJaggedTensor
from dataset import RandomRecDataset, Batch
from model import ModelEc as Model
from util import setup_logging

_SAVE_PATH = "save_dir/sparse"

WORLD_SIZE_STR = str(os.environ.get("WORLD_SIZE", "2"))
WORLD_SIZE = int(WORLD_SIZE_STR) if WORLD_SIZE_STR.isalnum() else 2
LOOP_TIMES = 500
EVICT_STEP_INTERVAL = LOOP_TIMES // 4
BATCH_NUM = LOOP_TIMES


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


def execute(rank: int, config: ExecuteConfig):
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

    dataset = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num, is_enable_score=True)
    dataset_golden = RandomRecDataset(BATCH_NUM, lookup_len, num_embeddings, table_num, is_enable_score=True)
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
    if enable_admit and enable_evict:
        logging.info("enable admit and evict")
        showclick_params = ShowClickParams(
            alpha=1,
            beta=1,
            admit_threshold=0.1,
            evict_percentage=0.1,
            score_decay=0.9
        )
    elif enable_admit and not enable_evict:
        logging.info("enable admit only")
        showclick_params = ShowClickParams(
            alpha=1,
            beta=1,
            admit_threshold=2.2,
            evict_percentage=0.0,
            score_decay=0.9
        )
    elif not enable_admit and enable_evict:
        logging.info("enable evict only")
        showclick_params = ShowClickParams(
            alpha=1,
            beta=1,
            admit_threshold=0.0,
            evict_percentage=0.1,
            score_decay=0.9
        )
    for i in range(table_num):
        admit_and_evict_config = AdmitAndEvictConfig(evict_step_interval=EVICT_STEP_INTERVAL,
                                                     not_admitted_default_value=0.999,
                                                     showclick_params=showclick_params,
                                                     policy_type=AdmitAndEvictPolicyType.POLICY_SHOWCLICK)
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
        logging.info(f"showclick admit={admit_and_evict_config.is_feature_admit_enabled()}, \
                     evict={admit_and_evict_config.is_feature_evict_enabled()}, ")

    test_model = TestModel(rank, world_size, device)
    test_result_golden = []

    # 校验开启准入淘汰后的loss
    test_result_golden = test_model.cpu_golden_loss(embedding_configs, data_loader_golden, 0, rank)
    test_results = test_model.test_loss(embedding_configs, data_loader, sharding_type, enable_evict, training=True)

    for i, result in enumerate(test_results):
        logging.debug("")
        logging.debug("==============batch %d================", i // 2)
        logging.debug("result test %s", result)
        # check evict ret
        if not enable_admit and enable_evict:
            golden = test_result_golden[i]
            logging.debug("golden test %s", golden)
            assert torch.allclose(
                golden, result, rtol=1e-04, atol=1e-04
            ), "golden and result is not closed"

    dist.destroy_process_group()


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.linspace(0, 1, steps=param.shape[1]).unsqueeze(0).repeat(param.shape[0], 1)
    param.data.copy_(result)


def _get_init_weight(table_dims: List[int]):
    init_embs = []
    for dim in table_dims:
        emb = torch.linspace(0, 1, steps=dim)
        init_embs.append(emb)
    return init_embs


def _get_init_optimizer_slot(table_dims: List[int]):
    init_slots = []
    for dim in table_dims:
        slot = torch.zeros((dim,))
        init_slots.append(slot)
    return init_slots


class TestModel:
    def __init__(self, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.pg_method = "hccl" if device == "npu" else "gloo"
        if device == "npu":
            torch_npu.npu.set_device(rank)
        self.setup(rank=rank, world_size=world_size)
        self.emb_configs: List[EmbCacheEmbeddingConfig] = []

        # for admit
        self.count_for_table: List[List[dict]] = [[] for _ in range(world_size)]
        self.label_for_table: List[List[dict]] = [[] for _ in range(world_size)]

        # for evict
        self.score_for_table: List[List[dict]] = [[] for _ in range(world_size)]

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
        training: bool = True,
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
            optimizer_kwargs={"lr": 0.02, "eps": 1e-8},
        )
        # Shard
        constrains = {
            f"table{i}": ParameterConstraints(sharding_types=[sharding_type], compute_kernels=['fused'])
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
            constraints=constrains,
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
                results.append(out.detach().cpu())
                results.append(loss.detach().cpu())

        return results


    def cpu_golden_loss(self, embedding_configs: List[EmbCacheEmbeddingConfig], dataloader: DataLoader[Batch],
                        evict_threshold: int, rank_id: int):
        pg = dist.new_group(backend="gloo")
        self.emb_configs = embedding_configs
        table_num = len(embedding_configs)
        ec = EmbeddingCollection(device=torch.device("cpu"), tables=embedding_configs)

        num_features = sum([c.num_features() for c in embedding_configs])
        ec_wrap = Model(ec, num_features)
        model = DDP(ec_wrap, process_group=pg)

        opt = torch.optim.Adagrad(model.parameters(), lr=0.02, eps=1e-8)
        results = []
        batch: Batch
        iter_ = iter(dataloader)
        for i in range(LOOP_TIMES):
            batch = next(iter_)
            opt.zero_grad()
            loss, outputs = model(batch)
            results.append(outputs.detach().cpu())
            results.append(loss.detach().cpu())
            loss.backward()
            opt.step()

            for rank_id in range(self.world_size):
                # 0 static key count data
                self._static_key_count(batch, table_num, rank_id)
            
            if LOOP_TIMES < 30:
                for rank_id in range(self.world_size):
                    for table_index in range(table_num):
                        logging.info("mock all2all iterId:%d, rankId:%d, tableIndex:%d, count data:%s, label data:%s",
                                    i, rank_id, table_index, self.count_for_table[rank_id][table_index],
                                    self.label_for_table[rank_id][table_index])

            for rank_id in range(self.world_size):
                no_admit_ids = [[] for _ in range(table_num)]
                # 1 record batch label data
                self._record_showclick_info_cpu(batch, table_num, rank_id, i, no_admit_ids)
                # 2 evict emb and optimizer data
                if self.emb_configs[0].admit_and_evict_config.is_feature_evict_enabled():
                    if i > 0 and (i + 1) % EVICT_STEP_INTERVAL == 0:
                        self._evict_embedding_cpu(evict_threshold, ec.embeddings, opt, i, rank_id)
        return results


    def _static_key_count(self, batch, table_num, rank_id):
        sparse_tensor: KeyedJaggedTensor = batch.sparse_features
        labels: Tensor = batch.click_labels
        values = sparse_tensor.values()
        offset_per_key = sparse_tensor.offset_per_key()
        # init data structure
        if len(self.count_for_table[rank_id]) == 0:
            for _ in range(table_num):
                self.count_for_table[rank_id].append(dict())
        if len(self.label_for_table[rank_id]) == 0:
            for _ in range(table_num):
                self.label_for_table[rank_id].append(dict())

        for table_index in range(table_num):
            start = offset_per_key[table_index]
            end = offset_per_key[table_index + 1]
            values_per_table = values[start:end]

            # world size为2 且输入数据相同 所以统计时数量乘以2
            for index, ids in enumerate(values_per_table):
                ids = ids.item()
                # 分桶 + unique
                if ids % WORLD_SIZE != rank_id:
                    continue
                if ids in self.count_for_table[rank_id][table_index]:
                    self.count_for_table[rank_id][table_index][ids] = \
                        self.count_for_table[rank_id][table_index][ids] + 1 * 2
                else:
                    self.count_for_table[rank_id][table_index][ids] = 1 * 2

                if ids in self.label_for_table[rank_id][table_index]:
                    self.label_for_table[rank_id][table_index][ids] = \
                        self.label_for_table[rank_id][table_index][ids] + labels[index].item() * 2
                else:
                    self.label_for_table[rank_id][table_index][ids] = labels[index].item() * 2
            if LOOP_TIMES < 30:
                logging.debug("static key count rankId:%d, tableIndex:%d, count:%s, label:%s",
                              rank_id, table_index, self.count_for_table[rank_id][table_index],
                              self.label_for_table[rank_id][table_index])
               

    def _record_showclick_info_cpu(self, batch, table_num, rank_id, batch_id, no_admit_ids):
        sparse_tensor: KeyedJaggedTensor = batch.sparse_features
        values = sparse_tensor.values()
        offset_per_key = sparse_tensor.offset_per_key()
        # init data structure
        if len(self.score_for_table[rank_id]) == 0:
            for _ in range(table_num):
                self.score_for_table[rank_id].append(dict())

        # record score data
        for table_index in range(table_num):
            start = offset_per_key[table_index]
            end = offset_per_key[table_index + 1]
            values_per_table = values[start:end]

            show_click_params = self.emb_configs[table_index].admit_and_evict_config.showclick_params

            processed_ids = []
            for ids in values_per_table:
                ids = ids.item()
                if ids % WORLD_SIZE != rank_id:
                    continue
                if ids in processed_ids:
                    continue
                processed_ids.append(ids)
                # 准入计数
                admit_score = show_click_params.alpha * self.count_for_table[rank_id][table_index][ids] + \
                    show_click_params.beta * self.label_for_table[rank_id][table_index][ids]
                if self.emb_configs[table_index].admit_and_evict_config.is_feature_admit_enabled():
                    if admit_score < show_click_params.admit_threshold:
                        no_admit_ids[table_index].append(ids)
                        continue
                # 淘汰计数
                if self.emb_configs[table_index].admit_and_evict_config.is_feature_evict_enabled():
                    if ids not in self.score_for_table[rank_id][table_index]:
                        self.score_for_table[rank_id][table_index][ids] = admit_score
                    else:
                        old_score = self.score_for_table[rank_id][table_index][ids]
                        new_score = (old_score + admit_score) * show_click_params.score_decay
                        self.score_for_table[rank_id][table_index][ids] = new_score
            if rank_id == self.rank and LOOP_TIMES < 30:
                logging.debug("record showclick info rankId:%d, batchId:%d, tableIndex:%d, score data:%s",
                              rank_id, batch_id, table_index, self.score_for_table[rank_id][table_index])
                    

    def _evict_embedding_cpu(self, evict_threshold: int, embeddings: nn.ModuleDict,
                             opt: torch.optim.Adagrad, batch_id: int, rank_id: int):
        logging.info("Start cpu embedding evict, current step:%d, rank_id:%d", batch_id, rank_id)
        emb_dims: List[int] = [c.embedding_dim for c in self.emb_configs]
        table_names = [c.name for c in self.emb_configs]
        table_num = len(table_names)
        emb_init_values: List[Tensor] = _get_init_weight(emb_dims)
        optimizer_init_values: List[Tensor] = _get_init_optimizer_slot(emb_dims)
        for table_index in range(table_num):
            sorted_score = sorted(self.score_for_table[rank_id][table_index].items(), key=lambda x: (x[1], x[0]))
            evict_percentage = self.emb_configs[table_index].admit_and_evict_config.showclick_params.evict_percentage
            evict_num = math.floor(len(sorted_score) * evict_percentage)
            if evict_num == 0 and len(sorted_score) > 0:
                evict_num = 1
            evict_ids_per_table, evict_ids_score_per_table = zip(*sorted_score[:evict_num])

            table_name = table_names[table_index]
            # get slot tensor of Adagrad optimizer
            op_t = opt.param_groups[0]["params"][table_index]
            slot_tensor = opt.state[op_t]["sum"]
            for ids in evict_ids_per_table:
                # step1 delete timestamp record for ids
                self.count_for_table[rank_id][table_index].pop(ids)
                self.score_for_table[rank_id][table_index].pop(ids)
                self.label_for_table[rank_id][table_index].pop(ids)

                # step2 reset emb and optimizer slot as init value
                with torch.no_grad():
                    # init emb
                    embeddings[table_name].weight[ids].data.copy_(emb_init_values[table_index])
                    # init optimizer slot
                    slot_tensor[ids].data.copy_(optimizer_init_values[table_index])
            if LOOP_TIMES < 30:
                logging.info("batchId:%d, table name:%s, evict ids num:%d, rank_id:%d, evict ids:%s, evict scores:%s",
                             batch_id, table_name, len(evict_ids_per_table),
                             rank_id, evict_ids_per_table, evict_ids_score_per_table)


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
@pytest.mark.skip(reason="暂时无法通过loss进行正确性验证，待保存和加载功能完善后再启用该测试，" \
    "将保存的admit key count与手动统计的key count进行对比验证")
def test_admit_count_correctness(config: ExecuteConfig):
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
    "enable_admit": [False],
    "enable_evict": [True],
}


@pytest.mark.parametrize("config", [
    ExecuteConfig(*v) for v in itertools.product(*params.values())
])
def test_evict_correctness(config: ExecuteConfig):
    mp.spawn(
        execute,
        args=(config,),
        nprocs=WORLD_SIZE,
        join=True,
    )
