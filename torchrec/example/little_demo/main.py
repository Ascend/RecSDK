# coding: UTF-8
# Copyright 2025. Huawei Technologies Co.,Ltd.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import logging

from hybrid_torchrec.distributed.hybrid_train_pipeline import (
    HybridTrainPipelineSparseDist,
)
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders

from dataset import RandomRecDataset
from model import TestModel

import torch.distributed as dist
from torch.utils.data import DataLoader
import torch


from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)


logging.getLogger().setLevel(logging.INFO)
FEAT_NAMES = [["phone", "clothes"], ["user"]]
TABLE_NAMES = ["product", "user"]
EMBEBD_DIMS = [1024, 1024]
NUM_EMBEBDS = [10240, 10240]
ID_RANGES = [1024, 1024, 1024]
BATCH_SIZE = 32
BATCH_NUM = 32


def get_distribute_env():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "6000"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(backend="hccl")
    return rank, world_size


def create_ddp(test_model):
    host_gp = dist.new_group(backend="gloo")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
    hybrid_sharder = get_default_hybrid_sharders(host_env=host_env)
    constraints = {
        table_name: ParameterConstraints(
            sharding_types=["row_wise"], compute_kernels=["fused"]
        )
        for table_name in TABLE_NAMES
    }

    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size=world_size, compute_device="npu"),
        constraints=constraints,
    )

    plan = planner.collective_plan(test_model, hybrid_sharder, dist.GroupMember.WORLD)
    logging.info(plan)
    ddp_model = DistributedModelParallel(
        test_model, device=torch.device("npu"), plan=plan, sharders=hybrid_sharder
    )
    return ddp_model


def invoke_main():
    get_distribute_env()
    device = torch.device("npu")

    dataset = RandomRecDataset(BATCH_SIZE, BATCH_NUM, FEAT_NAMES, ID_RANGES)
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        prefetch_factor=32,
        pin_memory_device="npu",
        num_workers=4,
    )

    test_model = TestModel(TABLE_NAMES, FEAT_NAMES, EMBEBD_DIMS, NUM_EMBEBDS)

    # Optimizer
    embedding_optimizer = torch.optim.Adagrad
    optimizer_kwargs = {"lr": 0.001, "eps": 0.1}
    apply_optimizer_in_backward(
        embedding_optimizer,
        test_model.ebc.parameters(),
        optimizer_kwargs=optimizer_kwargs,
    )

    # Shard
    ddp_model = create_ddp(test_model)

    # Optimizer filer
    dense_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(ddp_model.named_parameters())),
        lambda params: torch.optim.Adagrad(params, lr=0.1),
    )
    optimizer = CombinedOptimizer([ddp_model.fused_optimizer, dense_optimizer])

    pipeline = HybridTrainPipelineSparseDist(
        ddp_model, optimizer, device, execute_all_batches=True
    )

    batched_iterator = iter(data_loader)

    for i in range(20):
        logging.info("step %s done", i)
        pipeline.progress(batched_iterator)
    logging.info("demo done")


if __name__ == "__main__":
    invoke_main()
