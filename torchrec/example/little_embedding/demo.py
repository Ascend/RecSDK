#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import sysconfig
from typing import Dict

import torch
import torch.distributed as dist
import torch_npu

from hybrid_torchrec.modules.little_embedding import (
    HashEmbeddingModuleCollection,
    Awaitable,
    EmbeddingConfig,
    OptimizerArgs,
    OptimType,
    LookupContext,
)
import torch_npu.npu
from torchrec import KeyedJaggedTensor, JaggedTensor


torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_distribute_env():
    rank_id = int(os.environ["LOCAL_RANK"])
    total_world_size = int(os.environ["WORLD_SIZE"])
    return rank_id, total_world_size


rank, world_size = get_distribute_env()
torch_npu.npu.set_device(rank)
dist.init_process_group(backend="hccl")


def dataset_getnext():
    # 1.使用List
    # jagged0：当前卡需要从table0查询的所有ids(所有卡分桶+all2all后,分给当前卡的ids)
    jagged0 = JaggedTensor(
        values=torch.Tensor([1, 3, 5, 11, 13, 15, 17]).long(),
        lengths=torch.Tensor([1, 1, 1, 1, 1, 1, 1]).long(),
    )
    jagged1 = JaggedTensor(
        values=torch.Tensor([1, 3, 5, 11, 13, 15]).long(),
        lengths=torch.Tensor([1, 1, 1, 1, 1, 1]).long(),
    )
    input_for_rank0 = KeyedJaggedTensor.from_jt_dict(
        {"table0": jagged0, "table1": jagged0}
    )
    input_for_rank1 = KeyedJaggedTensor.from_jt_dict(
        {"table0": jagged1, "table1": jagged1}
    )

    # embedding表 1-> 0.1, 0.1
    # 2-> 0.2, 0.2
    sparse_fid_list = [input_for_rank0, input_for_rank1]
    return sparse_fid_list


optimizer_type = OptimType.EXACT_ADAGRAD
optimizer_args = OptimizerArgs(
    learning_rate=0.1,
    eps=0.1,
    beta1=0.1,
    beta2=0.1,
)

config0 = EmbeddingConfig(
    table_name="table0",
    num_embedding=100,
    embedding_dim=32,
    optimizer_args=optimizer_args,
)
config1 = EmbeddingConfig(
    table_name="table1",
    num_embedding=100,
    embedding_dim=32,
    optimizer_args=optimizer_args,
)
embedding = HashEmbeddingModuleCollection(configs=[config0, config1])


for i in range(3):
    data = dataset_getnext()
    awaitables: Dict[str, Awaitable]
    context: LookupContext
    awaitables, context = embedding(data)
    result = [awaitables["table0"].wait()[rank], awaitables["table1"].wait()[rank]]
    logging.info("result %s", result)
    logging.info("context.ids2lookup_index %s", context.ids2lookup_index["table0"])
    loss = torch.concat(result).sum()
    loss.backward()
    # result =
    # 结果：
    # [
    #   [torch.Tensor([[0.1, 0.1], [0.3, 0.3], [0.5, 0.5], [1.1, 1.1], [1.3, 1.3], [1.5, 1.5]])],
    #   [torch.Tensor([[0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [1.2, 1.2], [1.4, 1.4], [1.6, 1.6]])],
    # ]
    # context.ids2lookup_index["table0"] = {1: 0, 3: 1, 5: 2, ...,}

# 2 step pipe_line
pipe1 = []
for i in range(10):
    data = dataset_getnext()
    awaitables: Dict[str, Awaitable]
    context: LookupContext
    awaitables, context = embedding(data)
    pipe1.append(awaitables)

pipe2 = []
for p in pipe1:
    # 同步开始查表和All2Al
    result = p["table0"].wait() + p["table1"].wait()
    loss = torch.concat(result).sum()
    loss.backward()
logging.info("demo done")
