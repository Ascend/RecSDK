#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import logging

import torch
import torch.distributed as dist

from hybrid_torchrec.modules.little_embedding import (
    HashEmbeddingModule,
    Awaitable,
    EmbeddingConfig,
)


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_distribute_env():
    rank_id = int(os.environ["LOCAL_RANK"])
    total_world_size = int(os.environ["WORLD_SIZE"])
    return rank_id, total_world_size


rank, world_size = get_distribute_env()
dist.init_process_group(backend="gloo")


def dataset_getnext():
    # 1.使用List
    inpurt_for_rank0 = [torch.Tensor([1, 3, 5]), torch.Tensor([2, 4, 6])]
    inpurt_for_rank1 = [torch.Tensor([11, 13, 15]), torch.Tensor([12, 14, 16])]
    # 2. flatten然后利用offset表示
    # input = torch.Tensor([1, 3, 5, 2, 4, 6, 11, 13, 15, 12, 14, 16]) offset = [0, 3, 6, 9, 12]

    # embeding表 1-> 0.1, 0.1
    # 2-> 0.2, 0.2
    sparse_fid_list = [inpurt_for_rank0, inpurt_for_rank1]
    return sparse_fid_list


config = EmbeddingConfig(num_embedding=100, embedding_dim=32, rank=rank)
embedding = HashEmbeddingModule(config=config)


for i in range(10):
    data = dataset_getnext()
    awaitable: Awaitable = embedding(data)
    result = awaitable.wait()
    logging.info("result %s", result)
    loss = torch.concat(result).sum()
    loss.backward()
    # result =
    # 结果：
    # [
    #   [torch.Tensor([[0.1, 0.1], [0.3, 0.3], [0.5, 0.5], [1.1, 1.1], [1.3, 1.3], [1.5, 1.5]])],
    #   [torch.Tensor([[0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [1.2, 1.2], [1.4, 1.4], [1.6, 1.6]])],
    # ]
    #

# 2 step pipe_line
pipe1 = []
for i in range(10):
    data = dataset_getnext()
    awaitable: Awaitable = embedding(data, is_full_pipe=False)
    pipe1.append(awaitable)

pipe2 = []
for p in pipe1:
    # 同步开始查表和All2Al
    r = p.wait()
    loss = torch.concat(r).sum()
    loss.backward()
logging.info("demo done")
