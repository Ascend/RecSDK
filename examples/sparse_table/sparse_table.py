#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
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

logger = logging.getLogger(__name__)
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import torch_npu
except ImportError:
    logger.warning("未检测到torch_npu包，当前运行在非NPU环境下")

torch.set_grad_enabled(True)


@dataclass
class DistributedEmbeddingFunctionInput:
    """封装分布式嵌入查找的输入参数"""
    input_ids: torch.Tensor  # 输入索引张量 (batch_size, seq_len)
    local_embedding: torch.Tensor  # 本地嵌入分片 (shard_size, embedding_dim)
    shard_start: int  # 当前分片处理的全局起始索引
    shard_end: int  # 当前分片处理的全局结束索引
    embedding_dim: int  # 嵌入向量维度


class DistributedEmbeddingFunction(Function):
    @staticmethod
    def forward(ctx, embedding_input):
        """
        Forward pass for distributed embedding lookup
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{rank}')
        elif torch.npu.is_available():
            device = torch.device(f'npu:{rank}')
        else:
            device = "cpu"

        input_ids = embedding_input.input_ids
        local_embedding = embedding_input.local_embedding
        shard_start = embedding_input.shard_start
        shard_end = embedding_input.shard_end
        embedding_dim = embedding_input.embedding_dim

        if world_size == 1:
            global_emb = local_embedding[input_ids]
            ctx.save_for_backward(input_ids)
            return global_emb

        # 1. Deduplicate input IDs
        flat_input = input_ids.flatten()
        unique_ids, inverse_indices, counts = torch.unique(
            flat_input, return_inverse=True, return_counts=True
        )

        # 2. All-to-all communication if needed
        # Calculate send counts
        send_counts = torch.zeros(world_size, dtype=torch.int64, device=device)
        for r in range(world_size):
            start = r * ((shard_end - shard_start) * world_size) // world_size
            end = (r + 1) * ((shard_end - shard_start) * world_size) // world_size
            mask = (unique_ids >= start) & (unique_ids < end)
            send_counts[r] = mask.sum()

        # Exchange counts
        recv_counts = torch.zeros_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)

        # Convert to element counts
        send_counts_emb = send_counts * embedding_dim
        recv_counts_emb = recv_counts * embedding_dim

        local_ids = torch.zeros(
            recv_counts.sum(),
            device=device,
            dtype=torch.int64
        )
        dist.all_to_all_single(
            local_ids,
            unique_ids,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
            async_op=False
        )

        local_ids_scratch = local_ids - shard_start
        local_emb_gather = local_embedding(local_ids_scratch)
        flattened_local_embeddings = local_emb_gather.flatten()

        unique_emb_all = torch.zeros(
            len(unique_ids) * embedding_dim,
            device=device
        )
        unique_emb_all_flatten = unique_emb_all.flatten()

        # 执行all2all通信
        dist.all_to_all_single(
            unique_emb_all_flatten,
            flattened_local_embeddings,
            output_split_sizes=send_counts_emb.tolist(),
            input_split_sizes=recv_counts_emb.tolist(),
            async_op=False
        )
        unique_emb = unique_emb_all_flatten.view(-1, embedding_dim)
        # Restore original order
        restore_emb = unique_emb[inverse_indices].view(*input_ids.shape, -1)

        # Save for backward
        ctx.save_for_backward(unique_ids, inverse_indices)
        return restore_emb


class DistributedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{dist.get_rank()}')
        elif torch.npu.is_available():
            self.device = torch.device(f'npu:{dist.get_rank()}')
        else:
            self.device = "cpu"
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Shard by embedding table
        self.shard_size = (num_embeddings + self.world_size - 1) // self.world_size
        self.shard_start = self.rank * self.shard_size
        self.shard_end = min((self.rank + 1) * self.shard_size, num_embeddings)
        self.local_dim = embedding_dim
        self.local_num_embeddings = self.shard_end - self.shard_start

        # Initialize local embedding
        self.local_embedding = nn.Embedding(
            self.local_num_embeddings,
            self.local_dim,
            device=self.device,
            dtype=torch.float32
        )
        nn.init.xavier_uniform_(self.local_embedding.weight)

    def forward(self, input_ids):
        embedding_input_params = DistributedEmbeddingFunctionInput(
            input_ids,
            self.local_embedding,
            self.shard_start,
            self.shard_end,
            self.embedding_dim)
        return DistributedEmbeddingFunction.apply(embedding_input_params)


def initialize_distributed():
    if torch.distributed.is_initialized():
        return
    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    elif torch.npu.is_available():
        device: torch.device = torch.device(f"npu:{rank}")
        backend = "hccl"
        torch_npu.npu.set_device(device)

    torch.distributed.init_process_group(backend=backend)


def setup_ddp():
    initialize_distributed()


def main():
    setup_ddp()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{dist.get_rank()}')
    elif torch.npu.is_available():
        device = torch.device(f'npu:{dist.get_rank()}')
    else:
        device = "cpu"

    # 参数配置
    num_embeddings = 1000
    embedding_dim = 64
    batch_size = 32
    seq_len = 10

    # 初始化模型
    model = DistributedEmbedding(
        num_embeddings,
        embedding_dim,
        device=device
    ).to(device)
    ddp_model = DDP(model, device_ids=[device.index])

    # 生成测试数据（需确保input_ids在有效范围内）
    input_ids = torch.randint(
        0, num_embeddings,
        (batch_size, seq_len),
        device=device
    )

    # 前向查表
    outputs = ddp_model(input_ids)

    base_tensor = torch.randn(
        (32, 10, 64),
        device=device,
        requires_grad=True  # 如果需要梯度反向传播
    )
    # 模拟损失计算和反向传播
    loss = torch.nn.functional.mse_loss(base_tensor, outputs)
    loss.backward()


if __name__ == '__main__':
    main()
