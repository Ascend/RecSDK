#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
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
import logging
import os
import random
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import torch_npu
except ImportError:
    logger.warning("未检测到torch_npu包，当前运行在非NPU环境下")

from data.reco_dataset import get_reco_dataset
from embedding_modules_dist import EmbeddingModule, LocalEmbeddingModule
from features import movielens_seq_features_from_row
from data_loader import create_data_loader, DataLoaderInput

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def cleanup():
    dist.destroy_process_group()


@dataclass
class TrainInput:
    dataset_name: str = "ml-1m"
    max_sequence_length: int = 200
    positional_sampling_ratio: float = 1.0
    local_batch_size: int = 128
    num_epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    save_ckpt_every_n: int = 10
    embedding_module_type: str = "local"
    item_embedding_dim: int = 240
    gr_output_length: int = 10
    random_seed: int = 42


def train_fn(train_input_params: TrainInput) -> None:
    dataset_name = train_input_params.dataset_name
    max_sequence_length = train_input_params.max_sequence_length
    positional_sampling_ratio = train_input_params.positional_sampling_ratio
    local_batch_size = train_input_params.local_batch_size
    num_epochs = train_input_params.num_epochs
    learning_rate = train_input_params.learning_rate
    weight_decay = train_input_params.weight_decay
    save_ckpt_every_n = train_input_params.save_ckpt_every_n
    embedding_module_type = train_input_params.embedding_module_type
    item_embedding_dim = train_input_params.item_embedding_dim
    gr_output_length = train_input_params.gr_output_length
    random_seed = train_input_params.random_seed
    # to enable more deterministic results.
    random.seed(random_seed)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    dataset = get_reco_dataset(
        dataset_name=dataset_name,
        max_sequence_length=max_sequence_length,
        chronological=True,
        positional_sampling_ratio=positional_sampling_ratio,
    )

    data_loader_input = DataLoaderInput(
        dataset=dataset.train_dataset,
        batch_size=local_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        drop_last=world_size > 1
    )
    train_data_sampler, train_data_loader = create_data_loader(data_loader_input)

    if embedding_module_type == "local":
        embedding_module: EmbeddingModule = LocalEmbeddingModule(
            num_items=4000,
            item_embedding_dim=item_embedding_dim,
        )
    else:
        raise ValueError(f"Unknown embedding_module_type {embedding_module_type}")

    model = embedding_module
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
    elif torch.npu.is_available():
        device = torch.device(f'npu:{rank}')
    else:
        device = "cpu"

    model = model.to(device)
    model = DDP(model, device_ids=[rank], broadcast_buffers=False)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=weight_decay,
    )

    batch_id = 0
    for epoch in range(num_epochs):
        if train_data_sampler is not None:
            train_data_sampler.set_epoch(epoch)
        model.train()
        for row in iter(train_data_loader):
            seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                row,
                device=device,
                max_output_length=gr_output_length + 1,
            )

            seq_features.past_ids.scatter_(
                dim=1,
                index=seq_features.past_lengths.view(-1, 1),
                src=target_ids.view(-1, 1),
            )

            opt.zero_grad()
            input_embeddings = model(seq_features.past_ids)
            bs, seq_len, emb_dim = input_embeddings[:, 1:, :].shape
            base_tensor = torch.randn(
                (bs, seq_len, emb_dim),
                device=device,
            )
            if rank == 0:
                logging.info(f"Rank {rank}: input embedding shape {input_embeddings[:, 1:, :].shape}.")
            loss = torch.nn.functional.mse_loss(base_tensor, input_embeddings[:, 1:, :])

            loss.backward()
            opt.step()
            batch_id += 1
            if rank == 0:
                logging.info(f"Rank {rank}: step: {batch_id}, loss: {loss}.")
        if rank == 0 and epoch > 0 and (epoch % save_ckpt_every_n) == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                },
                f"./ckpts/model_ep{epoch}",
            )


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


if __name__ == "__main__":
    setup_ddp()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{dist.get_rank()}')
    elif torch.npu.is_available():
        device = torch.device(f'npu:{dist.get_rank()}')
    else:
        device = "cpu"
    train_fn(TrainInput())
    cleanup()
