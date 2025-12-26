#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import argparse
import warnings
from typing import Union, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import collate_fn, MovieLensDataset
from model import create_model
from logger import logger

# Filter FBGEMM warning, make notebook clean
warnings.filterwarnings(
    "ignore", message=".*torch.library.impl_abstract.*", category=FutureWarning
)

backend = "hccl"
dist.init_process_group(backend=backend)
local_rank = dist.get_rank()  # for one node
world_size = dist.get_world_size()
torch.npu.set_device(local_rank)
device = torch.device(f"npu:{local_rank}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TorchRec MovieLens with dynamicemb")
    parser.add_argument("--train", action="store_true")

    parser.add_argument(
        "--data_path",
        type=str,
        default="./ml-1m",
        help="path to dataset MovieLens，and will download if non-existed",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="training optimizer"
    )
    parser.add_argument("--epochs", type=int, default=2, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--embedding_dim", type=int, default=64, help="embedding dimension"
    )
    parser.add_argument(
        "--num_embeddings", type=int, default=10000, help="number of embeddings"
    )
    parser.add_argument(
        "--mlp_dims",
        type=str,
        default="128,64,32",
        help="dimension of MLP layer，separating with commas",
    )
    parser.add_argument(
        "--seed", type=int, default=2025, help="random seed used for initialization"
    )
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: Tuple[int, int],
) -> None:
    model.train()
    total_loss = 0

    epoch, total_epochs = epochs
    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{total_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch + 1}/{total_epochs}, Average Loss: {avg_loss:.4f}")


def test_one_epoch(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    epochs: Tuple[int, int],
) -> None:
    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    epoch, total_epochs = epochs
    logger.info(f"Epoch {epoch + 1}/{total_epochs}, Test Loss: {avg_test_loss:.4f}")


def create_optimizer(args: argparse.Namespace, model: nn.Module) -> Union[Adam, AdamW]:
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def train(args: argparse.Namespace) -> None:
    train_dataset = MovieLensDataset(args.data_path, split="train")
    test_dataset = MovieLensDataset(args.data_path, split="test")
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        sampler=train_sampler,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        sampler=test_sampler,
    )

    model = create_model(args, device)
    logger.info(f"The model: {model}")

    optimizer = create_optimizer(args, model)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, optimizer, criterion, (epoch, args.epochs))
        test_one_epoch(model, test_loader, criterion, (epoch, args.epochs))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dist.barrier(device_ids=[local_rank])
    if args.train:
        train(args)
    logger.info("Demo done.")


if __name__ == "__main__":
    main()

dist.destroy_process_group()
