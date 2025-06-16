#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from typing import Dict, List

import pytest
import torch
from dataset import RandomRecDataset, Batch
from torch.utils.data import DataLoader
from hybrid_torchrec import (
    HashEmbeddingBagCollection,
    HashEmbeddingBagConfig,
)
from model import Model
from util import setup_logging

import torchrec
from torchrec import EmbeddingBagCollection, EmbeddingBagConfig

LOOP_TIMES = 8
BATCH_NUM = 32


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.randn((1, param.shape[1])).repeat(param.shape[0], 1)
    param.data.copy_(result)


def train_model(
    model: List[EmbeddingBagConfig], dataloader: DataLoader[Batch], device: str
):
    opt = torch.optim.Adagrad(model.parameters(), lr=0.02, eps=1e-8)
    results = []
    batch: Batch
    iter_ = iter(dataloader)
    for _ in range(LOOP_TIMES):
        batch = next(iter_).to(device)
        opt.zero_grad()
        loss, output = model(batch)
        results.append(loss.detach().cpu())
        results.append(output.detach().cpu())
        loss.backward()
        opt.step()

    for i in range(len(model.ebc.embedding_bags)):
        logging.debug(
            "single table%d weight %s",
            i,
            model.ebc.embedding_bags[f"table{i}"].weight,
        )
    return results


def generate_hash_config(
    embedding_dims, num_embeddings, pool_type
) -> List[HashEmbeddingBagConfig]:
    test_table_configs: List[HashEmbeddingBagCollection] = []
    for i, (table_dim, num_embedding) in enumerate(zip(embedding_dims, num_embeddings)):
        config = HashEmbeddingBagConfig(
            name=f"table{i}",
            embedding_dim=table_dim,
            num_embeddings=num_embedding,
            feature_names=[f"feat{i}"],
            pooling=pool_type,
            init_fn=weight_init,
        )
        test_table_configs.append(config)
    return test_table_configs


def generate_base_config(
    embedding_dims, num_embeddings, pool_type
) -> List[EmbeddingBagConfig]:
    test_table_configs: List[EmbeddingBagConfig] = []
    for i, (table_dim, num_embedding) in enumerate(zip(embedding_dims, num_embeddings)):
        config = EmbeddingBagConfig(
            name=f"table{i}",
            embedding_dim=table_dim,
            num_embeddings=num_embedding,
            feature_names=[f"feat{i}"],
            pooling=pool_type,
            init_fn=weight_init,
        )
        test_table_configs.append(config)
    return test_table_configs


@pytest.mark.parametrize("embedding_dims", [[32, 64, 128]])
@pytest.mark.parametrize("num_embeddings", [[400, 4000, 400]])
@pytest.mark.parametrize("pool_type", [torchrec.PoolingType.MEAN])
@pytest.mark.parametrize("lookup_len", [1024])
@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_hstu_dens_normal(
    embedding_dims, num_embeddings, pool_type, lookup_len, device
):
    setup_logging(rank=0)
    logging.info("this test %s", os.path.basename(__file__))
    test_table_configs = generate_hash_config(embedding_dims, num_embeddings, pool_type)
    gloden_table_configs = generate_base_config(
        embedding_dims, num_embeddings, pool_type
    )

    num_features = sum([c.num_features() for c in test_table_configs])
    dataset = RandomRecDataset(
        BATCH_NUM, lookup_len, num_embeddings, len(test_table_configs)
    )

    # Test Model
    ebc_hash = HashEmbeddingBagCollection(tables=test_table_configs, device=device)
    test_model = Model(ebc_hash, num_features).to(device)
    test_dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
    )

    # Golden Model
    ebc = EmbeddingBagCollection(tables=gloden_table_configs, device="cpu")
    gloen_model = Model(ebc, num_features)
    gloden_dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
    )

    test_results = train_model(test_model, test_dataloader, device)
    gloden_results = train_model(gloen_model, gloden_dataloader, "cpu")

    for gloden, result in zip(gloden_results, test_results):
        logging.debug("")
        logging.debug("===========================")
        logging.debug("result test %s", result)
        logging.debug("gloden test %s", gloden)
        assert torch.allclose(
            gloden, result, rtol=1e-04, atol=1e-04
        ), "gloden and result is not closed"
