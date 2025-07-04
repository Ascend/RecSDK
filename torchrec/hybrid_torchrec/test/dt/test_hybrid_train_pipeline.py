#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

import unittest
from unittest.mock import patch, MagicMock
import pytest
sys.modules['torch_npu'] = MagicMock

import torch
from torch.utils.data import DataLoader
from torch.optim import Adagrad

from hybrid_torchrec.distributed.hybrid_train_pipeline import (
    HybridTrainPipelineSparseDist,
    HybridTrainPipelineContext,
    _fuse_input_dist_splits
)
from hybrid_torchrec import HashEmbeddingBagCollection, HashEmbeddingBagConfig
from dataset import RandomRecDataset

import torchrec

BATCH_SIZE = 8


def get_embedding_config(embedding_dims, num_embeddings, table_num):
    embeding_config = []
    for i in range(table_num):
        ebc_config = HashEmbeddingBagConfig(
            name=f"table{i}",
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[f"feat{i}"],
            pooling=torchrec.PoolingType.SUM,
        )
        embeding_config.append(ebc_config)
    return embeding_config


class TestHybridTrainPipelineSparseDist(unittest.TestCase):
    def setUp(self):
        self.rank = 0
        self.world_size = 1
        lookup_len = 256
        embedding_dims = [32, 64]
        num_embeddings = [100, 200]
        table_num = len(embedding_dims)
        embedding_config = get_embedding_config(embedding_dims, num_embeddings, table_num)
        self.model = HashEmbeddingBagCollection(device=torch.device("cpu"), tables=embedding_config)
        dataset = RandomRecDataset(BATCH_SIZE, lookup_len, num_embeddings, table_num)
        self.data_loader = DataLoader(
            dataset,
            batch_size=None,
            batch_sampler=None,
            num_workers=1,
        )
    
    def test_pipeline_with_invalid_device(self):
        with pytest.raises(ValueError):
            pipe = HybridTrainPipelineSparseDist(
                self.model,
                optimizer=Adagrad,
                device=torch.device("cpu"),
                return_loss=True
            )
    
    def test_pipeline_with_invalid_pipe_n_batch(self):
        with pytest.raises(ValueError):
            pipe = HybridTrainPipelineSparseDist(
                self.model,
                optimizer=Adagrad,
                device=torch.device("cpu"),
                pipe_n_batch=0,
                return_loss=True
            )

    def test_pipeline_with_invalid_model(self):
        with pytest.raises(TypeError):
            pipe = HybridTrainPipelineSparseDist(
                None,
                optimizer=Adagrad,
                device=torch.device("cpu"),
                pipe_n_batch=6,
                return_loss=True
            )

    def test_hybrid_train_pipeline_context(self):
        iter_ = iter(self.data_loader)
        batch = next(iter_, None)

        if batch is None:
            init_context = HybridTrainPipelineContext(batch)
            _fuse_input_dist_splits(init_context)
