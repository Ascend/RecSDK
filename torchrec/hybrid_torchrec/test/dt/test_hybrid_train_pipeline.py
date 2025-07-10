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
from torch.optim import Adagrad, Adam, SGD
import torch.distributed as dist

from hybrid_torchrec.distributed.hybrid_train_pipeline import (
    HybridTrainPipelineSparseDist,
    HybridTrainPipelineContext,
    _fuse_input_dist_splits
)
from hybrid_torchrec import HashEmbeddingBagCollection, HashEmbeddingBagConfig
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders

from dataset import RandomRecDataset
from model import Model

import torchrec
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.keyed import CombinedOptimizer

OPTIMIZER_PARAM = {
    Adam: dict(lr=0.02),
    Adagrad: dict(lr=0.02, eps=1.0e-8),
    SGD: dict(lr=0.02),
}

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


def mock_copy_to_npu(batch, context):
    return batch


class TestHybridTrainPipelineSparseDist(unittest.TestCase):
    def setUp(self):
        self.rank = 0
        self.world_size = 1
        if not dist.is_initialized():
            dist.init_process_group(
                "gloo", init_method="tcp://127.0.0.1:6000",
                world_size=self.world_size, rank=self.rank)
        self.device = torch.device("cpu")
        lookup_len = 256
        embedding_dims = [32, 64]
        num_embeddings = [100, 200]
        table_num = len(embedding_dims)
        embedding_config = get_embedding_config(embedding_dims, num_embeddings, table_num)
        ebc = HashEmbeddingBagCollection(device=self.device, tables=embedding_config)
        num_features = sum([c.num_features() for c in embedding_config])
        self.model = Model(ebc, num_features)
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

    @patch("torchrec.distributed.planner.ParameterConstraints.__post_init__", return_value=None)
    @patch("torchrec.tensor_types.check", return_value=None)
    @patch("torchrec.distributed.model_parallel.check", return_value=None)
    @patch("torchrec.distributed.planner.types.check", return_value=None)
    @patch("hybrid_torchrec.distributed.hybrid_train_pipeline.HybridTrainPipelineSparseDist.param_check", \
            return_value=None)
    @patch("hybrid_torchrec.distributed.hybrid_train_pipeline.HybridTrainPipelineSparseDist._copy_to_npu", \
            side_effect=mock_copy_to_npu)
    def test_hybrid_train_pipeline_init_success(self, *mocks):
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=self.world_size, rank=self.rank, pg=host_gp)
    
        apply_optimizer_in_backward(
            optimizer_class=Adagrad,
            params=self.model.parameters(),
            optimizer_kwargs=OPTIMIZER_PARAM[Adagrad],
        )
        # Shard    
        constrans = {
            f"table{i}": ParameterConstraints(
                sharding_types=["table_wise"], compute_kernels=["fused"]
            )
            for i in range(2)
        }
        planner = EmbeddingShardingPlanner(
            topology=Topology(world_size=self.world_size, compute_device="cpu"),
            constraints=constrans,
        )
        plan = planner.collective_plan(
            self.model, get_default_hybrid_sharders(host_env), dist.GroupMember.WORLD
        )
        ddp_model = torchrec.distributed.DistributedModelParallel(
            self.model,
            sharders=get_default_hybrid_sharders(host_env),
            device=torch.device(self.device),
            plan=plan,
        )

        # Optimizer
        optimizer = CombinedOptimizer([ddp_model.fused_optimizer])

        ddp_model.train()
        pipe = HybridTrainPipelineSparseDist(
            ddp_model,
            optimizer=optimizer,
            device=torch.device(self.device),
            return_loss=True,
        )
        iter_ = iter(self.data_loader)
        pipe._fill_pipeline(iter_)
        assert(pipe._contexts[0][0].batch is not None)

    def test_hybrid_train_pipeline_context(self):
        iter_ = iter(self.data_loader)
        batch = next(iter_, None)

        if batch is None:
            init_context = HybridTrainPipelineContext(batch)
            _fuse_input_dist_splits(init_context)
