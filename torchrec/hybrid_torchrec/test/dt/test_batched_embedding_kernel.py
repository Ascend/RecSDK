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
from parameterized import parameterized
import torch
from torch.optim import Adam, Adagrad, SGD

from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)

sys.modules['torch_npu'] = MagicMock

from hybrid_torchrec.distributed.batched_embedding_kernel import ( 
    HybridSplitTableBatchedEmbeddingBagsCodegen,
)

from torchrec import ComputeDevice

TORCH_OPTIMIZER_TO_FBGEMM = {
    Adam: EmbOptimType.ADAM,
    Adagrad: EmbOptimType.EXACT_ADAGRAD,
    SGD: EmbOptimType.EXACT_SGD
}


class TestHybridSplitTableBatchedEmbeddingBagsCodegen(unittest.TestCase):
    def setUp(self):
        self.indices = torch.Tensor([0, 1, 2, 3, 1]).to(torch.int64)
        self.offsets = torch.Tensor([0, 2, 4, 5]).to(torch.int64)
        self.hash_indices = torch.Tensor([0, 1, 2, 3]).to(torch.int64)
        self.unique_indices = torch.Tensor([0, 1, 2, 3]).to(torch.int64)
        self.unique_inverse = torch.Tensor([0, 1, 2, 3, 1]).to(torch.int64)
        self.per_sample_weights = torch.Tensor([1.0, 2.0])
        self.batch_size_per_feature_per_rank = ([1, 1], [1, 1])
        tables = [[100, 32], [200, 64]]
        self.embedding_specs = [
            (num_embeddings, embedding_dim, EmbeddingLocation.DEVICE, ComputeDevice.NPU)
            for (num_embeddings, embedding_dim) in tables
        ]
    
    @parameterized.expand([
        ("SGD", SGD, "lookup_sgd.invoke"),
        ("Adagrad", Adagrad, "lookup_adagrad.invoke"),
        ("Adam", Adam, "lookup_adam.invoke"),
    ])
    def test_forward_with_all_parameter_return_success(self, name, optim, mock_target):
        # 1. Mock 优化器调用
        with patch(f"hybrid_torchrec.hybrid_lookup_invoke.{mock_target}") as mock_invoke:
            tbe = HybridSplitTableBatchedEmbeddingBagsCodegen(
                self.embedding_specs,
                optimizer=TORCH_OPTIMIZER_TO_FBGEMM[optim],
                pooling_mode=PoolingMode.SUM,
                device=torch.device("meta")
            )
            mock_result = torch.Tensor([1, 2, 3]).to(torch.float)
            tbe.iter = torch.Tensor([0]) # device为meta类型需要对使用的tensor进行初始化
            mock_invoke.return_value = mock_result
            result = tbe(self.indices,
                         self.offsets,
                         self.hash_indices,
                         self.unique_indices,
                         self.unique_inverse)
            assert torch.equal(mock_result, result)
    
    def test_forward_with_unsupported_optim(self):
        tbe = HybridSplitTableBatchedEmbeddingBagsCodegen(
            self.embedding_specs,
            optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            pooling_mode=PoolingMode.SUM,
            device=torch.device("meta")
        )
        tbe.iter = torch.Tensor([0])
        assert(tbe(self.indices,
                   self.offsets,
                   self.hash_indices,
                   self.unique_indices,
                   self.unique_inverse) == NotImplemented)