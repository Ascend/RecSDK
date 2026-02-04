#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import sys
import unittest
from unittest.mock import patch, MagicMock

import torch

from dynamic_emb.distributed.batched_dynamicemb_function import (
    DynamicEmbeddingFunctionV2, 
    DynamicEmbeddingFunctionV2Config,
)

_original_dynamic_emb_extensions = sys.modules.get("dynamic_emb_extensions")


def setup_module():
    mock_dynamicemb = MagicMock()
    sys.modules["dynamic_emb_extensions"] = mock_dynamicemb


def teardown_module():
    sys.modules["dynamic_emb_extensions"] = _original_dynamic_emb_extensions


class MockStorage:
    def __init__(self, emb_dim, dtype=torch.float32):
        self.emb_dim = emb_dim
        self.dtype = dtype
        self.data = {}  # {index: embedding tensor}

    def embedding_dim(self):
        return self.emb_dim

    def embedding_dtype(self):
        return self.dtype

    def lookup(self, indices, out, initializer, training):
        for i, idx in enumerate(indices.cpu().numpy()):
            if idx not in self.data:
                self.data[idx] = initializer.initialize(idx, self.emb_dim, self.dtype, indices.device)
            out[i] = self.data[idx].to(indices.device)

    def update(self, indices, grads, optimizer):
        for idx, grad in zip(indices.cpu().numpy(), grads):
            self.data[idx] = optimizer.update(self.data[idx], grad)


class MockInitializer:
    @staticmethod
    def initialize(idx, emb_dim, dtype, device):
        return torch.tensor([idx + i for i in range(emb_dim)], dtype=dtype, device=device)


class MockOptimizer:
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self):
        pass

    def update(self, embedding, grad):
        return embedding - self.lr * grad


class MockCache:
    def __init__(self):
        self.emb_dim = 0


class TestDynamicEmbeddingFunctionV2(unittest.TestCase):
    def setUp(self):
        self.emb_dim = 2
        self.table_num = 1
        self.storages = [MockStorage(emb_dim=self.emb_dim)]
        self.initializers = [MockInitializer()]
        self.optimizer = MockOptimizer(lr=0.1)
        self.caches = [None]  # 禁用缓存
        self.output_dtype = torch.float32
        self.unique_op = "mock"  # 不影响测试的假操作

    def test_forward(self):
        indices = torch.tensor([5], device="cpu")
        offsets = torch.tensor([0, 1])
        feature_offsets = torch.tensor([1])
        feature_offsets = torch.cat([offsets[0:1], feature_offsets])  # 符合get_table_range逻辑

        def set_val_fun1(storage, indices_per_table, embs_per_table, initializer, training):
            embs_per_table.fill_(1.0)

        # 创建多个patch上下文
        with patch("dynamic_emb.distributed.key_value_table.KeyValueTableFunction.lookup", side_effect=set_val_fun1), \
             patch("dynamic_emb.distributed.batched_dynamicemb_function.gather_embedding",
                   return_value=torch.tensor([[1.0, 1.0]], dtype=torch.float32)) as mocked_gather_embedding, \
             patch("dynamic_emb.distributed.batched_dynamicemb_function.get_table_range_op",
                   return_value=torch.tensor([0, 1], dtype=torch.long)) as mocked_get_table_range, \
                patch("dynamic_emb.distributed.batched_dynamicemb_function.segmented_unique_op", return_value=(
                        torch.tensor([5], device="cpu"),
                        torch.tensor([0], device="cpu"),
                        torch.tensor([0, 1], device="cpu"),
                        torch.tensor([0, 1], device="cpu")
                )) as mocked_segmented_unique:
            # 前向传播
            emb_config = DynamicEmbeddingFunctionV2Config(
                indices=indices,
                offsets=offsets,
                caches=self.caches,
                storages=self.storages,
                feature_offsets=feature_offsets,
                output_dtype=self.output_dtype,
                initializers=self.initializers,
                optimizer=self.optimizer,
                enable_prefetch=self.unique_op,
                input_dist_dedup=False,
                training=False
            )
            output_embs = DynamicEmbeddingFunctionV2.apply(
                emb_config,
                True,
            )

            # 验证前向传播结果
            expected = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
            self.assertTrue(torch.allclose(output_embs, expected))
            mocked_get_table_range.assert_called_once()

    def test_manual_backward(self):
        class MockCtx:
            def __init__(self, indices, unique_indices, unique_embs, inverse, indices_table_range,
                         h_unique_indices_table_range, caches, storages, optimizer):
                self.saved_tensors = (indices,)  # 接收外部传入的indices，而非内部self.indices
                self.unique_indices = unique_indices
                self.unique_embs = unique_embs
                self.inverse = inverse
                self.indices_table_range = indices_table_range
                self.h_indices_table_range = indices_table_range.cpu()
                self.h_unique_indices_table_range = h_unique_indices_table_range
                self.unique_indices_table_range = None
                self.caches = caches
                self.storages = storages
                self.optimizer = optimizer
                self.input_dist_dedup = False

        indices = torch.tensor([5], device="cpu")  # 原始输入索引
        unique_indices = torch.tensor([5], device="cpu")  # 去重后的索引
        inverse = torch.tensor([0], device="cpu")  # 去重映射关系
        indices_table_range = torch.tensor([0, 1], device="cpu")  # 索引表范围
        h_unique_indices_table_range = torch.tensor([0, 1], device="cpu")  # CPU端范围
        unique_embs = torch.tensor([[5.0, 6.0]], device="cpu")  # 索引5的初始嵌入：[5,6]

        ctx = MockCtx(
            indices=indices,
            unique_indices=unique_indices,
            unique_embs=unique_embs,
            inverse=inverse,
            indices_table_range=indices_table_range,
            h_unique_indices_table_range=h_unique_indices_table_range,
            caches=self.caches,
            storages=self.storages,
            optimizer=self.optimizer
        )
        grads = torch.tensor([[1.0, 1.0]], device="cpu")

        with patch("dynamic_emb.distributed.key_value_table.KeyValueTableFunction.update") as mocked_update, \
            patch("dynamic_emb.distributed.batched_dynamicemb_function.reduce_grads", return_value=(
                torch.tensor([5], device="cpu"),
                torch.tensor([[1.0, 1.0]], dtype=torch.float32)
            )) as mocked_reduce_grads:
            DynamicEmbeddingFunctionV2.backward(ctx, grads)
            mocked_update.assert_called_once()
            mocked_reduce_grads.assert_called_once()
