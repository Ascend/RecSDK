#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from typing import List
import unittest
from unittest.mock import patch, MagicMock
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
    HybridBatchedFusedEmbeddingBag,
    HybridBatchedFusedEmbedding,
)
from hybrid_torchrec.distributed.embedding_lookup import (
    HybridGroupedEmbeddingsLookup,
    HybridGroupedPooledEmbeddingsLookup
)
from hybrid_torchrec.sparse.jagged_tensor_with_looup_helper import (
    KeyedJaggedTensorWithLookHelper,
)

from torchrec import ComputeDevice
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.types import DataType
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardingType,
)

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
        self.tables = [[100, 32], [200, 64]]
        self.embedding_specs = [
            (num_embeddings, embedding_dim, EmbeddingLocation.DEVICE, ComputeDevice.NPU)
            for (num_embeddings, embedding_dim) in self.tables
        ]
        self.base_model = HybridSplitTableBatchedEmbeddingBagsCodegen(
            self.embedding_specs,
            optimizer=TORCH_OPTIMIZER_TO_FBGEMM[Adam],
            pooling_mode=PoolingMode.SUM,
            device=torch.device("cpu")
        )
    
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
                device=torch.device("cpu")
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

            # 测试indices.detype != offset.dtype
            self.indices = torch.tensor([0, 1, 2, 3, 1]).to(torch.float)
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
            device=torch.device("cpu")
        )
        tbe.iter = torch.Tensor([0])
        assert tbe(self.indices,
                   self.offsets,
                   self.hash_indices,
                   self.unique_indices,
                   self.unique_inverse) == NotImplemented

    def test_forward_check_vbe_metadata(self):
        with patch(f"hybrid_torchrec.hybrid_lookup_invoke.lookup_sgd.invoke") as mock_invoke:
            mock_result = torch.Tensor([1, 2, 3]).to(torch.float)
            mock_invoke.return_value = mock_result
            tbe = HybridSplitTableBatchedEmbeddingBagsCodegen(
                self.embedding_specs,
                optimizer=EmbOptimType.EXACT_SGD,
                pooling_mode=PoolingMode.SUM,
                device=torch.device("cpu")
            )
            tbe.iter = torch.Tensor([0])
            batch_size_per_feature_per_rank = [[2]]
            result = tbe(self.indices,
                self.offsets,
                self.hash_indices,
                self.unique_indices,
                self.unique_inverse,
                batch_size_per_feature_per_rank=batch_size_per_feature_per_rank)
            assert torch.equal(mock_result, result)

    def test_scatter_update_embs(self):
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        # 行保持不变，列4倍
        updates = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=torch.float).repeat(1, 4)
        # 默认参数不支持
        with self.assertRaises(ValueError) as cm:
            self.base_model.scatter_update_embs(indices, updates)
        self.assertIn("Mixed dimensions are not supported", str(cm.exception))
        # 修改默认参数能正常执行不报错
        self.base_model.is_mixed_dim = False
        self.base_model.scatter_update_embs(indices, updates)

    def test_gather_embs(self):
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        # 默认参数不支持
        with self.assertRaises(ValueError) as cm:
            _ = self.base_model.gather_embs(indices)
        self.assertIn("Mixed dimensions are not supported", str(cm.exception))
        # 修改默认参数能正常执行不报错
        self.base_model.is_mixed_dim = False
        result_mixed = self.base_model.gather_embs(indices)
        self.assertIsInstance(result_mixed, torch.Tensor)

    def test_gather_momentum(self):
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        # 默认参数不支持
        with self.assertRaises(ValueError) as cm:
            _ = self.base_model.gather_momentum(indices)
        self.assertIn("Mixed dimensions are not supported", str(cm.exception))
        # 修改默认参数能正常执行不报错
        self.base_model.is_mixed_dim = False
        result = self.base_model.gather_momentum(indices)
        # adam 优化器动量为2阶动量，保持为list
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_scatter_update_momentum(self):
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        updates = [torch.tensor([0.1, 0.2], dtype=torch.float).repeat(16),
                   torch.tensor([0.3, 0.4], dtype=torch.float).repeat(16)]  # 构造dim32
        # 默认参数不支持
        with self.assertRaises(ValueError) as cm:
            self.base_model.scatter_update_momentum(indices, updates)
        self.assertIn("Mixed dimensions are not supported", str(cm.exception))
        # 修改默认参数,验证动量是否更新
        self.base_model.is_mixed_dim = False
        # 初始化动量
        total_size = sum([t[0] * t[1] for t in self.tables])
        self.base_model.momentum1_dev = torch.zeros((total_size,), dtype=torch.float)
        self.base_model.momentum2_dev = torch.zeros((total_size,), dtype=torch.float)
        # 更新,不等于0
        self.base_model.scatter_update_momentum(indices, updates)
        self.assertNotEqual(torch.sum(self.base_model.momentum1_dev), 0)
        self.assertNotEqual(torch.sum(self.base_model.momentum2_dev), 0)

    def test_get_momentum(self):
        result = self.base_model.get_momentum()
        # 初始化使用的2阶动量
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)


class EmbeddingTable:
    def __init__(self,
                 name: str,
                 feature_names: List[str],
                 local_cols: int,
                 local_rows: int,
                 compute_kernel: str):
        self.name = name
        self.feature_names = feature_names
        self.local_cols = local_cols
        self.local_rows = local_rows
        self.compute_kernel = compute_kernel

    def num_features(self) -> int:
        return len(self.feature_names)


class TestHybridBatchedFusedEmbedding(unittest.TestCase):
    def setUp(self):
        # 构造一个简单的Embedding_table
        self.embedding_tables = [
            EmbeddingTable(
                name="table1",
                feature_names=["feature1"],
                local_cols=16,
                local_rows=200,
                compute_kernel=EmbeddingComputeKernel.DENSE
            ),
            EmbeddingTable(
                name="table2",
                feature_names=["feature2"],
                local_cols=16,
                local_rows=200,
                compute_kernel=EmbeddingComputeKernel.DENSE
            )
        ]
        # 构造GroupedEmbeddingConfig
        self.config = GroupedEmbeddingConfig(
            data_type=DataType.FP32,
            pooling=PoolingType.SUM,
            is_weighted=True,
            has_feature_processor=False,
            compute_kernel=EmbeddingComputeKernel.DENSE,
            embedding_tables=self.embedding_tables,
        )
        # EBC
        with patch(f"torchrec.distributed.batched_embedding_kernel.BaseBatchedEmbeddingBag.__init__") as mock_init:
            with patch(
                    f"hybrid_torchrec.distributed.batched_embedding_kernel.HybridBatchedFusedEmbeddingBag.__init__"
            ) as mock_lookup:
                mock_init.return_value = None
                mock_lookup.return_value = None
                self.model_ebc = HybridBatchedFusedEmbeddingBag(
                    config=self.config,
                    pg=None,
                    device=None,
                    sharding_type=None
                )

        # EC
        with patch(f"torchrec.distributed.batched_embedding_kernel.BaseBatchedEmbedding.__init__") as mock_init:
            with patch(
                    f"hybrid_torchrec.distributed.batched_embedding_kernel.HybridBatchedFusedEmbedding.__init__"
            ) as mock_lookup:
                mock_init.return_value = None
                mock_lookup.return_value = None
                self.model_ec = HybridBatchedFusedEmbedding(
                    config=self.config,
                    pg=None,
                    device=None
                )

    def test_init_local_cols_error(self):
        with patch(f"torchrec.distributed.batched_embedding_kernel.BaseBatchedEmbeddingBag.__init__") as mock_init:
            mock_init.return_value = None
            # 修改cols使其不满足校验
            self.embedding_tables[0].local_cols = 15
            # 初始化异常
            with self.assertRaises(ValueError) as cm:
                _ = HybridBatchedFusedEmbeddingBag(
                    config=self.config,
                    pg=None,
                    device=None,
                    sharding_type=None
                )
            self.assertIn("not divisible by 4", str(cm.exception))

    @parameterized.expand(
        [torch.device("cuda"),
        torch.device("mtia"),
        torch.device("cpu")]
    )
    def test_different_device(self, device):
        with patch(f"torchrec.distributed.batched_embedding_kernel.BaseBatchedEmbeddingBag.__init__") as mock_init:
            mock_init.return_value = None
            # 屏蔽了父类BaseBatchedEmbeddingBag初始化,会抛出指定异常
            with self.assertRaises(AttributeError):
                # ebc
                _ = HybridBatchedFusedEmbeddingBag(
                        config=self.config,
                        pg=None,
                        device=device,
                        sharding_type=None
                    )
        with patch(f"torchrec.distributed.batched_embedding_kernel.BaseBatchedEmbedding.__init__") as mock_init_ec:
            mock_init_ec.return_value = None
            with self.assertRaises(AttributeError):
                # ec
                _ = HybridBatchedFusedEmbedding(
                        config=self.config,
                        pg=None,
                        device=device
                    )

    @parameterized.expand([(True, False), (True, True), (False, False), (False, True)])
    def test_forward_variable_stride_per_key_true(self, variable_stride_per_key, weights_or_none):
        # 模拟KeyedJaggedTensorWithLookHelper
        features = MagicMock(spec=KeyedJaggedTensorWithLookHelper)
        features.values.return_value = torch.tensor([0, 1, 2])
        features.offsets.return_value = torch.tensor([0, 1, 2, 3])
        if weights_or_none:
            features.weights_or_none.return_value = torch.tensor([1, 2, 3], dtype=torch.int32)
        else:
            features.weights_or_none.return_value = None
        features.variable_stride_per_key.return_value = variable_stride_per_key
        # 设置self._emb_module 为 SplitTableBatchedEmbeddingBagsCodegen
        self.model_ebc._emb_module = MagicMock(spec=object)
        self.model_ec._emb_module = MagicMock(spec=object)
        # 设置返回值为torch.tensor
        self.model_ebc._emb_module.return_value = torch.tensor([1, 2, 3])
        self.model_ec._emb_module.return_value = torch.tensor([1, 2, 3])

        # 调用 forword 方法
        return_ebc = self.model_ebc.forward(features)
        return_ec = self.model_ec.forward(features)
        # weight不是None,也不是float时，会被赋值为None
        if weights_or_none:
            _, kwargs_ebc = self.model_ebc._emb_module.call_args
            _, kwargs_ec = self.model_ec._emb_module.call_args
            self.assertIs(kwargs_ebc.get("per_sample_weights"), None)
            self.assertIs(kwargs_ec.get("per_sample_weights"), None)
        else:
            # 验证是否正确调用了 emb_module 方法
            self.model_ebc._emb_module.assert_called_once()
            self.model_ec._emb_module.assert_called_once()
            # 验证返回值 Tensor
            self.assertIsInstance(return_ebc, torch.Tensor)
            self.assertIsInstance(return_ec, torch.Tensor)


class TestHybridGroupedEmbeddingsLookup(unittest.TestCase):
    def setUp(self):
        # 构造一个简单的Embedding_table
        self.embedding_tables = [
            EmbeddingTable(
                name="table1",
                feature_names=["feature1"],
                local_cols=16,
                local_rows=200,
                compute_kernel=EmbeddingComputeKernel.FUSED
            )]
        # 构造GroupedEmbeddingConfig
        self.group_config = [GroupedEmbeddingConfig(
            data_type=DataType.FP32,
            pooling=PoolingType.SUM,
            is_weighted=True,
            has_feature_processor=False,
            compute_kernel=EmbeddingComputeKernel.FUSED,
            embedding_tables=self.embedding_tables,
        )]
        with patch(
                f"hybrid_torchrec.distributed.batched_embedding_kernel.HybridBatchedFusedEmbeddingBag.__init__"
        ) as mock_lookup:
            mock_lookup.return_value = None
            # 创建测试对象
            self.lookup_ebc = HybridGroupedPooledEmbeddingsLookup(
                grouped_configs=self.group_config,
                device=torch.device("cpu"),
                pg=MagicMock(),
                sharding_type=ShardingType.ROW_WISE,
            )
        with patch(
                f"hybrid_torchrec.distributed.batched_embedding_kernel.HybridBatchedFusedEmbedding.__init__"
        ) as mock_lookup_ec:
            mock_lookup_ec.return_value = None
            # 创建测试对象
            self.lookup_ec = HybridGroupedEmbeddingsLookup(
                grouped_configs=self.group_config,
                device=torch.device("cpu"),
                pg=MagicMock()
            )

    def test_creat_lookup_success(self):
        self.assertEqual(len(self.lookup_ebc._emb_modules), 1)
        self.assertEqual(len(self.lookup_ec._emb_modules), 1)

    @parameterized.expand([EmbeddingComputeKernel.DENSE, EmbeddingComputeKernel.KEY_VALUE])
    def test_different_compute_kernel(self, kernel_type):
        # 更改参数
        self.group_config[0].compute_kernel = kernel_type
        if kernel_type == EmbeddingComputeKernel.DENSE:
            # EBC
            with self.assertRaises(TypeError) as cm:
                # 创建测试对象,抛出TypeError
                _ = HybridGroupedPooledEmbeddingsLookup(
                    grouped_configs=self.group_config,
                    device=torch.device("cpu"),
                    pg=MagicMock(),
                    sharding_type=ShardingType.ROW_WISE,
                )
            self.assertIn("NotImplementedType is not a Module subclass", str(cm.exception))
            # EC
            with self.assertRaises(TypeError) as cm:
                # 创建测试对象,抛出TypeError
                _ = HybridGroupedEmbeddingsLookup(
                    grouped_configs=self.group_config,
                    device=torch.device("cpu"),
                    pg=MagicMock(),
                )
            self.assertIn("NotImplementedType is not a Module subclass", str(cm.exception))
        if kernel_type == EmbeddingComputeKernel.KEY_VALUE:
            with patch(f"torchrec.distributed.batched_embedding_kernel.KeyValueEmbeddingBag.__init__") as mock_lookup:
                mock_lookup.return_value = None
                # 创建测试对象
                lookup_ebc = HybridGroupedPooledEmbeddingsLookup(
                    grouped_configs=self.group_config,
                    device=torch.device("cpu"),
                    pg=MagicMock(),
                    sharding_type=ShardingType.ROW_WISE,
                )
                self.assertEqual(len(lookup_ebc._emb_modules), 1)
            with patch(f"torchrec.distributed.batched_embedding_kernel.KeyValueEmbedding.__init__") as mock_lookup_ec:
                mock_lookup_ec.return_value = None
                # 创建测试对象
                lookup_ec = HybridGroupedEmbeddingsLookup(
                    grouped_configs=self.group_config,
                    device=torch.device("cpu"),
                    pg=MagicMock()
                )
                self.assertEqual(len(lookup_ec._emb_modules), 1)
