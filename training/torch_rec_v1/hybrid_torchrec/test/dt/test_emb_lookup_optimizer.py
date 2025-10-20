#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import pytest

import torch

from hybrid_torchrec.hybrid_lookup_invoke.hybrid_lookup_args import HybridCommonArgs

from hybrid_torchrec.hybrid_lookup_invoke.hybrid_lookup_adagrad import (
    check_unique_valid,
    invoke,
)
from fbgemm_gpu.split_embedding_codegen_lookup_invokers.lookup_adagrad import (
    CommonArgs,
    OptimizerArgs,
    VBEMetadata,
    Momentum,
)


class TestHybridOps(unittest.TestCase):
    def setUp(self):
        """生成测试所需的各种Tensor"""
        self.device = "cpu"
        self.placeholder = torch.randn(1, requires_grad=True)
        self.dev_weights = torch.randn(10, device=self.device)
        self.host_weights = torch.randn(10)
        self.uvm_weights = torch.randn(10)
        self.lxu_cache_weights = torch.randn(5, device=self.device)
        self.weights_placements = torch.tensor([0, 1], dtype=torch.long)
        self.D_offsets = torch.tensor([0, 5], dtype=torch.int32)
        self.hash_size_cumsum = torch.tensor([0, 100], dtype=torch.long)
        self.indices = torch.arange(10, dtype=torch.long)
        # 初始化待校验数据
        self.hash_indices = torch.tensor([11, 22, 11, 33, 22, 33], dtype=torch.long)
        self.unique_indices, self.unique_inverse = torch.unique(self.hash_indices, return_inverse=True)
        self.weights_offsets = torch.tensor([0, 3], dtype=torch.long)
        self.offsets = torch.tensor([0, 2, 4, 6], dtype=torch.long)  # 3个batch [[11,22],[11,33],[22,33]]
        self.unique_offset = torch.tensor([0, len(self.unique_indices)], dtype=torch.long)
        self.hash_indices2address = torch.zeros_like(self.hash_indices)
        self.lxu_cache_locations = torch.randint(0, 5, (10,), dtype=torch.long)

        self.vbe_metadata = VBEMetadata(
            B_offsets=None,
            output_offsets_feature_rank=None,
            B_offsets_rank_per_feature=None
        )

        self.args = HybridCommonArgs(
            placeholder_autograd_tensor=self.placeholder,
            dev_weights=self.dev_weights,
            host_weights=self.host_weights,
            uvm_weights=self.uvm_weights,
            lxu_cache_weights=self.lxu_cache_weights,
            weights_placements=self.weights_placements,
            weights_offsets=self.weights_offsets,
            D_offsets=self.D_offsets,
            total_D=10,
            max_D=5,
            hash_size_cumsum=self.hash_size_cumsum,
            total_hash_size_bits=7,
            indices=self.indices,
            offsets=self.offsets,
            hash_indices=self.hash_indices,
            unique_indices=self.unique_indices,
            unique_offset=self.unique_offset,
            unique_inverse=self.unique_inverse,
            hash_indices2address=self.hash_indices2address,
            pooling_mode=0,
            indice_weights=None,
            feature_requires_grad=None,
            lxu_cache_locations=self.lxu_cache_locations,
            uvm_cache_stats=None,
            output_dtype=torch.float32,
            vbe_metadata=self.vbe_metadata,
            is_experimental=False,
            use_uniq_cache_locations_bwd=False,
            use_homogeneous_placements=True
        )

    def create_optimizer_args(self):
        optimizer_args = OptimizerArgs(
            gradient_clipping=False,
            max_gradient=1.0,
            stochastic_rounding=False,
            learning_rate=0.01,
            eps=1e-8,
            # 必填参数默认值
            max_norm=None,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.0,
            weight_decay_mode="l2",
            eta=0.0,
            momentum=0.0,
            counter_halflife=None,
            adjustment_iter=0,
            adjustment_ub=None,
            learning_rate_mode="fixed",
            grad_sum_decay=0.99,
            tail_id_threshold=0,
            is_tail_id_thresh_ratio=False,
            total_hash_size=None,
            weight_norm_coefficient=0.0,
            lower_bound=None,
            regularization_mode="l2"
        )
        momentum1 = Momentum(
            host=torch.zeros_like(self.args.dev_weights),
            offsets=torch.tensor([0], dtype=torch.long),
            placements=torch.tensor([0], dtype=torch.long),
            dev=torch.empty(0),
            uvm=torch.empty(0)
        )
        return optimizer_args, momentum1

    def test_check_unique_valid_success(self):
        """测试数据结构初始化"""
        self.assertEqual(self.args.total_D, 10)
        self.assertEqual(self.args.max_D, 5)
        self.assertEqual(self.args.pooling_mode, 0)
        self.assertIsNone(self.args.indice_weights)

        # 抛出RuntimeError
        with self.assertRaises(RuntimeError):
            check_unique_valid(self.args)
        # 验证分支
        new_args = self.args._replace(hash_indices=None)
        assert check_unique_valid(new_args) is None

    def test_invoke(self):
        # 测试CPU路径
        optimizer_args, momentum1 = self.create_optimizer_args()
        # CPU分支正常执行
        output = invoke(self.args, optimizer_args, momentum1)
        self.assertEqual(output.shape, (3, 10))

        # 测试npu分支
        new_args2 = self.args._replace(host_weights=torch.empty(0))

        # 抛出异常: 走到NPU分支, 执行报错
        #   在有NPU的环境，调用到NPU算子，抛出RuntimeError
        #   无NPU环境，无fbgemm接口，抛出AttributeError
        with pytest.raises((AttributeError, RuntimeError)):
            _ = invoke(new_args2, optimizer_args, momentum1)

    def test_invoke_with_vbe_failed(self):
        optimizer_args, momentum1 = self.create_optimizer_args()
        vbe_metadata = VBEMetadata(
            B_offsets=torch.tensor([0, 3, 5], dtype=torch.long),
            output_offsets_feature_rank=torch.tensor([0, 2, 4], dtype=torch.long),
            B_offsets_rank_per_feature=torch.tensor([[0, 2], [3, 5]], dtype=torch.long),
            max_B=4,
            output_size=6
        )

        # 测试vbe分支
        new_args = self.args._replace(vbe_metadata=vbe_metadata)

        # 执行被测试代码会抛出异常
        with pytest.raises(ValueError):
            _ = invoke(new_args, optimizer_args, momentum1)
