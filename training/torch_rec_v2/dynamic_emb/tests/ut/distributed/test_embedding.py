#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from typing import Optional, Dict, Any
import unittest
from unittest.mock import patch, MagicMock, Mock

import pytest
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.fbgemm_qcomm_codec import get_qcomm_codecs_registry, QCommsConfig, CommType

from dynamic_emb.distributed.embedding import ShardedDynamicEmbeddingCollection, DynamicEmbeddingCollectionSharder
from dynamic_emb.distributed.sharding.rw_sequence_sharding import RwSequenceDynamicEmbeddingSharding

_original_dynamic_emb_extensions = sys.modules.get("dynamic_emb_extensions")


def setup_module():
    mock_dynamicemb = MagicMock()
    sys.modules["dynamic_emb_extensions"] = mock_dynamicemb


def teardown_module():
    sys.modules["dynamic_emb_extensions"] = _original_dynamic_emb_extensions


class MockKeyedJaggedTensor:
    def __init__(self, keys, lengths, offsets, values, variable_stride=False):
        self._keys = keys
        self._lengths = lengths
        self._offsets = offsets
        self._values = values
        self._variable_stride = variable_stride

    def keys(self):
        return self._keys

    def lengths(self):
        return self._lengths

    def offsets(self):
        return self._offsets

    def values(self):
        return self._values

    def variable_stride_per_key(self):
        return self._variable_stride

    def permute(self, order, order_tensor):
        return self

    def split(self, splits):
        return [self for _ in splits]


class MockEmbeddingCollectionContext:
    """仅模拟上下文所需接口"""

    def __init__(self):
        self.input_features = []
        self.reverse_indices = []
        self.sharding_contexts = []


class MockInputDist:
    """模拟输入分发器"""

    def __init__(self, unbucketize_permute_tensor=None):
        self.unbucketize_permute_tensor = unbucketize_permute_tensor
        self.features = None

    def __call__(self, features):
        self.features = features
        return Mock()


def create_mock_buffer(shape, dtype=torch.int64, device="cpu"):
    """生成模拟缓冲区"""
    return torch.randint(0, 100, shape, dtype=dtype, device=device)


class TestShardedDynamicEmbeddingCollection(unittest.TestCase):
    def setUp(self):
        """前置准备：初始化设备和基础 Mock 对象"""
        self.device = torch.device("cpu")

        # 模拟分片（用于 _create_lookups 测试）
        self.mock_sharding = Mock(RwSequenceDynamicEmbeddingSharding)
        self.mock_sharding._grouped_embedding_configs = [
            Mock(compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL, pooling=None, fused_params={})
        ]
        self.mock_sharding.create_lookup = Mock(return_value=Mock())

    @patch.object(ShardedEmbeddingCollection, "__init__", return_value=None)
    def _create_test_instance(self, mock_parent_init: Mock, **kwargs) -> ShardedDynamicEmbeddingCollection:
        """
        创建测试实例：Mock 父类 __init__，直接注入子类所需属性
        :param mock_parent_init: 父类 __init__ 的 Mock 对象（由 @patch 自动传入）
        :return: 配置好的子类实例
        """
        # 实例化子类（父类 __init__ 已被 Mock，无需传入任何父类参数）
        ec = ShardedDynamicEmbeddingCollection()
        ec._parameters = {}  # 模块参数（空字典即可）
        ec._modules = {}  # 子模块（空字典即可）
        ec._buffers = {}  # 缓冲区（空字典即可）
        ec._hooks = []  # 钩子函数（空列表即可）

        # 注入子类必需的属性（覆盖父类依赖，仅保留子类自身逻辑所需）
        ec._use_index_dedup = False
        ec._device = self.device
        ec._device_num_sms = 32  # 用于 _dedup_indices
        ec._unique_op = Mock()  # 用于 _dedup_indices
        ec._sharding_type_to_sharding = {"sequence": self.mock_sharding}  # 用于 _create_lookups
        ec._lookups = []  # 用于 _create_lookups
        ec._feature_splits = [1]  # 用于 input_dist
        ec._features_order = kwargs.get("features_order", None)  # 用于 input_dist 的 permute
        ec._has_uninitialized_input_dist = False  # 用于 input_dist
        ec._input_dists = [MockInputDist()]  # 用于 input_dist
        ec.use_index_dedup = kwargs.get("use_index_dedup", True)  # 用于 input_dist 和 _dedup_indices
        ec.module_fqn = "test.module.ec"  # 可选属性

        # 模拟 get_buffer 方法（返回测试所需缓冲区）
        ec.get_buffer = Mock(
            side_effect=lambda name: {
                "_hash_size_offset_tensor_0": create_mock_buffer((2,), device=self.device),
                "_nonfuse_table_feature_offsets_host_0": create_mock_buffer((3,), dtype=torch.uint64, device="cpu"),
                "_nonfuse_table_feature_offsets_device_0": create_mock_buffer(
                    (3,), dtype=torch.uint64, device=self.device
                ),
            }[name]
        )

        return ec

    @patch.object(ShardedEmbeddingCollection, "__init__", return_value=None)
    def test_create_lookups_no_pooling(self, mock_parent_init):
        self.mock_sharding._grouped_embedding_configs[0].pooling = None
        ec = self._create_test_instance()
        ec._create_lookups()
        # 验证配置未被修改
        config = self.mock_sharding._grouped_embedding_configs[0]
        self.assertNotIn("use_index_dedup", config.fused_params)

    @patch.object(ShardedEmbeddingCollection, "__init__", return_value=None)
    def test_dedup_indices_dtype_int32(self, mock_parent_init):
        input_values = torch.tensor([10, 20, 10, 30, 20], dtype=torch.int32, device=self.device)
        self._run_dedup_test(
            input_values=input_values,
            expected_unique_values=torch.tensor([10, 20, 30], dtype=torch.int32, device=self.device),
        )

    def _run_dedup_test(
        self,
        input_values: torch.Tensor,
        expected_unique_values: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        input_offsets: Optional[torch.Tensor] = None,
    ):
        try:
            ec = self._create_test_instance()
            if ec is None:
                self.fail("_create_test_instance() failed")
            if not isinstance(ec, ShardedDynamicEmbeddingCollection):
                self.fail(
                    f"_create_test_instance() failed: invalid return type, "
                    f"expected {ShardedDynamicEmbeddingCollection.__name__}, "
                    f"got {type(ec).__name__ if ec else 'NoneType'}"
                )
        except Exception as e:
            self.fail(f"Create test instance failed：{type(e).__name__}: {str(e)}")
        if input_lengths is None:
            input_lengths = torch.tensor([3, 2], device=self.device)  # 对应 input_values 长度 5
        if input_offsets is None:
            input_offsets = torch.tensor([0, 3, 5], device=self.device)  # 对应 lengths [3,2]

        # 构造输入 KJT
        input_kjt = MockKeyedJaggedTensor(
            keys=["feature1"], lengths=input_lengths, offsets=input_offsets, values=input_values
        )

        # 模拟 dedup_input_indices 算子输出（关键修复：匹配 unique_idx_list 长度）
        with patch("dynamic_emb.distributed.embedding.dedup_input_indices_op") as mock_dedup_op:

            def side_effect(*args):
                (
                    indices_input,
                    offsets,
                    d_table_offset,
                    table_num,
                    local_batchsize,
                    reverse_idx,
                    d_unique_nums,
                    d_unique_offsets,
                    unique_idx_list,
                    new_offsets,
                    new_lengths,
                ) = args

                # 有效长度：table0 占 2 个，table1 占 1 个，总有效长度 3
                d_unique_offsets.copy_(torch.tensor([0, 2, 3], dtype=torch.uint64, device="cpu"))
                d_unique_nums.copy_(torch.tensor([2, 1], dtype=torch.uint64, device="cpu"))

                # table0 有效数据：[10,20]，剩余部分填充无效值（后续会被 h_unique_offsets 截断）
                table0_data = torch.full_like(input_values, fill_value=-1, dtype=torch.int64, device=self.device)
                table0_data[:2] = torch.tensor([10, 20], dtype=torch.int64, device=self.device)
                unique_idx_list[0].copy_(table0_data)

                # table1 有效数据：[30]，剩余部分填充无效值
                table1_data = torch.full_like(input_values, fill_value=-1, dtype=torch.int64, device=self.device)
                table1_data[:1] = torch.tensor([30], dtype=torch.int64, device=self.device)
                unique_idx_list[1].copy_(table1_data)

                # 3. 模拟新的 offsets 和 lengths（与输入一致）
                new_offsets.copy_(input_kjt.offsets())
                new_lengths.copy_(input_kjt.lengths())

            mock_dedup_op.side_effect = side_effect

            # 执行去重
            ctx = MockEmbeddingCollectionContext()
            result = ec._dedup_indices(ctx, [input_kjt])

        # 验证输出
        self.assertEqual(len(result), 1)
        dedup_kjt = result[0]

        # 验证 dtype 正确
        self.assertEqual(dedup_kjt.values().dtype, expected_unique_values.dtype)

        # 验证去重后 values 正确（有效长度由 h_unique_offsets[-1] 决定）
        self.assertEqual(dedup_kjt.values().shape, expected_unique_values.shape)
        torch.testing.assert_close(dedup_kjt.values(), expected_unique_values)

        # 验证 offsets 和 lengths 与输入一致
        torch.testing.assert_close(dedup_kjt.offsets(), input_kjt.offsets())
        torch.testing.assert_close(dedup_kjt.lengths(), input_kjt.lengths())

        # 验证上下文存储
        self.assertEqual(len(ctx.input_features), 1)
        self.assertEqual(len(ctx.reverse_indices), 1)

    @patch.object(ShardedEmbeddingCollection, "__init__", return_value=None)
    def test_input_dist_without_dedup(self, mock_parent_init):
        ec = self._create_test_instance(use_index_dedup=False)

        input_kjt = MockKeyedJaggedTensor(
            keys=["feature1"],
            lengths=torch.tensor([2], device=self.device),
            offsets=torch.tensor([0, 2], device=self.device),
            values=torch.tensor([10, 20], dtype=torch.int64, device=self.device),
        )

        with patch.object(ec, "_dedup_indices") as mock_dedup:
            ctx = MockEmbeddingCollectionContext()
            ec.input_dist(ctx, input_kjt)

        mock_dedup.assert_not_called()


class TestDynamicEmbeddingCollectionSharder:
    @staticmethod
    def test_init_ok():
        optimizer_kwargs = {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0,
            "eps": 0.001,
        }
        fused_params: Dict[str, Any]
        fused_params = {"output_dtype": SparseType.FP32}
        fused_params.update(optimizer_kwargs)
        fused_params["prefetch_pipeline"] = False  # whether enable prefetch for embedding lookup module
        qcomm_codecs_registry = get_qcomm_codecs_registry(
            qcomms_config=QCommsConfig(
                forward_precision=CommType.FP32,
                backward_precision=CommType.FP32,
            )
        )
        try:
            DynamicEmbeddingCollectionSharder(
                qcomm_codecs_registry=qcomm_codecs_registry,
                fused_params=fused_params,
                use_index_dedup=False,
            )
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    @pytest.mark.parametrize("invalid_value", ["xxx", {123: "xxx"}])
    def test_fused_params_value_err(invalid_value):
        with pytest.raises(ValueError):
            DynamicEmbeddingCollectionSharder(fused_params=invalid_value)

    @staticmethod
    @pytest.mark.parametrize("invalid_value", ["xxx", {123: "xxx"}, {"123": "xxx"}])
    def test_qcomm_codecs_registry_value_err(invalid_value):
        with pytest.raises(ValueError):
            DynamicEmbeddingCollectionSharder(qcomm_codecs_registry=invalid_value)

    @staticmethod
    def test_use_index_dedup_value_err():
        with pytest.raises(ValueError):
            DynamicEmbeddingCollectionSharder(
                use_index_dedup="xxx",
            )
