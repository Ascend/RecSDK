#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
import warnings
from unittest.mock import patch, MagicMock

import torch
from torch import nn

from dynamic_emb.distributed.incremental_dump import (
    _is_valid_score_threshold,
    set_score,
    get_score,
)
from dynamic_emb.distributed.batched_dynamicemb_table import BatchedDynamicEmbeddingTablesV2
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbTableOptions,
    DynamicEmbPoolingMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
)

from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import EmbOptimType

_original_dynamic_emb_extensions = sys.modules.get("dynamic_emb_extensions")


def setup_module():
    mock_dynamicemb = MagicMock()
    sys.modules["dynamic_emb_extensions"] = mock_dynamicemb


def teardown_module():
    sys.modules["dynamic_emb_extensions"] = _original_dynamic_emb_extensions


class TestIsValidScoreThreshold(unittest.TestCase):
    """_is_valid_score_threshold 的 UT"""

    def test_valid_dict(self):
        """合法 Dict[str, Dict[str, int]] 返回 True"""
        self.assertTrue(_is_valid_score_threshold({"model.emb": {"table1": 1}}))
        self.assertTrue(_is_valid_score_threshold({"c1": {"t1": 0, "t2": 2}}))

    def test_not_dict_returns_false(self):
        """非 dict 返回 False"""
        self.assertFalse(_is_valid_score_threshold(None))
        self.assertFalse(_is_valid_score_threshold(1))
        self.assertFalse(_is_valid_score_threshold([]))
        self.assertFalse(_is_valid_score_threshold("path"))

    def test_inner_value_not_dict_returns_false(self):
        """外层 value 不是 dict 返回 False"""
        self.assertFalse(_is_valid_score_threshold({"model.emb": [1, 2]}))
        self.assertFalse(_is_valid_score_threshold({"model.emb": 1}))

    def test_inner_key_not_str_returns_false(self):
        """内层 key 不是 str 返回 False"""
        self.assertFalse(_is_valid_score_threshold({"model.emb": {1: 1}}))

    def test_inner_value_not_int_returns_false(self):
        """内层 value 不是 int 返回 False"""
        self.assertFalse(_is_valid_score_threshold({"model.emb": {"table1": "1"}}))
        self.assertFalse(_is_valid_score_threshold({"model.emb": {"table1": 1.0}}))

    def test_outer_key_not_str_returns_false(self):
        """外层 key 不是 str 返回 False"""
        self.assertFalse(_is_valid_score_threshold({1: {"table1": 1}}))

    def test_empty_dict_returns_true(self):
        """空 dict 视为合法"""
        self.assertTrue(_is_valid_score_threshold({}))


class TestSetScoreAndGetScore(unittest.TestCase):
    """set_score / get_score 的 UT，复用 dump_load 的 mock 模型环境"""

    def setUp(self):
        self.mock_table = MagicMock()
        self.mock_optimizer = MagicMock()
        self.mock_optimizer.get_state_dim.return_value = 1
        self.mock_optimizer.get_opt_args.return_value = {}
        self.mock_optimizer.set_opt_args = MagicMock()

        self.patched_device_timestamp = patch(
            "dynamic_emb.distributed.batched_dynamicemb_table.device_timestamp",
            return_value=5,
        )
        self.patched_create_table = patch(
            "dynamic_emb.distributed.key_value_table.create_dynamicemb_table",
            return_value=self.mock_table,
        )
        self.patched_device_timestamp.start()
        self.patched_create_table.start()

        outer_self = self

        def mock_create_optimizer(self_obj, *args, **kwargs):
            self_obj._optimizer = outer_self.mock_optimizer
            self_obj._optimizer_type = EmbOptimType.ADAM

        self.patched_create_optimizer = patch.object(
            BatchedDynamicEmbeddingTablesV2,
            "_create_optimizer",
            autospec=True,
            side_effect=mock_create_optimizer,
        )
        self.patched_create_optimizer.start()

        self.table_options = [
            DynamicEmbTableOptions(
                index_type=torch.int64,
                embedding_dtype=torch.float32,
                optimizer_type=EmbOptimType.ADAM,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.NORMAL,
                    value=0.0,
                ),
                dim=10,
                max_capacity=1024,
            )
        ]
        self.feature_table_map = [0]
        self.test_table_name = "test_table"

        class TestModel(nn.Module):
            def __init__(self, table_options, feature_table_map, test_table_name):
                super().__init__()
                self.embedding = BatchedDynamicEmbeddingTablesV2(
                    table_options=table_options,
                    pooling_mode=DynamicEmbPoolingMode.NONE,
                    feature_table_map=feature_table_map,
                    table_names=[test_table_name],
                    device=torch.device("cpu"),
                )

        self.model = TestModel(
            table_options=self.table_options,
            feature_table_map=self.feature_table_map,
            test_table_name=self.test_table_name,
        )
        # Wrap methods for call assertions
        self.model.embedding.set_score = MagicMock()  # 完全替换为 Mock，不执行原逻辑
        self.model.embedding.get_score = MagicMock(return_value={"test_table": 0})  # 添加 get_score Mock

    def tearDown(self):
        self.patched_device_timestamp.stop()
        self.patched_create_table.stop()
        self.patched_create_optimizer.stop()

    def test_set_score_all_tables_int(self):
        """set_score(table_score=int) 对所有表设置同一分数"""
        with (
            patch("dynamic_emb.distributed.incremental_dump.find_sharded_modules") as mock_find,
            patch("dynamic_emb.distributed.incremental_dump.get_dynamic_emb_module") as mock_get,
        ):
            mock_find.return_value = [("model.embedding", "embedding", self.model.embedding)]
            mock_get.return_value = [self.model.embedding]

            set_score(self.model, 100)

            self.model.embedding.set_score.assert_called_once()
            call_args = self.model.embedding.set_score.call_args[0][0]
            self.assertEqual(call_args, {"test_table": 100})

    def test_set_score_per_collection_dict(self):
        """set_score(table_score=Dict[collection_path, Dict[table_name, int]]) 按表设置"""
        with (
            patch("dynamic_emb.distributed.incremental_dump.find_sharded_modules") as mock_find,
            patch("dynamic_emb.distributed.incremental_dump.get_dynamic_emb_module") as mock_get,
        ):
            mock_find.return_value = [("model.embedding", "embedding", self.model.embedding)]
            mock_get.return_value = [self.model.embedding]

            set_score(
                self.model,
                {"model.embedding": {"test_table": 42}},
            )

            self.model.embedding.set_score.assert_called_once()
            call_args = self.model.embedding.set_score.call_args[0][0]
            self.assertEqual(call_args, {"test_table": 42})

    def test_set_score_invalid_type_raises(self):
        """set_score(table_score=非法类型) 抛出 ValueError"""
        with self.assertRaises(ValueError) as ctx:
            set_score(self.model, 1.5)
        self.assertIn("table_score should be int or Dict", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            set_score(self.model, {"model.emb": {"t": "not_int"}})
        self.assertIn("Invalid parameter type of para 'score of table_score'", str(ctx.exception))

    def test_set_score_negative_int_raises(self):
        """set_score(table_score=负数) 抛出 ValueError"""
        with self.assertRaises(ValueError) as ctx:
            set_score(self.model, -1)
        self.assertIn("'table_score' is less than 0", str(ctx.exception))

        with (
            patch("dynamic_emb.distributed.incremental_dump.find_sharded_modules") as mock_find,
            patch("dynamic_emb.distributed.incremental_dump.get_dynamic_emb_module") as mock_get,
        ):
            mock_find.return_value = [("model.embedding", "embedding", self.model.embedding)]
            mock_get.return_value = [self.model.embedding]

            with self.assertRaises(ValueError) as ctx:
                set_score(self.model, {"model.embedding": {"test_table": -100}})
            self.assertIn("'score of table_score' is less than 0", str(ctx.exception))
            self.model.embedding.set_score.assert_not_called()

    def test_set_score_no_collections_warns_and_returns(self):
        """模型中没有 ShardedDynamicEmbeddingCollection 时告警并直接返回"""

        class EmptyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

        empty_model = EmptyModel()
        with patch(
            "dynamic_emb.distributed.incremental_dump.find_sharded_modules",
            return_value=[],
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                set_score(empty_model, 1)
            self.assertEqual(len(w), 1)
            self.assertIn("don't have any ShardedDynamicEmbeddingCollection", str(w[0].message))

    def test_set_score_no_dynamic_emb_warns_and_returns(self):
        """有 collection 但无 dynamic emb 模块时告警并返回"""
        with (
            patch("dynamic_emb.distributed.incremental_dump.find_sharded_modules") as mock_find,
            patch(
                "dynamic_emb.distributed.incremental_dump.get_dynamic_emb_module",
                return_value=[],
            ),
        ):
            mock_find.return_value = [("model.embedding", "embedding", self.model.embedding)]

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                set_score(self.model, 1)
            self.assertEqual(len(w), 1)
            self.assertIn("don't have any Dynamic embedding tables", str(w[0].message))

    def test_set_score_unknown_collection_in_dict_warns(self):
        """table_score 中指定了模型中不存在的 collection 时告警"""
        with (
            patch("dynamic_emb.distributed.incremental_dump.find_sharded_modules") as mock_find,
            patch("dynamic_emb.distributed.incremental_dump.get_dynamic_emb_module") as mock_get,
        ):
            mock_find.return_value = [("model.embedding", "embedding", self.model.embedding)]
            mock_get.return_value = [self.model.embedding]

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                set_score(
                    self.model,
                    {
                        "model.embedding": {"test_table": 10},
                        "nonexistent.collection": {"t": 20},
                    },
                )
            self.model.embedding.set_score.assert_called_once()
            self.assertEqual(len(w), 1)
            self.assertIn("nonexistent.collection", str(w[0].message))

    def test_get_score_returns_dict_when_has_dynamic_emb(self):
        """有 dynamic emb 时 get_score 返回 collection_path -> table_name -> score"""
        with (
            patch("dynamic_emb.distributed.incremental_dump.find_sharded_modules") as mock_find,
            patch("dynamic_emb.distributed.incremental_dump.get_dynamic_emb_module") as mock_get,
        ):
            mock_find.return_value = [("model.embedding", "embedding", self.model.embedding)]
            mock_get.return_value = [self.model.embedding]
            self.model.embedding.get_score.return_value = {"test_table": 5}

            result = get_score(self.model)

            self.assertIsNotNone(result)
            self.assertIn("model.embedding", result)
            self.assertEqual(result["model.embedding"], {"test_table": 5})

    def test_get_score_no_collections_returns_none_and_warns(self):
        """无 collection 时 get_score 返回 None 并告警"""

        class EmptyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

        empty_model = EmptyModel()
        with patch(
            "dynamic_emb.distributed.incremental_dump.find_sharded_modules",
            return_value=[],
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = get_score(empty_model)
            self.assertIsNone(result)
            self.assertEqual(len(w), 1)
            self.assertIn("don't have any ShardedDynamicEmbeddingCollection", str(w[0].message))
