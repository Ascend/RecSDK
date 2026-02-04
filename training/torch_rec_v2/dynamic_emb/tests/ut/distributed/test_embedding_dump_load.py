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

import unittest
import os
import sys
import shutil
import json
import warnings
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
import torch.distributed as dist

from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import EmbOptimType
from dynamic_emb.distributed.batched_dynamicemb_table import BatchedDynamicEmbeddingTablesV2
from dynamic_emb.distributed.types import (
    KEY_TYPE, 
    EMBEDDING_TYPE, 
    SCORE_TYPE, 
    OPT_STATE_TYPE
)

from dynamic_emb.distributed.dump_load import (
    DynamicEmbDump, 
    DynamicEmbLoad, 
    find_sharded_modules, 
    get_dynamic_emb_module, 
    check_emb_collection_modules,
)

from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbTableOptions,
    DynamicEmbPoolingMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbDataType,
)

_original_dynamic_emb_extensions = sys.modules.get("dynamic_emb_extensions")


def setup_module():
    mock_dynamicemb = MagicMock()
    sys.modules["dynamic_emb_extensions"] = mock_dynamicemb


def teardown_module():
    sys.modules["dynamic_emb_extensions"] = _original_dynamic_emb_extensions


class TestDynamicEmbDumpLoad(unittest.TestCase):
    def setUp(self):
        # 创建mock的分布式环境
        self.distributed_patcher = patch('torch.distributed.is_initialized', return_value=True)
        self.mock_is_initialized = self.distributed_patcher.start()

        self.dist_rank_patcher = patch('torch.distributed.get_rank', return_value=0)
        self.mock_get_rank = self.dist_rank_patcher.start()

        self.dist_world_size_patcher = patch('torch.distributed.get_world_size', return_value=1)
        self.mock_get_world_size = self.dist_world_size_patcher.start()

        self.init_pg_patcher = patch('torch.distributed.init_process_group', return_value=None)
        self.mock_init_pg = self.init_pg_patcher.start()

        self.dist_barrier_patcher = patch('torch.distributed.barrier', return_value=None)
        self.mock_barrier = self.dist_barrier_patcher.start()
        
        # 创建测试目录
        self.test_dir = "test_dump_load"
        os.makedirs(self.test_dir, exist_ok=True)

        # 创建mock dynamicemb op接口
        self.mock_key_value_table_load = MagicMock()
        self.mock_load_validate = MagicMock()

        # 创建mock的optimizer
        self.mock_optimizer = MagicMock()
        self.mock_optimizer.get_state_dim.return_value = 1
        self.mock_optimizer.get_opt_args = MagicMock(return_value={"lr": "0.1", "opt_type": "ADAM"})
        self.mock_optimizer.set_opt_args = MagicMock()
        
        # 创建mock的 KeyValueTable
        self.mock_key_value_table = MagicMock()
        self.mock_key_value_table.get_max_capacity.return_value = 5
        self.mock_key_value_table.get_emb_cols.return_value = 10
        self.mock_key_value_table.get_evict_strategy.return_value = "kLru"
        self.mock_key_value_table.get_key_type.return_value = DynamicEmbDataType.Int64
        self.mock_key_value_table.get_value_type.return_value = DynamicEmbDataType.Float32
        self.mock_key_value_table.optstate_dim.return_value = 2
        
        # 创建mock的dynamic emb op
        self.mock_key_value_table.export_batch.return_value = None
        self.mock_key_value_table.load.return_value = None
        
        # Mock create_dynamicemb_table
        self.creat_table_patcher = patch(
            'dynamic_emb.distributed.key_value_table.create_dynamicemb_table',
            return_value=self.mock_key_value_table
        )
        self.mock_create_dynamicemb_table = self.creat_table_patcher.start()
    
        # 创建表名
        self.test_table_name = 'test_table'
        self.test_table_names = ['test_table']
        
        # Mock device_timestamp
        with patch("dynamic_emb.distributed.batched_dynamicemb_table.device_timestamp", return_value=5):
            # 创建mock的table options
            self.feature_table_map = [0]
            self.table_options = [
                DynamicEmbTableOptions(
                    index_type=torch.int64,
                    embedding_dtype=torch.float32,
                    optimizer_type=EmbOptimType.ADAM,
                    initializer_args=DynamicEmbInitializerArgs(
                        mode=DynamicEmbInitializerMode.NORMAL,
                        value=0.0,
                    ),
                    dim=10
                )
            ]
            self.model = self._create_test_model()
            
    def tearDown(self):
        self.distributed_patcher.stop()
        self.dist_rank_patcher.stop()
        self.dist_world_size_patcher.stop()
        self.init_pg_patcher.stop()
        self.dist_barrier_patcher.stop()       
        self.creat_table_patcher.stop()
        
        # 清理测试目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_test_model(self):
        """创建简单的测试模型，包含BatchedDynamicEmbeddingTablesV2"""
        
        # 创建optimizer
        def mock_create_optimizer(self_obj, *args, **kwargs):
            self_obj._optimizer = self.mock_optimizer
            self_obj._optimizer_type = EmbOptimType.ADAM
        
        BatchedDynamicEmbeddingTablesV2._create_optimizer = mock_create_optimizer
        
        class TestModel(nn.Module):
            def __init__(self, table_options, feature_table_map, test_table_name):
                super().__init__()
                self.embedding = BatchedDynamicEmbeddingTablesV2(
                    table_options=table_options,
                    pooling_mode=DynamicEmbPoolingMode.NONE,
                    feature_table_map=feature_table_map,
                    table_names=[test_table_name],
                    device=torch.device("npu:0")
                )
        
        return TestModel(
            table_options=self.table_options,
            feature_table_map=self.feature_table_map,
            test_table_name=self.test_table_name
        )
    
    def _create_model_with_various_emb_attrs(self):
        """创建包含_emb_module/_emb_modules/_lookups/ModuleList的测试模型"""
        table_options = self.table_options
        feature_table_map = self.feature_table_map
        test_table_name = self.test_table_name
        
        class SubEmbModule(nn.Module):
            def __init__(self, table_options, feature_table_map, test_table_name):
                super().__init__()
                self.emb = BatchedDynamicEmbeddingTablesV2(
                    table_options=table_options,
                    pooling_mode=DynamicEmbPoolingMode.NONE,
                    feature_table_map=feature_table_map,
                    table_names=[test_table_name],
                    device=torch.device("npu:0")
                )
        
        class TestModelWithAttrs(nn.Module):
            def __init__(self, table_options, feature_table_map, test_table_name):
                super().__init__()
                sub_emb = SubEmbModule(
                    table_options=table_options,
                    feature_table_map=feature_table_map,
                    test_table_name=test_table_name
                )
                # 测试_emb_module属性
                self._emb_module = sub_emb.emb
                # 测试_emb_modules属性
                self._emb_modules = sub_emb.emb
                # 测试_lookups属性（包含ModuleList，让函数遍历）
                self._lookups = [
                    sub_emb.emb,
                    nn.ModuleList([sub_emb.emb, sub_emb.emb])  # 嵌套ModuleList
                ]
        
        return TestModelWithAttrs(
            table_options=table_options,
            feature_table_map=feature_table_map,
            test_table_name=test_table_name
        )

    def test_dynamic_emb_dump(self):
        """测试DynamicEmbDump的正确调用流程"""
        
        with patch('dynamic_emb.distributed.dump_load.find_sharded_modules') as mock_find_sharded, \
             patch('dynamic_emb.distributed.dump_load.get_dynamic_emb_module') as mock_get_dynamic_emb:
            
            mock_find_sharded.return_value = [('embedding', 'embedding', self.model.embedding)]
            mock_get_dynamic_emb.return_value = [self.model.embedding]
            
            # 调用DynamicEmbDump
            DynamicEmbDump(
                path=self.test_dir,
                model=self.model,
                optim=True,
                allow_overwrite=True
            )
            
            # 验证create_dynamicemb_table被正确调用
            self.mock_create_dynamicemb_table.assert_called_once()
                       
            # 验证export_batch被正确调用
            self.mock_key_value_table.export_batch.assert_called_once()


    def test_dynamic_emb_load(self):
        """测试DynamicEmbLoad的正确调用流程"""
        
        TEST_NUM_KEYS = 10
        dim = 10
        optstate_dim = 2
        table_name = "test_table"
        rank = 0
        world_size = 1
      
        with patch('dynamic_emb.distributed.dump_load.find_sharded_modules') as mock_find_sharded, \
             patch('dynamic_emb.distributed.dump_load.get_dynamic_emb_module') as mock_get_dynamic_emb:

            mock_find_sharded.return_value = [('embedding', 'embedding', self.model.embedding)]
            mock_get_dynamic_emb.return_value = [self.model.embedding]

            # 创建文件
            emb_subdir = os.path.join(self.test_dir, "embedding")
            os.makedirs(emb_subdir, exist_ok=True)

            # 计算文件大小（严格匹配校验）
            key_file_size = TEST_NUM_KEYS * KEY_TYPE.itemsize
            embedding_file_size = TEST_NUM_KEYS * EMBEDDING_TYPE.itemsize * dim
            score_file_size = TEST_NUM_KEYS * SCORE_TYPE.itemsize
            opt_file_size = TEST_NUM_KEYS * OPT_STATE_TYPE.itemsize * optstate_dim

            # 文件路径
            opt_args_file = os.path.join(emb_subdir, f"{table_name}_opt_args.json")
            emb_keys_file = os.path.join(emb_subdir, f"{table_name}_emb_keys.rank_{rank}.world_size_{world_size}")
            emb_values_file = os.path.join(emb_subdir, f"{table_name}_emb_values.rank_{rank}.world_size_{world_size}")
            emb_scores_file = os.path.join(emb_subdir, f"{table_name}_emb_scores.rank_{rank}.world_size_{world_size}")
            opt_values_file = os.path.join(
                                emb_subdir, f"{table_name}_emb_opt_values.rank_{rank}.world_size_{world_size}"
                            )

            meta_data = {
                "evict_strategy": "kLru",
                "opt_type": "ADAM"
            }

            with open(opt_args_file, "w") as f:
                json.dump(meta_data, f)

            # 写入数据文件
            with open(emb_keys_file, "wb") as f:
                f.write(b'\x00' * key_file_size)
            with open(emb_values_file, "wb") as f:
                f.write(b'\x00' * embedding_file_size)
            with open(emb_scores_file, "wb") as f:
                f.write(b'\x00' * score_file_size)
            with open(opt_values_file, "wb") as f:
                f.write(b'\x00' * opt_file_size)

            DynamicEmbLoad(
                path=self.test_dir,
                model=self.model,
                optim=True
            )

            # 验证load被正确调用
            self.mock_key_value_table.load.assert_called_once()


    def test_find_sharded_modules_invalid_input(self):
        """测试find_sharded_modules输入非nn.Module抛异常"""
        with self.assertRaises(ValueError) as ctx:
            find_sharded_modules(module="not a module")
        self.assertIn("param `module` must be an instance of torch.nn.Module", str(ctx.exception))


    def test_find_sharded_modules_empty(self):
        """测试find_sharded_modules返回空列表（模型无ShardedDynamicEmbeddingCollection）"""
        # 创建空模型（无ShardedDynamicEmbeddingCollection）
        class EmptyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
        
        empty_model = EmptyModel()
        result = find_sharded_modules(empty_model)
        self.assertEqual(len(result), 0)


    def test_get_dynamic_emb_module_invalid_input(self):
        """测试get_dynamic_emb_module输入非nn.Module抛异常"""
        with self.assertRaises(ValueError) as ctx:
            get_dynamic_emb_module(model="not a module")
        self.assertIn("param `model` must be an instance of torch.nn.Module", str(ctx.exception))


    def test_check_emb_collection_modules_invalid_input(self):
        """测试check_emb_collection_modules输入非nn.Module抛异常"""
        ret_list = []
        with self.assertRaises(ValueError) as ctx:
            check_emb_collection_modules(module="not a module", ret_list=ret_list)
        self.assertIn("param `module` must be an instance of torch.nn.Module", str(ctx.exception))


    def test_check_emb_collection_modules_various_attributes(self):
        """测试check_emb_collection_modules处理_emb_module/_emb_modules/_lookups/ModuleList"""
        model = self._create_model_with_various_emb_attrs()
        ret_list = []
        check_emb_collection_modules(model, ret_list)
        self.assertEqual(len(ret_list), 5)
        self.assertTrue(all(isinstance(m, BatchedDynamicEmbeddingTablesV2) for m in ret_list))


    def test_dynamic_emb_dump_path_exists_no_overwrite(self):
        """测试DynamicEmbDump路径已存在且不允许覆盖抛异常"""
        # 先创建非空目录
        non_empty_dir = os.path.join(self.test_dir, "non_empty")
        os.makedirs(non_empty_dir, exist_ok=True)
        with open(os.path.join(non_empty_dir, "test.txt"), "w") as f:
            f.write("test")
     
        with patch('dynamic_emb.distributed.dump_load.find_sharded_modules') as mock_find_sharded:
            mock_find_sharded.return_value = [('embedding', 'embedding', self.model.embedding)]
            
            # 调用dump，不允许覆盖
            with self.assertRaises(RuntimeError) as ctx:
                DynamicEmbDump(
                    path=non_empty_dir,
                    model=self.model,
                    allow_overwrite=False
                )
            self.assertIn("Cannot dump to", str(ctx.exception))
            self.assertIn("because it already contains files", str(ctx.exception))


    def test_dynamic_emb_load_empty_collections(self):
        """测试DynamicEmbLoad没有找到collections_list的警告"""
        # 创建空模型
        class EmptyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
        
        empty_model = EmptyModel()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DynamicEmbLoad(
                path=self.test_dir,
                model=empty_model
            )
            # 验证警告被触发
            self.assertEqual(len(w), 1)
            self.assertIn("Input model don't have any ShardedDynamicEmbeddingCollection", str(w[0].message))


if __name__ == '__main__':
    unittest.main()
