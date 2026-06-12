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
from unittest.mock import patch, MagicMock, Mock

import torch

from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import (
    OptimizerArgs,
    EmbOptimType,
    string_to_opt_type,
    convert_optimizer_type,
    get_required_arg,
)
from dynamic_emb.distributed.dynamicemb_config import DynamicEmbTable, DynamicEmbTableOptions
from dynamic_emb.distributed.optimizers.adam_dynamicemb_optimizer import (
    AdamDynamicEmbeddingOptimizer,
    AdamDynamicEmbeddingOptimizerV2,
)
from dynamic_emb.distributed.optimizers.adamw_dynamicemb_optimizer import (
    AdamWDynamicEmbeddingOptimizerV2,
)
from dynamic_emb.distributed.optimizers.adagrad_dynamicemb_optimizer import (
    AdagradDynamicEmbeddingOptimizer,
    AdagradDynamicEmbeddingOptimizerV2,
)
from dynamic_emb.distributed.optimizers.rowwise_adagrad_dynamicemb_optimizer import (
    RowWiseAdagradDynamicEmbeddingOptimizer,
    RowWiseAdagradDynamicEmbeddingOptimizerV2,
)
from dynamic_emb.distributed.optimizers.sgd_dynamicemb_optimizer import (
    SGDDynamicEmbeddingOptimizer,
    SGDDynamicEmbeddingOptimizerV2,
)

_original_dynamic_emb_extensions = sys.modules.get("dynamic_emb_extensions")


def setup_module():
    mock_dynamicemb = MagicMock()
    sys.modules["dynamic_emb_extensions"] = mock_dynamicemb


def teardown_module():
    sys.modules["dynamic_emb_extensions"] = _original_dynamic_emb_extensions


def _make_mock_table_options(num_tables: int = 1, embedding_dtype: torch.dtype = torch.float32) -> list:
    table_options = []
    for _ in range(num_tables):
        option = MagicMock(spec=DynamicEmbTableOptions)
        option.embedding_dtype = embedding_dtype
        table_options.append(option)
    return table_options


def _make_mock_hashtables(num_tables: int = 1) -> list:
    hashtables = []
    for _ in range(num_tables):
        ht = MagicMock(spec=DynamicEmbTable)
        ht.set_initial_optstate = MagicMock()
        hashtables.append(ht)
    return hashtables


class TestOptimizers(unittest.TestCase):
    def test_optimizer_args_defaults(self):
        args = OptimizerArgs()
        self.assertEqual(args.learning_rate, 0.01)
        self.assertEqual(args.beta1, 0.9)
        self.assertEqual(args.beta2, 0.999)
        self.assertEqual(args.eps, 1e-8)

    def test_emb_optim_type_enum(self):
        self.assertEqual(EmbOptimType.ADAM.value, "adam")
        self.assertEqual(EmbOptimType.ADAMW.value, "adamW")
        self.assertEqual(EmbOptimType.NONE.value, "none")
        self.assertEqual(str(EmbOptimType.ADAM), "adam")

    def test_string_to_opt_type(self):
        self.assertEqual(string_to_opt_type("adam"), EmbOptimType.ADAM)
        self.assertEqual(string_to_opt_type("adamW"), EmbOptimType.ADAMW)
        self.assertEqual(string_to_opt_type("none"), EmbOptimType.NONE)

        with self.assertRaises(ValueError):
            string_to_opt_type("invalid")

    def test_convert_optimizer_type(self):
        with patch(
            "dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer.OptimizerType"
        ) as mocked_OptimizerType:
            mocked_OptimizerType.Adam = Mock()
            self.assertEqual(convert_optimizer_type(EmbOptimType.ADAM), mocked_OptimizerType.Adam)

            mocked_OptimizerType.AdamW = Mock()
            self.assertEqual(convert_optimizer_type(EmbOptimType.ADAMW), mocked_OptimizerType.AdamW)

            with self.assertRaises(ValueError):
                convert_optimizer_type(EmbOptimType.NONE)

    def test_get_required_arg(self):
        args = {
            "learning_rate": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
        }
        self.assertEqual(get_required_arg(args, "learning_rate"), 0.01)
        self.assertEqual(get_required_arg(args, "beta1"), 0.9)
        self.assertEqual(get_required_arg(args, "beta2"), 0.999)
        with self.assertRaises(ValueError):
            get_required_arg(args, "beta3")


class TestAdamDynamicEmbeddingOptimizerV2(unittest.TestCase):
    def test_adam_optimizer_initialization(self):
        opt_args = OptimizerArgs(learning_rate=0.001, beta1=0.95, beta2=0.995)
        optimizer = AdamDynamicEmbeddingOptimizerV2(opt_args)

        self.assertEqual(optimizer.get_opt_args()["lr"], 0.001)
        self.assertEqual(optimizer.get_opt_args()["beta1"], 0.95)
        self.assertEqual(optimizer.get_opt_args()["iters"], 0)
        self.assertEqual(optimizer.get_state_dim(10), 20)

    def test_adam_step_increment(self):
        optimizer = AdamDynamicEmbeddingOptimizerV2(OptimizerArgs())
        self.assertEqual(optimizer.get_opt_args()["iters"], 0)
        optimizer.step()
        self.assertEqual(optimizer.get_opt_args()["iters"], 1)

    def test_set_learning_rate(self):
        optimizer = AdamDynamicEmbeddingOptimizerV2(OptimizerArgs(learning_rate=0.01))
        self.assertEqual(optimizer.get_opt_args()["lr"], 0.01)
        optimizer.set_learning_rate(0.005)
        self.assertEqual(optimizer.get_opt_args()["lr"], 0.005)

    def test_set_opt_args(self):
        optimizer = AdamDynamicEmbeddingOptimizerV2(OptimizerArgs())
        new_args = {"lr": 0.02, "iters": 100, "beta1": 0.8, "beta2": 0.99, "eps": 1e-7, "weight_decay": 0.001}

        optimizer.set_opt_args(new_args)
        opt_args = optimizer.get_opt_args()

        self.assertEqual(opt_args["lr"], 0.02)
        self.assertEqual(opt_args["iters"], 100)
        self.assertEqual(opt_args["beta1"], 0.8)
        self.assertEqual(opt_args["beta2"], 0.99)
        self.assertEqual(opt_args["eps"], 1e-7)
        self.assertEqual(opt_args["weight_decay"], 0.001)

    def test_get_state_dim(self):
        emb_dim = 128
        optimizer = AdamDynamicEmbeddingOptimizerV2(OptimizerArgs())
        self.assertEqual(optimizer.get_state_dim(emb_dim), emb_dim * 2)

    def test_initial_optim_states(self):
        initial_val = 0.5
        opt_args = OptimizerArgs(initial_accumulator_value=initial_val)
        optimizer = AdamDynamicEmbeddingOptimizerV2(opt_args)
        self.assertEqual(optimizer.get_initial_optim_states(), initial_val)

        new_initial = 0.1
        optimizer.set_initial_optim_states(new_initial)
        self.assertEqual(optimizer.get_initial_optim_states(), new_initial)

    def test_fused_update_with_pointer_shape(self):
        optimizer = AdamDynamicEmbeddingOptimizerV2(OptimizerArgs())
        batch_size = 32
        emb_dim = 16

        grads = torch.randn(batch_size, emb_dim)
        value_ptr = torch.tensor(list(range(batch_size)), dtype=torch.int64)
        value_type = torch.float32

        with patch(
            "dynamic_emb.distributed.optimizers.adam_dynamicemb_optimizer.dynamic_emb_adamW_with_pointer"
        ) as mocked_func:
            mocked_func.return_value = ()
            optimizer.fused_update_with_pointer(grads, value_ptr, value_type)
            mocked_func.assert_called_once()


class TestAdamWDynamicEmbeddingOptimizerV2(unittest.TestCase):
    def test_fused_update_with_pointer_shape(self):
        optimizer = AdamWDynamicEmbeddingOptimizerV2(OptimizerArgs())
        batch_size = 32
        emb_dim = 16

        grads = torch.randn(batch_size, emb_dim)
        value_ptr = torch.tensor(list(range(batch_size)), dtype=torch.int64)
        value_type = torch.float32

        with patch(
            "dynamic_emb.distributed.optimizers.adamw_dynamicemb_optimizer.dynamic_emb_adamW_with_pointer"
        ) as mocked_func:
            mocked_func.return_value = ()
            optimizer.fused_update_with_pointer(grads, value_ptr, value_type)
            mocked_func.assert_called_once()


class TestAdagradDynamicEmbeddingOptimizerV2(unittest.TestCase):
    def test_adagrad_optimizer_initialization(self):
        opt_args = OptimizerArgs(learning_rate=0.001)
        optimizer = AdagradDynamicEmbeddingOptimizerV2(opt_args)

        opt_args_dict = optimizer.get_opt_args()
        self.assertEqual(opt_args_dict["opt_type"], "exact_adagrad")
        self.assertEqual(opt_args_dict["lr"], 0.001)
        self.assertEqual(opt_args_dict["eps"], 1e-8)
        self.assertEqual(opt_args_dict["initial_accumulator_value"], 0.0)
        self.assertEqual(optimizer.get_state_dim(10), 10)

    def test_set_learning_rate(self):
        optimizer = AdagradDynamicEmbeddingOptimizerV2(OptimizerArgs(learning_rate=0.01))
        self.assertEqual(optimizer.get_opt_args()["lr"], 0.01)
        optimizer.set_learning_rate(0.005)
        self.assertEqual(optimizer.get_opt_args()["lr"], 0.005)

    def test_set_opt_args(self):
        optimizer = AdagradDynamicEmbeddingOptimizerV2(OptimizerArgs())
        new_args = {"lr": 0.02, "eps": 1e-7, "initial_accumulator_value": 0.1}

        optimizer.set_opt_args(new_args)
        opt_args = optimizer.get_opt_args()

        self.assertEqual(opt_args["lr"], 0.02)
        self.assertEqual(opt_args["eps"], 1e-7)
        self.assertEqual(opt_args["initial_accumulator_value"], 0.1)

    def test_get_state_dim(self):
        emb_dim = 128
        optimizer = AdagradDynamicEmbeddingOptimizerV2(OptimizerArgs())
        self.assertEqual(optimizer.get_state_dim(emb_dim), emb_dim)

    def test_initial_optim_states(self):
        initial_val = 0.5
        opt_args = OptimizerArgs(initial_accumulator_value=initial_val)
        optimizer = AdagradDynamicEmbeddingOptimizerV2(opt_args)
        self.assertEqual(optimizer.get_initial_optim_states(), initial_val)

        new_initial = 0.1
        optimizer.set_initial_optim_states(new_initial)
        self.assertEqual(optimizer.get_initial_optim_states(), new_initial)

    def test_fused_update_with_pointer_shape(self):
        optimizer = AdagradDynamicEmbeddingOptimizerV2(OptimizerArgs())
        batch_size = 32
        emb_dim = 16

        grads = torch.randn(batch_size, emb_dim)
        value_ptr = torch.tensor(list(range(batch_size)), dtype=torch.int64)
        value_type = torch.float32

        with patch(
            "dynamic_emb.distributed.optimizers.adagrad_dynamicemb_optimizer.dynamic_emb_adagrad_with_pointer"
        ) as mocked_func:
            mocked_func.return_value = ()
            optimizer.fused_update_with_pointer(grads, value_ptr, value_type)
            mocked_func.assert_called_once()


class TestRowWiseAdagradDynamicEmbeddingOptimizerV2(unittest.TestCase):
    def test_row_wise_adagrad_optimizer_initialization(self):
        opt_args = OptimizerArgs(learning_rate=0.001)
        emb_dtype = torch.float32
        optimizer = RowWiseAdagradDynamicEmbeddingOptimizerV2(opt_args, emb_dtype)

        opt_args_dict = optimizer.get_opt_args()
        self.assertEqual(opt_args_dict["opt_type"], "exact_row_wise_adagrad")
        self.assertEqual(opt_args_dict["lr"], 0.001)
        self.assertEqual(opt_args_dict["eps"], 1e-8)
        self.assertEqual(opt_args_dict["initial_accumulator_value"], 0.0)
        self.assertEqual(optimizer.get_state_dim(10), 4)  # 16 // 4 = 4 for float32

    def test_set_learning_rate(self):
        emb_dtype = torch.float32
        optimizer = RowWiseAdagradDynamicEmbeddingOptimizerV2(OptimizerArgs(learning_rate=0.01), emb_dtype)
        self.assertEqual(optimizer.get_opt_args()["lr"], 0.01)
        optimizer.set_learning_rate(0.005)
        self.assertEqual(optimizer.get_opt_args()["lr"], 0.005)

    def test_set_opt_args(self):
        emb_dtype = torch.float32
        optimizer = RowWiseAdagradDynamicEmbeddingOptimizerV2(OptimizerArgs(), emb_dtype)
        new_args = {"lr": 0.02, "eps": 1e-7, "initial_accumulator_value": 0.1}

        optimizer.set_opt_args(new_args)
        opt_args = optimizer.get_opt_args()

        self.assertEqual(opt_args["lr"], 0.02)
        self.assertEqual(opt_args["eps"], 1e-7)
        self.assertEqual(opt_args["initial_accumulator_value"], 0.1)

    def test_get_state_dim_float32(self):
        emb_dim = 128
        emb_dtype = torch.float32
        optimizer = RowWiseAdagradDynamicEmbeddingOptimizerV2(OptimizerArgs(), emb_dtype)
        self.assertEqual(optimizer.get_state_dim(emb_dim), 4)  # 16 // 4

    def test_get_state_dim_float16(self):
        emb_dim = 128
        emb_dtype = torch.float16
        optimizer = RowWiseAdagradDynamicEmbeddingOptimizerV2(OptimizerArgs(), emb_dtype)
        self.assertEqual(optimizer.get_state_dim(emb_dim), 8)  # 16 // 2

    def test_initial_optim_states(self):
        emb_dtype = torch.float32
        initial_val = 0.5
        opt_args = OptimizerArgs(initial_accumulator_value=initial_val)
        optimizer = RowWiseAdagradDynamicEmbeddingOptimizerV2(opt_args, emb_dtype)
        self.assertEqual(optimizer.get_initial_optim_states(), initial_val)

        new_initial = 0.1
        optimizer.set_initial_optim_states(new_initial)
        self.assertEqual(optimizer.get_initial_optim_states(), new_initial)

    def test_fused_update_with_pointer_shape(self):
        emb_dtype = torch.float32
        optimizer = RowWiseAdagradDynamicEmbeddingOptimizerV2(OptimizerArgs(), emb_dtype)
        batch_size = 32
        emb_dim = 16

        grads = torch.randn(batch_size, emb_dim)
        value_ptr = torch.tensor(list(range(batch_size)), dtype=torch.int64)
        value_type = torch.float32

        with patch(
            "dynamic_emb.distributed.optimizers.rowwise_adagrad_dynamicemb_optimizer.dynamic_emb_rowwise_adagrad_with_pointer"
        ) as mocked_func:
            mocked_func.return_value = ()
            optimizer.fused_update_with_pointer(grads, value_ptr, value_type)
            mocked_func.assert_called_once()


class TestSGDDynamicEmbeddingOptimizerV2(unittest.TestCase):
    def test_sgd_optimizer_initialization(self):
        opt_args = OptimizerArgs(learning_rate=0.001)
        optimizer = SGDDynamicEmbeddingOptimizerV2(opt_args)

        opt_args_dict = optimizer.get_opt_args()
        self.assertEqual(opt_args_dict["opt_type"], "sgd")
        self.assertEqual(opt_args_dict["lr"], 0.001)
        self.assertEqual(optimizer.get_state_dim(10), 0)

    def test_set_learning_rate(self):
        optimizer = SGDDynamicEmbeddingOptimizerV2(OptimizerArgs(learning_rate=0.01))
        self.assertEqual(optimizer.get_opt_args()["lr"], 0.01)
        optimizer.set_learning_rate(0.005)
        self.assertEqual(optimizer.get_opt_args()["lr"], 0.005)

    def test_set_opt_args(self):
        optimizer = SGDDynamicEmbeddingOptimizerV2(OptimizerArgs())
        new_args = {"lr": 0.02}

        optimizer.set_opt_args(new_args)
        opt_args = optimizer.get_opt_args()

        self.assertEqual(opt_args["lr"], 0.02)

    def test_get_state_dim(self):
        emb_dim = 128
        optimizer = SGDDynamicEmbeddingOptimizerV2(OptimizerArgs())
        self.assertEqual(optimizer.get_state_dim(emb_dim), 0)

    def test_initial_optim_states(self):
        initial_val = 0.5
        opt_args = OptimizerArgs(initial_accumulator_value=initial_val)
        optimizer = SGDDynamicEmbeddingOptimizerV2(opt_args)
        self.assertEqual(optimizer.get_initial_optim_states(), initial_val)

        new_initial = 0.1
        optimizer.set_initial_optim_states(new_initial)
        self.assertEqual(optimizer.get_initial_optim_states(), new_initial)

    def test_fused_update_with_pointer_shape(self):
        optimizer = SGDDynamicEmbeddingOptimizerV2(OptimizerArgs())
        batch_size = 32
        emb_dim = 16

        grads = torch.randn(batch_size, emb_dim)
        value_ptr = torch.tensor(list(range(batch_size)), dtype=torch.int64)
        value_type = torch.float32

        with patch(
            "dynamic_emb.distributed.optimizers.sgd_dynamicemb_optimizer.dynamic_emb_sgd_with_pointer"
        ) as mocked_func:
            mocked_func.return_value = ()
            optimizer.fused_update_with_pointer(grads, value_ptr, value_type)
            mocked_func.assert_called_once()


class TestSGDDynamicEmbeddingOptimizer(unittest.TestCase):
    def _create_optimizer(self, opt_args=None, num_tables=1):
        if opt_args is None:
            opt_args = OptimizerArgs()
        table_options = _make_mock_table_options(num_tables)
        hashtables = _make_mock_hashtables(num_tables)
        return SGDDynamicEmbeddingOptimizer(opt_args, table_options, hashtables), hashtables

    def test_sgd_v1_optimizer_initialization(self):
        opt_args = OptimizerArgs(learning_rate=0.001)
        optimizer, _ = self._create_optimizer(opt_args)

        opt_args_dict = optimizer.get_opt_args()
        self.assertEqual(opt_args_dict["opt_type"], "exact_sgd")
        self.assertEqual(opt_args_dict["lr"], 0.001)
        self.assertEqual(optimizer._num_tables, 1)

    def test_set_learning_rate(self):
        optimizer, _ = self._create_optimizer(OptimizerArgs(learning_rate=0.01))
        self.assertEqual(optimizer.get_opt_args()["lr"], 0.01)
        optimizer.set_learning_rate(0.005)
        self.assertEqual(optimizer.get_opt_args()["lr"], 0.005)

    def test_set_opt_args(self):
        optimizer, _ = self._create_optimizer(OptimizerArgs())
        optimizer.set_opt_args({"lr": 0.02})
        self.assertEqual(optimizer.get_opt_args()["lr"], 0.02)

    def test_update_with_table(self):
        optimizer, hashtables = self._create_optimizer()
        batch_size = 32
        emb_dim = 16
        indices = [torch.arange(batch_size, dtype=torch.int64)]
        grads = [torch.randn(batch_size, emb_dim)]

        mock_weight_dtype = MagicMock()
        with (
            patch(
                "dynamic_emb.distributed.optimizers.sgd_dynamicemb_optimizer.dynamic_emb_sgd_with_table"
            ) as mocked_func,
            patch(
                "dynamic_emb.distributed.optimizers.sgd_dynamicemb_optimizer.torch_to_dyn_emb",
                return_value=mock_weight_dtype,
            ),
        ):
            optimizer.update(hashtables, indices, grads)
            mocked_func.assert_called_once()
            args = mocked_func.call_args[0]
            self.assertEqual(args[0], hashtables[0])
            self.assertEqual(args[1], batch_size)
            self.assertEqual(args[4], optimizer._opt_args.learning_rate)
            self.assertEqual(args[5], mock_weight_dtype)

    def test_update_unknown_hashtable_raises(self):
        optimizer, hashtables = self._create_optimizer()
        unknown_ht = MagicMock()
        with self.assertRaises(ValueError):
            optimizer.update(
                [unknown_ht],
                [torch.tensor([0], dtype=torch.int64)],
                [torch.randn(1, 16)],
            )


class TestAdamDynamicEmbeddingOptimizer(unittest.TestCase):
    def _create_optimizer(self, opt_args=None, num_tables=1):
        if opt_args is None:
            opt_args = OptimizerArgs()
        table_options = _make_mock_table_options(num_tables)
        hashtables = _make_mock_hashtables(num_tables)
        return AdamDynamicEmbeddingOptimizer(opt_args, table_options, hashtables), hashtables

    def test_adam_v1_optimizer_initialization(self):
        opt_args = OptimizerArgs(learning_rate=0.001, beta1=0.95, beta2=0.995)
        optimizer, _ = self._create_optimizer(opt_args)

        opt_args_dict = optimizer.get_opt_args()
        self.assertEqual(opt_args_dict["opt_type"], "adam")
        self.assertEqual(opt_args_dict["lr"], 0.001)
        self.assertEqual(opt_args_dict["beta1"], 0.95)
        self.assertEqual(opt_args_dict["iters"], 0)
        self.assertEqual(optimizer.state_names(), ["m", "v"])

    def test_update_increments_iterations(self):
        optimizer, hashtables = self._create_optimizer()
        indices = [torch.arange(8, dtype=torch.int64)]
        grads = [torch.randn(8, 16)]

        with patch(
            "dynamic_emb.distributed.optimizers.adam_dynamicemb_optimizer.dynamic_emb_adamW_with_table"
        ) as mocked_func:
            optimizer.update(hashtables, indices, grads)
            self.assertEqual(optimizer.get_opt_args()["iters"], 1)
            self.assertEqual(mocked_func.call_args[0][9], 1)

            optimizer.update(hashtables, indices, grads)
            self.assertEqual(optimizer.get_opt_args()["iters"], 2)
            self.assertEqual(mocked_func.call_args[0][9], 2)

    def test_set_learning_rate(self):
        optimizer, _ = self._create_optimizer(OptimizerArgs(learning_rate=0.01))
        optimizer.set_learning_rate(0.005)
        self.assertEqual(optimizer.get_opt_args()["lr"], 0.005)

    def test_set_opt_args(self):
        optimizer, _ = self._create_optimizer(OptimizerArgs())
        new_args = {
            "lr": 0.02,
            "iters": 100,
            "beta1": 0.8,
            "beta2": 0.99,
            "eps": 1e-7,
            "weight_decay": 0.001,
        }
        optimizer.set_opt_args(new_args)
        opt_args = optimizer.get_opt_args()
        self.assertEqual(opt_args["lr"], 0.02)
        self.assertEqual(opt_args["iters"], 100)
        self.assertEqual(opt_args["beta1"], 0.8)
        self.assertEqual(opt_args["beta2"], 0.99)
        self.assertEqual(opt_args["eps"], 1e-7)
        self.assertEqual(opt_args["weight_decay"], 0.001)

    def test_update_with_table(self):
        optimizer, hashtables = self._create_optimizer()
        batch_size = 32
        emb_dim = 16
        indices = [torch.arange(batch_size, dtype=torch.int64)]
        grads = [torch.randn(batch_size, emb_dim)]

        with patch(
            "dynamic_emb.distributed.optimizers.adam_dynamicemb_optimizer.dynamic_emb_adamW_with_table"
        ) as mocked_func:
            optimizer.update(hashtables, indices, grads)
            mocked_func.assert_called_once()

    def test_update_unknown_hashtable_raises(self):
        optimizer, hashtables = self._create_optimizer()
        unknown_ht = MagicMock()
        with self.assertRaises(ValueError):
            optimizer.update(
                [unknown_ht],
                [torch.tensor([0], dtype=torch.int64)],
                [torch.randn(1, 16)],
            )


class TestAdagradDynamicEmbeddingOptimizer(unittest.TestCase):
    def _create_optimizer(self, opt_args=None, num_tables=1):
        if opt_args is None:
            opt_args = OptimizerArgs()
        table_options = _make_mock_table_options(num_tables)
        hashtables = _make_mock_hashtables(num_tables)
        return AdagradDynamicEmbeddingOptimizer(opt_args, table_options, hashtables), hashtables

    def test_adagrad_v1_optimizer_initialization(self):
        opt_args = OptimizerArgs(learning_rate=0.001, initial_accumulator_value=0.5)
        optimizer, hashtables = self._create_optimizer(opt_args)

        opt_args_dict = optimizer.get_opt_args()
        self.assertEqual(opt_args_dict["opt_type"], "exact_adagrad")
        self.assertEqual(opt_args_dict["lr"], 0.001)
        self.assertEqual(opt_args_dict["initial_accumulator_value"], 0.5)
        self.assertEqual(optimizer.get_state_by_name("Gt"), hashtables)
        hashtables[0].set_initial_optstate.assert_called_once_with(0.5)

    def test_set_learning_rate(self):
        optimizer, _ = self._create_optimizer(OptimizerArgs(learning_rate=0.01))
        optimizer.set_learning_rate(0.005)
        self.assertEqual(optimizer.get_opt_args()["lr"], 0.005)

    def test_set_opt_args(self):
        optimizer, hashtables = self._create_optimizer(OptimizerArgs())
        optimizer.set_opt_args({"lr": 0.02, "eps": 1e-7, "initial_accumulator_value": 0.1})
        opt_args = optimizer.get_opt_args()
        self.assertEqual(opt_args["lr"], 0.02)
        self.assertEqual(opt_args["eps"], 1e-7)
        self.assertEqual(opt_args["initial_accumulator_value"], 0.1)
        hashtables[0].set_initial_optstate.assert_called_with(0.1)

    def test_update_with_table(self):
        optimizer, hashtables = self._create_optimizer()
        batch_size = 32
        emb_dim = 16
        indices = [torch.arange(batch_size, dtype=torch.int64)]
        grads = [torch.randn(batch_size, emb_dim)]

        with patch(
            "dynamic_emb.distributed.optimizers.adagrad_dynamicemb_optimizer.dynamic_emb_adagrad_with_table"
        ) as mocked_func:
            optimizer.update(hashtables, indices, grads)
            mocked_func.assert_called_once()
            args = mocked_func.call_args[0]
            self.assertEqual(args[0], hashtables[0])
            self.assertEqual(args[1], batch_size)
            self.assertEqual(args[4], optimizer._opt_args.learning_rate)
            self.assertEqual(args[5], optimizer._opt_args.eps)

    def test_update_unknown_hashtable_raises(self):
        optimizer, _ = self._create_optimizer()
        unknown_ht = MagicMock()
        with self.assertRaises(ValueError):
            optimizer.update(
                [unknown_ht],
                [torch.tensor([0], dtype=torch.int64)],
                [torch.randn(1, 16)],
            )


class TestRowWiseAdagradDynamicEmbeddingOptimizer(unittest.TestCase):
    def _create_optimizer(self, opt_args=None, num_tables=1):
        if opt_args is None:
            opt_args = OptimizerArgs()
        table_options = _make_mock_table_options(num_tables)
        hashtables = _make_mock_hashtables(num_tables)
        return RowWiseAdagradDynamicEmbeddingOptimizer(opt_args, table_options, hashtables), hashtables

    def test_row_wise_adagrad_v1_optimizer_initialization(self):
        opt_args = OptimizerArgs(learning_rate=0.001, initial_accumulator_value=0.5)
        optimizer, hashtables = self._create_optimizer(opt_args)

        opt_args_dict = optimizer.get_opt_args()
        self.assertEqual(opt_args_dict["opt_type"], "exact_row_wise_adagrad")
        self.assertEqual(opt_args_dict["lr"], 0.001)
        self.assertEqual(opt_args_dict["initial_accumulator_value"], 0.5)
        self.assertEqual(optimizer.get_state_by_name("Gt"), hashtables)
        hashtables[0].set_initial_optstate.assert_called_once_with(0.5)

    def test_set_learning_rate(self):
        optimizer, _ = self._create_optimizer(OptimizerArgs(learning_rate=0.01))
        optimizer.set_learning_rate(0.005)
        self.assertEqual(optimizer.get_opt_args()["lr"], 0.005)

    def test_set_opt_args(self):
        optimizer, hashtables = self._create_optimizer(OptimizerArgs())
        optimizer.set_opt_args({"lr": 0.02, "eps": 1e-7, "initial_accumulator_value": 0.1})
        opt_args = optimizer.get_opt_args()
        self.assertEqual(opt_args["lr"], 0.02)
        self.assertEqual(opt_args["eps"], 1e-7)
        self.assertEqual(opt_args["initial_accumulator_value"], 0.1)
        hashtables[0].set_initial_optstate.assert_called_with(0.1)

    def test_update_with_table(self):
        optimizer, hashtables = self._create_optimizer()
        batch_size = 32
        emb_dim = 16
        indices = [torch.arange(batch_size, dtype=torch.int64)]
        grads = [torch.randn(batch_size, emb_dim)]

        with patch(
            "dynamic_emb.distributed.optimizers.rowwise_adagrad_dynamicemb_optimizer.dynamic_emb_rowwise_adagrad_with_table"
        ) as mocked_func:
            optimizer.update(hashtables, indices, grads)
            mocked_func.assert_called_once()
            args = mocked_func.call_args[0]
            self.assertEqual(args[0], hashtables[0])
            self.assertEqual(args[1], batch_size)

    def test_update_unknown_hashtable_raises(self):
        optimizer, _ = self._create_optimizer()
        unknown_ht = MagicMock()
        with self.assertRaises(ValueError):
            optimizer.update(
                [unknown_ht],
                [torch.tensor([0], dtype=torch.int64)],
                [torch.randn(1, 16)],
            )
