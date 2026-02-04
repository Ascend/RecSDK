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
from dynamic_emb.distributed.optimizers.adam_dynamicemb_optimizer import AdamDynamicEmbeddingOptimizerV2
from dynamic_emb.distributed.optimizers.adamw_dynamicemb_optimizer import AdamWDynamicEmbeddingOptimizerV2

_original_dynamic_emb_extensions = sys.modules.get("dynamic_emb_extensions")


def setup_module():
    mock_dynamicemb = MagicMock()
    sys.modules["dynamic_emb_extensions"] = mock_dynamicemb


def teardown_module():
    sys.modules["dynamic_emb_extensions"] = _original_dynamic_emb_extensions


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
        value_ptr = torch.tensor([i for i in range(batch_size)], dtype=torch.int64)
        value_type = torch.float32

        with patch(
            "dynamic_emb.distributed.optimizers.adam_dynamicemb_optimizer.dynamic_emb_Adam_with_pointer"
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
        value_ptr = torch.tensor([i for i in range(batch_size)], dtype=torch.int64)
        value_type = torch.float32

        with patch(
            "dynamic_emb.distributed.optimizers.adamw_dynamicemb_optimizer.dynamic_emb_AdamW_with_pointer"
        ) as mocked_func:
            mocked_func.return_value = ()
            optimizer.fused_update_with_pointer(grads, value_ptr, value_type)
            mocked_func.assert_called_once()
