#!/usr/bin/env python3
# coding: UTF-8
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
from unittest import mock

import tensorflow as tf

from mx_rec.validator.emb_validator import (
    check_emb_init_params,
    check_emb_lookup_params,
    check_emb_multi_lookup_times,
    check_and_format_emb_padding_keys,
)
from mx_rec.core.asc.feature_spec import FeatureSpec
from core.mock_class import MockConfigInitializer


class TestCheckEmbInitParams(unittest.TestCase):
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    def test_freeze_err(self, validator_config_initializer):
        mock_config_init = MockConfigInitializer(freeze=True)
        validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        with self.assertRaises(EnvironmentError):
            check_emb_init_params(True, tf.TensorShape([8]))

    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    def test_use_dynamic_expansion_err(self, validator_config_initializer):
        mock_config_init = MockConfigInitializer(use_dynamic_expansion=True)
        validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        with self.assertRaises(ValueError) as e:
            check_emb_init_params(False, tf.TensorShape([8]))

        self.assertIn("not support embedding dynamic expansion", str(e.exception))

    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    def test_ndims_err(self, validator_config_initializer):
        mock_config_init = MockConfigInitializer(use_dynamic_expansion=True)
        validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        with self.assertRaises(ValueError) as e:
            check_emb_init_params(True, tf.TensorShape([8, 1]))

        self.assertIn("can only be one dim shape", str(e.exception))


class TestCheckEmbLookupParams(unittest.TestCase):
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    def test_feat_spec_init_err(self, validator_config_initializer):
        mock_config_init = MockConfigInitializer()
        validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        feat_spec = FeatureSpec("test_feat")
        with self.assertRaises(RuntimeError) as e:
            check_emb_lookup_params({}, feat_spec, None, False)

        self.assertIn("Feature Spec has not been initialized", str(e.exception))

    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    def test_feat_spec_is_training_err(self, validator_config_initializer):
        mock_config_init = MockConfigInitializer()
        validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        feat_spec = FeatureSpec("test_feat")
        with self.assertRaises(RuntimeError) as e:
            feat_spec.initialized = True
            check_emb_lookup_params({}, feat_spec, None, False)

        self.assertIn("You have not config feature for is training mode", str(e.exception))

    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    def test_slice_host_vocabulary_size_err(self, validator_config_initializer):
        mock_config_init = MockConfigInitializer()
        validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        table_params = {
            "slice_device_vocabulary_size": 8,
            "slice_host_vocabulary_size": 10**10,
            "table_name": "test_table",
        }
        with self.assertRaises(RuntimeError) as e:
            check_emb_lookup_params(table_params, tf.constant([0, 1], dtype=tf.float32), None, True)

        self.assertIn("Given host_vocabulary_size was too big for table", str(e.exception))

    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    def test_use_static_return_ok(self, validator_config_initializer):
        mock_config_init = MockConfigInitializer(use_static=False)
        validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        table_params = {
            "slice_device_vocabulary_size": 8,
            "slice_host_vocabulary_size": 16,
            "table_name": "test_table",
        }
        check_emb_lookup_params(table_params, tf.constant([0, 1], dtype=tf.float32), None, True)
        self.assertTrue(callable(check_emb_lookup_params))

    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    def test_send_count_not_int(self, validator_config_initializer):
        mock_config_init = MockConfigInitializer(use_static=True)
        validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        table_params = {
            "slice_device_vocabulary_size": 8,
            "slice_host_vocabulary_size": 16,
            "table_name": "test_table",
        }
        with self.assertRaises(ValueError) as e:
            check_emb_lookup_params(table_params, tf.constant([0, 1], dtype=tf.float32), None, True)

        self.assertIn("Send count must be a integer which is larger than 0", str(e.exception))

    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    def test_use_dynamic_expansion_return_ok(self, validator_config_initializer):
        mock_config_init = MockConfigInitializer(use_static=True, use_dynamic_expansion=True)
        validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        table_params = {
            "slice_device_vocabulary_size": 8,
            "slice_host_vocabulary_size": 16,
            "table_name": "test_table",
        }
        check_emb_lookup_params(table_params, tf.constant([0, 1], dtype=tf.float32), 8, True)
        self.assertTrue(callable(check_emb_lookup_params))

    @mock.patch.multiple(
        "mx_rec.validator.emb_validator",
        get_rank_size=mock.MagicMock(return_value=1),
    )
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    def test_slice_device_vocabulary_size_err(self, validator_config_initializer):
        mock_config_init = MockConfigInitializer(use_static=True)
        validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        table_params = {
            "slice_device_vocabulary_size": 8,
            "slice_host_vocabulary_size": 16,
            "table_name": "test_table",
        }
        with self.assertRaises(ValueError) as e:
            check_emb_lookup_params(table_params, tf.constant([0, 1], dtype=tf.float32), 10, True)

        self.assertIn("Given device_vocabulary_size was too small", str(e.exception))

    @mock.patch.multiple(
        "mx_rec.validator.emb_validator",
        get_rank_size=mock.MagicMock(return_value=1),
    )
    @mock.patch("mx_rec.validator.emb_validator.ConfigInitializer")
    def test_slice_host_vocabulary_size_err(self, validator_config_initializer):
        mock_config_init = MockConfigInitializer(use_static=True)
        validator_config_initializer.get_instance = mock.Mock(return_value=mock_config_init)
        table_params = {
            "slice_device_vocabulary_size": 32,
            "slice_host_vocabulary_size": 16,
            "table_name": "test_table",
        }
        with self.assertRaises(ValueError) as e:
            check_emb_lookup_params(table_params, tf.constant([0, 1], dtype=tf.float32), 17, True)

        self.assertIn("Given host_vocabulary_size was too small", str(e.exception))


class TestCheckEmbMultiLookupTimes(unittest.TestCase):
    def test_raise_ok(self):
        with self.assertRaises(RuntimeError) as e:
            check_emb_multi_lookup_times(129, "test_table")

        self.assertIn("The number of multiple sparse lookup for a table", str(e.exception))


class TestCheckAndFormatEmbPaddingKeys(unittest.TestCase):
    def test_padding_keys_value_err(self):
        big_number = 123456789012345678901234567890
        with self.assertRaises(ValueError) as e:
            check_and_format_emb_padding_keys(padding_keys=big_number, padding_keys_mask=False, padding_keys_len=None)

        self.assertIn("the padding keys should be between", str(e.exception))

    def test_padding_keys_list_err(self):
        with self.assertRaises(ValueError) as e:
            check_and_format_emb_padding_keys(padding_keys=[], padding_keys_mask=False, padding_keys_len=None)

        self.assertIn("the length of the padding keys should be between", str(e.exception))

    def test_padding_keys_list_value_err(self):
        big_number = 123456789012345678901234567890
        with self.assertRaises(ValueError) as e:
            check_and_format_emb_padding_keys(padding_keys=[big_number], padding_keys_mask=False, padding_keys_len=None)

        self.assertIn("the padding keys should be between", str(e.exception))

    def test_padding_keys_none_err(self):
        with self.assertRaises(ValueError) as e:
            check_and_format_emb_padding_keys(padding_keys=None, padding_keys_mask=True, padding_keys_len=None)

        self.assertIn("the padding keys mask be False when padding keys is None", str(e.exception))

    def test_padding_keys_none_ok(self):
        check_and_format_emb_padding_keys(padding_keys=None, padding_keys_mask=False, padding_keys_len=None)
        self.assertTrue(callable(check_and_format_emb_padding_keys))

    def test_padding_keys_type_err(self):
        with self.assertRaises(TypeError) as e:
            check_and_format_emb_padding_keys(padding_keys="xxx", padding_keys_mask=True, padding_keys_len=None)

        self.assertIn("the padding keys must be None/int/list type", str(e.exception))

    def test_padding_keys_mask_err(self):
        with self.assertRaises(ValueError) as e:
            check_and_format_emb_padding_keys(padding_keys=8, padding_keys_mask=False, padding_keys_len=None)

        self.assertIn("Padding keys mask be True when padding keys is not None", str(e.exception))

    def test_padding_keys_len_err(self):
        with self.assertRaises(ValueError) as e:
            check_and_format_emb_padding_keys(padding_keys=8, padding_keys_mask=True, padding_keys_len=None)

        self.assertIn("Padding keys length cannot be None when padding keys is not None", str(e.exception))

    def test_padding_keys_int_ok(self):
        check_and_format_emb_padding_keys(padding_keys=8, padding_keys_mask=True, padding_keys_len=16)
        self.assertTrue(callable(check_and_format_emb_padding_keys))

    def test_padding_keys_list_ok(self):
        check_and_format_emb_padding_keys(padding_keys=[8, 8], padding_keys_mask=True, padding_keys_len=16)
        self.assertTrue(callable(check_and_format_emb_padding_keys))
