#!/usr/bin/env python3
# coding: UTF-8
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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

from mx_rec.util.config_utils.feature_spec_utils import FeatureSpecConfig
from tests.mx_rec.core.mock_class import MockFeatureSpec


class TestFeatureSpecConfig(unittest.TestCase):
    def test_clear_same_table_feature_spec_empty(self):
        feat_spec_config = FeatureSpecConfig()
        with self.assertRaises(KeyError):
            feat_spec_config.clear_same_table_feature_spec(table_name="xx", is_training=True)

    def test_clear_same_table_feature_spec_ok(self):
        feat_spec_config = FeatureSpecConfig()
        feature_spec = MockFeatureSpec(name="feature_spec", table_name="test_table")
        feat_spec_config.insert_feature_spec(feature_spec=feature_spec, is_training=True)
        feat_spec_config.clear_same_table_feature_spec(table_name="test_table", is_training=True)
        self.assertEqual(feat_spec_config.table_name_to_feature_spec.get(feature_spec.table_name).get(True), [])
