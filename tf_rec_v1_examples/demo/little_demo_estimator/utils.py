#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

from mx_rec.core.asc.helper import FeatureSpec


class FeatureSpecIns:
    _single_instance = None
    _train_feature_spec_list = None
    _eval_feature_spec_list = None

    @staticmethod
    def set_instance():
        if FeatureSpecIns._single_instance is not None:
            raise RuntimeError("`FeatureSpecIns` single instance cannot be set once.")
        FeatureSpecIns._single_instance = FeatureSpecIns()

    @staticmethod
    def get_instance():
        if FeatureSpecIns._single_instance is None:
            raise RuntimeError("Please set `FeatureSpecIns` before get `FeatureSpecIns`.")
        return FeatureSpecIns._single_instance

    @staticmethod
    def set_train_feature_spec_list(fs_list):
        FeatureSpecIns._train_feature_spec_list = fs_list

    @staticmethod
    def get_train_feature_spec_list():
        if FeatureSpecIns._train_feature_spec_list is None:
            raise RuntimeError("Please set `train_feature_spec_list` before get `train_feature_spec_list`.")
        return FeatureSpecIns._train_feature_spec_list

    @staticmethod
    def set_eval_feature_spec_list(fs_list):
        FeatureSpecIns._eval_feature_spec_list = fs_list

    @staticmethod
    def get_eval_feature_spec_list():
        if FeatureSpecIns._eval_feature_spec_list is None:
            raise RuntimeError("Please set `eval_feature_spec_list` before get `eval_feature_spec_list`.")
        return FeatureSpecIns._eval_feature_spec_list


def create_feature_spec_list(cfg, use_timestamp=False, use_multi_lookup=False, multi_lookup_times=2):
    access_threshold = cfg.access_threshold if use_timestamp else None
    eviction_threshold = cfg.eviction_threshold if use_timestamp else None
    feature_spec_list = [FeatureSpec("user_ids", table_name="user_table",
                                     access_threshold=access_threshold, eviction_threshold=eviction_threshold),
                         FeatureSpec("item_ids", table_name="item_table",
                                     access_threshold=access_threshold, eviction_threshold=eviction_threshold)]
    if use_multi_lookup:
        for _ in range(multi_lookup_times):
            feature_spec_list.append(FeatureSpec("user_ids", table_name="user_table", access_threshold=access_threshold,
                                                 eviction_threshold=eviction_threshold, faae_coefficient=1))
    if use_timestamp:
        feature_spec_list.append(FeatureSpec("timestamp", is_timestamp=True))
    return feature_spec_list
