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

from typing import Union, Optional
from functools import reduce

import tensorflow as tf

from mx_rec.util.atomic import AtomicInteger
from mx_rec.util.initialize import insert_feature_spec, insert_training_mode_channel_id, get_use_static
from mx_rec.util.normalization import fix_invalid_table_name
from mx_rec.constants.constants import MAX_INT32
from mx_rec.validator.validator import ClassValidator, StringValidator, para_checker_decorator, \
    OptionalStringValidator, OptionalIntValidator
from mx_rec.util.log import logger

feature_spec_global_id = AtomicInteger()


class FeatureSpec:
    instance_count_train = 0
    instance_count_eval = 0
    use_timestamp_train = False
    use_timestamp_eval = False

    @para_checker_decorator(check_option_list=[
        ("name", StringValidator, {"min_len": 1, "max_len": 255}, ["check_string_length"]),
        ("table_name", OptionalStringValidator, {"min_len": 1, "max_len": 255}, ["check_string_length"]),
        ("table_name", ClassValidator, {"classes": (str, type(None))}),
        ("index_key", OptionalStringValidator, {"min_len": 1, "max_len": 255}, ["check_string_length"]),
        ("index_key", OptionalIntValidator, {"min_value": 0, "max_value": 255}, ["check_value"]),
        ("index_key", ClassValidator, {"classes": (str, int, type(None))}),
        ("access_threshold", OptionalIntValidator, {"min_value": -1, "max_value": MAX_INT32}, ["check_value"]),
        ("access_threshold", ClassValidator, {"classes": (int, type(None))}),
        ("eviction_threshold", OptionalIntValidator, {"min_value": -1, "max_value": MAX_INT32}, ["check_value"]),
        ("eviction_threshold", ClassValidator, {"classes": (int, type(None))}),
        ("is_timestamp", ClassValidator, {"classes": (bool, type(None))}),
        ("batch_size", OptionalIntValidator, {"min_value": 1, "max_value": MAX_INT32}, ["check_value"]),
        ("batch_size", ClassValidator, {"classes": (int, type(None))}),
        ("faae_coefficient", OptionalIntValidator, {"min_value": 1, "max_value": MAX_INT32}, ["check_value"])
    ])
    def __init__(self, name: str,
                 table_name: Optional[str] = None,
                 index_key: Union[None, int, str] = None,
                 access_threshold: Optional[int] = None,
                 eviction_threshold: Optional[int] = None, is_timestamp: Optional[bool] = None,
                 batch_size: Optional[int] = None, faae_coefficient: Optional[int] = 1):
        feature_spec_global_id.increase()
        spec_name = name + f"_{feature_spec_global_id}"
        self.name = spec_name
        # 防止当index_key=0时，判断条件被误判为False
        if isinstance(index_key, int):
            self._index_key = index_key
        else:
            self._index_key = index_key if index_key else name
        self._table_name = fix_invalid_table_name(table_name if table_name else name)
        self._feat_cnt = None
        self._access_threshold = access_threshold
        self._eviction_threshold = eviction_threshold
        self._faae_coefficient = faae_coefficient
        self._is_timestamp = is_timestamp
        self.feat_pos_train = None
        self.feat_pos_eval = None
        self.dims = None
        self.rank = None
        self.batch_size = batch_size
        self.split = None  # usually split == batch_size * feature_count
        self.initialized = False
        self._pipeline_mode = set()

        if self._access_threshold is None and self._eviction_threshold is not None:
            raise ValueError(f"Access_threshold should be configured before eviction_threshold.")

    @property
    def is_timestamp(self):
        return self._is_timestamp

    @property
    def access_threshold(self):
        return self._access_threshold

    @property
    def eviction_threshold(self):
        return self._eviction_threshold

    @property
    def faae_coefficient(self):
        return self._faae_coefficient

    @property
    def index_key(self):
        return self._index_key

    @property
    def table_name(self):
        return self._table_name

    @property
    def feat_cnt(self):
        return self._feat_cnt

    @property
    def pipeline_mode(self):
        return self._pipeline_mode

    @feat_cnt.setter
    def feat_cnt(self, feat_cnt: int):
        self._feat_cnt = feat_cnt

    @staticmethod
    def include_timestamp(is_training):
        if is_training:
            if FeatureSpec.use_timestamp_train:
                raise EnvironmentError(f"Timestamp was set twice for training mode.")
            FeatureSpec.use_timestamp_train = True
        else:
            FeatureSpec.use_timestamp_eval = True

    @staticmethod
    def use_timestamp(is_training):
        return FeatureSpec.use_timestamp_train if is_training else FeatureSpec.use_timestamp_eval

    def set_feat_pos(self, is_training):
        if is_training:
            self.feat_pos_train = FeatureSpec.instance_count_train
            FeatureSpec.instance_count_train += 1
        else:
            self.feat_pos_eval = FeatureSpec.instance_count_eval
            FeatureSpec.instance_count_eval += 1

    def insert_pipeline_mode(self, mode):
        if not isinstance(mode, bool):
            raise TypeError("Is training mode must be a boolean.")

        if mode and mode in self._pipeline_mode:
            logger.info("FeatureSpec%s. Is training mode [%s] has been set.", self.name, mode)
            return

        insert_training_mode_channel_id(is_training=mode)

        self._pipeline_mode.add(mode)

    def set_feat_attribute(self, tensor, is_training):
        self.insert_pipeline_mode(is_training)
        self.set_feat_pos(is_training)
        if not self.initialized:
            self.initialized = True

            if get_use_static():
                self.dims = tensor.shape.as_list()
                self.rank = tensor.shape.rank
                if self.rank < 1:
                    raise ValueError(f"Given tensor rank cannot be smaller than 1, which is {self.rank} now.")

                inferred_feat_cnt = 1 if self.rank == 1 else reduce(lambda x, y: x * y, self.dims[1:])
                logger.debug("update feature_spec[%s] feature_count to %s via %s", self.name, inferred_feat_cnt,
                             self.dims)
                self.batch_size = self.dims[0]
                self._feat_cnt = inferred_feat_cnt
                self.split = self.batch_size * self._feat_cnt
            else:
                tensor = tf.reshape(tensor, [-1])
                self.dims = tf.shape(tensor)
                self.rank = 1
                self.split = tf.math.reduce_prod(tf.shape(tensor))
                self.batch_size = self.split
                self._feat_cnt = 1

        else:
            logger.debug("The initialized Feature Spec was set once again.")
            if get_use_static():
                if self.dims != tensor.shape.as_list():
                    raise ValueError(f"Given static Tensor shape mismatches with the last one, whose is_training mode "
                                     f"is not {is_training}. ")
            else:
                if self.dims.shape.as_list() != tf.shape(tf.reshape(tensor, [-1])).shape.as_list():
                    raise ValueError(f"Given dynamic Tensor shape mismatches with the last one, whose is_training mode "
                                     f"is not {is_training}. ")

        insert_feature_spec(self, is_training)
        result = {
            'tensor': tensor,
            'table_name': self.table_name,
            'feat_count': self.feat_cnt,
            'split': self.split,
        }
        return result


def get_feature_spec(table_name, access_and_evict_config):
    access_threshold = None
    eviction_threshold = None
    faae_coefficient = None
    if access_and_evict_config:
        access_threshold = access_and_evict_config.get("access_threshold")
        eviction_threshold = access_and_evict_config.get("eviction_threshold")
        faae_coefficient = access_and_evict_config.get("faae_coefficient", 1)
    return FeatureSpec(table_name, access_threshold=access_threshold, eviction_threshold=eviction_threshold,
                       faae_coefficient=faae_coefficient)


def set_temporary_feature_spec_attribute(mock_feature_spec: FeatureSpec, total_feature_count: Union[int, tf.Tensor]):
    """
    Set properties for a temporary feature_spec.

    Args:
        mock_feature_spec: A temporary feature_spec consisting of multiple feature_spec with the same table.
        total_feature_count: Inner product of the shape of a tensor.

    Returns: None

    """
    mock_feature_spec.batch_size = total_feature_count
    mock_feature_spec.feat_cnt = 1
    mock_feature_spec.dims = [total_feature_count, 1]
    mock_feature_spec.initialized = True
    mock_feature_spec.pipeline_mode.add(True)
    mock_feature_spec.pipeline_mode.add(False)
