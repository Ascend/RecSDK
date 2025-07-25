#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from typing import Optional, Union

from tensorflow.python.framework.ops import Operation

from mx_rec.constants.constants import ASCEND_GLOBAL_HASHTABLE_COLLECTION, TRAIN_CHANNEL_ID, EVAL_CHANNEL_ID
from mx_rec.validator.validator import para_checker_decorator, StringValidator


class TrainParamsConfig:
    """
    Configuration of training job parameters, such as dataset iterator type.
    """

    def __init__(self):
        self._ascend_global_hashtable_collection = ASCEND_GLOBAL_HASHTABLE_COLLECTION
        self._training_mode_channel_dict = dict()
        self._bool_gauge_set = set()
        self._is_graph_modify_hook_running = False
        self._is_last_round = False
        self._merged_multi_lookup = dict()
        self._target_batch = dict()
        self._iterator_type = ""
        self._sparse_dir = ""
        self._initializer_dict = dict()
        self._dataset_element_spec = None
        self._experimental_mode = None

    @property
    def iterator_type(self):
        return self._iterator_type

    @property
    def is_last_round(self):
        return self._is_last_round

    @property
    def is_graph_modify_hook_running(self):
        return self._is_graph_modify_hook_running

    @property
    def sparse_dir(self):
        return self._sparse_dir

    @property
    def ascend_global_hashtable_collection(self):
        return self._ascend_global_hashtable_collection

    @property
    def dataset_element_spec(self) -> Optional[Union[list, tuple, dict]]:
        return self._dataset_element_spec

    @property
    def experimental_mode(self) -> str:
        return self._experimental_mode

    @experimental_mode.setter
    def experimental_mode(self, mode: str):
        self._experimental_mode = mode

    @iterator_type.setter
    def iterator_type(self, iterator_type):
        self._iterator_type = iterator_type

    @is_graph_modify_hook_running.setter
    def is_graph_modify_hook_running(self, is_hook_running):
        self._is_graph_modify_hook_running = is_hook_running

    @sparse_dir.setter
    def sparse_dir(self, sparse_dir):
        self._sparse_dir = sparse_dir

    @is_last_round.setter
    def is_last_round(self, last_round):
        self._is_last_round = last_round

    @ascend_global_hashtable_collection.setter
    @para_checker_decorator(
        check_option_list=[("name", StringValidator, {"min_len": 1, "max_len": 255}, ["check_string_length"])]
    )
    def ascend_global_hashtable_collection(self, name):
        self._ascend_global_hashtable_collection = name

    @dataset_element_spec.setter
    def dataset_element_spec(self, dataset_element_spec: Union[list, tuple, dict]):
        self._dataset_element_spec = dataset_element_spec

    @property
    def bool_gauge_set(self):
        return self._bool_gauge_set

    def insert_training_mode_channel_id(self, is_training: bool) -> None:
        if is_training not in self._training_mode_channel_dict:
            # mx_rec has 2 channel for data input.
            # train_model bind to channel TRAIN_CHANNEL_ID
            # eval_model bind to channel EVAL_CHANNEL_ID
            self._training_mode_channel_dict[is_training] = TRAIN_CHANNEL_ID if is_training else EVAL_CHANNEL_ID

    def get_training_mode_channel_id(self, is_training: bool) -> bool:
        return self._training_mode_channel_dict.get(is_training)

    def insert_bool_gauge(self, name: Optional[str]) -> None:
        self._bool_gauge_set.add(name)

    def insert_merged_multi_lookup(self, is_training: bool, value: bool = True) -> None:
        self._merged_multi_lookup[is_training] = value

    def get_merged_multi_lookup(self, is_training: bool) -> None:
        return self._merged_multi_lookup.get(is_training)

    def set_target_batch(self, is_training: bool, batch: dict) -> None:
        self._target_batch[is_training] = batch

    def get_target_batch(self, is_training: bool) -> Optional[dict]:
        return self._target_batch.get(is_training)

    def get_initializer(self, is_training: bool) -> Optional[Operation]:
        return self._initializer_dict.get(is_training)

    def set_initializer(self, is_training: bool, initializer: Optional[Operation]) -> None:
        self._initializer_dict[is_training] = initializer
