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
import os
import logging
import re
from typing import List
import six

import tensorflow as tf
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.training import warm_starting_util
if tf.__version__.startswith("1"):
    from npu_bridge.npu_init import NPUEstimator
else:
    from npu_device.compat.v1.npu_init import NPUEstimator

from mx_rec.util.log import logger
from mx_rec.saver.saver import Saver


class WarmStartController:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WarmStartController, cls).__new__(cls)
            cls._instance._warm_start_dict = {}
            cls._instance.table_name_to_prev_table_name = {}
        return cls._instance

    def __init__(self):
        logging.info("start to build WarmStartController.")

    def add_element(self, path: str, table_list: List[str]):
        if path not in self._warm_start_dict:
            self._warm_start_dict[path] = table_list
        else:
            self._warm_start_dict[path] += table_list

    def add_table_to_prev_table(self, table: str, prev_table: str):
        self.table_name_to_prev_table_name[table] = prev_table

    def get_elements(self):
        return self._warm_start_dict


def patch_for_warm_start():
    estimator_lib.Estimator.__init__ = patch_estimator_init(estimator_lib.Estimator.__init__)
    warm_starting_util.warm_start = patch_for_func_warm_start(warm_starting_util.warm_start)
    NPUEstimator.train = patch_for_estimator_train(NPUEstimator.train)


def patch_estimator_init(func):
    def wrapper(*args, **kwargs):
        warm_start_from = kwargs.get('warm_start_from', None)
        if warm_start_from:
            kwargs['warm_start_from'] = warm_settings_filter(warm_start_from)
        return func(*args, **kwargs)
    return wrapper


def patch_for_func_warm_start(func):
    def wrapper(*args, **kwargs):
        ckpt_to_initialize_from = args[0]
        if isinstance(ckpt_to_initialize_from, (list, tuple)):
            vars_to_warm_start_list = args[1]
            var_name_to_prev_var_name_list = args[3]
            warm_start_num = len(ckpt_to_initialize_from)
            for i in range(warm_start_num):
                f = func(ckpt_to_initialize_from[i], vars_to_warm_start_list[i], args[2],
                         var_name_to_prev_var_name_list[i], **kwargs)
            return f
        else:
            return func(*args, **kwargs)
    return wrapper


def patch_for_estimator_train(func):
    def wrapper(*args, **kwargs):
        hooks = kwargs.get('hooks', [])
        if WarmStartController().get_elements():
            hooks.append(SparseRestoreHook())
        return func(*args, **kwargs)
    return wrapper


def warm_settings_filter(warm_start_from):
    warm_start_from_res = None
    if isinstance(warm_start_from, estimator_lib.WarmStartSettings):
        if isinstance(warm_start_from.ckpt_to_initialize_from, (list, tuple)):
            out_setting_list = []
            logger.info("According to warm_start_settings, warm start will load from more than one checkpoint path.")
            warm_start_settings_list = _build_warm_settings_list(warm_start_from)
            for setting in warm_start_settings_list:
                filter_setting = _warm_settings_filter(setting)
                if filter_setting:
                    out_setting_list.append(filter_setting)
            if out_setting_list:
                warm_start_from_res = recover_warm_settings(out_setting_list)
        elif isinstance(warm_start_from.ckpt_to_initialize_from, (six.string_types, six.binary_type)):
            logger.info("According to warm_start_settings, warm start will load from only one checkpoint path.")
            filter_setting = _warm_settings_filter(warm_start_from)
            if filter_setting:
                warm_start_from_res = filter_setting
    elif isinstance(warm_start_from, (six.string_types, six.binary_type)):
        table_name_list = get_table_name_set_by_ckpt_path(warm_start_from)
        WarmStartController().add_element(warm_start_from, table_name_list)
        warm_start_from_res = warm_start_from
    else:
        raise ValueError("Invalid parameter: warm_start_from. ")
    return warm_start_from_res


def recover_warm_settings(setting_list):
    ckpt_to_initialize_from_list = []
    vars_to_warm_start_list = []
    var_name_to_prev_var_name_list = []
    for setting in setting_list:
        ckpt_to_initialize_from_list.append(setting.ckpt_to_initialize_from)
        vars_to_warm_start_list.append(setting.vars_to_warm_start)
        var_name_to_prev_var_name_list.append(setting.var_name_to_prev_var_name)

    return estimator_lib.WarmStartSettings(
        ckpt_to_initialize_from=ckpt_to_initialize_from_list,
        vars_to_warm_start=vars_to_warm_start_list,
        var_name_to_prev_var_name=var_name_to_prev_var_name_list)


def _build_warm_settings_list(warm_start_from):
    ckpt_to_initialize_from = warm_start_from.ckpt_to_initialize_from
    vars_to_warm_start = warm_start_from.vars_to_warm_start
    var_name_to_prev_var_name = warm_start_from.var_name_to_prev_var_name
    # 类型校验
    for params in [vars_to_warm_start, var_name_to_prev_var_name]:
        if not isinstance(params, (list, tuple)):
            raise ValueError("If you choose to load from multiple model paths through the warm start option, "
                             "then the parameter type in the warm settings should be a list.")
    # 长度校验
    if not (len(ckpt_to_initialize_from) == len(vars_to_warm_start) == len(var_name_to_prev_var_name)):
        raise ValueError("If you choose to load from multiple model paths through the warm start option, "
                         "then the parameter list list should be the same length. ")
    warm_start_settings_count = len(ckpt_to_initialize_from)

    warm_start_settings_list = []
    for i in range(warm_start_settings_count):
        tmp_settings = estimator_lib.WarmStartSettings(
            ckpt_to_initialize_from=ckpt_to_initialize_from[i],
            vars_to_warm_start=vars_to_warm_start[i],
            var_name_to_prev_var_name=var_name_to_prev_var_name[i])
        warm_start_settings_list.append(tmp_settings)
    return warm_start_settings_list


def _warm_settings_filter(warm_start_setting):
    vars_to_warm_start = warm_start_setting.vars_to_warm_start
    var_name_to_prev_var_name = warm_start_setting.var_name_to_prev_var_name
    vars_to_warm_start_res = []
    warm_start_setting_res = None
    table_name_list = get_table_name_set_by_ckpt_path(warm_start_setting.ckpt_to_initialize_from)
    if isinstance(vars_to_warm_start, str):
        matching_tables = [table for table in table_name_list if re.match(vars_to_warm_start, table)]
        if matching_tables:
            WarmStartController().add_element(warm_start_setting.ckpt_to_initialize_from, matching_tables)
            if vars_to_warm_start != ".*":
                return warm_start_setting_res
        warm_start_setting_res = warm_start_setting
    elif all(isinstance(v, str) for v in vars_to_warm_start):
        sparse_vars = []
        for v in vars_to_warm_start:
            matching_tables = [table for table in table_name_list if re.match(v, table)]
            if matching_tables:
                sparse_vars.append(v)
                WarmStartController().add_element(warm_start_setting.ckpt_to_initialize_from, matching_tables)
        vars_to_warm_start_res = [v for v in vars_to_warm_start if v not in sparse_vars]
        if not vars_to_warm_start_res:
            warm_start_setting = None
        else:
            warm_start_setting.vars_to_warm_start = vars_to_warm_start_res
        warm_start_setting_res = warm_start_setting
    else:
        raise ValueError("vars_to_warm_start must be list or str!")
    return warm_start_setting_res


def get_table_name_set_by_ckpt_path(warm_start_path: str) -> List[str]:
    '''
    Get the list of sparse table names saved under the path 'warm_start_path'.
    '''
    table_name_list = []
    if tf.io.gfile.isdir(warm_start_path):
        restore_path = get_latest_ckpt(warm_start_path)
    else:
        restore_path = warm_start_path
    directory, base_name = os.path.split(restore_path)
    ckpt_name = f"sparse-{base_name}"
    sparse_path = os.path.join(directory, ckpt_name)
    if not tf.io.gfile.isdir(sparse_path):
        logger.info(f"under the warm start path {warm_start_path}, sparse directory {sparse_path} not exists.")
    else:
        for dirname in tf.io.gfile.listdir(sparse_path):
            table_name_list.append(dirname)
    return table_name_list


def get_latest_ckpt(warm_start_path: str) -> str:
    ckpt_path = os.path.join(warm_start_path, "checkpoint")
    if not tf.io.gfile.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file is missing under the warm start model path {warm_start_path}")
    with tf.io.gfile.GFile(ckpt_path, "r") as f:
        latest_ckpt = f.readline().rstrip()
        latest_ckpt = latest_ckpt.split(":")[1].strip(' ').replace('"', '')
        latest_ckpt = latest_ckpt.split("/")[-1]
    path = os.path.join(warm_start_path, latest_ckpt)
    return path


class SparseRestoreHook(tf.estimator.SessionRunHook):
    def __init__(self):
        logging.info("In warm start mode, SparseRestoreHook has been initialized.")
        self._is_warm_start = False
        self._saver = None
        self._warm_start_dict = {}

    def begin(self):
        self._saver = Saver()
        logging.info("In warm start mode, begin SparseRestoreHook.")

    def after_create_session(self, session, coord):
        if not self._is_warm_start:
            self._warm_start_dict = WarmStartController().get_elements()
            for path, restore_tables in self._warm_start_dict.items():
                restore_path = get_latest_ckpt(path)
                self._saver.restore(session, restore_path, restore_tables)
            self._is_warm_start = True
