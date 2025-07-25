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

import os

import tensorflow as tf

SLICE_PREFIX = "slice_"
SPARSE_FILE_PREFIX = "sparse-"
DATA_SUFFIX = ".data"
ATTRIBUTE_SUFFIX = ".attribute"
MIN_FILE_SIZE = 1
MAX_FILE_SIZE = 1024 * 1024 * 1024 * 1024


def get_attribute_and_data_file(table_path):
    if not tf.io.gfile.exists(table_path):
        raise FileNotFoundError(f"the input table path {table_path} does not exists.")

    attribute_file_list = []
    data_file_list = []
    for file_name in tf.io.gfile.listdir(table_path):
        if file_name.endswith(ATTRIBUTE_SUFFIX):
            attribute_file_list.append(file_name)
        if file_name.endswith(DATA_SUFFIX):
            data_file_list.append(file_name)
    if len(attribute_file_list) != 1:
        raise AssertionError(f"under the table path {table_path}, ther must only one attribute file. "
                             f"In fact, {len(attribute_file_list)} attribute file exists. ")
    if len(data_file_list) != 1:
        raise AssertionError(f"under the table path {table_path}, ther must only one data file. "
                             f"In fact, {len(data_file_list)} data file exists. ")
    attribute_file = os.path.join(table_path, attribute_file_list[0])
    data_file = os.path.join(table_path, data_file_list[0])
    return attribute_file, data_file


def generate_upper_dir(sparse_file, dir_prefix_list, table_name, data_type):
    temp_dir = sparse_file
    for dir_prefix in dir_prefix_list:
        temp_dir = os.path.join(temp_dir, dir_prefix)
    return os.path.join(temp_dir, table_name, data_type)


def generate_attribute_dir(sparse_file, dir_prefix_list, table_name, data_type, rank_id):
    temp_dir = sparse_file
    for dir_prefix in dir_prefix_list:
        temp_dir = os.path.join(temp_dir, dir_prefix)
    return os.path.join(temp_dir, table_name, data_type, f"{SLICE_PREFIX}{rank_id}{ATTRIBUTE_SUFFIX}")


def generate_data_dir(sparse_file, dir_prefix_list, table_name, data_type, rank_id):
    temp_dir = sparse_file
    for dir_prefix in dir_prefix_list:
        temp_dir = os.path.join(temp_dir, dir_prefix)
    return os.path.join(temp_dir, table_name, data_type, f"{SLICE_PREFIX}{rank_id}{DATA_SUFFIX}")


def validate_read_file(read_path):
    if os.path.abspath(read_path) != os.path.realpath(read_path):
        raise ValueError(f"the path {read_path} to be read is soft link.")
    file_stat = tf.io.gfile.stat(read_path)
    if not MIN_FILE_SIZE < file_stat.length <= MAX_FILE_SIZE:
        raise ValueError(f"file size: {file_stat.length} is invalid, not in ({MIN_FILE_SIZE}, {MAX_FILE_SIZE}]")
