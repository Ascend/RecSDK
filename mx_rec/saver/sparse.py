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
import json
from typing import List

import numpy as np
import tensorflow as tf

from mx_rec.constants.constants import MAX_INT32
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.validator.validator import para_checker_decorator, ClassValidator, ListValidator, OrValidator, AndValidator
from mx_rec.util.log import logger
from mx_rec.saver.saver import validate_read_file


class SparseProcessor:
    single_instance = None

    def __init__(self, table_list: List[str]):
        self.export_name = "key-emb"
        self.device_dir_list = ["HashTable", "HBM"]
        self.host_dir_list = ["HashTable", "DDR"]
        self.device_emb_dir = "embedding"
        self.host_emb_dir = "embedding_data"
        self.device_hashmap_dir = "key"
        self.host_hashmap_dir = "embedding_hashmap"
        self.data_suffix = ".data"
        self.attrib_suffix = ".attribute"
        self.json_attrib_dtype = "data_type"
        self.json_attrib_shape = "shape"
        self.table_list = table_list
        self.default_table_list = list(ConfigInitializer.get_instance().sparse_embed_config.table_name_set)

        if not self.table_list:
            logger.debug("table list not be set, use default value : all table created ")
            self.table_list = self.default_table_list
        else:
            self.table_list = check_table_param(self.table_list, self.default_table_list)

    @staticmethod
    def set_instance(table_list):
        SparseProcessor.single_instance = SparseProcessor(table_list)

    @staticmethod
    def _get_data(data_dir, dtype, data_shape):
        try:
            with tf.io.gfile.GFile(data_dir, "rb") as file:
                validate_read_file(data_dir)
                data = file.read()
                data = np.fromstring(data, dtype=dtype)
            data = data.reshape(data_shape)
        except Exception as err:
            raise RuntimeError(f"error happened when get data from data file {data_dir}, "
                               f"the error is `{err}`.") from err
        return data

    @staticmethod
    def _get_shape_from_attrib(attribute_dir, is_json):
        try:
            if is_json:
                with tf.io.gfile.GFile(attribute_dir, "r") as file:
                    validate_read_file(attribute_dir)
                    attributes = json.load(file)
            else:
                with tf.io.gfile.GFile(attribute_dir, "rb") as file:
                    validate_read_file(attribute_dir)
                    attributes = file.read()
                    attributes = np.fromstring(attributes, dtype=np.uint64)
        except Exception as err:
            raise RuntimeError(f"error happened when get shape from attribute file {attribute_dir}, "
                               f"the error is `{err}`.") from err
        return attributes

    def export_sparse_data(self):
        logger.info("table list to be exported is %s", self.table_list)
        sparse_dir = ConfigInitializer.get_instance().train_params_config.sparse_dir
        for table in self.table_list:
            table_dir = os.path.join(sparse_dir, table)
            key = self._get_key(table_dir)
            emb_data = self.get_embedding(table_dir)
            transformed_data = dict(zip(key[:], emb_data[:]))
            save_path = os.path.join(table_dir, self.export_name + ".npy")
            with tf.io.gfile.GFile(save_path, "wb") as file:
                np.save(file, transformed_data)

    def get_embedding(self, table_dir):
        emb_dir = os.path.join(table_dir, self.device_emb_dir)
        data_file, attribute_file = self._get_file_names(emb_dir)
        device_attribute = self._get_shape_from_attrib(attribute_file, is_json=False)
        data_shape = [device_attribute[0], device_attribute[1]]
        emb_data = self._get_data(data_file, np.float32, data_shape)
        return emb_data

    def _get_key(self, table_dir):
        key_dir = os.path.join(table_dir, self.device_hashmap_dir)
        data_file, attribute_file = self._get_file_names(key_dir)
        raw_key = self._get_data(data_file, np.uint64, -1)
        return raw_key

    def _get_file_names(self, directory):
        data_file = None
        attribute_file = None
        files = tf.io.gfile.listdir(directory)
        if not files:
            raise FileExistsError(f"There is no files under the {directory}.")
        for file in files:
            if file.find(self.data_suffix) != -1:
                data_file = file
            elif file.find(self.attrib_suffix) != -1:
                attribute_file = file
        if not data_file:
            raise FileNotFoundError(f"There is no data file under the {directory}.")
        if not attribute_file:
            raise FileNotFoundError(f"There is no attribute file under the {directory}.")

        data_file = os.path.join(directory, data_file)
        attribute_file = os.path.join(directory, attribute_file)
        if not tf.io.gfile.exists(data_file):
            raise FileExistsError(f"embedding data file {data_file} does not exist when reading.")
        if not tf.io.gfile.exists(attribute_file):
            raise FileExistsError(f"attribute file {attribute_file} does not exist when reading.")
        return data_file, attribute_file


@para_checker_decorator(check_option_list=[
    ("table_list", OrValidator, {"options": [
        (ClassValidator, {"classes": type(None)}),
        (AndValidator, {"options": [
            (ClassValidator, {"classes": list}),
            (ListValidator, {
                "sub_checker": ClassValidator,
                "list_max_length": MAX_INT32,
                "list_min_length": 1,
                "sub_args": {
                    "classes": str
                }
            },
             ["check_list_length"])
        ]})
    ]})
])
def export(table_list=None):
    empty_value = 0
    SparseProcessor.set_instance(table_list)
    if SparseProcessor.single_instance.table_list:
        return SparseProcessor.single_instance.export_sparse_data()
    else:
        logger.warning("no table can be exported ,please check if you have saved or created tables")
        return empty_value


def check_table_param(table_list, default_table_list):
    out_list = []
    for table in table_list:
        if table in default_table_list:
            out_list.append(table)
        else:
            logger.warning("%s not be created , please check your table name.", table)

    return out_list


def set_upper_dir(model_dir, dir_list):
    for directory in dir_list:
        model_dir = os.path.join(model_dir, directory)
    return model_dir
