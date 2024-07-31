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

import copy
import logging
import os

from typing import Tuple


import numpy as np
import tensorflow as tf


META_LENGTH = "length"
META_EMB_DIM = "emb_dim"
META_DTYPE = "dtype"
META_DATA = "data"
META_BASIC_CONTENT_LIST = [META_EMB_DIM, META_DTYPE, META_LENGTH]

META_NAME_KEY = "key"
SPARSE_ALLCLOSE_RTOL = 1e-10


class SparseModel:
    def __init__(self, data_dir:str, data_step:int):
        self.sparse_path = os.path.join(
            data_dir, "02dump_model", f"sparse-model-{data_step}"
        )
        self.table_path_list = list_model_files(self.sparse_path)
        self.table_data_dict = self.parse_sparse_info()
        self.emb_name_list = sorted(list(self.table_data_dict.keys()))

    def parse_sparse_info(self) -> dict:
        table_data_dict = {}

        for table_path in self.table_path_list:
            logging.debug(f"Table path is {table_path}")
            table_name, table_data = parse_table_info(table_path)
            table_data_dict[table_name] = table_data

        return table_data_dict

    def __eq__(self, other) -> bool:
        logging.info(f"Sparse model comparison start......")
        if not isinstance(other, SparseModel):
            logging.error(
                f"Sparse model comparison must between SparseModel, but {other.__class__} is given"
            )
            return False
        if not compare_ckpt(self, other):
            return False
        return True


def compare_ckpt(test_data:SparseModel, golden_data:SparseModel) -> bool:
    test_emb_name_list = test_data.emb_name_list
    golden_emb_name_list = golden_data.emb_name_list
    logging.debug(
        f"Test Data tables: {test_emb_name_list} Golden Data tables: {golden_emb_name_list}."
    )

    if test_emb_name_list != golden_emb_name_list:
        logging.error(
            f"Test Data tables: {test_emb_name_list} Golden Data tables: {golden_emb_name_list}."
            + "Comparision table names not match! Please check your input sparse model."
        )
        return False

    for table_name in golden_emb_name_list:
        test_data_dict = test_data.table_data_dict
        golden_data_dict = golden_data.table_data_dict
        if not compare_single_table_equal(test_data_dict, golden_data_dict, table_name):
            logging.error(
                f"Comparision sparse table {table_name} not equal! Please check your input sparse model."
            )
            return False
    return True


def compare_single_table_equal(test_data_dict:dict, golden_data_dict:dict, table_name:str) -> bool:
    test_table_data = test_data_dict[table_name]
    golden_table_data = golden_data_dict[table_name]
    test_meta_list = list(test_table_data.keys())
    golden_meta_list = list(golden_table_data.keys())

    test_meta_list = sorted(test_meta_list)
    golden_meta_list = sorted(golden_meta_list)
    if not check_basic_meta_equal(
        test_meta_list, golden_meta_list, test_table_data, golden_table_data
    ):
        logging.error(f"meta items not equal!")
        return False

    test_key = test_table_data[META_NAME_KEY][META_DATA]
    golden_key = golden_table_data[META_NAME_KEY][META_DATA]
    key_intersection = parse_key_data_and_cmp(test_key, golden_key, table_name)

    test_key_emb_dict = construct_key_embedding_dict(
        test_key, test_meta_list, test_table_data
    )
    golden_key_emb_dict = construct_key_embedding_dict(
        golden_key, golden_meta_list, golden_table_data
    )
    if not check_key_emb_equal(
        key_intersection, test_key_emb_dict, golden_key_emb_dict
    ):
        return False
    return True


def check_basic_meta_equal(
    test_meta_list:list, golden_meta_list:list, test_table_data:dict, golden_table_data:dict
):
    logging.debug(
        f"Test Data Meta: {test_meta_list} Golden Data Meta: {golden_meta_list}."
    )
    if test_meta_list != golden_meta_list:
        logging.error(
            f"Test Data meta: {test_meta_list} Golden Data meta: {golden_meta_list}."
            + "Comparision meta not match! Please check your input sparse model."
        )
        return False

    for meta_name in golden_meta_list:
        test_meta_data = test_table_data[meta_name]
        golden_meta_data = golden_table_data[meta_name]

        logging.debug(f"++++++++++++ {meta_name} ++++++++++++")
        for meta_basic_content in META_BASIC_CONTENT_LIST:
            test_meta_content = test_meta_data[meta_basic_content]
            golden_meta_content = golden_meta_data[meta_basic_content]
            logging.debug(
                f"[{meta_basic_content}]Test Data tables: {test_meta_content} Golden Data tables: {golden_meta_content}."
            )

            if (
                test_meta_content != golden_meta_content
                and meta_basic_content != META_LENGTH
            ):
                logging.error(
                    f"{meta_name} {meta_basic_content} does not match. Basic Meta must have the same basic content."
                )
                return False
        print()
    return True


def parse_key_data_and_cmp(test_key_data:list, golden_key_data:list, table_name:str) -> list:
    test_key_set = set(test_key_data)
    test_key_set_len = len(test_key_set)

    golden_key_set = set(golden_key_data)
    golden_key_set_len = len(golden_key_set)

    key_intersection = list(test_key_set & golden_key_set)
    intersection_len = len(key_intersection)

    logging.info(
        f"Intersection part for table [{table_name}] Test data: [{intersection_len}/{test_key_set_len}]  "
        f"Golden data: [{intersection_len}/{golden_key_set_len}]"
    )
    return key_intersection


def construct_key_embedding_dict(key_data:list, meta_name_list:list, table_data:dict) -> dict:
    key_embedding_dict = {}
    temp_meta_name_list = copy.deepcopy(meta_name_list)
    temp_meta_name_list.remove(META_NAME_KEY)

    logging.info(f"Constructing key embedding dict, this may take a few minutes......")
    for i, key_value in enumerate(key_data):
        tmp_emb = [[]] * len(temp_meta_name_list)
        for j, meta_name in enumerate(temp_meta_name_list):
            tmp_emb[j] = table_data[meta_name][META_DATA][i]
        key_embedding_dict[key_value] = tmp_emb
    logging.info(f"Constructing key embedding succeed.")
    return key_embedding_dict


def check_key_emb_equal(key_intersection:list, test_key_emb_dict:dict, golden_key_emb_dict:dict) -> bool:
    logging.info(f"Comparing embeddings, this may take a few minutes......")
    for key_value in key_intersection:
        test_emb = test_key_emb_dict[key_value]
        golden_emb = golden_key_emb_dict[key_value]
        for i in range(len(test_emb)):
            if not np.allclose(test_emb[i], golden_emb[i], rtol=SPARSE_ALLCLOSE_RTOL):
                logging.error(
                    f"KEY {key_value} Embedding value not equal."
                    f"Test Embedding:\n {test_emb[i]}\n"
                    f"Golden Embedding:\n {golden_emb[i]}\n"
                )
                return False
    return True


def parse_table_info(tabel_path:str) -> Tuple[str, dict]:
    meta_list = list_model_files(tabel_path)
    table_name = tabel_path.split("/")[-1]

    table_meta_dict = {}

    for meta_path in meta_list:
        logging.debug(f"Meta path is {meta_path}")

        meta_name, meta_dict = parse_single_meta_data(meta_path)
        logging.debug(f"{meta_name}, {meta_dict.keys()}")

        table_meta_dict[meta_name] = meta_dict

    return table_name, table_meta_dict


def parse_single_meta_data(meta_path: str) -> Tuple[str, dict]:
    meta_name = meta_path.split("/")[-1]
    meta_attribute_path = os.path.join(meta_path, "slice.attribute")
    meta_data_path = os.path.join(meta_path, "slice.data")
    logging.debug(
        f"Meta name: {meta_name}. Meta attribute path: {meta_attribute_path}."
        + f"Meta data path: {meta_data_path}."
    )

    try:
        with tf.io.gfile.GFile(meta_attribute_path, "rb") as fin:
            meta_attributes_file = fin.read()
            try:
                meta_attributes = np.fromstring(meta_attributes_file, dtype=np.int64)
            except ValueError as err:
                raise RuntimeError(
                    f"get attributes from file {meta_attribute_path} failed."
                ) from err

            meta_length = meta_attributes[0]
            meta_embd_dim = meta_attributes[1]

            logging.debug(f"{meta_attribute_path} {meta_length}, {meta_embd_dim}")
            logging.debug(f"{meta_attributes}")

            if len(meta_attributes) == 3:
                meta_dtype = np.float32
                meta_data = np.fromfile(meta_data_path, meta_dtype)
                meta_data = meta_data.reshape(meta_length, meta_embd_dim)
            else:
                meta_dtype = np.int64
                meta_data = np.fromfile(meta_data_path, meta_dtype)

            meta_data_dict = {
                META_LENGTH: meta_length,
                META_EMB_DIM: meta_embd_dim,
                META_DTYPE: meta_dtype,
                META_DATA: meta_data,
            }
    except:
        raise ValueError(f"Some accident occur during parse {meta_path}")

    return meta_name, meta_data_dict


def list_model_files(directory:str) -> list:
    current_sudirs_abspaths = []
    for dirpath, dirs, files in os.walk(directory):
        if dirpath != directory:
            continue
        if files:
            raise ValueError(
                f"Find unexpected files{files}, saved model may have been tampered, please check."
            )

        for dir in dirs:
            current_sudirs_abspaths.append(os.path.join(dirpath, dir))
    return current_sudirs_abspaths
