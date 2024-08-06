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

DUMP_MODEL_STR = "02dump_model"
META_LENGTH = "length"
META_EMB_DIM = "emb_dim"
META_DTYPE = "dtype"
META_DATA = "data"
META_BASIC_CONTENT_LIST = [META_EMB_DIM, META_DTYPE, META_LENGTH]

META_NAME_KEY = "key"
SPARSE_ALLCLOSE_RTOL = 1e-10


class SparseModel:
    """
    This class is used to represent parsed sparse data for choosen step and rank.
    """

    def __init__(self, data_dir: str, data_step: int):
        self.sparse_path = os.path.join(
            data_dir, DUMP_MODEL_STR, f"sparse-model-{data_step}"
        )
        self.table_path_list = list_model_dirs(self.sparse_path)
        self.table_data_dict = self.parse_sparse_info()
        self.emb_name_list = sorted(list(self.table_data_dict.keys()))

    def __eq__(self, other) -> bool:
        logging.info("Sparse model comparison start......")
        target_class = other.__class__
        if not isinstance(other, SparseModel):
            logging.error(
                "Sparse model comparison must between SparseModel, but %s is given",
                target_class,
            )
            return False
        if not compare_ckpt(self, other):
            return False
        return True

    def parse_sparse_info(self) -> dict:
        """
        Parse sparse info from sparse model.
        """
        table_data_dict = {}

        for table_path in self.table_path_list:
            logging.debug("Table path is %s", table_path)
            table_name, table_data = parse_table_info(table_path)
            table_data_dict[table_name] = table_data

        return table_data_dict


def compare_ckpt(test_data: SparseModel, golden_data: SparseModel) -> bool:
    """
    Compare two sparse model.
    """
    test_emb_name_list = test_data.emb_name_list
    golden_emb_name_list = golden_data.emb_name_list
    logging.debug(
        "Test Data tables: %s Golden Data tables: %s.",
        test_emb_name_list,
        golden_emb_name_list,
    )

    if test_emb_name_list != golden_emb_name_list:
        logging.error(
            "Test Data tables: %s Golden Data tables: %s. "
            "Comparision table names not match! Please check your input sparse model.",
            test_emb_name_list,
            golden_emb_name_list,
        )
        return False

    for table_name in golden_emb_name_list:
        test_data_dict = test_data.table_data_dict
        golden_data_dict = golden_data.table_data_dict
        if not compare_single_table_equal(test_data_dict, golden_data_dict, table_name):
            logging.error(
                "Comparision sparse table %s not equal! Please check your input sparse model.",
                table_name,
            )
            return False
    return True


def compare_single_table_equal(
    test_data_dict: dict, golden_data_dict: dict, table_name: str
) -> bool:
    """
    Compare two sparse model for the same single table.
    """
    test_table_data = test_data_dict[table_name]
    golden_table_data = golden_data_dict[table_name]
    test_meta_list = list(test_table_data.keys())
    golden_meta_list = list(golden_table_data.keys())

    test_meta_list = sorted(test_meta_list)
    golden_meta_list = sorted(golden_meta_list)
    if not check_basic_meta_equal(
        test_meta_list, golden_meta_list, test_table_data, golden_table_data
    ):
        logging.error("meta items not equal!")
        return False

    test_key = test_table_data[META_NAME_KEY][META_DATA]
    golden_key = golden_table_data[META_NAME_KEY][META_DATA]
    key_intersection = parse_key_data_and_cmp(test_key, golden_key, table_name)
    if not key_intersection:
        logging.error("Sparse Model keys have no intersection!")
        return False

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
    test_meta_list: list,
    golden_meta_list: list,
    test_table_data: dict,
    golden_table_data: dict,
) -> bool:
    """
    Compare two sparse model meta items like key. embedding...
    """
    logging.debug(
        "Test Data Meta: %s Golden Data Meta: %s.", test_meta_list, golden_meta_list
    )
    if test_meta_list != golden_meta_list:
        logging.error(
            "Test Data meta: %s Golden Data meta: %s.Comparision meta not match! Please check your input sparse model.",
            test_meta_list,
            golden_meta_list,
        )
        return False

    for meta_name in golden_meta_list:
        test_meta_data = test_table_data[meta_name]
        golden_meta_data = golden_table_data[meta_name]

        logging.debug("++++++++++++ %s ++++++++++++", meta_name)
        for meta_basic_content in META_BASIC_CONTENT_LIST:
            if meta_basic_content == META_LENGTH:
                logging.debug("[%s] no need to be the same, skip", META_LENGTH)
                continue
            test_meta_content = test_meta_data[meta_basic_content]
            golden_meta_content = golden_meta_data[meta_basic_content]
            logging.debug(
                "[%s]Test Data tables: %s Golden Data tables: %s.",
                meta_basic_content,
                test_meta_content,
                golden_meta_content,
            )

            if test_meta_content != golden_meta_content:
                logging.error(
                    "%s %s does not match. Basic Meta must have the same basic content.",
                    meta_name,
                    meta_basic_content,
                )
                return False
    return True


def parse_key_data_and_cmp(
    test_key_data: list, golden_key_data: list, table_name: str
) -> list:
    """
    Parse sparse modle as key_value and embedding pair and compare.
    """
    test_key_set = set(test_key_data)
    test_key_set_len = len(test_key_set)

    golden_key_set = set(golden_key_data)
    golden_key_set_len = len(golden_key_set)

    key_intersection = list(test_key_set & golden_key_set)
    intersection_len = len(key_intersection)

    logging.info(
        "Intersection part for table [%s] Test data: [%s/%s]  Golden data: [%s/%s]",
        table_name,
        intersection_len,
        test_key_set_len,
        intersection_len,
        golden_key_set_len,
    )
    return key_intersection


def construct_key_embedding_dict(
    key_data: list, meta_name_list: list, table_data: dict
) -> dict:
    """
    Parse sparse modle as key_value and embedding pair.
    """
    key_embedding_dict = {}
    temp_meta_name_list = copy.deepcopy(meta_name_list)
    temp_meta_name_list.remove(META_NAME_KEY)

    logging.info("Constructing key embedding dict, this may take a few minutes......")
    for i, key_value in enumerate(key_data):
        tmp_emb = [[] for _ in range(len(temp_meta_name_list))]
        for j, meta_name in enumerate(temp_meta_name_list):
            tmp_emb[j] = table_data[meta_name][META_DATA][i]
        key_embedding_dict[key_value] = tmp_emb
    logging.info("Constructing key embedding succeed.")
    return key_embedding_dict


def check_key_emb_equal(
    key_intersection: list, test_key_emb_dict: dict, golden_key_emb_dict: dict
) -> bool:
    """
    Comparse sparse modle as key_value and embedding pair.
    """
    logging.info("Comparing embeddings, this may take a few minutes......")
    for key_value in key_intersection:
        test_emb = test_key_emb_dict[key_value]
        golden_emb = golden_key_emb_dict[key_value]
        if len(test_emb) != len(golden_emb):
            logging.error(
                "Key %s Embedding shape not equal.Test shape: %s Golden shape: %s",
                key_value,
                len(test_emb),
                len(golden_emb),
            )
            return False
        for i, cur_test_emb in enumerate(test_emb):
            if not np.allclose(cur_test_emb, golden_emb[i], rtol=SPARSE_ALLCLOSE_RTOL):
                logging.error(
                    "KEY %s Embedding value not equal.Test Embedding:\n %s\nGolden Embedding:\n %s\n",
                    key_value,
                    cur_test_emb,
                    golden_emb[i],
                )
                return False
    return True


def parse_table_info(tabel_path: str) -> Tuple[str, dict]:
    """
    Parse sparse modle table info.
    """
    meta_list = list_model_dirs(tabel_path)
    table_name = os.path.basename(tabel_path)

    table_meta_dict = {}

    for meta_path in meta_list:
        logging.debug("Meta path is %s", meta_path)

        meta_name, meta_dict = parse_single_meta_data(meta_path)
        logging.debug("%s, %s", meta_name, meta_dict.keys())

        table_meta_dict[meta_name] = meta_dict

    return table_name, table_meta_dict


def parse_single_meta_data(meta_path: str) -> Tuple[str, dict]:
    """
    Parse single meta(like key or emb) data info.
    """
    meta_name = os.path.basename(meta_path)
    meta_attribute_path = os.path.join(meta_path, "slice.attribute")
    meta_data_path = os.path.join(meta_path, "slice.data")
    logging.debug(
        "Meta name: %s. Meta attribute path: %s. Meta data path: %s.",
        meta_name,
        meta_attribute_path,
        meta_data_path,
    )

    with tf.io.gfile.GFile(meta_attribute_path, "rb") as fin:
        meta_attributes_file = fin.read()
        try:
            meta_attributes = np.fromstring(meta_attributes_file, dtype=np.int64)
        except ValueError as err:
            raise RuntimeError(
                f"get attributes from file {meta_attribute_path} failed."
            ) from err

        meta_length = meta_attributes[0]
        meta_emb_dim = meta_attributes[1]

        if len(meta_attributes) == 3:
            meta_dtype = np.float32
            meta_data = np.fromfile(meta_data_path, meta_dtype)
            meta_data = meta_data.reshape(meta_length, meta_emb_dim)
        else:
            meta_dtype = np.int64
            meta_data = np.fromfile(meta_data_path, meta_dtype)

        meta_data_dict = {
            META_LENGTH: meta_length,
            META_EMB_DIM: meta_emb_dim,
            META_DTYPE: meta_dtype,
            META_DATA: meta_data,
        }
    return meta_name, meta_data_dict


def list_model_dirs(directory: str) -> list:
    """
    List given directory dirs and files and get subdirs.
    """
    current_sudirs_abspaths = []
    for dirpath, dirs, files in os.walk(directory):
        if dirpath != directory:
            continue
        if files:
            raise ValueError(
                f"find unexpected files{files}, saved model may have been tampered, please check."
            )

        for cur_dir in dirs:
            current_sudirs_abspaths.append(os.path.join(dirpath, cur_dir))
    return current_sudirs_abspaths
