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

import logging
import sys
import os
import re
from typing import Dict, List, Tuple

import numpy as np

from utils import (
    init_logger,
    parse_input_param,
    validate_path,
    nested_dict_to_str,
    DEBUG_LEVEL,
    INFO_LEVEL,
    ERROR_LEVEL,
)
from dump_info import DumpInfo
from data_set import BatchDataSet
from sparse_ckpt import SparseModel
from dense_ckpt import DenseModel
from loss import Loss
from ops import OpData


OUTPUT_REGEX_STR = r"^\d{8}_\d{6}$"
DUMP_INFO_FUNC_KEY = "DumpInfo"
BATCH_DATASET_FUNC_KEY = "BatchDataSet"
SPARSE_MODEL_FUNC_KEY = "SparseModel"
DENSE_MODEL_FUNC_KEY = "DenseModel"
LOSS = "Loss"

OP_DATA_FUNC_KEY = "Opdata"


class PrecisionData:
    """
    This class is used to represent the whole package of parsed data for choosen step and rank
    """

    def __init__(self, data_path: str, data_name: str):
        logging.info(f"DataDir of {data_name}:\n{data_path} ......\n")
        validate_path(data_path, "PRECISION DATA PATH", OUTPUT_REGEX_STR)
        self.data_path = data_path
        self.data_name = data_name

        logging.info(
            f"=========== {self.data_name} [{DUMP_INFO_FUNC_KEY}] parsing start...... ==========="
        )
        self.dump_info = DumpInfo(self.data_path)
        logging.info(
            f"=========== {self.data_name} [{DUMP_INFO_FUNC_KEY}] parsing succeed...... ==========="
        )

        self.rank_size = self.dump_info.task_config["global_rank_size"]
        self.dump_step_list = self.dump_info.task_config["precision_dump_step"]

        self.data_dict = {}

        for step in self.dump_step_list:
            self.data_dict[step] = {}
            for rank in range(self.rank_size):
                self.data_dict[step][rank] = {}

        self.data_parse_func_map = {
            BATCH_DATASET_FUNC_KEY: self.parse_batch_data,
            SPARSE_MODEL_FUNC_KEY: self.parse_sparse_model,
            DENSE_MODEL_FUNC_KEY: self.parse_dense_model,
            LOSS: self.parse_loss_data,
            OP_DATA_FUNC_KEY: self.parse_op_data,
        }

    def parse_batch_data(self, data_step: int, rank_id: int) -> BatchDataSet:
        print()
        logging.info(
            f"=========== Step-{data_step} Rank-{rank_id} {self.data_name} [{BATCH_DATASET_FUNC_KEY}] "
            "parsing start...... ==========="
        )

        cur_batch_data = BatchDataSet(self.data_path, data_step, rank_id)
        self.data_dict[data_step][rank_id][BATCH_DATASET_FUNC_KEY] = cur_batch_data

        logging.info(
            f"=========== Step-{data_step} Rank-{rank_id} {self.data_name} [{BATCH_DATASET_FUNC_KEY}] "
            "parsing succeed...... ===========\n"
        )
        return cur_batch_data

    def parse_sparse_model(self, data_step: int, rank_id: int) -> SparseModel:
        print()
        logging.info(
            f"=========== Step-{data_step} Rank-{rank_id} {self.data_name} [{SPARSE_MODEL_FUNC_KEY}] "
            "parsing start...... ==========="
        )

        for i in range(self.rank_size):
            if self.data_dict[data_step][i].get(SPARSE_MODEL_FUNC_KEY):
                cur_sparse_model = self.data_dict[data_step][i].get(
                    SPARSE_MODEL_FUNC_KEY
                )
            else:
                cur_sparse_model = SparseModel(self.data_path, data_step)

        self.data_dict[data_step][rank_id][SPARSE_MODEL_FUNC_KEY] = cur_sparse_model

        logging.info(
            f"=========== Step-{data_step} Rank-{rank_id} {self.data_name} [{SPARSE_MODEL_FUNC_KEY}] "
            "parsing succeed...... ===========\n"
        )
        return cur_sparse_model

    def parse_dense_model(self, data_step: int, rank_id: int) -> DenseModel:
        print()
        logging.info(
            f"=========== Step-{data_step} Rank-{rank_id} {self.data_name} [{DENSE_MODEL_FUNC_KEY}] "
            "parsing start...... ==========="
        )

        for i in range(self.rank_size):
            if self.data_dict[data_step][i].get(DENSE_MODEL_FUNC_KEY):
                cur_dense_model = self.data_dict[data_step][i].get(DENSE_MODEL_FUNC_KEY)
            else:
                cur_dense_model = DenseModel(self.data_path, data_step)

        self.data_dict[data_step][rank_id][DENSE_MODEL_FUNC_KEY] = cur_dense_model

        logging.info(
            f"=========== Step-{data_step} Rank-{rank_id} {self.data_name} [{DENSE_MODEL_FUNC_KEY}] "
            "parsing succeed...... ===========\n"
        )
        return cur_dense_model

    def parse_loss_data(self, data_step: int, rank_id: int) -> Loss:
        print()
        logging.info(
            f"=========== Step-{data_step} Rank-{rank_id} {self.data_name} [{LOSS}] "
            "parsing start...... ==========="
        )

        cur_loss_data = Loss(self.data_path, data_step, rank_id)
        self.data_dict[data_step][rank_id][LOSS] = cur_loss_data

        logging.info(
            f"=========== Step-{data_step} Rank-{rank_id} {self.data_name} [{LOSS}] "
            "parsing succeed...... ===========\n"
        )
        return cur_loss_data

    def parse_op_data(self, data_step: int, rank_id: int) -> OpData:
        print()
        logging.info(
            f"=========== Step-{data_step} Rank-{rank_id} {self.data_name} [{OP_DATA_FUNC_KEY}] "
            "parsing start...... ==========="
        )

        cur_op_data = OpData(self.data_path, self.dump_info, data_step, rank_id)
        self.data_dict[data_step][rank_id][OP_DATA_FUNC_KEY] = cur_op_data

        logging.info(
            f"=========== Step-{data_step} Rank-{rank_id} {self.data_name} [{OP_DATA_FUNC_KEY}] "
            "parsing succeed...... ===========\n"
        )
        return cur_op_data


def construct_precision_comparison(
    test_data: SparseModel,
    golden_data: SparseModel,
    select_func_list: list,
    step_list=None,
    rank_list=None,
) -> Dict[int, Dict[int, Dict[str, bool]]]:
    parsed_step_list, parsed_rank_list = parse_step_and_rank(test_data, golden_data, step_list, rank_list)
    comparison_result = pared_and_compare_data(test_data, golden_data, select_func_list, parsed_step_list, parsed_rank_list)
    return comparison_result
    
def parse_step_and_rank(
    test_data: SparseModel,
    golden_data: SparseModel,
    step_list=None,
    rank_list=None,
) -> Tuple[List[int], List[int]]:
    # if given step not in parsed data, raise an error
    for step in step_list:
        if step not in test_data.dump_step_list:
            raise ValueError(f"Comparison Step must in Test dump_step_list.\n")
        if step not in golden_data.dump_step_list:
            raise ValueError(f"Comparison Step must in Golden dump_step_list.\n")

    # if step_list not given, compare all steps if Test and Golden steps equal
    if not step_list:
        if test_data.dump_step_list != golden_data.dump_step_list:
            raise ValueError(
                f"Custom Step List not set, Test Golden dump_step_list should be the same.\n"
                f"Test Step List:{test_data.dump_step_list}\n"
                f"Golden Step List:{golden_data.dump_step_list}."
            )
        step_list = test_data.dump_step_list

    # if test golden rank_size not equal, raise error
    if test_data.rank_size != golden_data.rank_size:
        raise ValueError(
            f"Test Golden rank_size should be the same.\n"
            f"Test Step List:{test_data.dump_step_list}\n"
            f"Golden Step List:{golden_data.dump_step_list}."
        )

    # if given rank id larger than rank size
    if max(rank_list) > test_data.rank_size:
        raise ValueError(f"Comparison Rank must smaller than rank_size.\n")

    # if rank_list not given, compare all ranks if Test and Golden ranks equal
    if not rank_list:
        rank_list = range(test_data.rank_size)
    return step_list, rank_list

def pared_and_compare_data(
    test_data: SparseModel,
    golden_data: SparseModel,
    select_func_list: list,
    step_list=None,
    rank_list=None,
) -> Dict[int, Dict[int, Dict[str, bool]]]:
    comparison_result_dict = {}
    for step in step_list:
        comparison_result_dict[step] = {}
        for rank_id in rank_list:
            comparison_result_dict[step][rank_id] = {}

    for step in step_list:
        for rank_rank, rank_id in enumerate(rank_list):
            for func_key in select_func_list:
                if rank_rank != 0 and (func_key == SPARSE_MODEL_FUNC_KEY or func_key == DENSE_MODEL_FUNC_KEY):
                    logging.info(
                        f"=========== Step-{step} Rank-{rank_id} [{func_key}] Has already been pared and compared, thus skip...... ==========="
                    )
                    compared_rank_id = rank_list[0]
                    test_data.data_dict[step][rank_id][func_key] = test_data.data_dict[step][compared_rank_id][func_key]
                    golden_data.data_dict[step][rank_id][func_key] = golden_data.data_dict[step][compared_rank_id][func_key]
                    comparison_result_dict[step][rank_id][func_key] = comparison_result_dict[step][compared_rank_id][func_key]
                    continue
                test_parsed_data, golden_parsed_data = pared_single_data(test_data, golden_data, func_key, step, rank_id)
                cur_comparison_result = compare_single_data(test_parsed_data, golden_parsed_data, func_key, step, rank_id)
                comparison_result_dict[step][rank_id][func_key] = cur_comparison_result
    return comparison_result_dict

def pared_single_data(
    test_data: SparseModel,
    golden_data: SparseModel,
    func_key:str,
    step:int,
    rank_id:int,
) -> None:
    test_parsed_data = test_data.data_parse_func_map[func_key](
        step, rank_id
    )
    golden_parsed_data = golden_data.data_parse_func_map[func_key](
        step, rank_id
    )

    test_data.data_dict[step][rank_id][func_key] = test_parsed_data
    golden_data.data_dict[step][rank_id][func_key] = golden_parsed_data
    return test_parsed_data, golden_parsed_data

def compare_single_data(
    test_parsed_data,
    golden_parsed_data,
    func_key:str,
    step:int,
    rank_id:int,
) -> bool:
    print()
    logging.info(
        f"=========== Step-{step} Rank-{rank_id} [{func_key}] comparison start ==========="
    )

    cur_comparison_result = (test_parsed_data == golden_parsed_data)
    if not cur_comparison_result:
        logging.error(
            f"===========[x] Step-{step} Rank-{rank_id} [{func_key}] does not match ===========\n"
        )
    else:
        logging.info(
            f"===========[√] Step-{step} Rank-{rank_id} [{func_key}] match ===========\n"
        )
    return cur_comparison_result

if __name__ == "__main__":
    """
    Please choose your log level before comparison
    could be DEBUG_LEVEL, INFO_LEVEL, ERROR_LEVEL
    """
    init_logger(INFO_LEVEL)

    """s
    Choose what data you want to compare
    could be any one or any combination of following choices
    BATCH_DATASET_FUNC_KEY, SPARSE_MODEL_FUNC_KEY, DENSE_MODEL_FUNC_KEY, OP_DATA_FUNC_KEY
    """
    func_key_list = [BATCH_DATASET_FUNC_KEY, SPARSE_MODEL_FUNC_KEY, DENSE_MODEL_FUNC_KEY, LOSS, OP_DATA_FUNC_KEY]
    func_key_list = [OP_DATA_FUNC_KEY]

    """
    Set up the steps of data you want to compare, could only be a list of int
    """
    step_list = [1]

    """
    Set up the ranks of data you want to compare, could only be a list of int
    """
    rank_list = range(8)

    # Parsing input param into Teat,Golden Path
    test_path, golden_path = parse_input_param()

    # Construct precision data for both test data and golden data
    test_data = PrecisionData(test_path, "[Test Data]")
    golden_data = PrecisionData(golden_path, "[Golden Data]")

    # Construct comparison using given parameter
    comparison_result_dict = construct_precision_comparison(
        test_data, golden_data, func_key_list, step_list, rank_list
    )

    # This act will show the result of all comparison that is much easier to understand
    print(nested_dict_to_str(comparison_result_dict))

    # Use pdb can look into the parsing data without running the whole process repeatly
    # print(nested_dict_to_str(test_data.data_dict))
    # print(nested_dict_to_str(golden_data.data_dict))
    # import pdb
    # pdb.set_trace()
