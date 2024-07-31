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
import os
from utils import parse_json_to_dict, parse_dict


TASK_CONFIG = "task_config"
DUMP_OP_INFO = "dump_op_info"
DUMP_EMB_OP_INFO = "dump_emb_op_info"


class DumpInfo:
    def __init__(self, data_dir:str):
        self.path = os.path.join(data_dir, "dump_info.json")
        self.dump_dict = parse_json_to_dict(self.path)

        self.task_config = parse_dict(self.dump_dict, TASK_CONFIG)
        self.task_config_key_list = sorted(list(self.task_config.keys()))
        self.dump_op_info = parse_dict(self.dump_dict, DUMP_OP_INFO)
        self.dump_emb_op_info = parse_dict(self.dump_dict, DUMP_EMB_OP_INFO)
        self.table_info = sorted(list(self.dump_emb_op_info.keys()))

    def __eq__(self, other) -> bool:
        logging.info(f"[DumpInfo] comparison start......\n")

        if not isinstance(other, DumpInfo):
            logging.error(
                f"[DumpInfo] comparison must between DumpInfo, but {other.__class__} is given\n"
            )
            return False

        if not parse_and_compare_dump_info(self, other):
            logging.error(f"[DumpInfo] invalid.....\n")
            return False

        logging.info(f"[DumpInfo] comparison Finished......\n")
        return True


def parse_and_compare_dump_info(test_info:DumpInfo, golden_info:DumpInfo) -> bool: 
    if test_info.table_info != golden_info.table_info:
        logging.error(f"Table info invalid......")
        logging.error(
            f"Test tables: {test_info.table_list} Golden tables: {golden_info.table_list}"
        )
        return False

    if test_info.dump_op_info != golden_info.dump_op_info:
        logging.error(f"Dump op info invalid......")
        logging.error(
            f"Test Dump op: {test_info.dump_op_info} Golden Dump op: {golden_info.dump_op_info}"
        )
        return False

    if not parse_and_compare_task_config(test_info, golden_info):
        logging.error(f"Task config invalid......")
        return False
    return True


def parse_and_compare_task_config(test_info:DumpInfo, golden_info:DumpInfo) -> bool:
    if test_info.task_config_key_list != golden_info.task_config_key_list:
        logging.error(
            f"Table keys invalid. Test keys: {test_info.task_config_key_list} Golden keys: {golden_info.task_config_key_list}"
        )
        return False
    logging.error(
        f"Table keys invalid. Test keys: {test_info.task_config_key_list} Golden keys: {golden_info.task_config_key_list}"
    )

    for key in test_info.task_config_key_list:
        test_value = test_info.task_config[key]
        golden_value = golden_info.task_config[key]
        if test_value != golden_value:
            logging.info(
                f"++++++++++ Comparision between {key.upper()}: Test[{test_value}] Golden[{golden_value}] ++++++++++"
            )

    return True
