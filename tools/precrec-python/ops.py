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

import json
import logging
import subprocess
import os
import re
from typing import Tuple


import numpy as np

from dataclasses import dataclass


from utils import validate_path, nested_dict_to_str
from dump_info import DumpInfo


DUMP_OP_NUMPY_REGEX_STR = r"^.+\.npy$"
OP_NUMPY_ATOL = 1e-10

DUMP_NP_LEN = 8

INSTRUCT_PYTHON = "python3"
INSTRUCT_CONVERT = "convert"
INSTRUCT_D = "-d"
INSTRUCT_OUT = "-out"

ASCEND_TOOLKIT_MSACCUCMP_PATH = (
    "/usr/local/Ascend/ascend-toolkit/latest/tools/operator_cmp/compare/msaccucmp.py"
)

DYN_EXP_OP_LIST = set(["EmbeddingLookupByAddress", "EmbeddingUpdateByAddress"])
NO_DYN_OP_LIST = set(["GatherV2", "ScatterNdAdd"])

DYN_ARCH_OP_TYPE = "EmbeddingLookupByAddress"
NODYN_ARCH_OP_TYPE = "GatherV2"

OP_TYPE = "op_type"
OP_DATA_PATH = "op_data_path"
DATA_TYPE = "data_type"
DATA_INDEX = "data_index"

LOOKUP_TABLE = "lookup_table"
UPDATE_TABLE = "update_table"
UPDATE_GRAD = "update_grad"


NODYN_STAMP_CONSTRUCT_DICT = {
    "lookup_table": ["output", "0"],
    "update_grad": ["input", "2"],
}

DYN_STAMP_CONSTRUCT_DICT = {
    "lookup_table": ["output", "0"],
    "update_grad": ["input", "1"],
}

class OpData:
    def __init__(self, data_dir:str, dump_info:DumpInfo, data_step:int, rank_id:int):
        self.dump_data_path = find_match_op_dump_data(data_dir, rank_id, data_step)
        self.dump_info = dump_info
        validate_path(ASCEND_TOOLKIT_MSACCUCMP_PATH, "msaccucmp.py")
        self.output_numpy_path = exe_msaccucmp_convert(self.dump_data_path)
        self.op_data_dict = self.parse_numpy_data()

    def __eq__(self, other) -> bool:
        test_op_data_dict = self.op_data_dict
        golden_op_data_dict = other.op_data_dict

        test_op_data_keys = test_op_data_dict.keys()
        golden_op_data_keys = golden_op_data_dict.keys()
        test_op_list = sorted(list(test_op_data_keys))
        golden_op_list = sorted(list(golden_op_data_keys))

        if test_op_list != golden_op_list:
            logging.error(
                f"[OpData]Test data and Golden data should have the same names, "
                + f"but Test:{test_op_list} Golden:{golden_op_list} are given."
            )
            return False

        for table_name in golden_op_list:
            test_table_data = test_op_data_dict[table_name]
            golden_table_data = golden_op_data_dict[table_name]
            for emb_type in golden_table_data.keys():
                test_emb_data = test_table_data[emb_type][0]
                golden_emb_data = golden_table_data[emb_type][0]

                logging.debug(
                    f"[OpData][{table_name}][{emb_type}] Data are shown as below.\n"
                    f"Test: {test_emb_data.dtype} {test_emb_data.shape}\n"
                    f"Golden: {golden_emb_data.dtype} {golden_emb_data.shape}\n"
                    f"Test: {test_emb_data}\n"
                    f"Golden: {golden_emb_data}\n"
                )

                if test_emb_data.dtype != golden_emb_data.dtype:
                    logging.error(
                        f"[OpData][{table_name}][{emb_type}] Test and Golden shape not equal.\n"
                        f"Test:{test_emb_data.dtype}\n"
                        f"Golden:{golden_emb_data.dtype}\n"
                    )
                    return False

                if test_emb_data.shape != golden_emb_data.shape:
                    logging.error(
                        f"[OpData][{table_name}][{emb_type}] Test and Golden shape not equal.\n"
                        f"Test:{test_emb_data.shape}\n"
                        f"Golden:{golden_emb_data.shape}\n"
                    )
                    return False

                if not np.allclose(test_emb_data, golden_emb_data, OP_NUMPY_ATOL):
                    logging.error(
                        f"[OpData][{table_name}][{emb_type}] Test and Golden value not equal.\n"
                        f"Test:{test_emb_data}\n"
                        f"Golden:{golden_emb_data}\n"
                    )
                    return False
        return True

    def parse_numpy_data(self):
        for dirpath, dirs, files in os.walk(self.output_numpy_path):
            if dirs:
                logging.warning(
                    f"Dump op numpy path should not contain any directory, your file may have been tampered: {self.output_numpy_path}"
                )
            files = sorted(files)

            logging.debug(
                f"parsing numpy op data to desc start......\nNumpy dir:{dirpath}  Numpy files: {files}."
            )
            op_numpy_des_dict = self.parse_op_numpy_to_desc(dirpath, files)
            logging.debug(f"Parsing numpy op data to desc succeed.")

            op_numpy_data_dict = parse_op_desc_to_data(
                self.dump_info.dump_emb_op_info, op_numpy_des_dict
            )
        return op_numpy_data_dict

    def parse_op_numpy_to_desc(self, dir_path:str, file_names:list) -> dict:
        op_desc_dict = {}
        for filename in file_names:
            cur_op_data_path = os.path.join(dir_path, filename)
            validate_path(cur_op_data_path, "op_dump_numpy", DUMP_OP_NUMPY_REGEX_STR)

            file_name_split = filename.split(".")
            if len(file_name_split) != DUMP_NP_LEN:
                raise ValueError(
                    f"Dump op numpy may have been tamperd or msaccucmp updated. Path:{cur_op_data_path}"
                )

            cur_op_type = file_name_split[0]
            cur_op_name = file_name_split[1]
            cur_op_data_type = file_name_split[-3]
            cur_op_data_index = file_name_split[-2]

            op_desc = {
                DATA_TYPE: cur_op_data_type,
                DATA_INDEX: cur_op_data_index,
                OP_TYPE: cur_op_type,
                OP_DATA_PATH: cur_op_data_path,
            }
            if not op_desc_dict.get(cur_op_name):
                op_desc_dict[cur_op_name] = []

            op_desc_dict[cur_op_name].append(op_desc)
            logging.debug(
                f"Parsing op name to op desc succeed. Cur_op_name:{cur_op_name} Op_desc: {op_desc}."
            )

        op_type_set = set([op_desc[0][OP_TYPE] for _, op_desc in op_desc_dict.items()])

        if op_type_set == DYN_EXP_OP_LIST:
            self.use_dyn_exp = True
        elif op_type_set == NO_DYN_OP_LIST:
            self.use_dyn_exp = False

        op_numpy_desc_dict = div_op_desc_by_table(self.use_dyn_exp, op_desc_dict)
        return op_numpy_desc_dict


def find_match_op_dump_data(dump_data_path:str, rank_id:int, step:int) -> str:
    # 算子最终要去解析的文件是 /xxxx/20240724_141123/03dump_op/20240724141134/0/ge_default_20240724141135_31/4/0
    pattern_str = (
        f"^{re.escape(dump_data_path)}/"  # 匹配 precision check的位置:/xxxx/20240724_141123
        r"04dump_op/"  # 匹配 03dump_op
        r"\d{14}/"  # 匹配 日期: 20240724141134
        f"{re.escape(str(rank_id))}/"  # 匹配 rankid: 0
        r"ge_default_\d{14}_\d+/"  # 匹配ge_default_后跟日期和时间戳
        r"\d+/"  # 匹配 model id: 4
        f"{re.escape(str(step-1))}"  # 匹配 step: 0
    )
    pattern = re.compile(pattern_str)
    for root, dirs, files in os.walk(dump_data_path):
        for dir in dirs:
            op_path = os.path.join(root, dir)
            if pattern.match(op_path):
                logging.debug(f"Find matched Dump op path: {op_path}")
                return op_path


def exe_msaccucmp_convert(op_data_path:str) -> str:
    output_numpy_path = os.path.join(op_data_path, "dump_op_np")
    logging.info(f"convert target path: {output_numpy_path}")

    if os.path.exists(output_numpy_path):
        logging.warning(
            f"Dump op data have already been parsed and Convertion stop!!! This may cause some mistakes, please check the path: {output_numpy_path}"
        )
        return output_numpy_path
    else:
        os.makedirs(output_numpy_path)
        logging.debug(f"Dump op parse output dir created, path: {output_numpy_path}")

    instruct_item_command = [
        INSTRUCT_PYTHON,
        ASCEND_TOOLKIT_MSACCUCMP_PATH,
        INSTRUCT_CONVERT,
        INSTRUCT_D,
        op_data_path,
        INSTRUCT_OUT,
        output_numpy_path,
    ]
    logging.debug(f"msaccucmp convert exec instruction: {instruct_item_command}")

    convert_result = subprocess.run(
        instruct_item_command, capture_output=True, text=True
    )

    # 检查命令是否成功执行
    if convert_result.returncode == 0:
        logging.info(f"Msaccucmp convert dump op to numpy succeed.")
    else:
        raise ValueError(
            f"Msaccucmp convert dump op to numpy Failed!\n" + f"{convert_result.stdout}"
        )

    return output_numpy_path


def div_op_desc_by_table(use_dyn_exp:bool, op_desc_dict:dict) -> dict:
    table_op_desc_dict = {"use_dyn_exp": use_dyn_exp}
    table_name_set = set()

    if use_dyn_exp:
        arch_op_type = DYN_ARCH_OP_TYPE
    else:
        arch_op_type = NODYN_ARCH_OP_TYPE

    op_name_list = sorted(list(op_desc_dict.keys()))
    logging.debug(
        f"Div op desc by table start......\nUse_dyn_exp:{use_dyn_exp} Op_name_list:{op_name_list}"
    )

    for op_name in op_name_list:
        op_desc_list = op_desc_dict[op_name]
        if op_desc_list[0][OP_TYPE] == arch_op_type:
            if not op_name.startswith("LazyAdam"):
                table_name = op_name.split("__")[0]
            else:
                op_name_str = op_name.split("_")
                table_name = "_".join([op_name_str[3], op_name_str[4]])

            table_name_set.add(table_name)
            table_op_desc_dict[table_name] = {}

    logging.debug(f"Parsing table names succeed.Table list: \n{table_name_set}")

    for table_name in table_name_set:
        for op_name in op_name_list:
            if table_name in op_name:
                table_op_desc_dict[table_name][op_name] = op_desc_dict[op_name]
    logging.debug(
        f"Div op desc by table succeed.\nTable_op_desc_dict:{nested_dict_to_str(table_op_desc_dict)}"
    )
    return table_op_desc_dict


@dataclass
class TABLE_EMB_DATA:
    lookup_result: np.array
    update_grad: np.array


def parse_op_desc_to_data(dump_emb_op_info:DumpInfo, op_numpy_des_dict:dict) -> dict:
    '''
    # dump_emb_op_info
    # {"user_table": {"emb_look_ops": ["user_table//user_table_lookup/gather_for_id_offsets", "LazyAdam_0/update_user_table/GatherV2", "LazyAdam_0/update_user_table/GatherV2_1"], "emb_update_ops": ["LazyAdam_0/update_user_table/ScatterNdAdd", "LazyAdam_0/update_user_table/ScatterNdAdd_1", "LazyAdam_0/update_user_table/ScatterNdAdd_2"]},
    #  "item_table": {"emb_look_ops": ["item_table//item_table_lookup/gather_for_id_offsets", "LazyAdam_0/update_item_table/GatherV2", "LazyAdam_0/update_item_table/GatherV2_1"], "emb_update_ops": ["LazyAdam_0/update_item_table/ScatterNdAdd", "LazyAdam_0/update_item_table/ScatterNdAdd_1", "LazyAdam_0/update_item_table/ScatterNdAdd_2"]}}

    # op_numpy_des_dict
    # {"use_dyn_exp":bool
    #  "user_table":{op_name:[OP_FILE_DESC]},
    #  "item_table":{op_name:[OP_FILE_DESC]}}
    '''
    table_list, dump_ops_info = parse_dump_data(dump_emb_op_info, op_numpy_des_dict)

    table_emb_dict = {}

    for table_name in table_list:
        table_op_info_list = dump_ops_info[table_name]
        table_op_des_list = op_numpy_des_dict[table_name]
        use_dyn_exp = op_numpy_des_dict["use_dyn_exp"]
        emb_data_dict = parse_table_des_to_data(
            table_op_info_list, table_op_des_list, use_dyn_exp
        )
        table_emb_dict[table_name] = emb_data_dict

    return table_emb_dict


def parse_dump_data(dump_emb_op_info:dict, op_numpy_des_dict:dict) -> Tuple[list, dict]:
    info_tables = sorted(dump_emb_op_info.keys())
    data_tables_list = list(op_numpy_des_dict.keys())
    data_tables_list.remove("use_dyn_exp")
    data_tables = sorted(data_tables_list)

    if info_tables != data_tables:
        raise ValueError(
            f"dump info and dump data table names not match! Your file may have been tampered.\n"
            f"Info:{info_tables}\nData:{data_tables}"
        )

    info_ops_list = []
    data_ops_list = []

    for table in info_tables:
        temp_emb_look_ops = dump_emb_op_info[table][LOOKUP_TABLE]
        dump_emb_op_info[table][LOOKUP_TABLE] = [
            ops_name.replace("/", "_") for ops_name in temp_emb_look_ops
        ]

        temp_emb_update_ops = dump_emb_op_info[table][UPDATE_TABLE]
        dump_emb_op_info[table][UPDATE_TABLE] = [
            ops_name.replace("/", "_") for ops_name in temp_emb_update_ops
        ]

        temp_emb_update_ops = dump_emb_op_info[table][UPDATE_GRAD]
        dump_emb_op_info[table][UPDATE_GRAD] = [
            ops_name.replace("/", "_") for ops_name in temp_emb_update_ops
        ]

        temp_info_ops_name = (
            dump_emb_op_info[table][LOOKUP_TABLE]
            + dump_emb_op_info[table][UPDATE_TABLE]
            + dump_emb_op_info[table][UPDATE_GRAD]
        )
        info_ops_list.extend(temp_info_ops_name)

        temp_data_ops_name = [
            op_name for op_name, _ in op_numpy_des_dict[table].items()
        ]
        data_ops_list.extend(temp_data_ops_name)

    info_ops_list = set(info_ops_list)
    data_ops_list = set(data_ops_list)

    if info_ops_list != data_ops_list:
        ops_inter = list(set(info_ops_list) & set(data_ops_list))
        # 求差集（list1 中不在 list2 中的元素）
        ops_diff1 = list(set(info_ops_list) - set(data_ops_list))
        # 求差集（list2 中不在 list1 中的元素）
        ops_diff2 = list(set(data_ops_list) - set(info_ops_list))
        logging.error(
            f"\nInfo Ops Intersection: {ops_inter}\n"
            f"\nOps in info but data: {ops_diff1}\n"
            f"\nOps in data but info: {ops_diff2}\n"
        )
        raise ValueError(
            f"dump info and dump data ops data names not match! Your file may have been tampered.\n"
        )

    return data_tables, dump_emb_op_info


def parse_table_des_to_data(table_op_info_list:list, table_op_des_dict:dict, use_dyn_exp:bool) -> dict:
    lookup_table_ops = table_op_info_list[LOOKUP_TABLE]
    update_grad_ops = table_op_info_list[UPDATE_GRAD]

    stamp_path_pair = construct_stamp_path_dict(table_op_des_dict)

    lookup_table_result = construc_numpy_data(
        lookup_table_ops, stamp_path_pair, LOOKUP_TABLE, use_dyn_exp
    )
    update_grad_result = construc_numpy_data(
        update_grad_ops, stamp_path_pair, UPDATE_GRAD, use_dyn_exp
    )

    emb_data_dict = {
        LOOKUP_TABLE: (lookup_table_result, lookup_table_result.shape),
        UPDATE_GRAD: (update_grad_result, update_grad_result.shape),
    }
    return emb_data_dict


def construc_numpy_data(ops_names:list, stamp_path_pair:dict, stamp_type:str, use_dyn_exp:bool) -> np.array:
    lookup_table_stamps = construc_stamp(ops_names, stamp_type, use_dyn_exp)
    numpy_list = []

    for lookup_table_stamp in lookup_table_stamps:
        temp_nump = np.load(stamp_path_pair[lookup_table_stamp])
        numpy_list.append(temp_nump)

    if len(numpy_list) > 1:
        numpy_data = np.concatenate(numpy_list, axis=1)
        return numpy_data
    elif len(numpy_list) == 1:
        return numpy_list[0]
    else:
        raise ValueError(f"Numpy list should not be empty for {stamp_type}")


def construc_stamp(ops_names:list, stamp_type:str, use_dyn_exp:bool) ->list:
    stamp_list = []

    if use_dyn_exp:
        construct_dict = DYN_STAMP_CONSTRUCT_DICT
    else:
        construct_dict = NODYN_STAMP_CONSTRUCT_DICT

    stamp_data_type = construct_dict[stamp_type][0]
    stamp_data_index = construct_dict[stamp_type][1]

    for op_name in ops_names:
        stamp_name = ".".join([op_name, stamp_data_type, stamp_data_index])
        stamp_list.append(stamp_name)
    print(stamp_list)
    return stamp_list


def construct_stamp_path_dict(table_op_des_dict:dict) -> dict:
    stamp_value_dict = {}
    for op_name, op_des_list in table_op_des_dict.items():
        for op_des in op_des_list:
            op_stamp = ".".join([op_name, op_des[DATA_TYPE], op_des[DATA_INDEX]])
            stamp_value_dict[op_stamp] = op_des[OP_DATA_PATH]
    return stamp_value_dict
