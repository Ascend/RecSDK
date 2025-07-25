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
import subprocess
import os
import re
from typing import Tuple

import numpy as np

from dump_info import DumpInfo
from utils import validate_path, nested_dict_to_str, sc_get_key_value


STAMP_TYPE_INDEX = 0
STAMP_INDEX_INDEX = 1

DUMP_OP_STR = "04dump_op"
USE_DYN_EXP_STR = "use_dyn_exp"
DUMP_OP_NUMPY_REGEX_STR = r"^.+\.npy$"
OP_NUMPY_ATOL = 1e-10

DUMP_NP_LEN = 8

INSTRUCT_PYTHON = "python"
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
    """
    This class is used to represent parsed ops data for choosen step and rank.
    """

    def __init__(
        self, data_dir: str, dump_info: DumpInfo, data_step: int, rank_id: int
    ):
        self.dump_data_path = find_match_op_dump_data(data_dir, rank_id, data_step)
        self.dump_info = dump_info
        validate_path(ASCEND_TOOLKIT_MSACCUCMP_PATH, "msaccucmp.py")
        self.output_numpy_path = exe_msaccucmp_convert(self.dump_data_path)
        self.op_data_dict, self.use_dyn_exp = self.parse_numpy_data()

    def __eq__(self, other) -> bool:
        test_op_data_dict = self.op_data_dict
        golden_op_data_dict = other.op_data_dict

        test_op_data_keys = test_op_data_dict.keys()
        golden_op_data_keys = golden_op_data_dict.keys()
        test_op_list = sorted(list(test_op_data_keys))
        golden_op_list = sorted(list(golden_op_data_keys))

        if test_op_list != golden_op_list:
            logging.error(
                "[OpData]Test data and Golden data should have the same names, but Test:%s Golden:%s are given.",
                test_op_list,
                golden_op_list,
            )
            return False

        for table_name in golden_op_list:
            test_table_data = sc_get_key_value(test_op_data_dict, table_name)
            golden_table_data = sc_get_key_value(golden_op_data_dict, table_name)
            for emb_type in golden_table_data:
                test_emb_data = test_table_data[emb_type][0]
                golden_emb_data = golden_table_data[emb_type][0]

                logging.debug(
                    "[OpData][%s][%s] Data are shown as below.\nTest: %s %s\nGolden: %s %s\nTest: %s\nGolden: %s\n",
                    table_name,
                    emb_type,
                    test_emb_data.dtype,
                    test_emb_data.shape,
                    golden_emb_data.dtype,
                    golden_emb_data.shape,
                    test_emb_data,
                    golden_emb_data,
                )

                if test_emb_data.dtype != golden_emb_data.dtype:
                    logging.error(
                        "[OpData][%s][%s] Test and Golden shape not equal.\nTest:%s\nGolden:%s\n",
                        table_name,
                        emb_type,
                        test_emb_data.dtype,
                        golden_emb_data.dtype,
                    )
                    return False

                if test_emb_data.shape != golden_emb_data.shape:
                    logging.error(
                        "[OpData][%s][%s] Test and Golden shape not equal.\nTest:%s\nGolden:%s\n",
                        table_name,
                        emb_type,
                        test_emb_data.shape,
                        golden_emb_data.shape,
                    )
                    return False

                if not np.allclose(test_emb_data, golden_emb_data, OP_NUMPY_ATOL):
                    logging.error(
                        "[OpData][%s][%s] Test and Golden value not equal.\nTest:%s\nGolden:%s\n",
                        table_name,
                        emb_type,
                        test_emb_data,
                        golden_emb_data,
                    )
                    return False
        return True

    def parse_numpy_data(self):
        """
        Parse numpy op data to desc and then parse them to numpy data.
        """
        for dirpath, dirs, files in os.walk(self.output_numpy_path):
            if dirs:
                logging.warning(
                    "Dump op numpy path should not contain any directory, your file may have been tampered: %s",
                    self.output_numpy_path,
                )
            files = sorted(files)

            logging.debug(
                "parsing numpy op data to desc start......\nNumpy dir:%s  Numpy files: %s.",
                dirpath,
                files,
            )
            op_numpy_des_dict, use_dyn_flag = parse_op_numpy_to_desc(dirpath, files)
            logging.debug("Parsing numpy op data to desc succeed.")

            op_numpy_data_dict = parse_op_desc_to_data(
                self.dump_info.dump_emb_op_info, op_numpy_des_dict, use_dyn_flag
            )
        return op_numpy_data_dict, use_dyn_flag


def parse_op_numpy_to_desc(dir_path: str, file_names: list) -> dict:
    """
    Parse numpy op data to desc(which is a dict to describe op).
    """
    op_desc_dict = {}
    for filename in file_names:
        cur_op_data_path = os.path.join(dir_path, filename)
        validate_path(cur_op_data_path, "op_dump_numpy", DUMP_OP_NUMPY_REGEX_STR)

        file_name_split = filename.split(".")
        if len(file_name_split) != DUMP_NP_LEN:
            raise ValueError(
                f"Dump op numpy may have been tamperd or msaccucmp updated. Path:{cur_op_data_path}"
            )

        # File_name_split example ['GatherV2', 'item_table__item_table_lookup_gather_for_id_offsets',
        #  '58', '29', '1722569783477282', 'input', '0', 'npy']
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
            "Parsing op name to op desc succeed. Cur_op_name:%s Op_desc: %s.",
            cur_op_name,
            op_desc,
        )

    op_type_set = set([op_desc[0][OP_TYPE] for _, op_desc in op_desc_dict.items()])

    if op_type_set == DYN_EXP_OP_LIST:
        use_dyn_exp = True
    elif op_type_set == NO_DYN_OP_LIST:
        use_dyn_exp = False
    else:
        raise ValueError(
            f"Unexpected op_type_set: {op_type_set}, your file may have been tampered"
        )

    op_numpy_desc_dict = div_op_desc_by_table(use_dyn_exp, op_desc_dict)
    return op_numpy_desc_dict, use_dyn_exp


def find_match_op_dump_data(dump_data_path: str, rank_id: int, step: int) -> str:
    """
    Find op path for given rank and step.
    """
    # target path example /xxxx/20240724_141123/03dump_op/20240724141134/0/ge_default_20240724141135_31/4/0
    pattern_str = (
        f"^{re.escape(dump_data_path)}/"  # match precision check data path:/xxxx/20240724_141123
        r"04dump_op/"  # match 04dump_op
        r"\d{14}/"  # match date: 20240724141134
        f"{re.escape(str(rank_id))}/"  # match rankid: 0
        r"ge_default_\d{14}_\d+/"  # match ge_default_{data}_{time}
        r"\d+/"  # match model id: 4
        f"{re.escape(str(step-1))}"  # match step: 0
    )

    pattern = re.compile(pattern_str)
    for root, dirs, _ in os.walk(dump_data_path):
        for cur_dir in dirs:
            op_path = os.path.join(root, cur_dir)
            if pattern.match(op_path):
                logging.debug("Find matched Dump op path: %s", op_path)
                return op_path

    raise ValueError(
        f"No matched Dump op path found for rank_id {rank_id} and step {step}"
    )


def exe_msaccucmp_convert(op_data_path: str) -> str:
    """
    Execute msaccucmp convert to convert op dump to numpy.
    """
    output_numpy_path = os.path.join(op_data_path, "dump_op_np")
    logging.info("convert target path: %s", output_numpy_path)

    if os.path.exists(output_numpy_path):
        logging.warning(
            "Dump op data have already been parsed and Convertion stop!!! "
            "This may cause some mistakes, please check the path: %s",
            output_numpy_path,
        )
        return output_numpy_path

    os.makedirs(output_numpy_path, mode=0o750)
    logging.debug("Dump op parse output dir created, path: %s", output_numpy_path)

    instruct_item_command = [
        INSTRUCT_PYTHON,
        ASCEND_TOOLKIT_MSACCUCMP_PATH,
        INSTRUCT_CONVERT,
        INSTRUCT_D,
        op_data_path,
        INSTRUCT_OUT,
        output_numpy_path,
    ]
    logging.debug("msaccucmp convert exec instruction: %s", instruct_item_command)

    convert_result = subprocess.run(
        instruct_item_command, capture_output=True, text=True, shell=False
    )

    # check convertion result
    convert_returncode = convert_result.returncode
    if convert_returncode != 0:
        raise ValueError(
            f"Msaccucmp convert dump op to numpy Failed!\n Command: {instruct_item_command}"
        )

    logging.info("Msaccucmp convert dump op to numpy succeed.")
    return output_numpy_path


def div_op_desc_by_table(use_dyn_exp: bool, op_desc_dict: dict) -> dict:
    """
    Diverse op desc to its belonging table.
    """
    table_op_desc_dict = {USE_DYN_EXP_STR: use_dyn_exp}
    table_name_set = set()

    if use_dyn_exp:
        arch_op_type = DYN_ARCH_OP_TYPE
    else:
        arch_op_type = NODYN_ARCH_OP_TYPE

    op_name_list = sorted(list(op_desc_dict.keys()))
    logging.debug(
        "Div op desc by table start......\nUse_dyn_exp:%s Op_name_list:%s",
        use_dyn_exp,
        op_name_list,
    )

    for op_name in op_name_list:
        op_desc_list = op_desc_dict[op_name]
        if op_desc_list[0][OP_TYPE] == arch_op_type:
            # only support LazyAdam for now
            if not op_name.startswith("LazyAdam"):
                # op name example item_table__item_table_lookup_gather_for_id_offsets
                table_name = op_name.split("__")[0]
            else:
                # op name example
                # LazyAdamByAddress_0_update_item_table__item_table_
                # lookup_id_offsets_item_table_GetNext_EmbeddingLookupByAddress
                op_name_str = op_name.split("_")
                table_name = "_".join([op_name_str[3], op_name_str[4]])

            table_name_set.add(table_name)
            table_op_desc_dict[table_name] = {}

    logging.debug("Parsing table names succeed.Table list: \n%s", table_name_set)

    for table_name in table_name_set:
        for op_name in op_name_list:
            if table_name in op_name:
                table_op_desc_dict[table_name][op_name] = op_desc_dict[op_name]
    logging.debug(
        "Div op desc by table succeed.\nTable_op_desc_dict:%s",
        nested_dict_to_str(table_op_desc_dict),
    )
    return table_op_desc_dict


def parse_op_desc_to_data(
    dump_emb_op_info: DumpInfo, op_numpy_des_dict: dict, use_dyn_flag: bool
) -> dict:
    """
    Parse op desc to numpy arrary.
    Input data dict example is shown as below.

    dump_emb_op_info
    {"user_table":
    {"emb_look_ops":
    ["user_table//user_table_lookup/gather_for_id_offsets", "LazyAdam_0/update_user_table/GatherV2",
    "LazyAdam_0/update_user_table/GatherV2_1"],
    "emb_update_ops":
    ["LazyAdam_0/update_user_table/ScatterNdAdd", "LazyAdam_0/update_user_table/ScatterNdAdd_1",
    "LazyAdam_0/update_user_table/ScatterNdAdd_2"]},

    "item_table":
    {"emb_look_ops":
    ["item_table//item_table_lookup/gather_for_id_offsets", "LazyAdam_0/update_item_table/GatherV2",
    "LazyAdam_0/update_item_table/GatherV2_1"],
    "emb_update_ops":
    ["LazyAdam_0/update_item_table/ScatterNdAdd", "LazyAdam_0/update_item_table/ScatterNdAdd_1",
    "LazyAdam_0/update_item_table/ScatterNdAdd_2"]}}

    op_numpy_des_dict
    {USE_DYN_EXP_STR:bool
     "user_table":{op_name:[OP_FILE_DESC]},
     "item_table":{op_name:[OP_FILE_DESC]}}
    """
    table_list, dump_ops_info = parse_dump_data(dump_emb_op_info, op_numpy_des_dict)

    table_emb_dict = {}

    for table_name in table_list:
        table_op_info_list = dump_ops_info[table_name]
        table_op_des_list = op_numpy_des_dict[table_name]
        use_dyn_exp = op_numpy_des_dict[USE_DYN_EXP_STR]
        emb_data_dict = parse_table_des_to_data(
            table_op_info_list, table_op_des_list, use_dyn_exp
        )
        table_emb_dict[table_name] = emb_data_dict

    return table_emb_dict


def parse_dump_data(
    dump_emb_op_info: dict, op_numpy_des_dict: dict
) -> Tuple[list, dict]:
    """
    Parse dump info and reconstruct its op name to suit dump op data.
    """
    info_tables = sorted(dump_emb_op_info.keys())
    data_tables_list = list(op_numpy_des_dict.keys())
    data_tables_list.remove(USE_DYN_EXP_STR)
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
            ops_name.replace("/", "_")
            for ops_name in temp_emb_look_ops
        ]

        temp_emb_update_ops = dump_emb_op_info[table][UPDATE_GRAD]
        dump_emb_op_info[table][UPDATE_GRAD] = [
            ops_name.replace("/", "_")
            for ops_name in temp_emb_update_ops
        ]

        temp_info_ops_name = (
            dump_emb_op_info[table][LOOKUP_TABLE] + dump_emb_op_info[table][UPDATE_GRAD]
        )
        info_ops_list.extend(temp_info_ops_name)

        temp_data_ops_name = [
            op_name
            for op_name, _ in op_numpy_des_dict[table].items()
        ]
        data_ops_list.extend(temp_data_ops_name)

    info_ops_list_set = set(info_ops_list)
    data_ops_list_set = set(data_ops_list)

    if info_ops_list_set != data_ops_list_set:
        ops_inter = list(info_ops_list_set & data_ops_list_set)

        # The difference set (the elements in info_ops_list that are not in data_ops_list).
        ops_diff1 = list(info_ops_list_set - data_ops_list_set)

        # The difference set (the elements in data_ops_list that are not in info_ops_list).
        ops_diff2 = list(info_ops_list_set - data_ops_list_set)
        logging.error(
            "Info Ops Intersection: %s\nOps in info but data: %s\nOps in data but info: %s\n",
            ops_inter,
            ops_diff1,
            ops_diff2,
        )
        raise ValueError(
            f"dump info and dump data ops data names not match! Your file may have been tampered.\n"
        )

    return data_tables, dump_emb_op_info


def parse_table_des_to_data(
    table_op_info_list: list, table_op_des_dict: dict, use_dyn_exp: bool
) -> dict:
    """
    Parse desc as one table's lookup result and emb update grade.
    """
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


def construc_numpy_data(
    ops_names: list, stamp_path_pair: dict, stamp_type: str, use_dyn_exp: bool
) -> np.array:
    """
    Parse osp data as lookup result os emb update grade.
    """
    lookup_table_stamps = construct_stamp(ops_names, stamp_type, use_dyn_exp)
    numpy_list = []

    for lookup_table_stamp in lookup_table_stamps:
        temp_nump = np.load(stamp_path_pair[lookup_table_stamp])
        numpy_list.append(temp_nump)

    if not numpy_list:
        raise ValueError(f"Numpy list should not be empty for {stamp_type}")

    if len(numpy_list) == 1:
        return numpy_list[0]

    numpy_data = np.concatenate(numpy_list, axis=1)
    return numpy_data


def construct_stamp(ops_names: list, stamp_type: str, use_dyn_exp: bool) -> list:
    """
    Parse ops stamp using its name.
    """
    stamp_list = []

    if use_dyn_exp:
        construct_dict = DYN_STAMP_CONSTRUCT_DICT
    else:
        construct_dict = NODYN_STAMP_CONSTRUCT_DICT

    construct_list = sc_get_key_value(construct_dict, stamp_type)
    stamp_data_type = construct_list[STAMP_TYPE_INDEX]
    stamp_data_index = construct_list[STAMP_INDEX_INDEX]

    for op_name in ops_names:
        stamp_name = ".".join([op_name, stamp_data_type, stamp_data_index])
        stamp_list.append(stamp_name)
    return stamp_list


def construct_stamp_path_dict(table_op_des_dict: dict) -> dict:
    """
    Parse ops and construct stamp, op file path dict.
    """
    stamp_value_dict = {}
    for op_name, op_des_list in table_op_des_dict.items():
        for op_des in op_des_list:
            op_stamp = ".".join([op_name, op_des[DATA_TYPE], op_des[DATA_INDEX]])
            stamp_value_dict[op_stamp] = op_des[OP_DATA_PATH]
    return stamp_value_dict
