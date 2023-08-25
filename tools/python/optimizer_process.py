# coding: UTF-8

#  Copyright (C)  2023. Huawei Technologies Co., Ltd. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import argparse
import os
import numpy as np
import json
from enum import Enum

# 每张卡处理自己的

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='path of the model file to be converted')
parser.add_argument('--step', type=int, required=True)

sparse_file_prefix = "sparse-model.ckpt-"
optimizer_prefix = "Optimizer"
data_suffix = ".data"
attribute_suffix = ".attribute"


class DataAttr(Enum):
    SHAPE = "shape"
    DATATYPE = "data_type"


def get_optimizer_name(sparse_file_path):
    optimizer_list = []
    for folder_name in os.listdir(sparse_file_path):
        optimizer_list.append(folder_name)
    return optimizer_list


def get_table_list(table_upper_path):
    table_list = []
    for folder_name in os.listdir(table_upper_path):
        table_list.append(folder_name+"/table")
    return table_list


def get_optimizer_param_name(table_path):
    param_list = []
    for folder_name in os.listdir(table_path):
        param_list.append(folder_name)
    return param_list


def get_optimizer_data():
    pass


def get_attribute_and_data_file(table_path):
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"the input table path {table_path} does not exists.")

    attribute_file_list = []
    data_file_list = []
    for file_name in os.listdir(table_path):
        if file_name.endswith(attribute_suffix):
            attribute_file_list.append(file_name)
        if file_name.endswith(data_suffix):
            data_file_list.append(file_name)
    if len(attribute_file_list) != 1:
        raise AssertionError(f"under the table path {table_path}, there must only one attribute file. "
                             f"In fact, {len(attribute_file_list)} attribute file exists.")
    if len(data_file_list) != 1:
        raise AssertionError(f"under the table path {table_path}, there must only one data file. "
                             f"In fact, {len(data_file_list)} data file exists.")
    attribute_file = os.path.join(table_path, attribute_file_list[0])
    data_file = os.path.join(table_path, data_file_list[0])
    return attribute_file, data_file


def process(path, step):
    save_dict = {}
    sparse_file_name = sparse_file_prefix + str(step)
    sparse_file_path = os.path.join(path, sparse_file_name,optimizer_prefix)
    optimizer_list = get_optimizer_name(sparse_file_path)
    for optimizer in optimizer_list:
        table_upper_path = os.path.join(sparse_file_path, optimizer, "HBM")
        table_list = get_table_list(table_upper_path)

        for table in table_list:
            table_path = os.path.join(table_upper_path, table)
            optimizer_param_list = get_optimizer_param_name(table_path)
            optimizer_dict = {}
            for param in optimizer_param_list:
                data_path = os.path.join(table_path, param)
                attribute_data_dir, target_data_dir = get_attribute_and_data_file(data_path)
                with open(attribute_data_dir, "r") as fin:
                    optimizer_attributes = json.load(fin)
                with open(target_data_dir, "r") as fin:
                    optimizer_data = np.fromfile(target_data_dir,
                                                 dtype=optimizer_attributes.pop(DataAttr.DATATYPE.value))
                data_shape = optimizer_attributes.pop(DataAttr.SHAPE.value)
                optimizer_data = optimizer_data.reshape(data_shape)
                optimizer_dict[param] = optimizer_data
            save_dict[table] = optimizer_dict
            np.save(path+"/optimizer_dict.npy", save_dict)


if __name__ == "__main__":
    args = parser.parse_args()
    process(args.path, args.step)