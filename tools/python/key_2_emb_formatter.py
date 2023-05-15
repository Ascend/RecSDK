# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
# Description:
# Author: MindX SDK
# Create: 2023-01-29

import argparse
import json
import logging
import os
import re
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='path of the root dir of saved file')
parser.add_argument('--name', type=str, default="key_2_embedding", help='name of output file')
parser.add_argument('--ddr', type=bool, default=False, help='if saved data was from ddr mode, default False')
parser.add_argument('--step', type=int, default=0, help='the step when the data was saved, default 0')


def get_verified_path(path):
    real_path = os.path.realpath(path)
    if os.path.exists(real_path):
        return real_path
    else:
        raise NotADirectoryError(f"{path} is not a valid directory")


def get_valid_file_name(name):
    invalid_symbols = r"[\/\\\:\*\?\"\<\>\|]"
    valid_name = re.sub(invalid_symbols, "_", name)
    return valid_name


class Formatter:
    def __init__(self, saved_file_path, out_file_name, is_ddr_mode, step):
        self._device_dir_list = ["HashTable", "HBM"]
        self._host_dir_list = ["HashTable", "DDR"]
        self._device_emb_dir = "embedding"
        self._host_emb_dir = "embedding_data"
        self._device_hashmap_dir = "key_offset_map"
        self._host_hashmap_dir = "embedding_hashmap"
        self._attrib_suffix = ".attribute"
        self._data_suffix = ".data"
        self._out_file_suffix = ".npy"

        self._saved_file_path = get_verified_path(saved_file_path)
        self._out_file_name = get_valid_file_name(out_file_name)
        self._sub_dirs = self._get_sub_dirs(step)
        self._table_names = None

        self._json_attrib_dtype = "data_type"
        self._json_attrib_shape = "shape"
        self._host_attrib_dtype = np.uint64
        self._hashmap_dtype = np.uint32
        self._raw_key_dtype = np.uint64
        self._key_dtype = np.int64
        self._raw_key_offset = np.iinfo(np.uint32).max
        self._data_dtype = None

        self._is_ddr_mode = is_ddr_mode

    def process(self):
        dev_dir = self._set_upper_dir(self._sub_dirs[0], self._device_dir_list)
        self._table_names = self._get_table_names(dev_dir)

        transformed_data = []
        for table_name in self._table_names:
            combined_key = None
            combined_emb = None
            for sub_dir in self._sub_dirs:
                dev_dir = self._set_upper_dir(sub_dir, self._device_dir_list)
                host_dir = self._set_upper_dir(sub_dir, self._host_dir_list)
                emb_data = self._data_process(dev_dir, host_dir, table_name)
                key, offset = self._hashmap_process(dev_dir, host_dir, table_name)
                emb_data = emb_data[offset]

                if combined_key is not None:
                    combined_key = np.append(combined_key, key, axis=0)
                else:
                    combined_key = key
                if combined_emb is not None:
                    combined_emb = np.append(combined_emb, emb_data, axis=0)
                else:
                    combined_emb = emb_data

                logging.debug(f"{table_name} has combined key {combined_key.shape}"
                              f" and combined emb {combined_emb.shape}")

            transformed_data.append(table_name)
            transformed_data.append(combined_key)
            transformed_data.append(combined_emb)

        np.save("./" + self._out_file_name + self._out_file_suffix, transformed_data)

    def _data_process(self, dev_dir, host_dir, table_name):
        dev_emb_dir = os.path.join(dev_dir, table_name, self._device_emb_dir)
        host_emb_dir = os.path.join(host_dir, table_name, self._host_emb_dir)

        data_file, attribute_file = self._get_file_names(dev_emb_dir)
        dev_attribute = self._get_attribute(dev_emb_dir, attribute_file, is_json=True)
        if not self._data_dtype:
            self._data_dtype = dev_attribute.pop(self._json_attrib_dtype)

        dev_data_shape = dev_attribute.pop(self._json_attrib_shape)
        emb_data = self._get_data(dev_emb_dir, data_file, self._data_dtype, dev_data_shape)

        if self._is_ddr_mode:
            data_file, attribute_file = self._get_file_names(host_emb_dir)
            host_attribute = self._get_attribute(host_emb_dir, attribute_file, is_json=False)
            host_data_shape = [host_attribute[0], host_attribute[1]]
            host_data = self._get_data(host_emb_dir, data_file, self._data_dtype, host_data_shape)
            host_data = host_data[:, :dev_data_shape[1]]
            emb_data = np.append(emb_data, host_data, axis=0)

        return emb_data

    def _hashmap_process(self, dev_dir, host_dir, table_name):
        dev_hashmap_dir = os.path.join(dev_dir, table_name, self._device_hashmap_dir)
        host_hashmap_dir = os.path.join(host_dir, table_name, self._host_hashmap_dir)

        if self._is_ddr_mode:
            data_file, attribute_file = self._get_file_names(host_hashmap_dir)
        else:
            data_file, attribute_file = self._get_file_names(dev_hashmap_dir)

        attribute = self._get_attribute(host_hashmap_dir, attribute_file, is_json=False)
        data_shape = attribute[:2]
        raw_hashmap = self._get_data(host_hashmap_dir, data_file, self._hashmap_dtype, data_shape)
        offset = raw_hashmap[:, -1]
        raw_key = raw_hashmap[:, :2].astype(self._raw_key_dtype)
        key = raw_key[:, 0] * self._raw_key_offset + raw_key[:, 1]
        key = key.astype(self._key_dtype)

        return key, offset

    def _get_sub_dirs(self, step):
        sub_dirs = []
        for _, sub_dir, _ in os.walk(self._saved_file_path):
            sub_dirs.append(sub_dir)

        if not sub_dirs or not sub_dirs[0]:
            raise FileNotFoundError(f"There is no sparse checkpoint for given root directory.")

        picked_sub_dirs = []
        for sub_dir in sub_dirs[0]:
            if int(sub_dir.split("-")[-1]) == step:
                picked_sub_dirs.append(sub_dir)

        if not picked_sub_dirs:
            raise FileNotFoundError(f"There is no sparse checkpoint for given training step {step}.")
        return picked_sub_dirs

    def _set_upper_dir(self, sub_dir, dir_list):
        temp_dir = os.path.join(self._saved_file_path, sub_dir)
        for directory in dir_list:
            temp_dir = os.path.join(temp_dir, directory)
        return temp_dir

    def _get_table_names(self, directory):
        if os.path.exists(directory):
            table_names = []
            for _, table_name, _ in os.walk(directory):
                table_names.append(table_name)
            return table_names[0]
        else:
            raise ValueError("given directory does not contain required subdirectories, cannot search for table names")

    def _get_file_names(self, directory):
        files = []
        data_file = None
        attribute_file = None
        for _, _, file in os.walk(directory):
            files.append(file)
        for file in files[0]:
            if file.find(self._data_suffix) != -1:
                data_file = file
            elif file.find(self._attrib_suffix) != -1:
                attribute_file = file
        return data_file, attribute_file

    def _get_attribute(self, directory, file_name, is_json):
        file_dir = os.path.join(directory, file_name)
        if is_json:
            with open(file_dir, "r") as fin:
                attributes = json.load(fin)
                return attributes
        else:
            attributes = np.fromfile(file_dir, self._host_attrib_dtype)
            return attributes

    def _get_data(self, directory, file_name, dtype, shape):
        file_dir = os.path.join(directory, file_name)
        data = np.fromfile(file_dir, dtype=dtype)
        data = data.reshape(shape)
        return data


if __name__ == "__main__":
    args = parser.parse_args()
    formatter = Formatter(saved_file_path=args.path, out_file_name=args.name, is_ddr_mode=args.ddr, step=args.step)
    formatter.process()