#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import os
import json

import numpy as np

from mx_rec.util.initialize import get_table_instance_by_name, export_table_name_set, get_sparse_dir
from mx_rec.validator.validator import FileValidator
from mx_rec.validator.validator import para_checker_decorator, ClassValidator
from mx_rec.util.log import logger


class SparseProcessor:
    single_instance = None

    def __init__(self, table_list):
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
        self.default_table_list = list(export_table_name_set())

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
        with open(data_dir, "rb") as file:
            # check whether data file is valid
            file_validator = FileValidator("data_dir", data_dir)
            # 1.check whether data_dir is soft link
            file_validator.check_not_soft_link()
            # 2.check data file size
            file_validator.check_file_size()
            file_validator.check()

        try:
            data = np.fromfile(data_dir, dtype=dtype)
        except FileNotFoundError as err:
            raise FileNotFoundError(f"data dir not found.") from err
        data = data.reshape(data_shape)
        return data

    @staticmethod
    def _get_shape_from_attrib(attribute_dir, is_json):
        if is_json:
            try:
                with open(attribute_dir, "r") as fin:
                    # check whether attribute file is valid
                    file_validator = FileValidator("attribute_dir", attribute_dir)
                    # 1.check whether attribute_dir is soft link
                    file_validator.check_not_soft_link()
                    # 2.check attribute file size
                    file_validator.check_file_size()
                    file_validator.check()
                    attributes = json.load(fin)
            except FileNotFoundError as err:
                raise FileNotFoundError("attribute dir not found.") from err
        else:
            try:
                attributes = np.fromfile(attribute_dir, np.uint64)
            except FileNotFoundError as err:
                raise FileNotFoundError("attribute dir not found.") from err

        return attributes

    def export_sparse_data(self):
        logger.info("table list to be exported is %s", self.table_list)
        sparse_dir = get_sparse_dir()
        ddr = False
        dev_dir = set_upper_dir(sparse_dir, self.device_dir_list)
        host_dir = set_upper_dir(sparse_dir, self.host_dir_list)
        for table in self.table_list:
            table_instance = get_table_instance_by_name(table)
            device_table_dir = os.path.join(dev_dir, table)
            host_table_dir = os.path.join(host_dir, table)
            if table_instance.host_vocabulary_size != 0:
                out_dir = host_table_dir
                key, offset = self._get_hashmap(host_table_dir, True)
                emb_data = self.get_embedding(device_table_dir, host_table_dir, True,
                                              table_instance.use_dynamic_expansion)
                emb_data = emb_data[offset]
            else:
                out_dir = device_table_dir
                key, _ = self._get_hashmap(device_table_dir, False)
                emb_data = self.get_embedding(device_table_dir, host_table_dir, False,
                                              table_instance.use_dynamic_expansion)
            transformed_data = dict(zip(key[:], emb_data[:]))
            save_path = os.path.join(out_dir, self.export_name + ".npy")
            np.save(save_path, transformed_data)

    def get_embedding(self, device_table_dir, host_table_dir, ddr, use_dynamic_expansion):
        emb_dir = os.path.join(device_table_dir, self.device_emb_dir)
        data_file, attribute_file = self._get_file_names(emb_dir)
        if not os.path.exists(data_file):
            raise FileExistsError(f"embedding data file {data_file} does not exist when reading.")
        if not os.path.exists(attribute_file):
            raise FileExistsError(f"attribute file {attribute_file} does not exist when reading.")

        if use_dynamic_expansion:
            device_attribute = self._get_shape_from_attrib(attribute_file, is_json=False)
            data_shape = [device_attribute[0], device_attribute[1]]
        else:
            device_attribute = self._get_shape_from_attrib(attribute_file, is_json=True)
            data_shape = device_attribute.pop(self.json_attrib_shape)
        emb_data = self._get_data(data_file, np.float32, data_shape)

        if ddr:
            emb_dir = os.path.join(host_table_dir, self.host_emb_dir)
            data_file, attribute_file = self._get_file_names(emb_dir)
            host_attribute = self._get_shape_from_attrib(attribute_file, is_json=False)
            host_data_shape = [host_attribute[0], host_attribute[1]]
            host_data = self._get_data(data_file, np.float32, host_data_shape)
            host_data = host_data[:, :data_shape[1]]
            emb_data = np.append(emb_data, host_data, axis=0)
        return emb_data

    def _get_hashmap(self, table_dir, ddr):
        if not ddr:
            hashmap_dir = os.path.join(table_dir, self.device_hashmap_dir)
        else:
            hashmap_dir = os.path.join(table_dir, self.host_hashmap_dir)
        data_file, attribute_file = self._get_file_names(hashmap_dir)
        if not os.path.exists(data_file):
            raise FileExistsError(f"hashmap data file {data_file} does not exist when reading.")
        if not os.path.exists(attribute_file):
            raise FileExistsError(f"hashmap attribute file {attribute_file} does not exist when reading.")

        shape_data = self._get_shape_from_attrib(attribute_file, is_json=False)
        if len(shape_data) < 2:
            raise ValueError(f"the attribute data from file {attribute_file} is invalid")
        data_shape = shape_data[:2]
        raw_hashmap = self._get_data(data_file, np.uint64, data_shape)
        offset = []
        if ddr:
            offset = raw_hashmap[:, -1]
        key = raw_hashmap[:, 0]
        return key, offset

    def _get_file_names(self, directory):
        files = []
        data_file = None
        attribute_file = None
        for _, _, file in os.walk(directory):
            files.append(file)
        if not files:
            raise FileExistsError(f"There is no files under the {directory} ")
        for file in files[0]:
            if file.find(self.data_suffix) != -1:
                data_file = file
            elif file.find(self.attrib_suffix) != -1:
                attribute_file = file
        data_file = os.path.join(directory, data_file)
        attribute_file = os.path.join(directory, attribute_file)
        return data_file, attribute_file


@para_checker_decorator(check_option_list=[
    ("table_list", ClassValidator, {"classes": (list, type(None))})
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
        if table not in default_table_list:
            logger.warning("%s not be created , please check your table name.", table)
        out_list.append(table)
    return out_list


def set_upper_dir(model_dir, dir_list):
    for directory in dir_list:
        model_dir = os.path.join(model_dir, directory)
    return model_dir
