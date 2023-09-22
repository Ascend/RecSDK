import argparse
import json
import os
import re
from enum import Enum

import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True, help='path of the model file to be converted')
parser.add_argument('--output_path', type=str, required=True, help='output path must be local path')
parser.add_argument('--rank_size', type=int, choices=range(1, 17), default=8, required=False)
parser.add_argument('--estimator', type=int, choices=[0, 1], default=1, required=False)
parser.add_argument('--ddr', type=int, choices=[0, 1], default=0, required=False)
parser.add_argument('--save_easy', type=int, choices=[0, 1], default=1, required=False)

slice_prefix = "slice_"
sparse_file_prefix = "sparse-"
data_suffix = ".data"
attribute_suffix = ".attribute"
hbm_prefix_list = ["HashTable", "HBM"]
ddr_prefix_list = ["HashTable", "DDR"]
min_file_size = 1
max_file_size = 1024 * 1024 * 1024 * 1024


class DataAttr(Enum):
    SHAPE = "shape"
    DARATYPE = "data_type"


class ModelConverter:
    def __init__(self, input_model_path, output_model_path, rank_size, estimator, ddr, save_easy):
        self._input_path = input_model_path
        self._output_path = output_model_path
        self._rank_size = rank_size
        self._is_estimator = bool(estimator)
        self._is_ddr = bool(ddr)
        self._is_save_easy = bool(save_easy)
        self._load_ckpt_path = None
        self._input_model_path_list = []
        self._table_list = []
        self.table_info_dict = {}
        self.sparse_file_list = []

        if not tf.io.gfile.exists(self._input_path):
            raise FileNotFoundError(f"the input path {self._input_path} does not exists. please check it.")
        if not tf.io.gfile.exists(self._output_path):
            tf.io.gfile.makedirs(self._output_path)
        self._build_input_model_list(self._is_estimator)
        self._build_sparse_file_list()
        self._build_table_info_dict()

    def convert(self):
        for table_name, emb_size in self.table_info_dict.items():
            result_key = np.array([])
            result_embedding = np.array([])
            for rank in range(self._rank_size):
                if not self._is_save_easy:
                    offset, key = self._get_key_and_offset(self.sparse_file_list[rank], table_name)
                    emb_data = self._get_embedding_array(self.sparse_file_list[rank], table_name)[list(offset)]
                else:
                    key = self._get_key_array(self.sparse_file_list[rank], table_name)
                    emb_data = self._get_embedding_array(self.sparse_file_list[rank], table_name)

                if rank == 0:
                    result_key = key
                    result_embedding = emb_data
                else:
                    result_key = np.concatenate((result_key, key), axis=0)
                    result_embedding = np.concatenate((result_embedding, emb_data), axis=0)

            # save result every table
            transformed_data = dict(zip(result_key[:], result_embedding[:]))
            save_path = os.path.join(self._output_path, table_name + "_key_embedding" + ".npy")
            save_dir = os.path.dirname(save_path)
            os.makedis(save_dir, exist_ok=True)
            np.save(save_path, transformed_data)

    def _get_key_and_offset(self, sparse_file_path, table_name):
        if self._is_ddr:
            upper_dir = generate_upper_dir(sparse_file_path, ddr_prefix_list, table_name, "embedding_hashmap")
        else:
            upper_dir = generate_upper_dir(sparse_file_path, hbm_prefix_list, table_name, "key_offset_map")
        attribute_data_dir, target_data_dir = get_attribute_and_data_file(upper_dir)

        with open(attribute_data_dir, "r") as fin:
            attributes = np.fromfile(attribute_data_dir, dtype=np.uint64)
        data_shape = attributes[:2]

        with open(target_data_dir, "r") as fin:
            key_offset_data = np.fromfile(target_data_dir, dtype=np.int64)
        key_offset_data = key_offset_data.reshape(data_shape)
        offset = key_offset_data[:, 1]
        key = key_offset_data[:, 0]
        return offset, key

    def _get_key_array(self, sparse_file_path, table_name):
        upper_dir = generate_upper_dir(sparse_file_path, hbm_prefix_list, table_name, "key")
        attribute_data_dir, target_data_dir = get_attribute_and_data_file(upper_dir)
        with tf.io.gfile.GFile(attribute_data_dir, "r") as fin:
            emb_attributes = json.load(fin)
        with tf.io.gfile.GFile(target_data_dir, "rb") as fin:
            key_data = fin.read()
            key_data = np.fromstring(key_data, dtype=emb_attributes.pop(DataAttr.DARATYPE.value))

        data_shape = emb_attributes.pop(DataAttr.SHAPE.value)
        key = key_data.reshape(data_shape)
        return key

    def _get_embedding_array(self, sparse_file_path, table_name):
        upper_dir = generate_upper_dir(sparse_file_path, hbm_prefix_list, table_name, "embedding")
        attribute_data_dir, target_data_dir = get_attribute_and_data_file(upper_dir)
        with tf.io.gfile.GFile(attribute_data_dir, "r") as fin:
            emb_attributes = json.load(fin)

        with tf.io.gfile.GFile(target_data_dir, "rb") as fin:
            emb_data = fin.read()
            emb_data = np.fromstring(emb_data, dtype=emb_attributes.pop(DataAttr.DARATYPE.value))
        data_shape = emb_attributes.pop(DataAttr.SHAPE.value)
        emb_data = emb_data.reshape(data_shape)

        if self._is_ddr:
            ddr_upper_dir = generate_upper_dir(sparse_file_path, ddr_prefix_list, table_name, "embedding_data")
            attribute_data_dir, target_data_dir = get_attribute_and_data_file(ddr_upper_dir)
            with open(attribute_data_dir, "r") as fin:
                attributes = np.fromfile(attribute_data_dir, dtype=np.uint64)
                data_shape = attributes[:2]
            with open(target_data_dir, "r") as fin:
                ddr_emb_data = np.fromfile(target_data_dir, dtype=np.float32)
            ddr_emb_data = ddr_emb_data.reshape(data_shape)
            emb_data = np.concatenate((emb_data, ddr_emb_data[:, :self.table_info_dict[table_name]]), axis=0)
        return emb_data

    def _build_sparse_file_list(self):
        if self._is_estimator:
            latest_ckpt = self._get_latest_ckpt_name()
            sparse_file_name = sparse_file_prefix + latest_ckpt
            for rank in range(self._rank_size):
                sparse_file_path = os.path.join(self._input_model_path_list[rank], sparse_file_name)
                self.sparse_file_list.append(sparse_file_path)
        else:
            pattern = re.compile(r"sparse-.+")
            for folder_name in tf.io.gfile.listdir(self._input_path):
                if tf.io.gfile.isdir(os.path.join(self._input_path, folder_name)) and pattern.match(folder_name):
                    sparse_file_path = os.path.join(self._input_path, folder_name)
                    self.sparse_file_list.append(sparse_file_path)
            if len(self.sparse_file_list) != self._rank_size:
                raise AssertionError(
                    f"the sparse file num should be {self._rank_size} rather than {len(self.sparse_file_list)}")

    def _build_input_model_list(self, is_estimator):
        if is_estimator:
            for i in range(self._rank_size):
                # for mt, need two rank id
                model_path = os.path.join(self._input_path, str(i))
                self._input_model_path_list.append(model_path)
        else:
            self._input_model_path_list.append(self._input_path)
        self._load_ckpt_path = self._input_model_path_list[0]

    def _get_latest_ckpt_name(self):
        ckpt_path = os.path.join(self._load_ckpt_path, "checkpoint")
        if not tf.io.gfile.exists(ckpt_path):
            raise FileNotFoundError(f"the input path you provided {ckpt_path} miss checkpoint file.please check it.")
        with tf.io.gfile.GFile(ckpt_path, "r") as fin:
            # validate open file
            latest_ckpt = fin.readline().rstrip()
            latest_ckpt = latest_ckpt.split(":")[1].strip(' ').replace('"', '')
            latest_ckpt = latest_ckpt.split("/")[-1]
        return latest_ckpt

    def _build_table_info_dict(self):
        tmp_file_list = []
        table_upper_file = os.path.join(self.sparse_file_list[0], "HashTable", "HBM")
        if not tf.io.gfile.exists(table_upper_file):
            raise FileNotFoundError(f"the sparse file path {table_upper_file} does not exists.")
        for _, table_name, _ in tf.io.gfile.walk(table_upper_file):
            tmp_file_list.append(table_name)


        if not tmp_file_list:
            raise FileNotFoundError(f"under the sparse file path {table_upper_file}, no file exists.")
        self._table_list = tmp_file_list[0]
        for table_name in self._table_list:
            table_name = f"{table_name}/table"
            table_path = os.path.join(table_upper_file, table_name, "embedding")
            attribute_file = get_attribute_and_data_file(table_path)[0]
            with tf.io.gfile.GFile(attribute_file, "r") as fin:
                emb_attributes = json.load(fin)
                data_shape = emb_attributes.pop(DataAttr.SHAPE.value)
                self.table_info_dict[table_name] = data_shape[1]


def get_attribute_and_data_file(table_path):
    if not tf.io.gfile.exists(table_path):
        raise FileNotFoundError(f"the input table path {table_path} does not exists.")

    attribute_file_list = []
    data_file_list = []
    for file_name in tf.io.gfile.listdir(table_path):
        if file_name.endswith(attribute_suffix):
            attribute_file_list.append(file_name)
        if file_name.endswith(data_suffix):
            data_file_list.append(file_name)
    if len(attribute_file_list) != 1:
        raise AssertionError(f"under the table path {table_path}, ther must only one attribute file. "
                             f"In fact, {len(attribute_file_list)} attribute file exists. ")
    if len(data_file_list) != 1:
        raise AssertionError(f"under the table path {table_path}, ther must only one data file. "
                             f"In fact, {len(data_file_list)} data file exists. ")
    attribute_file = os.path.join(table_path, attribute_file_list[0])
    data_file = os.path.join(table_path, data_file_list[0])
    return attribute_file, data_file


def generate_upper_dir(sparse_file, dir_prefix_list, table_name, data_type):
    temp_dir = sparse_file
    for dir in dir_prefix_list:
        temp_dir = os.path.join(temp_dir, dir)
    return os.path.join(temp_dir, table_name, data_type)


def generate_attribute_dir(sparse_file, dir_prefix_list, table_name, data_type, rank_id):
    temp_dir = sparse_file
    for dir in dir_prefix_list:
        temp_dir = os.path.join(temp_dir, dir)
    return os.path.join(temp_dir, table_name, data_type, f"{slice_prefix}{rank_id}{attribute_suffix}")


def generate_data_dir(sparse_file, dir_prefix_list, table_name, data_type, rank_id):
    temp_dir = sparse_file
    for dir in dir_prefix_list:
        temp_dir = os.path.join(temp_dir, dir)
    return os.path.join(temp_dir, table_name, data_type, f"{slice_prefix}{rank_id}{data_suffix}")


if __name__ == "__main__":
    args = parser.parse_args()
    convert_instance = ModelConverter(input_model_path=args.input_path, output_model_path=args.output_path,
                                      rank_size=args.rank_size,
                                      estimator=args.estimator, ddr=args.ddr, save_easy=args.save_easy)
    convert_instance.convert()
    print(f"sparse table has been converted to numpy file. output path is {args.output_path}")

