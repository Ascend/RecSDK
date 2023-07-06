#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import json
import os
import logging
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.util import compat

from mx_rec.constants.constants import DataName, DataAttr
from mx_rec.util.initialize import get_rank_id, get_rank_size, get_customized_ops, get_table_instance, \
    get_table_instance_by_name, is_asc_manager_initialized, save_host_data, restore_host_data, get_host_data, \
    send_host_data, get_ascend_global_hashtable_collection
from mx_rec.util.perf import performance


class Saver(object):
    customized_ops = get_customized_ops()

    def __init__(self, var_list=None, max_to_keep=3, prefix_name="checkpoint"):
        self.max_to_keep = max_to_keep
        self._prefix_name = prefix_name
        self.var_list = var_list
        self.rank_id = get_rank_id()
        self.local_rank_id = self.rank_id % 8
        self.rank_size = get_rank_size()
        self.local_rank_size = min(self.rank_size, 8)
        self.save_op_dict = defaultdict(dict)
        self.restore_fetch_list = []
        self.placeholder_dict = defaultdict(dict)
        # save_easy_mode : only save the embedding and key data of sparse tables
        self.save_easy_mode = os.getenv("SAVE_EASY", 0)
        self.build()
        # since tf 2.6.0, tf needs tensorflow_io to support hdfs path
        if tf.__version__.startswith("2"):
            import tensorflow_io as tfio

    def build(self):
        if self.var_list is None:
            self.var_list = []
            logging.debug(f"optimizer collection name: {get_ascend_global_hashtable_collection()}")
            temp_var_list = tf.compat.v1.get_collection(get_ascend_global_hashtable_collection())
            for var in temp_var_list:
                table_instance = get_table_instance(var)
                if table_instance.is_save:
                    self.var_list.append(var)

        with tf.compat.v1.variable_scope("mx_rec_save"):
            self._build_save()
        with tf.compat.v1.variable_scope("mx_rec_restore"):
            self._build_restore()

        logging.debug("Save & Restore graph was built.")

    @performance("Save")
    def save(self, sess, save_path="model", global_step=None):
        """
        Save sparse tables
        :param sess: A Session to use to save the sparse table variables
        :param save_path: Only absolute path supported
        :param global_step: If provided the global step number is appended to save_path to create
         the checkpoint filenames. The optional argument can be a Tensor, a Tensor name or an integer.
        :return: None
        """
        logging.debug(f"======== Start saving for rank id {self.rank_id} ========")
        save_path = save_path if save_path else self._prefix_name
        directory, base_name = os.path.split(save_path)

        if global_step:
            if not isinstance(global_step, compat.integral_types):
                global_step = int(sess.run(global_step))
            ckpt_name = "sparse-%s-%d" % (base_name, global_step)
        else:
            ckpt_name = "sparse-%s" % base_name

        integrated_path = os.path.join(directory, ckpt_name)
        saving_path = integrated_path

        if tf.io.gfile.exists(saving_path):
            tf.io.gfile.rmtree(saving_path)
            logging.debug(f"rank id {self.rank_id} | Saving_path '{saving_path}' has been deleted.")
        tf.io.gfile.makedirs(saving_path)
        logging.debug(f"rank id {self.rank_id} | Saving_path '{saving_path}' has been made.")

        self._save(sess, saving_path)
        logging.info(f"sparse model was saved in dir '{saving_path}' .")
        logging.debug(f"======== Saving finished for rank id {self.rank_id} ========")

    @performance("Restore")
    def restore(self, sess, reading_path):
        logging.debug("======== Start restoring ========")
        directory, base_name = os.path.split(reading_path)
        ckpt_name = "sparse-%s" % base_name

        reading_path = os.path.join(directory, ckpt_name)
        if not tf.io.gfile.exists(reading_path):
            raise FileExistsError(f"Given dir {reading_path} does not exist, please double check.")

        self._restore(sess, reading_path)
        logging.info(f"sparse model was restored from dir '{reading_path}' .")
        logging.debug("======== Restoring finished ========")

    def _build_save(self):
        for var in self.var_list:
            table_instance = get_table_instance(var)
            table_name = table_instance.table_name
            with tf.compat.v1.variable_scope(table_name):
                sub_dict = self.save_op_dict[table_name]
                sub_dict[DataName.EMBEDDING.value] = var
                if table_instance.optimizer:
                    sub_dict["optimizer"] = table_instance.optimizer

    def _build_restore(self):
        for var in self.var_list:
            table_instance = get_table_instance(var)
            sub_placeholder_dict = self.placeholder_dict[table_instance.table_name]
            with tf.compat.v1.variable_scope(table_instance.table_name):
                sub_placeholder_dict[DataName.EMBEDDING.value] = variable = \
                    tf.compat.v1.placeholder(dtype=tf.float32, shape=[table_instance.slice_device_vocabulary_size,
                                                                      table_instance.scalar_emb_size],
                                             name=DataName.EMBEDDING.value)
                assign_op = var.assign(variable)
                self.restore_fetch_list.append(assign_op)

                if table_instance.optimizer:
                    self._build_optimizer_restore(sub_placeholder_dict, table_instance)

    def _build_optimizer_restore(self, sub_placeholder_dict, table_instance):
        sub_placeholder_dict["optimizer"] = optimizer_placeholder_dict = dict()
        optimizer_states = table_instance.optimizer
        for optimizer_name, optimizer_state_dict in optimizer_states.items():
            optimizer_placeholder_dict[optimizer_name] = sub_optimizer_placeholder_dict = \
                dict([(state_key, tf.compat.v1.placeholder(dtype=tf.float32,
                                                           shape=[table_instance.slice_device_vocabulary_size,
                                                                  table_instance.scalar_emb_size],
                                                           name=state_key))
                      for state_key, state in optimizer_state_dict.items()])
            for key_state, state in optimizer_state_dict.items():
                assign_op = state.assign(sub_optimizer_placeholder_dict.get(key_state))
                self.restore_fetch_list.append(assign_op)

    def _save(self, sess, root_dir):
        result = sess.run(self.save_op_dict)
        for table_name, dump_data_dict in result.items():
            if is_asc_manager_initialized() and self.save_easy_mode:
                host_data = get_host_data(table_name)
                key = np.array(list(host_data.keys()))
                offset = list(host_data.values())
                get_valid_dict_data(dump_data_dict, offset)
                save_key_data(root_dir, table_name, key, self.rank_id)
            if is_asc_manager_initialized() and not self.save_easy_mode:
                save_host_data(root_dir)
                logging.debug(f"host data was saved.")
            save_embedding_data(root_dir, table_name, dump_data_dict, self.rank_id)
            table_instance = get_table_instance_by_name(table_name)

            if table_instance.use_feature_mapping:
                save_feature_mapping_data(root_dir, table_name, dump_data_dict, self.rank_id)
                save_offset_data(root_dir, table_name, dump_data_dict, self.rank_id)
            if "optimizer" in dump_data_dict:
                dump_optimizer_data_dict = dump_data_dict.get("optimizer")
                for optimizer_name, dump_optimizer_data in dump_optimizer_data_dict.items():
                    save_optimizer_state_data(root_dir, table_name, optimizer_name, dump_optimizer_data,
                                              self.rank_id)

    def _restore(self, sess, reading_path):
        restore_feed_dict = defaultdict(dict)
        key_offset_dict = defaultdict(dict)
        for table_name, sub_placeholder_dict in self.placeholder_dict.items():
            fill_placeholder(reading_path, sub_placeholder_dict, restore_feed_dict, self.rank_id,
                             NameDescriptor(table_name, DataName.EMBEDDING.value))
            if self.save_easy_mode:
                fill_key_offset_dict(reading_path, self.rank_id, table_name, key_offset_dict)
            table_instance = get_table_instance_by_name(table_name)

            if table_instance.use_feature_mapping:
                fill_placeholder(reading_path, sub_placeholder_dict, restore_feed_dict, self.rank_id,
                                 NameDescriptor(table_name, DataName.FEATURE_MAPPING.value))
                fill_placeholder(reading_path, sub_placeholder_dict, restore_feed_dict, self.rank_id,
                                 NameDescriptor(table_name, DataName.OFFSET.value))

            if "optimizer" in sub_placeholder_dict:
                optimizer_state_placeholder_dict_group = sub_placeholder_dict.get("optimizer")
                for optimizer_name, optimizer_state_placeholder_dict in optimizer_state_placeholder_dict_group.items():
                    for state_key in optimizer_state_placeholder_dict:
                        fill_placeholder(reading_path=reading_path,
                                         placeholder_dict=optimizer_state_placeholder_dict,
                                         feed_dict=restore_feed_dict,
                                         suffix=self.rank_id,
                                         name_descriptor=NameDescriptor(table_name, state_key,
                                                                        optimizer_name=optimizer_name))

        if is_asc_manager_initialized() and self.save_easy_mode:
            send_host_data(key_offset_dict)
            logging.debug(f"host data was sent to the host pipeline.")
        if is_asc_manager_initialized() and not self.save_easy_mode:
            restore_host_data(reading_path)
            logging.debug(f"host data was restored.")
        sess.run(self.restore_fetch_list, feed_dict=restore_feed_dict)


class NameDescriptor:
    def __init__(self, table_name, data_name, optimizer_name=None):
        self.table_name = table_name
        self.data_name = data_name
        self.optimizer_name = optimizer_name


def get_valid_dict_data(dump_data_dict: dict, offset: list):
    """
    Extract embedding and optimizer data from the dict based on offset.
    :param dump_data_dict: sparse data dict to be saved
    :param offset: offset of the sparse table
    """
    embedding_data = dump_data_dict.get(DataName.EMBEDDING.value)[offset, :]
    dump_data_dict[DataName.EMBEDDING.value] = embedding_data
    if "optimizer" in dump_data_dict:
        dump_optimizer_data_dict = dump_data_dict.get("optimizer")
        for optimizer_name, dump_optimizer_data in dump_optimizer_data_dict.items():
            for state_key, state in dump_optimizer_data.items():
                state = state[offset, :]
                dump_optimizer_data[state_key] = state
            dump_optimizer_data_dict[optimizer_name] = dump_optimizer_data
        dump_data_dict["optimizer"] = dump_optimizer_data_dict


def fill_key_offset_dict(reading_path: str, rank_id: int, table_name: str, key_offset_dict: dict):
    """
    Filling data in the key-offset dictionary , which is sent to the host pipeline.
    :param reading_path: the path restoring the model
    :param rank_id: rank id
    :param table_name: the sparse table name
    :param key_offset_dict: key-offset dictionary saving mapping relationship
    """
    target_path = generate_path(reading_path, "HashTable", "HBM", table_name,
                                DataName.KEY.value)
    key = read_binary_data(target_path, rank_id, DataName.KEY.value, table_name)
    key = key.get(DataName.KEY.value)
    offsets = list(range(key.shape[0]))
    key_offset_map = dict(zip(key, offsets))
    key_offset_dict[table_name] = key_offset_map


def fill_placeholder(reading_path, placeholder_dict, feed_dict, suffix, name_descriptor):
    if name_descriptor.optimizer_name:
        target_path = generate_path(reading_path, "Optimizer", name_descriptor.optimizer_name, "HBM",
                                    name_descriptor.table_name, name_descriptor.data_name)
    else:
        target_path = generate_path(reading_path, "HashTable", "HBM", name_descriptor.table_name,
                                    name_descriptor.data_name)
    restore_data_dict = read_binary_data(target_path, suffix, name_descriptor.data_name, name_descriptor.table_name)

    for key, data in restore_data_dict.items():
        embedding_placeholder = placeholder_dict.get(key)
        feed_dict[embedding_placeholder] = data


def save_embedding_data(root_dir, table_name, dump_data_dict, suffix):
    target_path = generate_path(root_dir, "HashTable", "HBM", table_name, DataName.EMBEDDING.value)
    data_to_write = dump_data_dict.get(DataName.EMBEDDING.value)

    attribute = dict()
    attribute[DataAttr.DATATYPE.value] = data_to_write.dtype.name
    attribute[DataAttr.SHAPE.value] = data_to_write.shape
    write_binary_data(target_path, suffix, data_to_write, attributes=attribute)


def save_key_data(root_dir: str, table_name: str, data_to_write: np.ndarray, suffix: int):
    """
    Save the keys of the sparse table
    :param root_dir: the root path saving the model
    :param table_name: the sparse table name
    :param data_to_write: the key array to be written
    :param suffix: suffix of sparse data
    """
    target_path = generate_path(root_dir, "HashTable", "HBM", table_name, DataName.KEY.value)
    attribute = dict()
    attribute[DataAttr.DATATYPE.value] = data_to_write.dtype.name
    attribute[DataAttr.SHAPE.value] = data_to_write.shape
    write_binary_data(target_path, suffix, data_to_write, attributes=attribute)


def save_feature_mapping_data(root_dir, table_name, dump_data_dict, suffix):
    target_path = generate_path(root_dir, "HashTable", "HBM", table_name, DataName.FEATURE_MAPPING.value)
    data_to_write = dump_data_dict.get(DataName.FEATURE_MAPPING.value)
    valid_len = dump_data_dict.get(DataName.VALID_LEN.value)
    data_to_write = data_to_write[:valid_len * 3]

    attribute = dict()
    attribute[DataAttr.DATATYPE.value] = data_to_write.dtype.name
    attribute[DataName.THRESHOLD.value] = int(dump_data_dict.get(DataName.THRESHOLD.value))
    write_binary_data(target_path, suffix, data_to_write, attributes=attribute)


def save_offset_data(root_dir, table_name, dump_data_dict, suffix):
    target_path = generate_path(root_dir, "HashTable", "HBM", table_name, DataName.OFFSET.value)
    data_to_write = dump_data_dict.get(DataName.OFFSET.value)
    valid_bucket_num = dump_data_dict.get(DataName.VALID_BUCKET_NUM.value)
    data_to_write = data_to_write[:valid_bucket_num]

    attribute = dict()
    attribute[DataAttr.DATATYPE.value] = data_to_write.dtype.name
    write_binary_data(target_path, suffix, data_to_write, attributes=attribute)


def save_optimizer_state_data(root_dir, table_name, optimizer_name, dump_optimizer_data, suffix):
    for state_key, state in dump_optimizer_data.items():
        target_path = generate_path(root_dir, "Optimizer", optimizer_name, "HBM", table_name, state_key)
        data_to_write = state

        attribute = dict()
        attribute[DataAttr.DATATYPE.value] = data_to_write.dtype.name
        attribute[DataAttr.SHAPE.value] = data_to_write.shape
        write_binary_data(target_path, suffix, data_to_write, attributes=attribute)


def generate_path(*args):
    return os.path.join(*args)


def generate_file_name(suffix):
    return "slice_%d.data" % suffix, "slice_%d.attribute" % suffix


def write_binary_data(writing_path, suffix, data, attributes=None):
    tf.io.gfile.makedirs(writing_path)
    data_file, attribute_file = generate_file_name(suffix)
    target_data_dir = os.path.join(writing_path, data_file)
    target_attribute_dir = os.path.join(writing_path, attribute_file)
    if tf.io.gfile.exists(target_data_dir):
        raise FileExistsError(f"Target_data_dir {target_data_dir} exists before writing.")
    if tf.io.gfile.exists(target_attribute_dir):
        raise FileExistsError(f"Target_attribute_dir {target_attribute_dir} exists before writing.")

    if target_data_dir.find("://") != -1:
        logging.debug(f"use hdfs path {target_data_dir} to save sparse data.")
        with tf.io.gfile.GFile(target_data_dir, "w") as file:
            data = json.dumps(data.flatten().tolist())
            file.write(data)
    else:
        logging.debug(f"use local file path {target_data_dir} to save sparse data.")
        data.tofile(target_data_dir)

    if attributes is not None:
        if not isinstance(attributes, dict):
            raise TypeError(f"Parameter 'attributes' must be one dict instance, instead of {type(attributes)}")

        with tf.io.gfile.GFile(target_attribute_dir, "w") as file:
            file.write(json.dumps(attributes))


def read_binary_data(reading_path: str, suffix: int, data_name: str, table_name: str) -> dict:
    """
    Read sparse origin data from binary file
    :param reading_path: sparse data path
    :param suffix: suffix of sparse data
    :param data_name: the data type,including embedding, offset, etc.
    :param table_name: the sparse table name
    :return: the sparse data dict
    """
    data_file, attribute_file = generate_file_name(suffix)
    target_data_dir = os.path.join(reading_path, data_file)
    target_attribute_dir = os.path.join(reading_path, attribute_file)
    if not tf.io.gfile.exists(target_data_dir):
        raise FileExistsError(f"Target_data_dir {target_data_dir} does not exist when reading.")
    if not tf.io.gfile.exists(target_attribute_dir):
        raise FileExistsError(f"Target_attribute_dir {target_attribute_dir} does not exist when reading.")

    with tf.io.gfile.GFile(target_attribute_dir, "r") as fin:
        attributes = json.load(fin)

    if DataAttr.DATATYPE.value not in attributes:
        raise AttributeError(f"Lack of attribute {DataAttr.DATATYPE.value}.")

    if target_data_dir.find("://") != -1:
        logging.debug(f"use hdfs path {target_data_dir} to restore sparse data.")
        with tf.io.gfile.GFile(target_data_dir, "r") as file:
            data_to_restore = file.read()
            data_to_restore = np.array(json.loads(data_to_restore))
    else:
        logging.debug(f"use local file path {target_data_dir} to restore sparse data.")
        data_to_restore = np.fromfile(target_data_dir, dtype=attributes.pop(DataAttr.DATATYPE.value))

    if DataAttr.SHAPE.value in attributes and data_name != DataName.KEY.value:
        data_shape = attributes.pop(DataAttr.SHAPE.value)
        data_to_restore = data_to_restore.reshape(data_shape)
        table_instance = get_table_instance_by_name(table_name)
        current_data_shape = [table_instance.slice_device_vocabulary_size, table_instance.scalar_emb_size]
        if data_shape != current_data_shape:
            data_to_restore = process_embedding_data(data_to_restore, current_data_shape, data_shape)

    data_dict = {data_name: data_to_restore}
    logging.debug(f"Attribute: '{target_attribute_dir}' and data file: '{target_data_dir}' have been read.")
    logging.debug(f"Reading shape is {data_to_restore.shape}.")

    return data_dict


def process_embedding_data(data_to_restore: np.ndarray, current_data_shape: list, data_shape: list) -> np.ndarray:
    """
    Process embedding data when reading binary file
    :param data_to_restore: the embedding data reading from the binary file
    :param current_data_shape: current embedding data shape set by user
    :param data_shape: embedding data shape saved in the binary file
    :return: the embedding data
    """
    try:
        restore_vocab_size, restore_emb_size = current_data_shape
        vocab_size, emb_size = data_shape
    except ValueError as err:
        raise ValueError(f"The shape dimension of a sparse table cannot exceed two dimensions. ") from err

    if restore_vocab_size > vocab_size:
        pad_count = restore_vocab_size - vocab_size
        pad_matrix = np.zeros((pad_count, restore_emb_size))
        data_to_restore = np.concatenate((data_to_restore, pad_matrix), axis=0)

    elif restore_vocab_size < vocab_size:
        raise Exception(f"restore vocabulary size {restore_vocab_size} cannot be less than "
                        f"saved vocabulary size {vocab_size},which would loss the mapping between keys and embeddings ")

    return data_to_restore
