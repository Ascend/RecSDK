#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import json
import os
import shutil
import logging
import stat
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.util import compat

from mx_rec.constants.constants import DataName, DataAttr
from mx_rec.util.initialize import get_rank_id, get_rank_size, get_customized_ops, get_table_instance, \
    get_table_instance_by_name, is_asc_manager_initialized, save_host_data, restore_host_data, \
    get_ascend_global_hashtable_collection
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
        self.build()

    def build(self):
        if self.var_list is None:
            logging.debug(f"optimizer collection name: {get_ascend_global_hashtable_collection()}")
            self.var_list = tf.compat.v1.get_collection(get_ascend_global_hashtable_collection())

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
        if integrated_path.startswith("/"):
            saving_path = os.path.abspath(integrated_path)

        if os.path.exists(saving_path):
            shutil.rmtree(saving_path, ignore_errors=True)
            logging.debug(f"rank id {self.rank_id} | Saving_path '{saving_path}' has been deleted.")
        os.makedirs(saving_path, exist_ok=True)
        logging.debug(f"rank id {self.rank_id} | Saving_path '{saving_path}' has been made.")

        self._save(sess, saving_path)
        logging.info(f"sparse model was saved in dir '{saving_path}' .")
        logging.debug(f"======== Saving finished for rank id {self.rank_id} ========")

    @performance("Restore")
    def restore(self, sess, reading_path):
        logging.debug("======== Start restoring ========")
        directory, base_name = os.path.split(reading_path)
        ckpt_name = "sparse-%s" % base_name

        integrated_path = os.path.join(directory, ckpt_name)
        reading_path = os.path.abspath(integrated_path)
        if not os.path.exists(reading_path):
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
            save_embedding_data(root_dir, table_name, dump_data_dict, self.rank_id)
            table_instance = get_table_instance_by_name(table_name)
            if is_asc_manager_initialized():
                save_host_data(root_dir)
                logging.debug(f"host data was saved.")

            if table_instance.use_feature_mapping:
                save_feature_mapping_data(root_dir, table_name, dump_data_dict, self.rank_id)
                save_offset_data(root_dir, table_name, dump_data_dict, self.rank_id)
            if "optimizer" in dump_data_dict:
                dump_optimizer_data_dict = dump_data_dict.get("optimizer")
                for optimizer_name, dump_optimizer_data in dump_optimizer_data_dict.items():
                    save_optimizer_state_data(root_dir, table_name, optimizer_name, dump_optimizer_data,
                                              self.rank_id)

    def _restore(self, sess, reading_path):
        if is_asc_manager_initialized():
            restore_host_data(reading_path)
            logging.debug(f"host data was restored.")

        restore_feed_dict = defaultdict(dict)
        for table_name, sub_placeholder_dict in self.placeholder_dict.items():
            fill_placeholder(reading_path, sub_placeholder_dict, restore_feed_dict, self.rank_id,
                             NameDescriptor(table_name, DataName.EMBEDDING.value))
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

        sess.run(self.restore_fetch_list, feed_dict=restore_feed_dict)


class NameDescriptor:
    def __init__(self, table_name, data_name, optimizer_name=None):
        self.table_name = table_name
        self.data_name = data_name
        self.optimizer_name = optimizer_name


def fill_placeholder(reading_path, placeholder_dict, feed_dict, suffix, name_descriptor):
    if name_descriptor.optimizer_name:
        target_path = generate_path(reading_path, "Optimizer", name_descriptor.optimizer_name, "HBM",
                                    name_descriptor.table_name, name_descriptor.data_name)
    else:
        target_path = generate_path(reading_path, "HashTable", "HBM", name_descriptor.table_name,
                                    name_descriptor.data_name)
    restore_data_dict = read_binary_data(target_path, suffix, name_descriptor.data_name)

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
    os.makedirs(writing_path, exist_ok=True)
    data_file, attribute_file = generate_file_name(suffix)
    target_data_dir = os.path.join(writing_path, data_file)
    target_attribute_dir = os.path.join(writing_path, attribute_file)
    if os.path.exists(target_data_dir):
        raise FileExistsError(f"Target_data_dir {target_data_dir} exists before writing.")
    if os.path.exists(target_attribute_dir):
        raise FileExistsError(f"Target_attribute_dir {target_attribute_dir} exists before writing.")
    data.tofile(target_data_dir)

    if attributes is not None:
        if not isinstance(attributes, dict):
            raise TypeError(f"Parameter 'attributes' must be one dict instance, instead of {type(attributes)}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        mode = stat.S_IRUSR | stat.S_IWUSR
        with os.fdopen(os.open(target_attribute_dir, flags, mode), 'w') as file:
            file.write(json.dumps(attributes))


def read_binary_data(reading_path, suffix, data_name):
    data_file, attribute_file = generate_file_name(suffix)
    target_data_dir = os.path.join(reading_path, data_file)
    target_attribute_dir = os.path.join(reading_path, attribute_file)
    if not os.path.exists(target_data_dir):
        raise FileExistsError(f"Target_data_dir {target_data_dir} does not exist when reading.")
    if not os.path.exists(target_attribute_dir):
        raise FileExistsError(f"Target_attribute_dir {target_attribute_dir} does not exist when reading.")

    with open(target_attribute_dir, "r") as fin:
        attributes = json.load(fin)

    if DataAttr.DATATYPE.value not in attributes:
        raise AttributeError(f"Lack of attribute {DataAttr.DATATYPE.value}.")

    data_to_restore = np.fromfile(target_data_dir, dtype=attributes.pop(DataAttr.DATATYPE.value))
    if DataAttr.SHAPE.value in attributes:
        data_to_restore = data_to_restore.reshape(attributes.pop(DataAttr.SHAPE.value))

    data_dict = {data_name: data_to_restore}
    for key, item in attributes.items():
        data_dict[key] = item
    logging.debug(f"Attribute: '{target_attribute_dir}' and data file: '{target_data_dir}' have been read.")
    logging.debug(f"Reading shape is {data_to_restore.shape}.")

    return data_dict
