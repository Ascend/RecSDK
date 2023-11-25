#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import json
import os
import threading
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.util import compat

from mx_rec.constants.constants import DataName, DataAttr, MIN_SIZE, MAX_FILE_SIZE, Flag, TFDevice, \
    MAX_INT32, HDFS_FILE_PREFIX
from mx_rec.util.initialize import get_rank_id, get_rank_size, get_customized_ops, get_table_instance, \
    get_table_instance_by_name, is_asc_manager_initialized, save_host_data, restore_host_data, get_host_data, \
    send_host_data, get_ascend_global_hashtable_collection, set_sparse_dir, get_local_rank_size, \
    get_use_dynamic_expansion
from mx_rec.util.perf import performance
from mx_rec.validator.validator import DirectoryValidator, FileValidator, para_checker_decorator, ClassValidator, \
    IntValidator, OptionalStringValidator
from mx_rec.util.global_env_conf import global_env
from mx_rec.util.log import logger


# define save model thread
class SaveModelThread(threading.Thread):
    def __init__(self, sess, result, root_dir, table_name):
        super().__init__()
        self.result = result
        self.root_dir = root_dir
        self.table_name = table_name
        self.sess = sess

    def run(self):
        Saver().save_table_name_data(self.sess, self.result, self.root_dir, self.table_name)


class Saver(object):
    customized_ops = get_customized_ops()

    @staticmethod
    def _make_table_name_dir(root_dir, table_instance, table_name):
        if table_instance.host_vocabulary_size > 0:
            table_dir = os.path.join(root_dir, "HashTable", "DDR", table_name)
        else:
            table_dir = os.path.join(root_dir, "HashTable", "HBM", table_name)
        tf.io.gfile.makedirs(table_dir)

    @para_checker_decorator(check_option_list=[
        ("var_list", ClassValidator, {"classes": (list, type(None))}),
        ("max_to_keep", IntValidator, {"min_value": 0, "max_value": MAX_INT32}, ["check_value"]),
        ("prefix_name", ClassValidator, {"classes": (str, type(None))}),
        ("prefix_name", OptionalStringValidator, {"min_len": 1, "max_len": 50}, ["check_string_length"]),
    ])
    def __init__(self, var_list=None, max_to_keep=3, prefix_name="checkpoint"):
        self.max_to_keep = max_to_keep
        self._prefix_name = prefix_name
        self.var_list = var_list
        self.rank_id = get_rank_id()
        self.local_rank_size = get_local_rank_size()
        self.local_rank_id = self.rank_id % self.local_rank_size
        self.rank_size = get_rank_size()
        self.save_op_dict = defaultdict(dict)
        self.restore_fetch_list = []
        self.placeholder_dict = defaultdict(dict)
        # save_easy_mode : only save the embedding and key data of sparse tables
        self.save_easy_mode = (global_env.save_easy == Flag.TRUE.value)
        self._last_checkponts = []
        self.build()

    def build(self):
        if self.var_list is None:
            self.var_list = []
            logger.debug("optimizer collection name: %s", get_ascend_global_hashtable_collection())
            temp_var_list = tf.compat.v1.get_collection(get_ascend_global_hashtable_collection())
            for var in temp_var_list:
                table_instance = get_table_instance(var)
                if table_instance.is_save:
                    self.var_list.append(var)

        with tf.compat.v1.variable_scope("mx_rec_save"):
            self._build_save()
        with tf.compat.v1.variable_scope("mx_rec_restore"):
            self._build_restore()

        logger.debug("Save & Restore graph was built.")

    @performance("Save")
    def save(self, sess, save_path="model", global_step=None):
        """
        Save sparse tables. For local save, both save_easy mode and normal mode is supported.
        For easy_save mode, checkpoint is saved in under format:
        ./rank_id/HashTable/HBM/embed_table_name/key/xxx.data
        ./rank_id/HashTable/HBM/embed_table_name/key/xxx.attribute
        ./rank_id/HashTable/HBM/embed_table_name/embedding/xxx.data
        ./rank_id/HashTable/HBM/embed_table_name/embedding/xxx.attribute
        :param sess: A Session to use to save the sparse table variables
        :param save_path: Only absolute path supported
        :param global_step: If provided the global step number is appended to save_path to create
         the checkpoint filenames. The optional argument can be a Tensor, a Tensor name or an integer.
        :return: None
        """
        logger.debug("======== Start saving for rank id %s ========", self.rank_id)
        if not check_file_system_is_valid(save_path):
            raise ValueError("the path to save sparse embedding table data belong to invalid file system, "
                             "only local file system and hdfs file system supported. ")

        save_path = save_path if save_path else self._prefix_name
        directory, base_name = os.path.split(save_path)

        if global_step:
            if not isinstance(global_step, compat.integral_types):
                global_step = int(sess.run(global_step))
            ckpt_name = f"sparse-{base_name}-{global_step}"
        else:
            ckpt_name = f"sparse-{base_name}"

        saving_path = os.path.join(directory, ckpt_name)
        set_sparse_dir(saving_path)

        try:
            if not check_file_system_is_hdfs(saving_path):
                directory_validator = DirectoryValidator("saving_path", saving_path)
                directory_validator.check_not_soft_link()
                directory_validator.with_blacklist(exact_compare=False)
                directory_validator.check()
        except ValueError as err:
            raise ValueError(f"The saving path {saving_path} cannot be a system directory "
                             f"and cannot be soft link.") from err

        if tf.io.gfile.exists(saving_path):
            tf.io.gfile.rmtree(saving_path)
            logger.info("rank id %s | Saving_path '%s' has been deleted.", self.rank_id, saving_path)
        tf.io.gfile.makedirs(saving_path)
        logger.info("rank id %s | Saving_path '%s' has been made.", self.rank_id, saving_path)

        self._save(sess, saving_path)
        if self.max_to_keep:
            self._last_checkponts.append(saving_path)
            if len(self._last_checkponts) > self.max_to_keep:
                logger.info("checkpoints num %d > max_to_keep %d delete %s",
                            len(self._last_checkponts), self.max_to_keep,
                            self._last_checkponts[0])
                tf.io.gfile.rmtree(self._last_checkponts.pop(0))
        logger.info("sparse model was saved in dir '%s' .", saving_path)
        logger.info("======== Saving finished for rank id %s ========", self.rank_id)

    @performance("Restore")
    def restore(self, sess, reading_path):
        logger.debug("======== Start restoring ========")
        if not check_file_system_is_valid(reading_path):
            raise ValueError("the path to save sparse embedding table data belong to invalid file system, "
                             "only local file system and hdfs file system supported. ")

        directory, base_name = os.path.split(reading_path)
        ckpt_name = f"sparse-{base_name}"

        reading_path = os.path.join(directory, ckpt_name)
        set_sparse_dir(reading_path)
        if not tf.io.gfile.exists(reading_path):
            raise FileExistsError(f"Given dir {reading_path} does not exist, please double check.")

        self._restore(sess, reading_path)
        logger.info("sparse model was restored from dir '%s' .", reading_path)
        logger.debug("======== Restoring finished ========")

    @performance("save_table_name_data")
    def save_table_name_data(self, sess, result, root_dir, table_name):
        table_instance = get_table_instance_by_name(table_name)
        self._make_table_name_dir(root_dir, table_instance, table_name)

        dump_data_dict = sess.run(result.get(table_name))
        # when HBM mode is on, need to get host offset data, to process dump data dict for saving valid embedding.
        if is_asc_manager_initialized() and table_instance.host_vocabulary_size == 0:
            self._get_valid_dict_data(dump_data_dict, table_name)

        # save embedding
        save_embedding_data(root_dir, table_name, dump_data_dict, self.rank_id)

        # save optimizer data
        if "optimizer" in dump_data_dict:
            dump_optimizer_data_dict = dump_data_dict.get("optimizer")
            for optimizer_name, dump_optimizer_data in dump_optimizer_data_dict.items():
                save_optimizer_state_data(root_dir, table_name, optimizer_name, dump_optimizer_data, self.rank_id)

    @performance("_save")
    def _save(self, sess, root_dir):
        if is_asc_manager_initialized():
            save_host_data(root_dir)
            logger.debug(f"host data was saved.")

        if get_use_dynamic_expansion():
            # Data related to dynamic expansion needs to be saved only on the host side.
            return

        result = self.save_op_dict
        threads = []
        for table_name in result.keys():
            thread = SaveModelThread(sess, result, root_dir, table_name)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def _get_valid_dict_data(self, dump_data_dict, table_name):
        host_data = get_host_data(table_name)
        offset = list(host_data)

        get_valid_dict_data_from_host_offset(dump_data_dict, offset)

    def _build_save(self):
        for var in self.var_list:
            if global_env.tf_device == TFDevice.NPU.value and "merged" not in var.name:
                continue

            table_instance = get_table_instance(var)
            table_name = table_instance.table_name
            with tf.compat.v1.variable_scope(table_name):
                sub_dict = self.save_op_dict[table_name]
                sub_dict[DataName.EMBEDDING.value] = var
                if table_instance.optimizer:
                    sub_dict["optimizer"] = table_instance.optimizer

    def _build_restore(self):
        for var in self.var_list:
            if global_env.tf_device == TFDevice.NPU.value and "merged" not in var.name:
                continue
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


    def _restore(self, sess, reading_path):
        if is_asc_manager_initialized():
            restore_host_data(reading_path)
            logger.info("host data was restored.")

        if get_use_dynamic_expansion():
            # Data related to dynamic expansion needs to be restored only on the host side.
            return

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


def get_valid_dict_data_from_host_offset(dump_data_dict: dict, offset: list):
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


@performance("save_embedding_data")
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
    tf.io.gfile.makedirs(writing_path)
    data_file, attribute_file = generate_file_name(suffix)
    target_data_dir = os.path.join(writing_path, data_file)
    target_attribute_dir = os.path.join(writing_path, attribute_file)
    if tf.io.gfile.exists(target_data_dir):
        raise FileExistsError(f"Target_data_dir {target_data_dir} exists before writing.")
    if tf.io.gfile.exists(target_attribute_dir):
        raise FileExistsError(f"Target_attribute_dir {target_attribute_dir} exists before writing.")

    if check_file_system_is_hdfs(target_data_dir):
        with tf.io.gfile.GFile(target_data_dir, "wb") as file:
            data = data.tostring()
            file.write(data)
    else:
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
        validate_read_file(target_attribute_dir)
        attributes = json.load(fin)

    if DataAttr.DATATYPE.value not in attributes:
        raise AttributeError(f"Lack of attribute {DataAttr.DATATYPE.value}.")

    with tf.io.gfile.GFile(target_data_dir, "rb") as file:
        validate_read_file(target_data_dir)
        if check_file_system_is_hdfs(target_data_dir):
            data_to_restore = file.read()
            data_to_restore = np.fromstring(data_to_restore, dtype=attributes.pop(DataAttr.DATATYPE.value))
        else:
            data_to_restore = np.fromfile(target_data_dir, dtype=attributes.pop(DataAttr.DATATYPE.value))

    if DataAttr.SHAPE.value in attributes and data_name != DataName.KEY.value:
        data_shape = attributes.pop(DataAttr.SHAPE.value)
        data_to_restore = data_to_restore.reshape(data_shape)
        table_instance = get_table_instance_by_name(table_name)
        current_data_shape = [table_instance.slice_device_vocabulary_size, table_instance.scalar_emb_size]
        if data_shape != current_data_shape:
            data_to_restore = process_embedding_data(data_to_restore, current_data_shape, data_shape)

    data_dict = {data_name: data_to_restore}
    logger.debug("Attribute: '%s' and data file: '%s' have been read.", target_attribute_dir, target_data_dir)
    logger.debug("Reading shape is %s.", data_to_restore.shape)

    return data_dict


def validate_read_file(read_file_path):
    """
    Validate file before reading，including validating soft link, file size
    :param read_file_path: the file path to be validated
    """
    file_validator = FileValidator("read_file_path", read_file_path)
    file_validator.check_file_size(MAX_FILE_SIZE, MIN_SIZE)
    file_validator.check_user_group()
    if not check_file_system_is_hdfs(read_file_path):
        file_validator.check_not_soft_link()
    file_validator.check()


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


def check_file_system_is_valid(file_path):
    if file_path.find("://") == -1 or check_file_system_is_hdfs(file_path):
        return True
    return False


def check_file_system_is_hdfs(file_path):
    for prefix in HDFS_FILE_PREFIX:
        if file_path.startswith(prefix):
            return True
    return False
