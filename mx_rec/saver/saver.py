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
import os
import threading
from collections import defaultdict
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.python.util import compat

from mx_rec.constants.constants import DataName, DataAttr, MIN_SIZE, MAX_FILE_SIZE, Flag, TFDevice, \
    MAX_INT32, HDFS_FILE_PREFIX
from mx_rec.util.communication.hccl_ops import get_rank_id, get_rank_size, get_local_rank_size
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.perf import performance
from mx_rec.validator.validator import DirectoryValidator, FileValidator, para_checker_decorator, ClassValidator, \
    IntValidator, OptionalStringValidator
from mx_rec.util.global_env_conf import global_env
from mx_rec.util.log import logger
from mx_rec.optimizers.base import CustomizedOptimizer


# define save model thread
class SaveModelThread(threading.Thread):
    def __init__(self, saver, sess, result, root_dir, table_name):
        super().__init__()
        self.result = result
        self.root_dir = root_dir
        self.table_name = table_name
        self.sess = sess
        self.saver = saver

    def run(self):
        self.saver.save_table_name_data(self.sess, self.result, self.root_dir, self.table_name)


class Saver(object):
    @staticmethod
    def _make_table_name_dir(root_dir, table_instance, table_name):
        if not table_instance.is_hbm:
            table_dir = os.path.join(root_dir, "HashTable", "DDR", table_name)
        else:
            table_dir = os.path.join(root_dir, "HashTable", "HBM", table_name)
        try:
            tf.io.gfile.makedirs(table_dir)
        except Exception as err:
            raise RuntimeError(f"make dir {table_dir} for saving sparse table failed!") from err

    @para_checker_decorator(check_option_list=[
        ("var_list", ClassValidator, {"classes": (list, type(None))}),
        ("max_to_keep", IntValidator, {"min_value": 0, "max_value": MAX_INT32}, ["check_value"]),
        ("prefix_name", ClassValidator, {"classes": (str, type(None))}),
        ("prefix_name", OptionalStringValidator, {"min_len": 1, "max_len": 50}, ["check_string_length"]),
    ])
    def __init__(self, var_list=None, max_to_keep=3, prefix_name="checkpoint", warm_start_tables=None):
        self.max_to_keep = max_to_keep
        self._prefix_name = prefix_name
        self.var_list = var_list
        self.rank_id = get_rank_id()
        self.local_rank_size = get_local_rank_size()
        self.local_rank_id = self.rank_id % self.local_rank_size
        self.save_op_dict = defaultdict(dict)
        self.restore_fetch_dict = defaultdict()
        self.placeholder_dict = defaultdict(dict)
        self._last_checkponts = []
        self.config_instance = ConfigInitializer.get_instance()
        self.build()
        self.warm_start_tables = warm_start_tables

    def build(self):
        if self.var_list is None:
            self.var_list = []
            logger.debug("optimizer collection name: %s",
                         self.config_instance.train_params_config.ascend_global_hashtable_collection)
            temp_var_list = tf.compat.v1.get_collection(
                self.config_instance.train_params_config.ascend_global_hashtable_collection)
            for var in temp_var_list:
                table_instance = self.config_instance.sparse_embed_config.get_table_instance(var)
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
        Save sparse tables. checkpoint is saved in under format:
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
        self.config_instance.train_params_config.sparse_dir = saving_path

        try:
            if not check_file_system_is_hdfs(saving_path):
                directory_validator = DirectoryValidator("saving_path", saving_path)
                directory_validator.check_not_soft_link()
                directory_validator.with_blacklist(exact_compare=False)
                directory_validator.check()
        except ValueError as err:
            raise ValueError(f"The saving path {saving_path} cannot be a system directory "
                             f"and cannot be soft link.") from err

        if not tf.io.gfile.exists(saving_path):
            try:
                tf.io.gfile.makedirs(saving_path)
            except Exception as err:
                raise RuntimeError(f"make dir {saving_path} for saving sparse table failed!") from err
            logger.info("Saving_path '%s' has been made.", saving_path)

        self._save(sess, saving_path)
        if self.max_to_keep:
            self._last_checkponts.append(saving_path)
            if len(self._last_checkponts) > self.max_to_keep:
                logger.info("checkpoints num %d > max_to_keep %d delete %s",
                            len(self._last_checkponts), self.max_to_keep,
                            self._last_checkponts[0])
                try:
                    tf.io.gfile.rmtree(self._last_checkponts.pop(0))
                except tf.errors.NotFoundError as e:
                    logger.warning("oldest checkpoint file is not exist, maybe it has been deleted.")

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        comm.Barrier()
        if rank == 0:
            table_list = self.save_op_dict.keys()
            for table_name in table_list:
                self.merge_sparse_file(saving_path, table_name)
        comm.Barrier()

        logger.info("sparse model was saved in dir '%s' .", saving_path)
        logger.info("======== Saving finished for rank id %s ========", self.rank_id)

    @performance("Restore")
    def restore(self, sess, reading_path, warm_start_tables=None):
        logger.debug("======== Start restoring ========")
        if not check_file_system_is_valid(reading_path):
            raise ValueError("the path to save sparse embedding table data belong to invalid file system, "
                             "only local file system and hdfs file system supported. ")

        directory, base_name = os.path.split(reading_path)
        ckpt_name = f"sparse-{base_name}"

        reading_path = os.path.join(directory, ckpt_name)
        if not tf.io.gfile.exists(reading_path):
            raise FileExistsError(f"Given dir {reading_path} does not exist, please double check.")

        self._restore(sess, reading_path, warm_start_tables)
        logger.info("sparse model was restored from dir '%s' .", reading_path)
        logger.debug("======== Restoring finished ========")

    @performance("save_table_name_data")
    def save_table_name_data(self, sess, result, root_dir, table_name):
        dump_data_dict = sess.run(result.get(table_name))
        self._get_valid_dict_data(dump_data_dict, table_name)

        # save embedding
        save_embedding_data(root_dir, table_name, dump_data_dict, self.rank_id)

        # save optimizer data
        if "optimizer" in dump_data_dict:
            dump_optimizer_data_dict = dump_data_dict.get("optimizer")
            for optimizer_name, dump_optimizer_data in dump_optimizer_data_dict.items():
                save_optimizer_state_data(root_dir, table_name, optimizer_name, dump_optimizer_data, self.rank_id)

    def merge_sparse_file(self, root_dir: str, table_name: str):
        """
        将多卡保存下来的多个二进制文件合成一个

        Args:
            root_dir: 合并路径
            table_name: 被合并的表名

        Returns: None
        """
        table_dir = os.path.join(root_dir, table_name)
        table_instance = ConfigInitializer.get_instance().sparse_embed_config.get_table_instance_by_name(table_name)
        merge_type_list = get_merge_type_list(table_dir)

        for data_type in merge_type_list:
            upper_dir = os.path.join(table_dir, data_type)
            merge_multi_files(upper_dir)
            outfile_path = os.path.join(upper_dir, "slice.data")
            file_size = tf.io.gfile.stat(outfile_path).length
            if data_type == "key":
                attribute = np.array([file_size / 8, 8])
            else:
                attribute = np.array([file_size / 4 / table_instance.emb_size, table_instance.emb_size, 4])

            attribute = attribute.astype(np.int64)
            attribute_dir = os.path.join(upper_dir, "slice.attribute")
            with tf.io.gfile.GFile(attribute_dir, "wb") as file:
                attribute = attribute.tostring()
                file.write(attribute)

    @performance("_save")
    def _save(self, sess, root_dir):
        for table_name in self.save_op_dict:
            optimizer_instance = ConfigInitializer.get_instance().optimizer_config.optimizer_instance
            if optimizer_instance:
                set_optimizer_info(optimizer_instance, table_name)

        if self.config_instance.hybrid_manager_config.asc_manager:
            self.config_instance.hybrid_manager_config.save_host_data(root_dir)
            logger.debug(f"host data was saved.")

        if self.config_instance.use_dynamic_expansion:
            # Data related to dynamic expansion needs to be saved only on the host side.
            return

        result = self.save_op_dict
        threads = []
        for table_name in result.keys():
            thread = SaveModelThread(self, sess, result, root_dir, table_name)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def _get_valid_dict_data(self, dump_data_dict, table_name):
        host_data = self.config_instance.hybrid_manager_config.get_host_data(table_name)
        offset = list(host_data)

        get_valid_dict_data_from_host_offset(dump_data_dict, offset)

    def _build_save(self):
        for var in self.var_list:
            if global_env.tf_device == TFDevice.NPU.value and "merged" not in var.name:
                continue

            table_instance = self.config_instance.sparse_embed_config.get_table_instance(var)
            table_name = table_instance.table_name
            with tf.compat.v1.variable_scope(table_name):
                sub_dict = self.save_op_dict[table_name]
                sub_dict[DataName.EMBEDDING.value] = var
                optimizer = ConfigInitializer.get_instance().optimizer_config.get_optimizer_by_table_name(table_name)
                if optimizer:
                    sub_dict["optimizer"] = optimizer

    def _build_restore(self):
        # 这里build_restore的地方不变
        for var in self.var_list:
            if global_env.tf_device == TFDevice.NPU.value and "merged" not in var.name:
                continue
            table_instance = self.config_instance.sparse_embed_config.get_table_instance(var)
            sub_placeholder_dict = self.placeholder_dict[table_instance.table_name]
            with tf.compat.v1.variable_scope(table_instance.table_name):
                sub_placeholder_dict[DataName.EMBEDDING.value] = variable = \
                    tf.compat.v1.placeholder(dtype=tf.float32, shape=[table_instance.slice_device_vocabulary_size,
                                                                      table_instance.emb_size],
                                             name=DataName.EMBEDDING.value)
                assign_op = var.assign(variable)
                self.restore_fetch_dict[table_instance.table_name] = [assign_op]
                optimizer = ConfigInitializer.get_instance().optimizer_config.get_optimizer_by_table_name(
                    table_instance.table_name)
                if optimizer:
                    self._build_optimizer_restore(sub_placeholder_dict, table_instance, optimizer)

    def _build_optimizer_restore(self, sub_placeholder_dict, table_instance, optimizer):
        sub_placeholder_dict["optimizer"] = optimizer_placeholder_dict = dict()
        for optimizer_name, optimizer_state_dict in optimizer.items():
            optimizer_placeholder_dict[optimizer_name] = sub_optimizer_placeholder_dict = \
                dict([(state_key, tf.compat.v1.placeholder(dtype=tf.float32,
                                                           shape=[table_instance.slice_device_vocabulary_size,
                                                                  table_instance.emb_size],
                                                           name=state_key))
                      for state_key, state in optimizer_state_dict.items()])
            for key_state, state in optimizer_state_dict.items():
                if sub_optimizer_placeholder_dict.get(key_state).graph is not state.graph:
                    continue
                assign_op = state.assign(sub_optimizer_placeholder_dict.get(key_state))
                self.restore_fetch_dict[table_instance.table_name].append(assign_op)

    def get_warm_start_dict(self, table_list):
        placeholder_dict = defaultdict(dict)
        restore_fetch_list = []
        for table_name, v in self.placeholder_dict.items():
            if table_name in table_list:
                placeholder_dict[table_name] = v
                restore_fetch_list.append(self.restore_fetch_dict.get(table_name))

        if not restore_fetch_list:
            logger.warning("no tables can be warm start restored.")
        return placeholder_dict, restore_fetch_list

    def _restore(self, sess, reading_path, warm_start_tables=None):
        # 根据table_list去改造
        if warm_start_tables:
            placeholder_dict, restore_fetch_list = self.get_warm_start_dict(warm_start_tables)
        else:
            placeholder_dict, restore_fetch_list = self.placeholder_dict, self.restore_fetch_dict


        for table_name in placeholder_dict:
            optimizer_instance = ConfigInitializer.get_instance().optimizer_config.optimizer_instance
            if optimizer_instance:
                set_optimizer_info(optimizer_instance, table_name)

        if self.config_instance.hybrid_manager_config.asc_manager:
            self.config_instance.hybrid_manager_config.restore_host_data(reading_path, warm_start_tables)
            logger.info("host data was restored.")

        if self.config_instance.use_dynamic_expansion:
            # Data related to dynamic expansion needs to be restored only on the host side.
            return

        restore_feed_dict = defaultdict(dict)

        for table_name, sub_placeholder_dict in placeholder_dict.items():
            load_offset = self.config_instance.hybrid_manager_config.get_load_offset(table_name)
            fill_placeholder(reading_path, sub_placeholder_dict, restore_feed_dict,
                             NameDescriptor(table_name, DataName.EMBEDDING.value), load_offset)

            if "optimizer" in sub_placeholder_dict:
                optimizer_state_placeholder_dict_group = sub_placeholder_dict.get("optimizer")
                _fill_placeholder_for_optimizer(optimizer_state_placeholder_dict_group, reading_path,
                                                restore_feed_dict, table_name, load_offset)

        sess.run(restore_fetch_list, feed_dict=restore_feed_dict)


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


def fill_placeholder(reading_path: str, placeholder_dict: Dict[str, tf.Tensor],
                     feed_dict: Dict[str, Dict[str, tf.Tensor]],
                     name_descriptor: NameDescriptor, load_offset: List[int]):
    if name_descriptor.optimizer_name:
        target_path = generate_path(reading_path, name_descriptor.table_name,
                                    name_descriptor.optimizer_name + "_" + name_descriptor.data_name)
    else:
        target_path = generate_path(reading_path, name_descriptor.table_name, name_descriptor.data_name)
    restore_data_dict = read_binary_data(target_path, name_descriptor.data_name, name_descriptor.table_name,
                                         load_offset)

    for key, data in restore_data_dict.items():
        embedding_placeholder = placeholder_dict.get(key)
        feed_dict[embedding_placeholder] = data


@performance("save_embedding_data")
def save_embedding_data(root_dir, table_name, dump_data_dict, suffix):
    target_path = generate_path(root_dir, table_name, DataName.EMBEDDING.value)
    data_to_write = dump_data_dict.get(DataName.EMBEDDING.value)

    attribute = dict()
    attribute[DataAttr.DATATYPE.value] = data_to_write.dtype.name
    attribute[DataAttr.SHAPE.value] = data_to_write.shape
    write_binary_data(target_path, suffix, data_to_write)


def save_feature_mapping_data(root_dir, table_name, dump_data_dict, suffix):
    target_path = generate_path(root_dir, "HashTable", "HBM", table_name, DataName.FEATURE_MAPPING.value)
    data_to_write = dump_data_dict.get(DataName.FEATURE_MAPPING.value)
    valid_len = dump_data_dict.get(DataName.VALID_LEN.value)
    data_to_write = data_to_write[:valid_len * 3]

    attribute = dict()
    attribute[DataAttr.DATATYPE.value] = data_to_write.dtype.name
    attribute[DataName.THRESHOLD.value] = int(dump_data_dict.get(DataName.THRESHOLD.value))
    write_binary_data(target_path, suffix, data_to_write)


def save_offset_data(root_dir, table_name, dump_data_dict, suffix):
    target_path = generate_path(root_dir, "HashTable", "HBM", table_name, DataName.OFFSET.value)
    data_to_write = dump_data_dict.get(DataName.OFFSET.value)
    valid_bucket_num = dump_data_dict.get(DataName.VALID_BUCKET_NUM.value)
    data_to_write = data_to_write[:valid_bucket_num]

    attribute = dict()
    attribute[DataAttr.DATATYPE.value] = data_to_write.dtype.name
    write_binary_data(target_path, suffix, data_to_write)


def save_optimizer_state_data(root_dir, table_name, optimizer_name, dump_optimizer_data, suffix):
    for state_key, state in dump_optimizer_data.items():
        target_path = generate_path(root_dir, table_name, optimizer_name + "_" + state_key)
        data_to_write = state

        attribute = dict()
        attribute[DataAttr.DATATYPE.value] = data_to_write.dtype.name
        attribute[DataAttr.SHAPE.value] = data_to_write.shape
        write_binary_data(target_path, suffix, data_to_write)


def generate_path(*args):
    return os.path.join(*args)


def generate_file_name(suffix):
    return "slice_%d.data" % suffix, "slice_%d.attribute" % suffix


def write_binary_data(writing_path: str, suffix: int, data: np.ndarray):
    try:
        tf.io.gfile.makedirs(writing_path)
    except Exception as err:
        raise RuntimeError(f"make dir {writing_path} for writing data failed!") from err
    data_file, attribute_file = generate_file_name(suffix)
    target_data_dir = os.path.join(writing_path, data_file)
    # append mode of hdfs system supports not well when the file not exists.
    file_mode = "wb" if not tf.io.gfile.exists(target_data_dir) else "ab"
    with tf.io.gfile.GFile(target_data_dir, file_mode) as file:
        data = data.tostring()
        file.write(data)


def read_binary_data(reading_path: str, data_name: str, table_name: str, load_offset) -> dict:
    """
    Read sparse origin data from binary file
    :param reading_path: sparse data path
    :param suffix: suffix of sparse data
    :param data_name: the data type,including embedding, offset, etc.
    :param table_name: the sparse table name
    :return: the sparse data dict
    """
    data_file, attribute_file = "slice.data", "slice.attribute"
    target_data_dir = os.path.join(reading_path, data_file)
    target_attribute_dir = os.path.join(reading_path, attribute_file)
    if not tf.io.gfile.exists(target_data_dir):
        raise FileExistsError(f"Target_data_dir {target_data_dir} does not exist when reading.")
    if not tf.io.gfile.exists(target_attribute_dir):
        raise FileExistsError(f"Target_attribute_dir {target_attribute_dir} does not exist when reading.")

    with tf.io.gfile.GFile(target_attribute_dir, "rb") as fin:
        validate_read_file(target_attribute_dir)
        attributes = fin.read()
        try:
            attributes = np.fromstring(attributes, dtype=np.int64)
        except ValueError as err:
            raise RuntimeError(f"get attributes from file {target_attribute_dir} failed.") from err

    with tf.io.gfile.GFile(target_data_dir, "rb") as file:
        validate_read_file(target_data_dir)
        if check_file_system_is_hdfs(target_data_dir):
            data_to_restore = file.read()
            data_to_restore = np.fromstring(data_to_restore, dtype=np.float32)
        else:
            data_to_restore = np.fromfile(target_data_dir, dtype=np.float32)
    try:
        embedding_size = list(attributes)[1]
    except Exception as err:
        raise RuntimeError(f"get embedding size from attribute file {target_attribute_dir} failed.") from err

    data_to_restore = data_to_restore.reshape(-1, embedding_size)
    if load_offset:
        data_to_restore = data_to_restore[load_offset, :]
    data_shape = list(data_to_restore.shape)
    table_instance = ConfigInitializer.get_instance().sparse_embed_config.get_table_instance_by_name(table_name)
    current_data_shape = [table_instance.slice_device_vocabulary_size, table_instance.emb_size]
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
    if not check_file_system_is_hdfs(read_file_path):
        file_validator.check_not_soft_link()
        file_validator.check_user_group()
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
        data_to_restore = data_to_restore[:restore_vocab_size, :]

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


def _fill_placeholder_for_optimizer(optimizer_state_placeholder_dict_group: dict, reading_path: str,
                                    restore_feed_dict: dict, table_name: str, load_offset: list):
    """
    给优化器填充加载的数据.

    Args:
        optimizer_state_placeholder_dict_group: 待填充优化器的字典
        reading_path: 读取路径
        restore_feed_dict: session run的feed dict
        suffix: rank id
        table_name: 表名

    Returns: None
    """
    for optimizer_name, optimizer_state_placeholder_dict in optimizer_state_placeholder_dict_group.items():
        for state_key in optimizer_state_placeholder_dict:
            fill_placeholder(reading_path=reading_path,
                             placeholder_dict=optimizer_state_placeholder_dict,
                             feed_dict=restore_feed_dict,
                             name_descriptor=NameDescriptor(table_name, state_key, optimizer_name=optimizer_name),
                             load_offset=load_offset)


def get_merge_type_list(table_dir: str):
    """
    获取表路径下需要合入的数据类型list

    Args:
        table_dir: 稀疏表存储路径

    Returns: None
    """
    merge_type_list = []
    for item in tf.io.gfile.listdir(table_dir):
        if tf.io.gfile.isdir(os.path.join(table_dir, item)):
            merge_type_list.append(item)
    return merge_type_list


def merge_multi_files(upper_dir: str):
    """
    合并多个二进制文件

    Args:
        upper_dir: 合并路径

    Returns: None
    """
    data_files = [file for file in tf.io.gfile.listdir(upper_dir) if file.startswith("slice_")]
    data_files = sorted(data_files, key=os.path.basename)
    outfile_path = os.path.join(upper_dir, "slice.data")
    with tf.io.gfile.GFile(outfile_path, "wb") as outfile:
        for file in data_files:
            file_dir = os.path.join(upper_dir, file)
            with tf.io.gfile.GFile(file_dir, "rb") as file:
                outfile.write(file.read())
            tf.io.gfile.remove(file_dir)


def set_optimizer_info(optimizer: CustomizedOptimizer, table_name: str):
    """
    往host侧传递稀疏表的优化器名称信息

    Args:
        optimizer_dict: 优化器字典
        table_name: 表名

    Returns: None
    """
    from mxrec_pybind import OptimizerInfo
    optim_info = OptimizerInfo(optimizer.optimizer_type, optimizer.optim_param_list)
    ConfigInitializer.get_instance().hybrid_manager_config.set_optim_info(table_name, optim_info)