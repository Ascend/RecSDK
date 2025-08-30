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
import os
import threading
import glob
import struct
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Union, Generator, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.util import compat

from rec_sdk_common.log import logger
from rec_sdk_common.communication.hccl.hccl_info import (
    get_rank_id,
    get_rank_size,
    get_local_rank_size,
)
from rec_sdk_common.constants.constants import (
    FileParams,
    DeviceType,
    ValidatorParams,
)
from rec_sdk_common.util.tf_adapter import npu_ops
from rec_sdk_common.validator.validator import (
    DirectoryValidator,
    para_checker_decorator,
    ClassValidator,
    IntValidator,
    OptionalStringValidator,
)
from mx_rec.validator.validator import FileValidator
from mx_rec.constants.constants import (
    DataName,
    DataAttr,
    HDFS_FILE_PREFIX,
    TRAIN_CHANNEL_ID,
    BASE_MODEL,
    DELTA_MODEL,
    SAVE_DIR_MODE,
    SAVE_FILE_MODE,
    SAVE_FILE_FLAG,
    FLOAT32_BYTES,
    UINT64_BYTES,
    UINT32_BYTES,
)
from mx_rec.saver.constants import FILE_BUFFER_SIZE
from mx_rec.saver.utils import check_files_in_directories, get_optimizer_dict_by_table_name
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.perf import performance

from mx_rec.util.global_env_conf import global_env
from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.graph.merge_lookup import do_merge_lookup
from mx_rec.graph.modifier import replace_anchor_for_ddr_ssd, change_ext_emb_size_by_opt

SAVE_SPARSE_PATH_PREFIX = "sparse"
SAVE_DELTA_SPARSE_PATH_PREFIX = "delta-sparse"
GLOBAL_STEP_STR = "global_step"
SSD_SAVE_PATH_PREFIX = "ssd_sparse_model_rank_"
SSD_SAVE_FILE_PATTERNS = ["*.meta.*"]
SSD_DATA_FILE_MIN_SIZE = 0


@dataclass
class KeyInfo:
    offset: int
    emb_size: int
    embedding: List[float]


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
    @para_checker_decorator(check_option_list=[
        ("var_list", ClassValidator, {"classes": (list, type(None))}),
        ("max_to_keep", IntValidator, {"min_value": 0, "max_value": ValidatorParams.MAX_INT32.value}, ["check_value"]),
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
        self.rank_size = get_rank_size()
        self.save_op_dict = defaultdict(dict)
        self.restore_fetch_dict = defaultdict()
        self.placeholder_dict = defaultdict(dict)
        self._last_checkponts = []
        self.config_instance = ConfigInitializer.get_instance()
        self.build()
        self.warm_start_tables = warm_start_tables

    @staticmethod
    def _check_file_system_is_valid(save_path):
        if not check_file_system_is_valid(save_path):
            raise ValueError("the path to save sparse embedding table data belong to invalid file system, "
                            "only local file system and hdfs file system supported. ")

    def build(self):
        # If the 'export_saved_model' interface is called, the graph modification is required.
        self._modify_graph_for_export_model()

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
    def save(self, sess, save_path="model", global_step=None, save_delta=False):
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
        :param save_delta: check if save delta model in incremental checkpoint pattern
        :return: None
        """
        logger.debug("======== Start saving for rank id %s ========", self.rank_id)
        self._check_file_system_is_valid(save_path)

        save_path = save_path if save_path else self._prefix_name
        directory, base_name = os.path.split(save_path)
        save_path_prefix = SAVE_SPARSE_PATH_PREFIX if not save_delta else SAVE_DELTA_SPARSE_PATH_PREFIX

        if global_step:
            if not isinstance(global_step, compat.integral_types):
                global_step = int(sess.run(global_step))
            ckpt_name = f"{save_path_prefix}-{base_name}-{global_step}"
        else:
            ckpt_name = f"{save_path_prefix}-{base_name}"

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
                if check_file_system_is_hdfs(saving_path):
                    tf.io.gfile.makedirs(saving_path)
                else:
                    os.makedirs(saving_path, SAVE_DIR_MODE, exist_ok=True)
            except Exception as err:
                raise RuntimeError(f"make dir {saving_path} for saving sparse table failed!") from err
            logger.info("Saving_path '%s' has been made.", saving_path)

        self._save(sess, saving_path, save_delta)
        if self.max_to_keep:
            self._last_checkponts.append(saving_path)
            if len(self._last_checkponts) > self.max_to_keep:
                logger.info("checkpoints num %d > max_to_keep %d delete %s",
                            len(self._last_checkponts), self.max_to_keep,
                            self._last_checkponts[0])
                checkpoint_path = self._last_checkponts.pop(0)
                file_validator = FileValidator("checkpoint_path", checkpoint_path)
                if not check_file_system_is_hdfs(checkpoint_path):
                    file_validator.check_not_soft_link()
                file_validator.check()
                try:
                    tf.io.gfile.rmtree(checkpoint_path)
                except tf.errors.NotFoundError as e:
                    logger.warning("oldest checkpoint file is not exist, maybe it has been deleted.")

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        comm.Barrier()
        if should_write_data(rank, saving_path):
            table_list = self.save_op_dict.keys()
            for table_name in table_list:
                self.merge_sparse_file(saving_path, table_name)
        comm.Barrier()

        logger.info("sparse model was saved in dir '%s' .", saving_path)
        logger.info("======== Saving finished for rank id %s ========", self.rank_id)

    @performance("Restore")
    def restore(self, sess, reading_path, warm_start_tables=None, model_type="base"):
        logger.debug("======== Start restoring ========")
        if not check_file_system_is_valid(reading_path):
            raise ValueError("the path to save sparse embedding table data belong to invalid file system, "
                             "only local file system and hdfs file system supported. ")

        directory, base_name = os.path.split(reading_path)
        if model_type == BASE_MODEL:
            ckpt_name = f"{SAVE_SPARSE_PATH_PREFIX}-{base_name}"
        else:
            ckpt_name = f"tmp-{SAVE_SPARSE_PATH_PREFIX}-{base_name}"

        reading_path = os.path.join(directory, ckpt_name)
        if not tf.io.gfile.exists(reading_path):
            raise FileExistsError(f"Given dir {reading_path} does not exist, please double check.")

        self._restore(sess, reading_path, warm_start_tables)
        if model_type == DELTA_MODEL:
            file_validator = FileValidator("reading_path", reading_path)
            if not check_file_system_is_hdfs(reading_path):
                file_validator.check_not_soft_link()
            file_validator.check()
            try:
                tf.io.gfile.rmtree(reading_path)
            except tf.errors.NotFoundError:
                logger.warning("%s is not exists, maybe it has been deleted.", reading_path)
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
        logger.info("Start merge sparse file, merge dir:%s, table_name:%s.", root_dir, table_name)
        table_dir = os.path.join(root_dir, table_name)
        table_instance = ConfigInitializer.get_instance().sparse_embed_config.get_table_instance_by_name(table_name)
        merge_type_list = get_merge_type_list(table_dir)

        if not check_file_system_is_hdfs(root_dir):
            dir_validator = DirectoryValidator("root_dir", root_dir)
            dir_validator.check_not_soft_link()
            try:
                dir_validator.check()
            except ValueError as e:
                raise ValueError(f"root_dir:{root_dir} can't be soft link") from e

        for data_type in merge_type_list:
            upper_dir = os.path.join(table_dir, data_type)
            if table_instance.is_dp:
                # All card embeddings are the same in DP mode, only one copy needs to be kept and no merging is needed.
                rename_file_and_remove_others(upper_dir)
            else:
                # Different card embeddings in MP mode are different and need to be merged.
                merge_multi_files(upper_dir)
            outfile_path = os.path.join(upper_dir, "slice.data")
            file_size = tf.io.gfile.stat(outfile_path).length
            if data_type == "key":
                attribute = np.array([file_size / 8, 8])
            else:
                attribute = np.array([file_size / 4 / table_instance.emb_size, table_instance.emb_size, 4])

            attribute = attribute.astype(np.int64)
            attribute_dir = os.path.join(upper_dir, "slice.attribute")
            if check_file_system_is_hdfs(attribute_dir):
                with tf.io.gfile.GFile(attribute_dir, "wb") as file:
                    attribute = attribute.tostring()
                    file.write(attribute)
            else:
                with os.fdopen(os.open(attribute_dir, SAVE_FILE_FLAG, SAVE_FILE_MODE), "wb") as file:
                    file.write(attribute.tostring())

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

    @performance("_save")
    def _save(self, sess, root_dir, save_delta):
        for table_name in self.save_op_dict:
            optimizer_instance = ConfigInitializer.get_instance().optimizer_config.optimizer_instance
            if optimizer_instance:
                set_optimizer_info(optimizer_instance, table_name)

        table_instance0 = self.config_instance.sparse_embed_config.get_table_instance(self.var_list[0])
        if table_instance0.is_hbm:
            self._save_hbm(sess, root_dir, save_delta)
        else:
            self._save_ddr(sess, root_dir, save_delta)
        logger.debug(f"Host data was saved.")

    def _save_hbm(self, sess, root_dir, save_delta):
        self.config_instance.hybrid_manager_config.save_host_data(root_dir, save_delta)
        if self.config_instance.use_dynamic_expansion:
            # Data related to dynamic expansion needs to be saved only on the host side.
            return

        result = self.save_op_dict
        threads = []
        for table_name in result.keys():
            table_instance = ConfigInitializer.get_instance().sparse_embed_config.get_table_instance_by_name(table_name)
            # In DP mode, only one copy of embedding needs to be saved.
            if not should_save_sparse_embedding(table_instance.is_dp, root_dir):
                continue
            thread = SaveModelThread(self, sess, result, root_dir, table_name)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def _save_ddr(self, sess, root_dir, save_delta):
        # start host's threads for syncing data between device and host
        self.config_instance.hybrid_manager_config.start_sync_thread()

        # let hybridMgmt send swap_out offset from host
        self.config_instance.hybrid_manager_config.fetch_device_emb()

        # In DDR mode, within the save process, the graph has been fixed and cannot execute the get_next op.
        # The _unsafe_unfinalize operation can modify the state of the graph being fixed.
        sess.graph._unsafe_unfinalize()

        for var in self.var_list:
            table_instance = self.config_instance.sparse_embed_config.get_table_instance(var)
            table_name = table_instance.table_name

            # receive syncing info from host
            use_static = ConfigInitializer.get_instance().use_static
            max_lookup_vec_size = None
            if use_static:
                max_lookup_vec_size = table_instance.send_count * self.rank_size
            swap_out_pos, swap_out_len, sync_remain_flag = npu_ops.gen_npu_ops.get_next(
                output_types=[tf.int32, tf.int32, tf.bool],
                output_shapes=[[max_lookup_vec_size], [], []],
                channel_name=f"{table_name}_save_h2d_{TRAIN_CHANNEL_ID}")
            if use_static:
                swap_out_pos = swap_out_pos[:swap_out_len]

            table = [var]
            optimizer = get_optimizer_dict_by_table_name(table_name)
            if optimizer is not None:
                for slots in optimizer.values():
                    table += list(slots.values())
            swap_outs = [tf.gather(one_table, swap_out_pos) for one_table in table]
            swap_out = tf.concat(swap_outs, axis=1)
            channel_name = f"{table_name}_save_d2h_{TRAIN_CHANNEL_ID}"
            logger.info("Channel %s was built for op swap_out_op.", channel_name)
            swap_out_op = npu_ops.outfeed_enqueue_op(channel_name=channel_name, inputs=[swap_out])

            # send embedding to host
            sync_cnt = 0
            is_sync_remain = True
            while is_sync_remain:
                _, is_sync_remain = sess.run([swap_out_op, sync_remain_flag])
                sync_cnt += 1
                logger.info("Sending embedding to host, table:%s, sync_cnt:%d, is_sync_remain:%d.",
                            table_name, sync_cnt, is_sync_remain)
            logger.info("Finish sending embedding to host, table:%s.", table_name)

        self._save_host_data(root_dir, save_delta, sess)

    def _get_valid_dict_data(self, dump_data_dict, table_name):
        host_data = self.config_instance.hybrid_manager_config.get_host_data(table_name)
        offset = list(host_data)

        get_valid_dict_data_from_host_offset(dump_data_dict, offset)

    def _build_save(self):
        for var in self.var_list:
            if global_env.tf_device == DeviceType.NPU.value and "merged" not in var.name:
                continue

            table_instance = self.config_instance.sparse_embed_config.get_table_instance(var)
            table_name = table_instance.table_name
            with tf.compat.v1.variable_scope(table_name):
                sub_dict = self.save_op_dict[table_name]
                sub_dict[DataName.EMBEDDING.value] = var
                optimizer = get_optimizer_dict_by_table_name(table_name)
                if optimizer:
                    sub_dict["optimizer"] = optimizer

    def _build_restore(self):
        for var in self.var_list:
            if global_env.tf_device == DeviceType.NPU.value and "merged" not in var.name:
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
                optimizer = get_optimizer_dict_by_table_name(table_instance.table_name)
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

        table_instance0 = self.config_instance.sparse_embed_config.get_table_instance(self.var_list[0])
        if not table_instance0.is_hbm:
            return

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

    def _modify_graph_for_export_model(self):
        experimental_mode = self.config_instance.train_params_config.experimental_mode
        if experimental_mode is None or not self.config_instance.modify_graph:
            return

        is_training = experimental_mode == tf.compat.v1.estimator.ModeKeys.TRAIN
        do_merge_lookup(is_train=is_training)

        slot_num = 0
        optimizer_ins = self.config_instance.optimizer_config.optimizer_instance
        if optimizer_ins is not None:
            change_ext_emb_size_by_opt(optimizer_ins)
            slot_num = optimizer_ins.slot_num
        channel_id = 0 if is_training else 1
        replace_anchor_for_ddr_ssd(tf.compat.v1.get_default_graph(), slot_num, channel_id)

    def _save_host_data(self, root_dir: str, save_delta: bool, sess: tf.compat.v1.Session):
        if ConfigInitializer.get_instance().train_params_config.experimental_mode is None:
            self.config_instance.hybrid_manager_config.save_host_data(root_dir, save_delta)
            return

        # In the export saved model mode, if the SSD ckpt has already been saved, it will not be saved again.
        all_saved = True
        global_step = sess.run(tf.compat.v1.train.get_global_step())
        if global_step is None:
            raise ValueError("the global step cannot be None")
        ssd_save_file_patterns = [pattern + str(global_step) for pattern in SSD_SAVE_FILE_PATTERNS]
        logger.info("The patterns of the ssd file is: %s.", ssd_save_file_patterns)
        for var in self.var_list:
            table_instance = self.config_instance.sparse_embed_config.get_table_instance(var)
            if table_instance.ssd_vocabulary_size == 0:
                continue
            for ssd_path in table_instance.ssd_data_path:
                data_path = os.path.join(ssd_path, SSD_SAVE_PATH_PREFIX + "*")
                is_exists = check_files_in_directories(data_path, ssd_save_file_patterns)
                all_saved &= is_exists

        is_save_l3_storage = not all_saved
        logger.info("The `is_save_l3_storage` is %s.", is_save_l3_storage)
        self.config_instance.hybrid_manager_config.save_host_data(root_dir, save_delta, is_save_l3_storage)


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
        if check_file_system_is_hdfs(writing_path):
            tf.io.gfile.makedirs(writing_path)
        else:
            os.makedirs(writing_path, SAVE_DIR_MODE, exist_ok=True)
    except Exception as err:
        raise RuntimeError(f"make dir {writing_path} for writing data failed!") from err
    data_file, _ = generate_file_name(suffix)
    target_data_dir = os.path.join(writing_path, data_file)
    # append mode of hdfs system supports not well when the file not exists.
    write_mode = "wb" if not tf.io.gfile.exists(target_data_dir) else "ab"
    if check_file_system_is_hdfs(target_data_dir):
        with tf.io.gfile.GFile(target_data_dir, write_mode) as file:
            data = data.tostring()
            file.write(data)
    else:
        with os.fdopen(os.open(target_data_dir, SAVE_FILE_FLAG, SAVE_FILE_MODE), write_mode) as file:
            file.write(data.tostring())


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

    attributes = read_attribute_file(target_attribute_dir)
    data_to_restore = read_data_file(target_data_dir)
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
    file_validator.check_file_size(FileParams.MAX_FILE_SIZE.value, FileParams.MIN_SIZE.value)
    if not check_file_system_is_hdfs(read_file_path):
        file_validator.check_not_soft_link()
        file_validator.check_user_group()
        file_validator.check_file_mode()
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


def get_hdfs_safemode_status():
    try:
        result = subprocess.run(["/usr/local/hadoop-3.3.6/bin/hdfs", "dfsadmin", "-safemode", "get"],
                                capture_output=True, text=True, check=True, shell=False)
        output = result.stdout.strip()
        logger.info(f"HDFS safemode status:{output}.")
        return output
    except FileNotFoundError as err:
        logger.warning(f"Command 'hdfs' not found. Ignore this exception in non-HDFS scenario. Please ensure Hadoop"
                       f"is installed and 'hdfs' is in your PATH in HDFS scenario.")
    except Exception as err:
        logger.warning(f"Failed to get HDFS safemode status:{err}. Ignore this exception in non-HDFS scenario.")

    return ""


def check_hdfs_safemode_status():
    status = get_hdfs_safemode_status()
    if "Safe mode is ON" in status:
        raise RuntimeError(
            "The current HDFS is in safe mode. It is recommended to check the server disk space and the usage of HDFS "
            "resources. Use 'hdfs dfsadmin -safemode leave' to set Safe mode is OFF, and then run again."
        )


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
    if check_file_system_is_hdfs(upper_dir):
        merge_hdfs_file(upper_dir)
        return
    merge_local_file(upper_dir)


def merge_hdfs_file(upper_dir: str):
    data_files = [file for file in tf.io.gfile.listdir(upper_dir) if file.startswith("slice_")]
    data_files = sorted(data_files, key=os.path.basename)
    outfile_path = os.path.join(upper_dir, "slice.data")
    outfile = tf.io.gfile.GFile(outfile_path, "wb")
    for file in data_files:
        file_dir = os.path.join(upper_dir, file)
        with tf.io.gfile.GFile(file_dir, "rb") as file:
            outfile.write(file.read())
        tf.io.gfile.remove(file_dir)
    outfile.close()


def merge_local_file(upper_dir: str) -> None:
    data_files = [file for file in os.listdir(upper_dir) if file.startswith("slice_")]
    data_files = sorted(data_files, key=os.path.basename)
    outfile_path = os.path.join(upper_dir, "slice.data")
    outfile = os.fdopen(os.open(outfile_path, SAVE_FILE_FLAG, SAVE_FILE_MODE), "wb", buffering=FILE_BUFFER_SIZE)
    for file in data_files:
        file_dir = os.path.join(upper_dir, file)
        if os.path.getsize(file_dir) == 0:
            os.remove(file_dir)
            continue
        f = open(file_dir, "rb", buffering=FILE_BUFFER_SIZE)
        while True:
            data = f.read(FILE_BUFFER_SIZE)
            if not data:
                break
            outfile.write(data)
        f.close()
        os.remove(file_dir)
    outfile.close()


def rename_file_and_remove_others(upper_dir: str):
    """
    In DP mode, the embeddings of all cards are the same, and there is no need to merge the saved files.

    Args:
        upper_dir: model save path.

    Returns: None

    """

    data_files = [
        file 
        for file in tf.io.gfile.listdir(upper_dir) 
        if file.startswith("slice_")
    ]
    if not data_files:
        raise RuntimeError(
            f"rename and remove file failed, slice*.data do not exist in {upper_dir}."
        )

    # Rename: slice_0.data -> slice.data.
    data_files = sorted(data_files, key=os.path.basename)
    data_file = os.path.join(upper_dir, data_files[0])
    output_file = os.path.join(upper_dir, "slice.data")
    tf.io.gfile.rename(data_file, output_file, overwrite=True)

    # Remove: slice_1.data ... slice_x.data.
    for file in data_files[1:]:
        file_dir = os.path.join(upper_dir, file)
        if tf.io.gfile.exists(file_dir):
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


def should_write_data(rank_id: int, save_path: str) -> bool:
    # When using hdfs filesystem, only the rank0 process execute write data operation, assuming use same hdfs path in
    #   multi-machine.
    # When using local filesystem, the process which `rank_id % local_rank_size == 0` execute write data operation.
    # When using hdfs filesystem, and use different hdfs path to save data, should modify check condition
    #    as same as local filesystem.
    is_hdfs = check_file_system_is_hdfs(save_path)
    local_rank_size = get_local_rank_size()
    return rank_id == 0 if is_hdfs else rank_id % local_rank_size == 0


def update_model_index(save_dir: str, model_index: Dict[str, Union[str, int]]):
    model_index_file = os.path.join(save_dir, "model_index.json")
    if not tf.io.gfile.exists(model_index_file):
        model_index_list = []
    else:
        with tf.io.gfile.GFile(model_index_file, "r") as f:
            model_index_list = json.load(f)
    model_index_list.append(model_index)

    if check_file_system_is_hdfs(model_index_file):
        with tf.io.gfile.GFile(model_index_file, "w") as f:
            json.dump(model_index_list, f, ensure_ascii=False, separators=(",", ": "), indent=4)
    else:
        dir_validator = DirectoryValidator("save_dir", save_dir)
        dir_validator.check_not_soft_link()
        try:
            dir_validator.check()
        except ValueError as e:
            raise ValueError(f"save_dir:{save_dir} can't be soft link") from e
        with os.fdopen(os.open(model_index_file, SAVE_FILE_FLAG, SAVE_FILE_MODE), "w") as f:
            json.dump(model_index_list, f, ensure_ascii=False, separators=(",", ": "), indent=4)


def write_delta_export_time_ms(save_dir: str, delta_export_time_ms: dict):
    delta_export_time_ms_file = os.path.join(save_dir, "delta_export_time_ms.json")
    if check_file_system_is_hdfs(delta_export_time_ms_file):
        with tf.io.gfile.GFile(delta_export_time_ms_file, "w") as f:
            json.dump(delta_export_time_ms, f, indent=4)
    else:
        dir_validator = DirectoryValidator("save_dir", save_dir)
        dir_validator.check_not_soft_link()
        try:
            dir_validator.check()
        except ValueError as e:
            raise ValueError(f"save_dir:{save_dir} can't be soft link") from e
        with os.fdopen(os.open(delta_export_time_ms_file, SAVE_FILE_FLAG, SAVE_FILE_MODE), "w") as f:
            json.dump(delta_export_time_ms, f, indent=4)


def get_model_type_by_version(save_dir: str, model_version: str):
    model_index_file = os.path.join(save_dir, "model_index.json")
    validate_read_file(model_index_file)
    with tf.io.gfile.GFile(model_index_file, "r") as f:
        model_index_list = json.load(f)

    model_type = None
    for model_index in model_index_list:
        try:
            model_version_int = int(model_version)
        except ValueError as err:
            raise ValueError("Can not transfer %s to integer.", model_version) from err
        if model_index[GLOBAL_STEP_STR] == model_version_int:
            model_type = model_index["type"]
            return model_type
    return model_type


def get_base_and_delta_models(save_dir: str, model_version: str):
    model_index_file = os.path.join(save_dir, "model_index.json")
    validate_read_file(model_index_file)
    with tf.io.gfile.GFile(model_index_file, "r") as f:
        model_index_list = json.load(f)
        model_index_list.reverse()

    base_model = ""
    delta_models = []
    found_delta_model = False
    for model_index in model_index_list:
        if model_index[GLOBAL_STEP_STR] == int(model_version):
            delta_models.append(model_version)
            found_delta_model = True
            continue
        if not found_delta_model:
            continue
        if model_index["type"] == DELTA_MODEL:
            delta_models.append(str(model_index[GLOBAL_STEP_STR]))
        else:
            base_model = str(model_index[GLOBAL_STEP_STR])
            break
    delta_models.reverse()
    return base_model, delta_models


def read_base_delta_and_write(save_dir: str, base_model: str, delta_models: list):
    table_name_set = ConfigInitializer.get_instance().sparse_embed_config.table_name_set
    optimizer = ConfigInitializer.get_instance().optimizer_config.optimizer_instance
    optimizer_type, optim_param_list, optimizer_param_name_list = None, None, []
    if optimizer:
        optimizer_type, optim_param_list = optimizer.optimizer_type, optimizer.optim_param_list
        optimizer_param_name_list = [f"{optimizer_type}_{optim_param}" for optim_param in optim_param_list]
    # restore base model's optimizer
    base_optimizer = None if not optimizer else get_base_optimizer(save_dir, table_name_set, base_model)
    # restore base model's key and embedding
    base_table = get_base_key_embedding(save_dir, table_name_set, base_model)

    # read delta model and update to base model one by one
    for delta_model in delta_models:
        delta_model_path = os.path.join(save_dir, f"{SAVE_DELTA_SPARSE_PATH_PREFIX}-model.ckpt-{delta_model}")
        delta_optimizer_params = {}
        for table_name in table_name_set:
            delta_key_data, delta_embedding_data = get_table_key_emb(delta_model_path, table_name)
            for optimizer_param_name in optimizer_param_name_list:
                delta_optimizer_params[optimizer_param_name] = \
                    get_table_optimizer_param(delta_model_path, table_name, optimizer_param_name)
            # update base table
            len_of_delta_table = len(delta_key_data)
            for i in range(len_of_delta_table):
                key = base_table[table_name]["key"]
                embed = base_table[table_name]["embedding"]
                idx = None
                k, v = delta_key_data[i], delta_embedding_data[i]
                if k in key:
                    idx = np.where(key == k)[0][0]
                    embed[idx] = v
                else:
                    base_table[table_name]["key"] = np.append(key, k)
                    base_table[table_name]["embedding"] = np.vstack([embed, v])
                if delta_optimizer_params:
                    for optimizer_param_name in optimizer_param_name_list:
                        tmp = delta_optimizer_params[optimizer_param_name][i]
                        optimizer_param = base_optimizer[table_name][optimizer_param_name]
                        if idx is not None:
                            optimizer_param[idx] = tmp
                        else:
                            base_optimizer[table_name][optimizer_param_name] = np.vstack([optimizer_param, tmp])

    # write base model data to file
    tmp_path = f"{save_dir}/tmp-{SAVE_SPARSE_PATH_PREFIX}-model.ckpt-{delta_models[-1]}"
    write_base_table_to_file(tmp_path, base_table)
    if optimizer:
        write_base_table_to_file(tmp_path, base_optimizer)
    return tmp_path


def get_table_key_emb(model_path: str, table_name: str):
    key_path = os.path.join(model_path, table_name, "key")
    data_file = os.path.join(key_path, "slice.data")
    key_data = read_attribute_file(data_file)
    embedding_path = os.path.join(model_path, table_name, "embedding")
    attribute_file = os.path.join(embedding_path, "slice.attribute")
    embed_attr = read_attribute_file(attribute_file)
    data_file = os.path.join(embedding_path, "slice.data")
    embedding_data = read_data_file(data_file).reshape(embed_attr[:-1])
    return key_data, embedding_data


def write_base_table_to_file(save_dir: str, base_table: dict):
    if not check_file_system_is_hdfs(save_dir):
        dir_validator = DirectoryValidator("save_dir", save_dir)
        dir_validator.check_not_soft_link()
        try:
            dir_validator.check()
        except ValueError as e:
            raise ValueError(f"save_dir:{save_dir} can't be soft link") from e

    for table_name, table in base_table.items():
        for k, v in table.items():
            writing_path = os.path.join(save_dir, table_name, k)
            try:
                if check_file_system_is_hdfs(writing_path):
                    tf.io.gfile.makedirs(writing_path)
                else:
                    os.makedirs(writing_path, SAVE_DIR_MODE, exist_ok=True)
            except Exception as err:
                raise RuntimeError(f"Create dir {writing_path} for writing data failed!") from err

            data_file, attribute_file = "slice.data", "slice.attribute"
            target_data_dir = os.path.join(writing_path, data_file)
            target_attribute_dir = os.path.join(writing_path, attribute_file)
            write_bytes = 8 if k == "key" else 4
            attribute = np.append(v.shape, write_bytes)

            if check_file_system_is_hdfs(writing_path):
                with tf.io.gfile.GFile(target_attribute_dir, "wb") as file:
                    file.write(attribute.tostring())
                with tf.io.gfile.GFile(target_data_dir, "wb") as file:
                    file.write(v.tostring())
            else:
                with os.fdopen(os.open(target_attribute_dir, SAVE_FILE_FLAG, SAVE_FILE_MODE), "wb") as file:
                    file.write(attribute.tostring())
                with os.fdopen(os.open(target_data_dir, SAVE_FILE_FLAG, SAVE_FILE_MODE), "wb") as file:
                    file.write(v.tostring())


def clear_delta_models(save_dir: str):
    delta_directories = glob.glob(os.path.join(save_dir, 'delta-sparse*'))
    for delta_dir in delta_directories:
        file_validator = FileValidator("delta_dir", delta_dir)
        if not check_file_system_is_hdfs(delta_dir):
            file_validator.check_not_soft_link()
        file_validator.check()
        try:
            tf.io.gfile.rmtree(delta_dir)
        except tf.errors.NotFoundError:
            logger.warning("%s is not exists, maybe it has been deleted.", delta_dir)


def get_table_optimizer_param(model_path: str, table_name: str, optimizer_param_name: str):
    attribute_file = os.path.join(model_path, table_name, optimizer_param_name, "slice.attribute")
    data_file = os.path.join(model_path, table_name, optimizer_param_name, "slice.data")
    attribute = read_attribute_file(attribute_file)
    data_to_restore = read_data_file(data_file).reshape(attribute[:-1])
    return data_to_restore


def read_attribute_file(target_attribute_dir: str):
    with tf.io.gfile.GFile(target_attribute_dir, "rb") as fin:
        validate_read_file(target_attribute_dir)
        attributes = fin.read()
        try:
            attributes = np.fromstring(attributes, dtype=np.int64)
        except ValueError as err:
            raise RuntimeError(f"get attributes from file {target_attribute_dir} failed.") from err
    return attributes


def read_data_file(target_data_dir: str):
    with tf.io.gfile.GFile(target_data_dir, "rb") as file:
        validate_read_file(target_data_dir)
        if check_file_system_is_hdfs(target_data_dir):
            data_to_restore = file.read()
            data_to_restore = np.fromstring(data_to_restore, dtype=np.float32)
        else:
            data_to_restore = np.fromfile(target_data_dir, dtype=np.float32)
    return data_to_restore


def get_base_optimizer(save_dir: str, table_name_set: set, base_model: str):
    optimizer = ConfigInitializer.get_instance().optimizer_config.optimizer_instance
    optimizer_type = optimizer.optimizer_type
    optim_param_list = optimizer.optim_param_list
    base_optimizer = {}
    # if optimizer's params exist, then restore them; otherwise no need to restore
    if optim_param_list:
        optimizer_status_name_list = [
            f"{optimizer_type}_{optim_param}" 
            for optim_param in optim_param_list
        ]
        base_optimizer = {
            table_name: {optimizer_status_name: None} 
            for table_name in table_name_set
            for optimizer_status_name in optimizer_status_name_list
        }

        base_model_path = os.path.join(save_dir, f"{SAVE_SPARSE_PATH_PREFIX}-model.ckpt-{base_model}")
        for table_name in table_name_set:
            for optimizer_status_name in optimizer_status_name_list:
                optimier_data = get_table_optimizer_param(base_model_path, table_name, optimizer_status_name)
                base_optimizer[table_name][optimizer_status_name] = optimier_data
    return base_optimizer


def get_base_key_embedding(save_dir: str, table_name_set: set, base_model: str):
    base_table = {table_name: {"key": None, "embedding": None} for table_name in table_name_set}
    base_model_path = os.path.join(save_dir, f"{SAVE_SPARSE_PATH_PREFIX}-model.ckpt-{base_model}")
    for table_name in table_name_set:
        key_data, embedding_data = get_table_key_emb(base_model_path, table_name)
        base_table[table_name]["key"] = key_data
        base_table[table_name]["embedding"] = embedding_data
    return base_table


def should_save_sparse_embedding(is_dp: bool, save_path: str) -> bool:
    """
    Whether embeddings need to be saved for each card.

    Args:
        is_dp: switch whether to enable dp.
        save_path: model save path.

    Returns:
        bool: whether to save.

    """

    if not is_dp:
        return True

    is_hdfs = check_file_system_is_hdfs(save_path)
    # In hdfs, all servers only need to save one copy in total.
    if is_hdfs and get_rank_id() % get_rank_size() == 0:
        return True
    # Without hdfs, each server needs to keep a copy.
    if not is_hdfs and get_rank_id() % get_local_rank_size() == 0:
        return True

    return False


def read_base_delta_and_write_for_ssd(save_dir: str, base_model: str, delta_models: List[str], rank: int) -> None:
    """
    read base model and delta models for incremental restore
    :param save_dir: model save dir
    :param base_model: full model step
    :param delta_models: incremental models
    :param rank: process id
    :return:
    """
    current_ssd_dir = os.path.join(os.path.dirname(save_dir), SSD_SAVE_PATH_PREFIX + str(rank))
    file_validator = FileValidator("current_ssd_dir", current_ssd_dir)
    if not check_file_system_is_hdfs(current_ssd_dir):
        file_validator.check_not_soft_link()
    file_validator.check()

    table_name_set = ConfigInitializer.get_instance().sparse_embed_config.table_name_set
    for table_name in table_name_set:
        key_info_map = defaultdict(list)
        # Read base model's meta file and get file count list.
        file_ids = _read_table_meta_data(current_ssd_dir, table_name, base_model)
        for fid in file_ids:
            _read_key_offset_and_embedding(os.path.join(current_ssd_dir, table_name), base_model, fid, False,
                                           key_info_map)
        # Read delta model's meta file and get file count list.
        for delta_model in delta_models:
            file_ids = _read_table_meta_data(current_ssd_dir, table_name, delta_model)
            for fid in file_ids:
                _read_key_offset_and_embedding(os.path.join(current_ssd_dir, table_name), delta_model, fid, True,
                                               key_info_map)
        # Write key_info_map into new files.
        _write_ssd_meta_and_data(current_ssd_dir, table_name, file_ids[0], delta_models[-1], key_info_map)


def _read_table_meta_data(current_ssd_dir: str, table_name: str, model: str) -> List[int]:
    """
    read table meta data for SSD
    :param current_ssd_dir: ssd model saved dir
    :param table_name: table name
    :param model: step for saving model
    :return: [table_name, [fileID]]
    """
    table_meta_file = os.path.join(current_ssd_dir, table_name, table_name + ".meta." + model)
    with tf.io.gfile.GFile(table_meta_file, 'rb') as file:
        validate_read_file(table_meta_file)
        # Read name_size(4bytes uint32_t).
        name_size_data = file.read(UINT32_BYTES)
        if len(name_size_data) < UINT32_BYTES:
            raise EOFError("End of file reached before reading name size, file maybe broken.")

        name_size = struct.unpack('I', name_size_data)[0]

        # Read name(name_size bytes).
        name_data = file.read(name_size)
        if len(name_data) < name_size:
            raise EOFError("End of file reached before reading name, file maybe broken.")

        # Read fileCnt(8bytes uint64_t).
        file_cnt_data = file.read(UINT64_BYTES)
        if len(file_cnt_data) < UINT64_BYTES:
            raise EOFError("End of file reached before reading file count, file maybe broken.")

        file_cnt = struct.unpack('Q', file_cnt_data)[0]

        # Read fileCnt fileID(every 8bytes, uint64_t).
        file_ids = []
        for _ in range(file_cnt):
            fid_data = file.read(UINT64_BYTES)
            if len(fid_data) < UINT64_BYTES:
                raise EOFError("End of file reached before reading all file IDs, file maybe broken.")

            fid = struct.unpack('Q', fid_data)[0]
            file_ids.append(fid)

        return file_ids


def _read_key_offset_and_embedding(current_dir: str, model: str, fid: int, is_delta: bool, key_info_map: dict) -> None:
    """
    :param current_dir: save dir
    :param model: step for saving model
    :param fid: file ID for SSD
    :param is_delta: the model is whether full or incremental model
    :param key_info_map: key info, include key, offset, embedding size and embedding
    :return:
    """
    table_meta_file = os.path.join(current_dir, str(fid) + ".meta." + model)
    table_data_file = os.path.join(current_dir, str(fid) + ".data." + model)
    if is_delta:
        table_meta_file = os.path.join(current_dir, "delta-" + str(fid) + ".meta." + model)
        table_data_file = os.path.join(current_dir, "delta-" + str(fid) + ".data." + model)
    key_offset_gen = _read_key_offset(table_meta_file)
    embedding_data_gen = _read_embedding_data(table_data_file)
    for (key, offset), (emb_size, embedding) in zip(key_offset_gen, embedding_data_gen):
        key_info_map[key] = KeyInfo(offset=offset, emb_size=emb_size, embedding=embedding)


def _read_key_offset(file_path: str) -> Generator[Tuple[int, int], None, None]:
    """
    read key and offset from meta file
    :param file_path: meta file dir
    :return:
    """
    with tf.io.gfile.GFile(file_path, 'rb') as file:
        if tf.io.gfile.stat(file_path).length == SSD_DATA_FILE_MIN_SIZE:
            return
        validate_read_file(file_path)
        every_key_offset_bytes = UINT64_BYTES + UINT32_BYTES
        while True:
            # Read key(8bytes)and offset(4bytes).
            data = file.read(every_key_offset_bytes)  # 8bytes key + 4bytes offset
            if len(data) == 0:
                break
            if len(data) < every_key_offset_bytes:
                raise EOFError("End of file reached before reading key_offset, meta file maybe broken.")

            # Unpack key and offset.
            key = struct.unpack('q', data[:UINT64_BYTES])[0]                           # 'q':8bytes
            offset = struct.unpack('I', data[UINT64_BYTES:every_key_offset_bytes])[0]  # 'I':4bytes
            yield key, offset


def _read_embedding_data(file_path: str) -> Generator[Tuple[int, List[float]], None, None]:
    """
    read embedding data from data file
    :param file_path:
    :return:
    """
    with tf.io.gfile.GFile(file_path, 'rb') as file:
        if tf.io.gfile.stat(file_path).length == SSD_DATA_FILE_MIN_SIZE:
            return
        validate_read_file(file_path)
        while True:
            emb_size_data = file.read(UINT64_BYTES)
            if len(emb_size_data) == 0:
                break
            if len(emb_size_data) < UINT64_BYTES:
                raise EOFError("End of file reached before reading embedding size, data file maybe broken.")

            emb_size, = struct.unpack('Q', emb_size_data)
            embeddings_data = file.read(emb_size * FLOAT32_BYTES)
            if embeddings_data == 0:
                break
            if len(embeddings_data) < emb_size * FLOAT32_BYTES:
                raise EOFError("End of file reached before reading embedding file, data file maybe broken.")

            embedding = list(struct.unpack(f'{emb_size}f', embeddings_data))
            yield emb_size, embedding


def _write_ssd_meta_and_data(current_ssd_dir: str, table_name: str, fid: int, step: str, key_info_map: dict) -> None:
    """
    write key, offset, embedding size and embedding into new file
    :param current_ssd_dir: current dir
    :param table_name: table name
    :param fid: file ID
    :param step: the step for saving model
    :param key_info_map: key info map, include key, offset, embedding size and embedding
    :return:
    """
    meta_file_path = os.path.join(current_ssd_dir, table_name, str(fid) + ".meta." + step)
    data_file_path = os.path.join(current_ssd_dir, table_name, str(fid) + ".data." + step)
    if check_file_system_is_hdfs(meta_file_path) and check_file_system_is_hdfs(data_file_path):
        with tf.io.gfile.GFile(meta_file_path, "wb") as meta_file, tf.io.gfile.GFile(data_file_path, "wb") as data_file:
            for key, value in key_info_map.items():
                offset, emb_size, embedding = value.offset, value.emb_size, value.embedding
                meta_file.write(struct.pack('qI', key, offset))
                data_file.write(struct.pack('q', emb_size))
                data_file.write(struct.pack(f'{emb_size}f', *embedding))
    else:
        with os.fdopen(os.open(meta_file_path, SAVE_FILE_FLAG, SAVE_FILE_MODE), "wb") as meta_file, \
                os.fdopen(os.open(data_file_path, SAVE_FILE_FLAG, SAVE_FILE_MODE), "wb") as data_file:
            for key, value in key_info_map.items():
                offset, emb_size, embedding = value.offset, value.emb_size, value.embedding
                meta_file.write(struct.pack('qI', key, offset))
                data_file.write(struct.pack('q', emb_size))
                data_file.write(struct.pack(f'{emb_size}f', *embedding))
