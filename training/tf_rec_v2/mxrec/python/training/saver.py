#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
from typing import List

import tensorflow as tf
from tensorflow.python.framework import ops

from rec_sdk_common.log.log import LoggingProxy as logger
from rec_sdk_common.communication.hccl.hccl_info import get_rank_id
from rec_sdk_common.util.tf_adapter import gen_npu_cpu_ops
from rec_sdk_common.validator.safe_checker import class_safe_check, dir_safe_check, int_safe_check
from mxrec.python.constants.constants import ValidatorParams
from mxrec.python.embedding.table.static_emb_table import StaticEmbTable


_PREFIX_FILE_NAME = "sparse"
_PATH_MODE = 0o644


class EmbeddingTableSaver:
    def __init__(self, emb_tables: List[StaticEmbTable]):
        """
        Create a saver to save and restore embedding tables.

        Args:
            emb_tables: An iterable of embedding tables.

        Raises:
            ValueError: If `emb_tables` is empty.
            TypeError: If `emb_tables` is not an list.
        """

        class_safe_check("emb_tables", emb_tables, list)

        if not emb_tables:
            raise ValueError("empty `emb_tables`")

        class_safe_check("emb_table", emb_tables[0], StaticEmbTable)

        self.emb_tables = emb_tables

    def save(self, sess: tf.compat.v1.Session, save_path: str, global_step: int):
        """
        Save the embedding tables and variables to file.

        Args:
            sess: A TensorFlow session.
            save_path: The save path.
            global_step: The step of the model.

        Raises:
            TypeError: If `sess` is not a TensorFlow session.
            TypeError: If `global_step` is not an integer.
        """

        class_safe_check("sess", sess, tf.compat.v1.Session)
        int_safe_check(
            "global_step",
            global_step,
            min_value=ValidatorParams.MIN_UINT32.value,
            max_value=ValidatorParams.MAX_UINT32.value,
        )
        dir_safe_check("save_path", save_path)

        save_path = self._format_and_mkdir_save_path(save_path)

        for emb_table in self.emb_tables:
            if emb_table.count_filter:
                _save_count_filter(emb_table, save_path, global_step)

            if emb_table.time_evictor:
                _run_evict_graph(emb_table, sess)
                _save_time_evictor(emb_table, save_path, global_step)

        file_path = os.path.join(save_path, _PREFIX_FILE_NAME)
        logger.info("In save mode, the save path is %s, and the file path is %s.", save_path, file_path)
        _run_export_subgraph(self.emb_tables, file_path, sess, global_step)

    def load(self, sess: tf.compat.v1.Session, save_path: str, global_step: int):
        """
        Restore the embedding tables and variables from file.

        Args:
            sess: A TensorFlow session.
            save_path: The save path.
            global_step: The step of the model.

        Raises:
            TypeError: If `sess` is not a TensorFlow session.
            TypeError: If `global_step` is not an integer.
        """

        class_safe_check("sess", sess, tf.compat.v1.Session)
        int_safe_check(
            "global_step",
            global_step,
            min_value=ValidatorParams.MIN_UINT32.value,
            max_value=ValidatorParams.MAX_UINT32.value,
        )
        dir_safe_check("save_path", save_path)

        save_path = self._format_and_mkdir_save_path(save_path, is_load=True)

        for emb_table in self.emb_tables:
            if emb_table.count_filter:
                _load_count_filter(emb_table, save_path, global_step)

            if emb_table.time_evictor:
                _load_time_evictor(emb_table, save_path, global_step)

        file_path = os.path.join(save_path, _PREFIX_FILE_NAME)
        logger.info("In load mode, the save path is %s, and the file path is %s.", save_path, file_path)
        _run_import_graph(self.emb_tables, file_path, sess, global_step)

    @staticmethod
    def _format_and_mkdir_save_path(save_path: str, is_load: bool = False) -> str:
        save_path = _convert_to_abs_path(save_path)
        cur_rank = str(get_rank_id())
        save_path_with_rank = os.path.join(save_path, cur_rank)
        if is_load:
            return save_path_with_rank
        if not os.path.exists(save_path_with_rank):
            os.makedirs(save_path_with_rank, mode=_PATH_MODE)
        return save_path_with_rank


def _save_count_filter(emb_table: StaticEmbTable, save_path: str, global_step: int):
    save_path = os.path.join(save_path, str(global_step))

    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=_PATH_MODE)

    emb_table.count_filter.save(save_path)


def _save_time_evictor(emb_table: StaticEmbTable, save_path: str, global_step: int):
    save_path = os.path.join(save_path, str(global_step))

    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=_PATH_MODE)

    emb_table.time_evictor.save(save_path)


def _load_count_filter(emb_table: StaticEmbTable, save_path: str, global_step: int):
    save_path = os.path.join(save_path, str(global_step))
    emb_table.count_filter.load(save_path)


def _load_time_evictor(emb_table: StaticEmbTable, save_path: str, global_step: int):
    save_path = os.path.join(save_path, str(global_step))
    emb_table.time_evictor.load(save_path)


def _run_export_subgraph(
    emb_tables: List[StaticEmbTable], file_path: str, sess: tf.compat.v1.Session, global_step: int
):
    table_ids: List[int] = []
    table_names: List[str] = []
    embedding_dims: List[int] = []
    bucket_sizes: List[int] = []

    for emb_table in emb_tables:
        table_ids.append(StaticEmbTable.get_table_ins_to_id()[emb_table])
        table_names.append(emb_table.name)
        embedding_dims.append(emb_table.dim)
        bucket_sizes.append(emb_table.slice_dev_vocab_size)

    table_handles = gen_npu_cpu_ops.table_to_resource_v2(table_ids)
    # For EmbeddingHashmapSize, 'filter_export_flag=True' indicates that all keys are exported.
    table_sizes = gen_npu_cpu_ops.embedding_hashmap_size(
        table_ids=ops.convert_to_tensor(table_ids), export_mode="all", filter_export_flag=True
    )
    table_sizes = sess.run(table_sizes)
    keys, cnts, flags, vals = gen_npu_cpu_ops.embedding_hash_table_export(
        table_handles=table_handles,
        table_sizes=table_sizes,
        embedding_dims=ops.convert_to_tensor(embedding_dims, name="embedding_dims", dtype=tf.int64),
        bucket_sizes=ops.convert_to_tensor(bucket_sizes, name="bucket_sizes", dtype=tf.int64),
        export_mode="all",
        filter_export_flag=False,
        num=len(table_names),
    )
    export_op = gen_npu_cpu_ops.embedding_hashmap_export(
        file_path=ops.convert_to_tensor(file_path, name="file_path"),
        table_ids=ops.convert_to_tensor(table_ids, name="table_ids"),
        table_names=ops.convert_to_tensor(table_names, name="table_names"),
        global_step=ops.convert_to_tensor(global_step, name="step"),
        keys=keys,
        counters=cnts,
        filter_flags=flags,
        values=vals,
    )

    sess.run(export_op)


def _run_import_graph(
    emb_tables: List[StaticEmbTable], file_path: str, sess: tf.compat.v1.Session, global_step: int
):
    table_ids: List[int] = []
    table_names: List[str] = []
    embedding_dims: List[int] = []
    bucket_sizes: List[int] = []

    for emb_table in emb_tables:
        table_ids.append(StaticEmbTable.get_table_ins_to_id()[emb_table])
        table_names.append(emb_table.name)
        embedding_dims.append(emb_table.dim)
        bucket_sizes.append(emb_table.slice_dev_vocab_size)

    table_handles = gen_npu_cpu_ops.table_to_resource_v2(table_ids)
    table_sizes = gen_npu_cpu_ops.embedding_hashmap_file_size(
        file_path=ops.convert_to_tensor(file_path, name="file_path"),
        table_ids=ops.convert_to_tensor(table_ids, name="table_ids"),
        table_names=ops.convert_to_tensor(table_names, name="table_names"),
        global_step=ops.convert_to_tensor(ops.convert_to_tensor(global_step, name="step")),
        embedding_dims=embedding_dims,
    )
    keys, cnts, flags, vals = gen_npu_cpu_ops.embedding_hashmap_import(
        file_path=ops.convert_to_tensor(file_path, name="file_path"),
        table_ids=ops.convert_to_tensor(table_ids, name="table_ids"),
        table_sizes=table_sizes,
        table_names=ops.convert_to_tensor(table_names, name="table_names"),
        global_step=ops.convert_to_tensor(global_step, name="step"),
        embedding_dims=embedding_dims,
        num=len(table_names),
    )
    import_op = gen_npu_cpu_ops.embedding_hash_table_import(
        table_handles=table_handles,
        embedding_dims=ops.convert_to_tensor(embedding_dims, name="embedding_dims", dtype=tf.int64),
        bucket_sizes=ops.convert_to_tensor(bucket_sizes, name="bucket_sizes", dtype=tf.int64),
        keys=keys,
        counters=cnts,
        filter_flags=flags,
        values=vals,
    )

    sess.run(import_op)


def _run_evict_graph(emb_table: StaticEmbTable, sess: tf.compat.v1.Session):
    evicted_keys = emb_table.time_evictor.get_evicted_keys(emb_table.name)
    table_handles = gen_npu_cpu_ops.table_to_resource_v2([emb_table.table_id])
    evict_op = gen_npu_cpu_ops.embedding_hash_table_evict(
        table_handle=table_handles,
        keys=evicted_keys,
        table_cap=emb_table.slice_dev_vocab_size,
        embedding_dim=emb_table.dim,
        init_mode="constant",
        const_val=0.0,
    )
    sess.run(evict_op)


def _convert_to_abs_path(save_path: str) -> str:
    if os.path.isabs(save_path):
        return save_path

    return os.path.join(os.getcwd(), save_path)
