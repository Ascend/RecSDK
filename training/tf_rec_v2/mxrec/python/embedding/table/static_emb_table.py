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

import tensorflow as tf

from rec_sdk_common.util.tf_adapter import gen_npu_cpu_ops
from mxrec.python.constants.constants import StaticEmbTableConfig
from mxrec.python.embedding.table.base_emb_table import BaseEmbTable
from mxrec.python.embedding.feature.filter import CountFilter
from mxrec.python.embedding.feature.evictor import TimeEvictor


class StaticEmbTable(BaseEmbTable):
    """An embedding table with a fixed table size. Currently, the table is stored in device memory."""

    def __init__(self, et_config: StaticEmbTableConfig):
        super(StaticEmbTable, self).__init__(et_config)

        self._count_filter = CountFilter(et_config.name, et_config.min_used_times) if et_config.min_used_times else None
        self._time_evictor = TimeEvictor(et_config.name, et_config.max_cold_secs) if et_config.max_cold_secs else None

    @property
    def count_filter(self) -> CountFilter:
        return self._count_filter

    @property
    def time_evictor(self) -> TimeEvictor:
        return self._time_evictor

    def _create_hashtable(self) -> tf.Tensor:
        # The load factor is used for the dynamic table and is currently not effective.
        load_factor = 0.8
        table_handle = gen_npu_cpu_ops.init_embedding_hashmap_v2(
            table_id=self._table_id,
            bucket_size=self.slice_dev_vocab_size,
            embedding_dim=self.dim,
            load_factor=load_factor,
            dtype=self.value_dtype,
        )
        sampled_values = self._et_config.initializer(shape=[self.slice_dev_vocab_size, self.dim], dtype=tf.float32)
        # The `initializer_mode` supports two modes: "random" and "constant". When the parameter is set to "constant",
        # it must be used together with `constant_value`. When the parameter is set to "random", it must be used
        # together with `sampled_values`.
        hashtable = gen_npu_cpu_ops.init_embedding_hash_table(
            table_handle=table_handle,
            bucket_size=self.slice_dev_vocab_size,
            embedding_dim=self.dim,
            initializer_mode="random",
            sampled_values=sampled_values,
        )
        return hashtable
