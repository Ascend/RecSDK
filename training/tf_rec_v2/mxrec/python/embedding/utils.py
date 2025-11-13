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
from tensorflow.python.framework import ops

from rec_sdk_common.util.tf_adapter import gen_npu_cpu_ops
from mxrec.python.embedding.table.base_emb_table import BaseEmbTable
from mxrec.python.embedding.lookup.base_lookup import BaseLookup


class EmbeddingResource:
    """Convert table id to table handle."""

    def __init__(self, table_id: int):
        self._name = table_id
        self._tensor: tf.Tensor = gen_npu_cpu_ops.table_to_resource_v2(ops.convert_n_to_tensor([table_id]))

    @property
    def name(self) -> int:
        return self._name

    @property
    def handle(self) -> tf.Tensor:
        return self._tensor

    @property
    def graph(self) -> tf.Graph:
        return self._tensor.graph

    @property
    def op(self) -> tf.Operation:
        return self._tensor.op

    @property
    def device(self) -> str:
        return self._tensor.op.device


def get_table_ins_by_local_embedding(local_embedding: tf.Tensor) -> BaseEmbTable:
    local_emb_to_table_ins = BaseLookup.get_local_emb_to_table_ins()
    if local_embedding not in local_emb_to_table_ins:
        raise KeyError(f"the local embedding {local_embedding} does not exist in {local_emb_to_table_ins}")
    return local_emb_to_table_ins.get(local_embedding)
