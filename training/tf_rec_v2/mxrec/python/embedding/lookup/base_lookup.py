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

import abc
from typing import Dict

import tensorflow as tf

from rec_sdk_common.communication.hccl.hccl_info import get_rank_id, get_rank_size
from mxrec.python.constants.constants import MPLookupParams
from mxrec.python.embedding.table.static_emb_table import StaticEmbTable


class BaseLookupInterface(metaclass=abc.ABCMeta):
    """Base class interface of embedding lookup."""

    @abc.abstractmethod
    def lookup(self) -> tf.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _process_ids(self, ids: tf.Tensor) -> MPLookupParams:
        raise NotImplementedError


class BaseLookup(BaseLookupInterface):
    """Base class of embedding lookup."""

    _default_name_count = -1
    _local_emb_to_table_ins = {}

    def __init__(self, emb_table: StaticEmbTable, ids: tf.Tensor):
        self._emb_table = emb_table
        self._ids = ids
        self._rank_size = get_rank_size()
        self._rank_id = get_rank_id()

    @classmethod
    def get_local_emb_to_table_ins(cls) -> Dict[tf.Tensor, StaticEmbTable]:
        return cls._local_emb_to_table_ins

    @classmethod
    def set_local_emb_to_table_ins(cls, local_emb: tf.Tensor, table_ins: StaticEmbTable):
        cls._local_emb_to_table_ins[local_emb] = table_ins

    def lookup(self) -> tf.Tensor:
        raise NotImplementedError

    def _process_ids(self, ids: tf.Tensor) -> MPLookupParams:
        raise NotImplementedError

    def _get_default_lookup_name(self) -> str:
        self._default_name_count += 1
        scope_name = "{0}//{1}".format(self._emb_table.name, "emb_lookup_%d" % self._default_name_count)
        return scope_name
