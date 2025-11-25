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

__version__ = "7.3.T50"
__all__ = [
    "init",
    "get_embedding_table",
    "get_existing_tables",
    "get_init_hashtable_op",
    "embedding_lookup",
    "get_sparse_embedding",
    "AdamWOptimizer",
    "EmbeddingTableSaver",
]


from mxrec.python.initializer.initializer import init
from mxrec.python.embedding.embedding import (
    get_embedding_table,
    get_existing_tables,
    get_init_hashtable_op,
    embedding_lookup,
    get_sparse_embedding,
)
from mxrec.python.optimizer.adam_w import AdamWOptimizer
from mxrec.python.training.saver import EmbeddingTableSaver
from mxrec.python.optimizer.utils import patch_for_update_op


def version():
    return __version__


def _patch_for_mxrec():
    patch_for_update_op()


_patch_for_mxrec()
