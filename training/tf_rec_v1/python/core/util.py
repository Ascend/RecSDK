#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor

from rec_sdk_common.log.log import LoggingProxy as logger
from mx_rec.constants import constants
from mx_rec.graph.constants import AnchorIteratorOp
from mx_rec.util.initialize import ConfigInitializer


def mark_orphan_lookup_key(lookup_key: Tensor) -> Tensor:
    """Upward search default TF::Graph, mark the key tensor without TF::Dataset as root op.

    Args:
        lookup_key: An 'tf.Tensor' represents ID-like keys.

    Return:
        marked_lookup_key: Identity of `lookup_key` with name prefix 'orphan'.
    """

    graph_def = tf.compat.v1.get_default_graph().as_graph_def()
    subgraph = tf.compat.v1.graph_util.extract_sub_graph(graph_def, [lookup_key.op.name])

    for node in subgraph.node:
        if node.op == AnchorIteratorOp.ITERATOR_GET_NEXT.value:
            return lookup_key

    name_prefix = constants.ORPHAN_LOOKUP_KEY_PREFIX
    marked_lookup_key = tf.identity(lookup_key, name="{}/{}".format(name_prefix, lookup_key.op.name))
    logger.info("Mark orphan lookup key %s as %s.", lookup_key, marked_lookup_key)

    return marked_lookup_key


def check_and_set_vocab_size(device_vocab_size: int, host_vocab_size: int, ssd_vocab_size: int) -> Tuple[int, int, int]:
    if ConfigInitializer.get_instance().use_dynamic_expansion:
        logger.info("In dyanmic expansion mode, DDR and SSD vocabulary size will be reset to 0 automatically!")
        return (device_vocab_size, 0, 0)

    if host_vocab_size == 0 and ssd_vocab_size > 0:
        raise ValueError("set SSD vocabulary size must set DDR vocabulary size first")

    return (device_vocab_size, host_vocab_size, ssd_vocab_size)
