#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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

from typing import List, Optional, Union

import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer as InitializerV1
from tensorflow.python.ops.init_ops_v2 import Initializer as InitializerV2

from rec_sdk_common.log.log import LoggingProxy as logger
from rec_sdk_common.validator.safe_checker import (
    class_safe_check,
    int_safe_check,
    str_safe_check,
)
from mxrec.python.constants.constants import (
    EMBEDDING_TABLE_COLLECTION,
    INIT_HASHTABLE_COLLECTION,
    LOCAL_EMBEDDING_COLLECTION,
    EmbDistributionStrategy,
    StaticEmbTableConfig,
    ValidatorParams,
)
from mxrec.python.embedding.table.base_emb_table import BaseEmbTable
from mxrec.python.embedding.table.static_emb_table import StaticEmbTable


_MAX_TABLE_NAME_LEN = 128
_MAX_EMB_DIM = 512
_MAX_DEV_VOC_SIZE = 10**9


def get_embedding_table(
    name: str,
    dimension: int,
    device_vocabulary_size: int,
    initializer: Union[InitializerV1, InitializerV2] = tf.compat.v1.random_normal_initializer(),
    key_dtype: tf.DType = tf.int64,
    value_dtype: tf.DType = tf.float32,
    distribution_strategy: str = EmbDistributionStrategy.MP.value,
    min_used_times: Optional[int] = None,
    max_cold_secs: Optional[int] = None,
) -> Union[StaticEmbTable]:
    """Get the embedding table.

    Args:
        name: The embedding table name.
        dimension: The last dimension of this table(that is, the embedding vector size of embedding table).
        device_vocabulary_size: The embedding vector numbers on device memory.
        initializer: The initializer for embedding values.
        key_dtype: Data type for feature ids.
        value_dtype: The type of the embedding values.
        distribution_strategy: Distributed storage strategies for embedding tables.
        min_used_times: For the feature admission functionality, the ID must have a history of occurrences exceeding
            the specified threshold in order to take effect. Otherwise, when querying the table with this ID,
            the default embedding value will be returned.
        max_cold_secs: For the feature eviction functionality, this value represents the maximum allowable duration
            of inactivity for an ID and its associated embedding since the last access. IDs that exceed this threshold
            will be deleted from the hash table storing the embeddings during the next save operation.

    Returns:
        The embedding table.

    ```python
    import mxrec
    import tensorflow as tf

    # MxRec init.
    mxrec.init("toml_path")

    # Create an embedding table.
    table = mxrec.get_embedding_table(
        name="example_name",
        dimension=8,
        device_vocabulary_size=10000,
        initializer=tf.truncated_normal_initializer(),
        key_dtype=tf.int64,
        value_dtype=tf.float32,
    )
    ```

    """
    str_safe_check("name", name, min_len=1, max_len=_MAX_TABLE_NAME_LEN)
    int_safe_check("dimension", dimension, min_value=1, max_value=_MAX_EMB_DIM)
    int_safe_check(
        "device_vocabulary_size",
        device_vocabulary_size,
        min_value=0,
        max_value=_MAX_DEV_VOC_SIZE,
    )

    if min_used_times:
        int_safe_check(
            name="min_used_times",
            value=min_used_times,
            min_value=0,
            max_value=ValidatorParams.MAX_INT32.value,
        )

    if max_cold_secs:
        int_safe_check(
            name="max_cold_secs",
            value=max_cold_secs,
            min_value=0,
            max_value=ValidatorParams.MAX_UINT64.value,
        )

    if device_vocabulary_size == 0:
        raise ValueError(
            f"currently, the embedding table only supports storage in device memory and is a fixed table size, "
            f"and the device vocabulary size must be greater than 0, but got {device_vocabulary_size}"
        )
    class_safe_check("initializer", initializer, (InitializerV1, InitializerV2))
    if key_dtype != tf.int64:
        raise ValueError(f"currently, the embedding table only supports key dtype of tf.int64, but got {key_dtype}")
    if value_dtype != tf.float32:
        raise ValueError(
            f"currently, the embedding table only supports value dtype of tf.float32, but got {value_dtype}"
        )
    str_safe_check("distribution_strategy", distribution_strategy)
    if distribution_strategy not in {e.value for e in EmbDistributionStrategy}:
        raise ValueError(
            f"currently, the embedding table only support MP(model parallelism), but got {distribution_strategy}"
        )

    table_ins = _get_exist_table(name)
    if table_ins is not None:
        logger.info("The embedding table %s already exists.", table_ins)
        return table_ins

    et_config = StaticEmbTableConfig(
        name=name,
        dim=dimension,
        dev_vocab_size=device_vocabulary_size,
        initializer=initializer,
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        dist_strategy=distribution_strategy,
        min_used_times=min_used_times,
        max_cold_secs=max_cold_secs,
    )
    table_ins = StaticEmbTable(et_config)
    tf.compat.v1.add_to_collection(EMBEDDING_TABLE_COLLECTION, table_ins)
    logger.info("The embedding table %s has been created.", table_ins)

    return table_ins


def get_init_hashtable_op() -> List[tf.Operation]:
    """Retrieves a list of init hashmap op from the TensorFlow collection.

    Returns:
        List[tf.Operation]: A list of tf.Operation that are part of the LOCAL_EMBEDDING_COLLECTION.

    ```python
    import mxrec
    import tensorflow as tf

    # MxRec init.
    mxrec.init("toml_path")

    # Create an embedding table.
    table = mxrec.get_embedding_table(
        name="example_name",
        dimension=8,
        device_vocabulary_size=10000,
        initializer=tf.truncated_normal_initializer(),
        key_dtype=tf.int64,
        value_dtype=tf.float32,
    )
    init_hashtable_op = mxrec.get_init_hashtable_op()
    with tf.compat.v1.Session as sess:
        sess.run(init_hashtable_op)
        sess.run(tf.compat.v1.global_variables_initializer())
    ```
    """
    return tf.compat.v1.get_collection(INIT_HASHTABLE_COLLECTION)


def get_existing_tables() -> List[BaseEmbTable]:
    """Get all existing tables.

    Returns:
        A list of all existing embedding table instances.
    """

    return tf.compat.v1.get_collection(EMBEDDING_TABLE_COLLECTION)


def _get_exist_table(name: str) -> Optional[Union[StaticEmbTable]]:
    """Get the existing table by name.

    Args:
        name: The embedding table name.

    Returns:
        If it exists, return the embedding table instance; otherwise, return None.
    """
    tables = tf.compat.v1.get_collection(EMBEDDING_TABLE_COLLECTION)
    for table in tables:
        if name == table.name:
            return table
    return None
