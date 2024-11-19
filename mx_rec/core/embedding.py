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
from typing import Optional, Union

import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer as InitializerV1
from tensorflow.python.ops.init_ops_v2 import Initializer as InitializerV2

from mx_rec.constants.constants import (
    FLOAT32_BYTES,
    MAX_DEVICE_VOCABULARY_SIZE,
    MAX_INT32,
    MAX_VOCABULARY_SIZE,
    All2allGradientsOp,
)
from mx_rec.core.asc.feature_spec import FeatureSpec
from mx_rec.core.emb.base_sparse_embedding import BaseSparseEmbedding
from mx_rec.core.emb.emb_factory import (
    ExternalStorageSparseEmbeddingFactory,
    HBMDynamicSparseEmbeddingFactory,
    HBMSparseEmbeddingFactory,
)
from mx_rec.core.embedding_proxy import UnionKey, create_mergeable_embedding
from mx_rec.core.util import check_and_set_vocab_size, mark_orphan_lookup_key
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.log import logger
from mx_rec.util.normalization import fix_invalid_table_name
from mx_rec.validator.emb_validator import check_emb_multi_lookup_times
from mx_rec.validator.validator import (
    ClassValidator,
    FloatValidator,
    IntValidator,
    ListValidator,
    OptionalIntValidator,
    OptionalStringValidator,
    OptionValidator,
    OrValidator,
    SSDFeatureValidator,
    StringValidator,
    TensorShapeValidator,
    para_checker_decorator,
)


@para_checker_decorator(
    check_option_list=[
        ("key_dtype", OptionValidator, {"options": (tf.int64, tf.int32)}),
        (
            "dim",
            OrValidator,
            {
                "options": [
                    (IntValidator, {"min_value": 1, "max_value": 8192}, ["check_value"]),
                    (
                        TensorShapeValidator,
                        {"int_checker_args": {"min_value": 1, "max_value": 8192}},
                    ),
                ]
            },
        ),
        ("name", StringValidator, {"min_len": 1, "max_len": 100}, ["check_string_length", "check_whitelist"]),
        ("emb_initializer", ClassValidator, {"classes": (InitializerV1, InitializerV2)}),
        (["ssd_vocabulary_size", "ssd_data_path", "host_vocabulary_size"], SSDFeatureValidator),
        (
            "device_vocabulary_size",
            IntValidator,
            {"min_value": 1, "max_value": MAX_DEVICE_VOCABULARY_SIZE},
            ["check_value"],
        ),
        ("host_vocabulary_size", IntValidator, {"min_value": 0, "max_value": MAX_VOCABULARY_SIZE}, ["check_value"]),
        ("ssd_vocabulary_size", IntValidator, {"min_value": 0, "max_value": MAX_VOCABULARY_SIZE}, ["check_value"]),
        (
            "ssd_data_path",
            ListValidator,
            {"sub_checker": ClassValidator, "list_max_length": MAX_INT32, "sub_args": {"classes": str}},
            ["check_list_length"],
        ),
        ("is_save", ClassValidator, {"classes": (bool,)}),
        ("is_dp", ClassValidator, {"classes": (bool,)}),
        ("init_param", FloatValidator, {"min_value": -10, "max_value": 10}, ["check_value"]),
        ("all2all_gradients_op", OptionValidator, {"options": [i.value for i in list(All2allGradientsOp)]}),
        ("enable_merge", ClassValidator, {"classes": (bool,)}),
        ("value_dtype", OptionValidator, {"options": [tf.float32]}),
        ("shard_num", IntValidator, {"min_value": 1, "max_value": 8192}, ["check_value"]),
        ("fusion_optimizer_var", ClassValidator, {"classes": (bool,)}),
        ("hashtable_threshold", IntValidator, {"min_value": 0, "max_value": MAX_INT32}, ["check_value"]),
    ]
)
def create_table(
    key_dtype: tf.DType,
    dim: tf.TensorShape,
    name: str,
    emb_initializer: Union[InitializerV1, InitializerV2],
    device_vocabulary_size: int = 1,
    host_vocabulary_size: int = 0,
    ssd_vocabulary_size: int = 0,
    ssd_data_path: str = (os.getcwd(),),
    is_save: bool = True,
    is_dp: bool = False,
    init_param: float = 1.0,
    all2all_gradients_op: str = All2allGradientsOp.SUM_GRADIENTS.value,
    enable_merge: bool = False,
    value_dtype: tf.DType = tf.float32,
    shard_num: int = 1,
    fusion_optimizer_var: bool = True,
    hashtable_threshold: int = 0,
):
    """
    Args:
        key_dtype: data type for feature id
        dim: embedding vector size
        name: hash table name
        emb_initializer: the initializer for embedding values
        device_vocabulary_size: embedding vector numbers on device
        host_vocabulary_size: embedding vector numbers on ddr
        ssd_vocabulary_size: embedding vector numbers on ssd
        ssd_data_path: ssd embedding data save and load path relation from feature to variable offset will be built
        is_save: switch whether to store sparse table data.
        is_dp: switch whether to enable data parallel.
        init_param: embedding init param-coefficient
        all2all_gradients_op: sum_grads (default) or sum_gradients_and_div_by_ranksize.
        enable_merge: enable merge sparse embedding table automatically.
        value_dtype: the type of the value tensors. only tf.float32 if supported for now.
        shard_num: embedding partition number
        fusion_optimizer_var: fusion optimizer variable with embedding
        hashtable_threshold: choose to implement based on hash table or linear layer
    """
    name = fix_invalid_table_name(name)

    dim_bytes = dim.as_list()[0] * FLOAT32_BYTES if isinstance(dim, tf.TensorShape) else dim * FLOAT32_BYTES
    (device_vocabulary_size, host_vocabulary_size, ssd_vocabulary_size) = check_and_set_vocab_size(
        device_vocabulary_size, host_vocabulary_size, ssd_vocabulary_size
    )

    config = dict(
        key_dtype=key_dtype,
        embedding_size=dim,
        table_name=name,
        emb_initializer=emb_initializer,
        device_vocabulary_size=device_vocabulary_size,
        host_vocabulary_size=host_vocabulary_size,
        ssd_vocabulary_size=ssd_vocabulary_size,
        ssd_data_path=ssd_data_path,
        init_param=init_param,
        is_save=is_save,
        all2all_gradients_op=all2all_gradients_op,
        is_dp=is_dp,
    )

    logger.info(
        "Create table: The table name is %s, the key type is %s, the embedding size is %s, "
        "the embedding initializer is %s, the device/host/ssd vocabulary size is %s/%s/%s, "
        "the ssd data path is %s, the init param is %s, the is_save is %s, the all2all_gradients_op is %s, "
        "and the is_dp is %s.",
        name,
        key_dtype,
        dim_bytes / FLOAT32_BYTES,
        emb_initializer,
        device_vocabulary_size,
        host_vocabulary_size,
        ssd_vocabulary_size,
        ssd_data_path,
        init_param,
        is_save,
        all2all_gradients_op,
        is_dp,
    )

    if enable_merge:
        if not ConfigInitializer.get_instance().use_dynamic_expansion:
            raise RuntimeError("merge table function requires dynamic expansion mode")

        union_key = UnionKey(
            key_dtype=key_dtype,
            emb_dim=dim.num_elements(),
            initializer_type=type(emb_initializer),
            is_save=is_save,
            is_dp=is_dp,
            init_param=init_param,
            all2all_gradients_op=all2all_gradients_op,
        )

        return create_mergeable_embedding(name, config, union_key)

    if ConfigInitializer.get_instance().use_dynamic_expansion:
        return HBMDynamicSparseEmbeddingFactory().create_embedding(config)
    if host_vocabulary_size > 0:
        return ExternalStorageSparseEmbeddingFactory().create_embedding(config)

    return HBMSparseEmbeddingFactory().create_embedding(config)


@para_checker_decorator(
    check_option_list=[
        ("hashtable", ClassValidator, {"classes": (BaseSparseEmbedding,)}),
        ("ids", ClassValidator, {"classes": (FeatureSpec, tf.Tensor)}),
        ("is_train", ClassValidator, {"classes": (bool,)}),
        ("send_count", ClassValidator, {"classes": (int, type(None))}),
        ("send_count", OptionalIntValidator, {"min_value": 1, "max_value": MAX_INT32}, ["check_value"]),
        ("name", ClassValidator, {"classes": (str, type(None))}),
        ("name", OptionalStringValidator, {"min_len": 1, "max_len": 255}, ["check_string_length"]),
        ("modify_graph", ClassValidator, {"classes": (bool, type(None))}),
        ("batch", ClassValidator, {"classes": (dict, type(None))}),
        ("access_and_evict_config", ClassValidator, {"classes": (dict, type(None))}),
        ("is_grad", ClassValidator, {"classes": (bool,)}),
        ("serving_default_value", ClassValidator, {"classes": (tf.Tensor, type(None))}),
    ]
)
def sparse_lookup(
    hashtable: BaseSparseEmbedding,
    ids: Union[FeatureSpec, tf.Tensor],
    send_count: Optional[int] = None,
    is_train: bool = True,
    name: Optional[str] = None,
    modify_graph: bool = False,
    batch: Optional[dict] = None,
    access_and_evict_config: Optional[dict] = None,
    is_grad: bool = True,
    serving_default_value: Optional[tf.Tensor] = None,
    **kwargs,
):
    """
    Args:
        hashtable: SparseEmbedding instance to be looked up
        ids: Tensor to lookup from hashtable
        send_count: used to config all2all communication parameters
        is_train: indicates whether the mode is train.
        name: identity for lookup ops, it will be used to build scope_name together with hashtable name
        modify_graph: if True, the original graph will be modified before building a Session instance
        batch: the value returned by the get_next() method of TF Dataset
        access_and_evict_config: the configuration for the feature of feature filtering and eviction
        is_grad: indicate whether this lookup requires update gradients
        serving_default_value: The hashtable misses the id, that is, the id that is lower than the threshold during
            training, and the newly appeared id during prediction, and the lookup return value, which can ensure that
            the return value of the new id is consistent during training and prediction. The default is None, and the
            return value of the hashtable corresponding to the missing id is based on the initializer of hashtable.
    Returns: Tensor for lookup result

    """

    kwargs["is_grad"] = is_grad
    kwargs["is_train"] = is_train
    kwargs["name"] = name if name is not None else hashtable.get_default_lookup_name()
    kwargs["modify_graph"] = modify_graph
    kwargs["batch"] = batch
    kwargs["access_and_evict_config"] = access_and_evict_config
    kwargs["serving_default_value"] = serving_default_value

    # Parameters are supposed to be created innernally.
    kwargs["feature_spec_name_ids_dict"] = None
    kwargs["multi_lookup"] = False
    kwargs["lookup_ids"] = None

    # When performing multiple queries on a single table, if any one of the queries requires gradients (grad),
    # then the entire table also needs gradients; otherwise, the whole table does not require gradients.
    # Additionally, in the case of global uniqueness, backend does not need to send data.
    hashtable.is_grad |= is_grad

    logger.info(
        "Lookup: The table name is %s, and the value of `is_grad` in this lookup (lookup name is %s) is %s.",
        hashtable.table_name,
        name,
        is_grad,
    )

    hashtable.increase_multi_lookup_times(is_train)
    check_emb_multi_lookup_times(hashtable.multi_lookup_times.get(is_train), hashtable.table_name)

    if isinstance(ids, tf.Tensor):
        ids = mark_orphan_lookup_key(ids)

    with tf.compat.v1.variable_scope("{0}//{1}".format(hashtable.table_name, kwargs.get("name"))):
        if isinstance(ids, FeatureSpec):
            # check whether the name of the table exists with FeatureSpec.
            if hashtable.table_name != ids.table_name:
                raise ValueError(
                    f"The table name '{ids.table_name}' specified by FeatureSpec is inconsistent with"
                    f" the SparseEmbedding table name '{hashtable.table_name}'."
                )

            return hashtable.lookup_for_feat_spec(ids, send_count, **kwargs)

        if not modify_graph:
            raise ValueError("'ids' is type of tf.Tensor, 'modify_graph' should be set to True")

        ConfigInitializer.get_instance().modify_graph = modify_graph
        return hashtable.lookup(ids, send_count, **kwargs)
