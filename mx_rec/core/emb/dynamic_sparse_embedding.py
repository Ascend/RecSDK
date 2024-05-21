#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import abc
from typing import Optional, Union, Callable

import tensorflow as tf

from mx_rec.constants.constants import ASCEND_TABLE_NAME_MUST_CONTAIN, ASCEND_SPARSE_LOOKUP_LOCAL_EMB, \
     ASCEND_SPARSE_LOOKUP_ID_OFFSET, ASCAnchorAttr
from mx_rec.core.asc.feature_spec import FeatureSpec
from mx_rec.core.emb.base_sparse_embedding import BaseSparseEmbedding
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.log import logger
from mx_rec.util.ops import import_host_pipeline_ops


class DynamicSparseEmbedding(BaseSparseEmbedding):
    """
    稀疏表，表的大小非固定，支持动态扩容
    """

    def __init__(self, config: dict):
        super(DynamicSparseEmbedding, self).__init__(config)

    def capacity(self) -> int:
        return ConfigInitializer.get_instance().hybrid_manager_config.asc_manager.get_table_capacity(self._table_name)

    @abc.abstractmethod
    def _set_slice_vocab_size(self):
        pass

    def _get_update_grad(self, local_grad: tf.Tensor, result: dict,
                         table: Union[tf.compat.v1.Variable, tf.Tensor]) -> Union[tf.IndexedSlices, tf.Tensor]:
        return local_grad

    def _get_local_embeddings(self, table: Union[tf.compat.v1.Variable, tf.Tensor], result: dict,
                              feature_spec: FeatureSpec, **kwargs) -> tf.Tensor:
        return tf.identity(table, name="identity_local_emb")

    def _get_sparse_forward_result(self, sparse_forward_fn: Callable, table: Union[tf.compat.v1.Variable, tf.Tensor],
                                   result: dict, is_training: bool) -> tf.Tensor:
        local_embeddings = import_host_pipeline_ops().embedding_lookup_by_address(
            result.get(str(ASCAnchorAttr.ID_OFFSETS)), embedding_dim=self._emb_size, embedding_type=1)

        add_collection_condition = is_training and (
                ASCEND_TABLE_NAME_MUST_CONTAIN is None or ASCEND_TABLE_NAME_MUST_CONTAIN in self._table_name)
        logger.debug("feature spec mode, table_name: %s, ASCEND_TABLE_NAME_MUST_CONTAIN: %s",
                     self._table_name, ASCEND_TABLE_NAME_MUST_CONTAIN)
        if not add_collection_condition:
            return sparse_forward_fn(local_embeddings)
        # 创建扩容查询tensor和table_instance的映射关系，以便优化器中使用
        ConfigInitializer.get_instance().sparse_embed_config.insert_table_instance_to_tensor_dict(
            result.get(str(ASCAnchorAttr.ID_OFFSETS)), self)
        tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_LOCAL_EMB, local_embeddings)
        tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_ID_OFFSET, result.get(str(ASCAnchorAttr.ID_OFFSETS)))
        return sparse_forward_fn(local_embeddings)


class HBMDynamicSparseEmbedding(DynamicSparseEmbedding):
    """
    稀疏表，表的大小非固定，支持动态扩容，HBM模式
    """

    def __init__(self, config: dict):
        super(DynamicSparseEmbedding, self).__init__(config)

    def _set_slice_vocab_size(self):
        # 动态扩容模式下，保留device侧variable，大小设置为1
        self._slice_device_vocabulary_size = 1

