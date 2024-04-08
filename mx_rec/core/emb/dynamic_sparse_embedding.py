#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import abc
from typing import Optional, Union, Callable

import tensorflow as tf

from mx_rec.constants.constants import ASCEND_TABLE_NAME_MUST_CONTAIN, ASCEND_SPARSE_LOOKUP_LOCAL_EMB, \
     ASCEND_SPARSE_LOOKUP_ID_OFFSET
from mx_rec.core.asc.feature_spec import FeatureSpec
from mx_rec.core.asc.build_graph import get_preprocessed_tensor_for_asc
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
    def set_optimizer(self, key: str, state_dict: dict):
        pass

    @abc.abstractmethod
    def _build_optimizer_states(self):
        pass

    @abc.abstractmethod
    def _set_ext_emb_size(self):
        pass

    @abc.abstractmethod
    def _set_slice_vocab_size(self):
        pass

    @abc.abstractmethod
    def _get_preprocessed_tensor(self, feature_spec: FeatureSpec, is_training: bool, send_count: Optional[int]) -> dict:
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
            result.get("id_offsets"), embedding_dim=self._emb_size, embedding_type=1)

        add_collection_condition = is_training and (
                ASCEND_TABLE_NAME_MUST_CONTAIN is None or ASCEND_TABLE_NAME_MUST_CONTAIN in self._table_name)
        logger.debug("feature spec mode, table_name: %s, ASCEND_TABLE_NAME_MUST_CONTAIN: %s",
                     self._table_name, ASCEND_TABLE_NAME_MUST_CONTAIN)
        if not add_collection_condition:
            return sparse_forward_fn(local_embeddings)

        tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_LOCAL_EMB, local_embeddings)
        tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_ID_OFFSET, result.get("id_offsets"))
        # tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_UNIQUE_KEYS, result.get("unique_keys"))
        return sparse_forward_fn(local_embeddings)


class HBMDynamicSparseEmbedding(DynamicSparseEmbedding):
    """
    稀疏表，表的大小非固定，支持动态扩容，HBM模式
    """

    def __init__(self, config: dict):
        super(DynamicSparseEmbedding, self).__init__(config)

    def set_optimizer(self, key: str, state_dict: dict):
        pass

    def _build_optimizer_states(self):
        pass

    def _set_ext_emb_size(self):
        self._ext_emb_size = self._emb_size * self._ext_coefficient
        logger.debug("init table, ext_emb_size is set to be %s.", self._ext_emb_size)

    def _set_slice_vocab_size(self):
        # 动态扩容模式下，保留device侧variable，大小设置为1
        self._slice_device_vocabulary_size = 1

    def _get_preprocessed_tensor(self, feature_spec: FeatureSpec, is_training: bool, send_count: Optional[int]) -> dict:
        channel_id = ConfigInitializer.get_instance().train_params_config.get_training_mode_channel_id(is_training)
        config = dict(batch_size=feature_spec.batch_size, feat_cnt=feature_spec.feat_cnt, send_count=send_count,
                      rank_size=self._rank_size, channel_id=channel_id, table_name=self._table_name,
                      is_hbm=self._is_hbm, ext_emb_size=self._ext_emb_size,
                      emb_size=self._emb_size, device_id=self._device_id, use_dynamic_expansion=True)

        return get_preprocessed_tensor_for_asc(self._variable, config)
