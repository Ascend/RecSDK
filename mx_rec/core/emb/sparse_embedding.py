#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import abc
import math
from typing import Optional, Union, Callable

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

from mx_rec.core.asc.feature_spec import FeatureSpec
from mx_rec.core.asc.build_graph import get_preprocessed_tensor_for_asc
from mx_rec.core.emb.base_sparse_embedding import BaseSparseEmbedding
from mx_rec.optimizers.emb_optimizer import EmbOptimizer
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.log import logger


class SparseEmbedding(BaseSparseEmbedding):
    """
    稀疏表，表的大小为固定大小，不支持动态扩容
    """

    def __init__(self, config: dict):
        super(SparseEmbedding, self).__init__(config)

    @abc.abstractmethod
    def capacity(self) -> int:
        pass

    @abc.abstractmethod
    def set_optimizer(self, key: str, state_dict: dict):
        pass

    @abc.abstractmethod
    def _set_ext_emb_size(self):
        pass

    @abc.abstractmethod
    def _build_optimizer_states(self):
        pass

    @abc.abstractmethod
    def _get_preprocessed_tensor(self, feature_spec: FeatureSpec, is_training: bool, send_count: Optional[int]) -> dict:
        pass

    def _set_slice_vocab_size(self):
        self._slice_device_vocabulary_size = math.ceil(self._device_vocabulary_size / self._rank_size)
        self._slice_host_vocabulary_size = math.ceil(self._host_vocabulary_size / self._rank_size)
        self._slice_ssd_vocabulary_size = math.ceil(self._ssd_vocabulary_size / self._rank_size)

    def _get_update_grad(self, local_grad: tf.Tensor, result: dict,
                         table: Union[tf.compat.v1.Variable, tf.Tensor]) -> Union[tf.IndexedSlices, tf.Tensor]:
        return ops.IndexedSlices(values=local_grad,
                                 indices=result.get("id_offsets"),
                                 dense_shape=tf.shape(table))

    def _get_local_embeddings(self, table: Union[tf.compat.v1.Variable, tf.Tensor], result: dict,
                              feature_spec: FeatureSpec, **kwargs) -> tf.Tensor:
        id_offsets_abs = tf.abs(result.get("id_offsets"))
        local_embeddings = tf.gather(table, id_offsets_abs, axis=0, name="gather_for_id_offsets")
        local_embeddings = _set_specific_value_for_non_valid_key(result.get("id_offsets"),
                                                                 local_embeddings,
                                                                 feature_spec.access_threshold,
                                                                 kwargs.get("serving_default_value"),
                                                                 is_training=kwargs.get("is_train"))
        return local_embeddings

    def _get_sparse_forward_result(self, sparse_forward_fn: Callable, table: Union[tf.compat.v1.Variable, tf.Tensor],
                                   result: dict, is_training: bool) -> tf.Tensor:
        return sparse_forward_fn(self._variable)


class HBMSparseEmbedding(SparseEmbedding):
    """
    稀疏表，表的大小为固定大小，HBM模式
    """

    def __init__(self, config: dict):
        super(HBMSparseEmbedding, self).__init__(config)

    def capacity(self) -> int:
        return self._device_vocabulary_size

    def set_optimizer(self, key: str, state_dict: dict):
        pass

    def _build_optimizer_states(self):
        pass

    def _set_ext_emb_size(self):
        self._ext_emb_size = self._emb_size * self._ext_coefficient
        logger.debug("Init table, ext_emb_size is set to be %s.", self._ext_emb_size)

    def _get_preprocessed_tensor(self, feature_spec: FeatureSpec, is_training: bool, send_count: Optional[int]) -> dict:
        channel_id = ConfigInitializer.get_instance().train_params_config.get_training_mode_channel_id(is_training)
        config = dict(batch_size=feature_spec.batch_size, feat_cnt=feature_spec.feat_cnt, send_count=send_count,
                      rank_size=self._rank_size, channel_id=channel_id, table_name=self._table_name,
                      is_hbm=self._is_hbm, ext_emb_size=self._ext_emb_size,
                      emb_size=self._emb_size, device_id=self._device_id)

        return get_preprocessed_tensor_for_asc(self._variable, config)


class ExternalStorageSparseEmbedding(SparseEmbedding):
    """
    稀疏表，表的大小为固定大小，DDR/SSD模式
    """

    def __init__(self, config: dict):
        self.emb_optimizer = EmbOptimizer(config.get("optimizer_list"))
        self.emb_optimizer.check_optimizer_instance_list()

        super(ExternalStorageSparseEmbedding, self).__init__(config)

    @property
    def optimizer(self):
        return self.emb_optimizer.optimizer

    @property
    def optimizer_instance_list(self):
        return self.emb_optimizer.optimizer_instance_list

    def capacity(self) -> int:
        # DDR
        if not self._ssd_vocabulary_size:
            return self._device_vocabulary_size + self._host_vocabulary_size
        # SSD
        return self._device_vocabulary_size + self._host_vocabulary_size + self._ssd_vocabulary_size

    def set_optimizer(self, key: str, state_dict: dict):
        self.emb_optimizer.set_optimizer(key, state_dict, self._table_name)

    def _set_ext_emb_size(self):
        self._ext_coefficient += len(self.emb_optimizer.optimizer_slot_info_list)
        self._ext_emb_size = self._emb_size * self._ext_coefficient
        logger.debug("Init table, ext_emb_size is set to be %s.", self._ext_emb_size)

    def _build_optimizer_states(self):
        for sparse_optimizer_instance in self.emb_optimizer.optimizer_instance_list:
            slot_info_list = sparse_optimizer_instance.initialize_slots(self._variable, self)
            self.emb_optimizer.optimizer_slot_info_list.extend(slot_info_list)

        for slot_info in self.emb_optimizer.optimizer_slot_info_list:
            self.emb_optimizer.set_optimizer_slot(slot_info)

    def _get_preprocessed_tensor(self, feature_spec: FeatureSpec, is_training: bool, send_count: Optional[int]) -> dict:
        channel_id = ConfigInitializer.get_instance().train_params_config.get_training_mode_channel_id(is_training)
        config = dict(batch_size=feature_spec.batch_size, feat_cnt=feature_spec.feat_cnt, send_count=send_count,
                      rank_size=self._rank_size, channel_id=channel_id, table_name=self._table_name,
                      is_hbm=self._is_hbm, ext_emb_size=self._ext_emb_size,
                      emb_size=self._emb_size, device_id=self._device_id)

        variable_list = [self._variable] + \
                        [slot_info.get("slot") for slot_info in self.emb_optimizer.optimizer_slot_info_list]
        return get_preprocessed_tensor_for_asc(variable_list, config)


def _set_specific_value_for_non_valid_key(id_offsets: Optional[tf.Tensor],
                                          embeddings: Optional[tf.Tensor],
                                          access_threshold: Optional[int],
                                          serving_default_value: Optional[tf.Tensor] = None,
                                          is_training: bool = True) -> tf.Tensor:
    """
    将key为-1(无效值)的特征对应的emb置为0或者指定值.

    Args:
        id_offsets: 特征索引
        embeddings: 稀疏表
        access_threshold: 准入阈值
        serving_default_value: 参考sparse_lookup接口描述
        is_training: 当前流程是训练还是推理

    Returns: embeddings
    """
    # 在训练时，仅当开启准入功能才会出现无效值；推理时，是否开启准入都可能存在无效值
    if is_training and (access_threshold is None or access_threshold < 0):
        return embeddings

    if serving_default_value is None:
        # 未设置时，默认无效值的emb为全0
        default_value = tf.zeros_like(embeddings)
    else:
        try:
            default_value = tf.broadcast_to(serving_default_value, tf.shape(embeddings))
        except ValueError as e:
            logger.error("failed to broadcast serving_default_value to target embedding , please check its shape.")
            raise e
        except Exception as e:
            logger.error("failed to process serving_default_value.")
            raise e

    if tf.__version__.startswith("1"):
        id_offsets_expand = tf.math.greater_equal(id_offsets, 0)
        embeddings = tf.where(id_offsets_expand, embeddings, default_value)
        return embeddings

    id_offsets_expand = tf.compat.v1.expand_dims(id_offsets >= 0, axis=-1)
    embeddings = tf.where(id_offsets_expand, embeddings, default_value)
    return embeddings
