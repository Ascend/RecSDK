#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
import math
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

from mx_rec.core.asc.build_graph import get_preprocessed_tensor_for_asc
from mx_rec.core.asc.feature_spec import FeatureSpec, get_feature_spec
from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.constants.constants import ASCEND_SPARSE_LOOKUP_ENTRANCE, ASCEND_SPARSE_LOOKUP_HOT_POS, \
    ASCEND_SPARSE_LOOKUP_ID_OFFSET, ASCEND_SPARSE_LOOKUP_RESTORE_VECTOR, MxRecMode, \
    ASCAnchorAttr, ASCEND_SPARSE_LOOKUP_ALL2ALL_MATRIX, ASCEND_SPARSE_LOOKUP_LOCAL_EMB, \
    DEFAULT_EVICT_TIME_INTERVAL, TRAIN_CHANNEL_ID
from mx_rec.util.initialize import get_rank_id, get_rank_size, is_mpi_in_use, is_asc_frozen, get_customized_ops, \
    insert_table_instance, get_training_mode_channel_id, get_use_static, get_name_to_var_dict, \
    clear_channel, trigger_evict, get_table_instance_by_name, get_use_hot, get_device_id, export_feature_spec, \
    ConfigInitializer, get_ascend_global_hashtable_collection, get_host_pipeline_ops, get_use_dynamic_expansion
from mx_rec.util.tf_version_adapter import npu_ops
from mx_rec.util.variable import remove_saving_var


def create_table(**kwargs):
    key_dtype = kwargs.get("key_dtype")
    dim = kwargs.get("dim")
    name = kwargs.get("name")
    emb_initializer = kwargs.get("emb_initializer")
    device_vocabulary_size = kwargs.get("device_vocabulary_size", 1)
    host_vocabulary_size = kwargs.get("host_vocabulary_size", 0)
    optimizer_list = kwargs.get("optimizer_list", None)
    mode = kwargs.get("mode", MxRecMode.ASC)
    value_dtype = kwargs.get("value_dtype", tf.float32)
    shard_num = kwargs.get("shard_num", 1)
    fusion_optimizer_var = kwargs.get("fusion_optimizer_var", True)
    hashtable_threshold = kwargs.get("hashtable_threshold", 0)

    """
    Args:
        key_dtype: data type for feature id
        dim: embedding vector size
        name: hash table name
        emb_initializer: the initializer for embedding values
        device_vocabulary_size: embedding vector numbers on device
        host_vocabulary_size: embedding vector numbers on ddr
        relation from feature to variable offset will be built
        optimizer_list: specify the optimizers to use for current hash table
        mode: specify which mode to run for current sparse table
        value_dtype: the type of the value tensors.
        shard_num: embedding partition number
        fusion_optimizer_var: fusion optimizer variable with embedding
        hashtable_threshold: choose to implement based on hash table or linear layer
    """

    config = dict(key_dtype=key_dtype, embedding_size=dim, table_name=name, emb_initializer=emb_initializer,
                  device_vocabulary_size=device_vocabulary_size, host_vocabulary_size=host_vocabulary_size,
                  optimizer_list=optimizer_list, mode=mode, value_dtype=value_dtype, shard_num=shard_num,
                  fusion_optimizer_var=fusion_optimizer_var, hashtable_threshold=hashtable_threshold)
    embedding = SparseEmbedding(config)
    return embedding


def sparse_lookup(hashtable, ids, send_count, **kwargs):
    """

    Args:
        hashtable: SparseEmbedding instance to be looked up
        ids: Tensor to lookup from hashtable
        send_count: used to config all2all communication parameters
        kwargs:
            dim: not in use
            is_train: not in use
            name: will be used to build scope_name together with hashtable name
            modify_graph: if True, the original graph will be modified before building a Session instance

    Returns: Tensor for lookup result

    """

    def check_lookup_kwargs():
        kwargs["name"] = kwargs.get("name", hashtable.get_default_lookup_name())
        if not isinstance(kwargs.get("name"), str):
            raise TypeError("Given name must be a string.")

        kwargs["modify_graph"] = kwargs.get("modify_graph", False)
        if not isinstance(kwargs.get("modify_graph"), bool):
            raise TypeError("Given name must be a boolean.")

    def check_table_legality_for_feature_spec(table, feature_spec):
        # check whether the name of the table exists with FeatureSpec.
        if table.table_name != feature_spec.table_name:
            raise ValueError(f"The table name '{feature_spec.table_name}' specified by FeatureSpec is inconsistent with"
                             f" the SparseEmbedding table name '{table.table_name}'.")

    def check_modify_graph():
        if not kwargs.get("modify_graph"):
            logging.warning(f"MxRecMode {MxRecMode.ASC} must config with a 'True' "
                            f"modify_graph.")

    check_lookup_kwargs()
    scope_name = "{0}//{1}".format(hashtable.table_name, kwargs.get("name"))
    with tf.compat.v1.variable_scope(scope_name):
        if hashtable.mode == MxRecMode.ASC:
            if isinstance(ids, FeatureSpec):
                check_table_legality_for_feature_spec(hashtable, ids)
                return hashtable.lookup_for_asc_with_feature_spec(ids, send_count, **kwargs)
            else:
                check_modify_graph()
                return hashtable.lookup_for_asc(ids, send_count, **kwargs)
        else:
            raise EnvironmentError(f"Invalid MxRec Mode.")


class SparseEmbedding:
    """
    each feat_name has its own sparse_embedding_layer.
    """
    customized_ops = get_customized_ops()
    anchor_tensor_specs = defaultdict(dict)

    def __init__(self, config):
        self.embedding_size = config.get("embedding_size")
        if isinstance(self.embedding_size, int):
            self.embedding_size = tf.TensorShape([self.embedding_size])
        self.device_vocabulary_size = config.get("device_vocabulary_size")
        self.host_vocabulary_size = config.get("host_vocabulary_size")
        self.table_name = config.get("table_name")
        self.key_dtype = config.get("key_dtype")
        self._optimizer_instance_list = config.get("optimizer_list")
        self.emb_initializer = config.get("emb_initializer")
        self._mode = config.get("mode")
        self.optimizer_slot_info_list = []
        self._slot_num = dict()
        self._send_count = 0
        self._use_feature_mapping = False
        self.skip_emb_transfer = True if self.host_vocabulary_size <= 0 else False
        self._default_name_count = -1
        self.emb_size = None
        self.ext_emb_size = None
        self.ext_coefficient = 1
        self._optimizer = dict()
        self.slice_device_vocabulary_size = 0
        self.slice_host_vocabulary_size = 0
        self.variable = None
        self.lookup_info = set()
        self.lookup_result = None
        self.use_dynamic_expansion = get_use_dynamic_expansion()
        self.channel_name_list = []
        self.send_count_map = dict()
        self.channel_name_dict = {True: [], False: []}
        self.modify_graph = False

        self.set_slice_vocab_size()
        self.set_emb_size()
        if self._mode == MxRecMode.ASC and is_asc_frozen() and self.table_name in get_name_to_var_dict():
            self.variable = tf.compat.v1.get_variable(self.table_name,
                                                      shape=(self.slice_device_vocabulary_size, self.emb_size))
            if not self.skip_emb_transfer:
                self.set_ext_emb_size()
        else:
            self.check_and_format_init_params()
            self._initialize_variables()
            self.set_ext_emb_size()
            tf.compat.v1.add_to_collection(get_ascend_global_hashtable_collection(), self.variable)

    @property
    def use_feature_mapping(self):
        return self._use_feature_mapping

    @property
    def scalar_emb_size(self):
        return self.emb_size

    @property
    def mode(self):
        return self._mode

    @property
    def send_count(self):
        return self._send_count

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def optimizer_instance_list(self):
        return self._optimizer_instance_list

    @staticmethod
    def get_anchor_attribute(anchor, attr):
        if not isinstance(anchor, tf.Tensor):
            raise ValueError("Anchor must be a Tensor.")

        if attr not in ASCAnchorAttr:
            raise ValueError("Given attr must be limited in Enum 'ASCAnchorAttr'.")

        specs = SparseEmbedding.anchor_tensor_specs.get(anchor)
        if specs is None:
            raise ValueError(f"Given anchor '{anchor}' was not registered.")

        return specs.get(attr)

    @staticmethod
    def set_optimizer_slot(slot_info):
        slot = slot_info.get("slot")
        slot_name = slot_info.get("slot_name")
        optimizer = slot_info.get("optimizer")
        named_slot_key = slot_info.get("named_slot_key")

        optimizer.insert_slot(slot, named_slot_key, slot_name)

    def check_optimizer_instance(self):
        for optimizer_instance in self._optimizer_instance_list:
            if tf.__version__.startswith("1"):
                from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
                if isinstance(optimizer_instance, NPULossScaleOptimizer):
                    optimizer_instance = getattr(optimizer_instance, '_opt')
            else:
                from npu_device.train.optimizer.npu_loss_scale_optimizer import NpuLossScaleOptimizer
                if isinstance(optimizer_instance, NpuLossScaleOptimizer):
                    optimizer_instance = getattr(optimizer_instance, '_opt')

            if not isinstance(optimizer_instance, CustomizedOptimizer):
                raise ValueError(f"args optimizer list must be a list or an instance of CustomizedOptimizer.")

    def check_and_format_init_params(self):
        if not isinstance(self.embedding_size, tf.TensorShape):
            raise TypeError("Parameter 'embedding_size' must be a tf.TensorShape instance.")

        if self.embedding_size.ndims != 1:
            raise ValueError("Parameter 'embedding_size' can only be one dim shape.")

        if self.mode == MxRecMode.ASC and is_asc_frozen():
            raise EnvironmentError(f"Emb cache management has been established, you cannot build new ASC hash table.")

        if self.mode != MxRecMode.ASC and self.host_vocabulary_size > 0:
            raise ValueError(f"Only ASC mode can use host_vocabulary_size > 0.")

        if self.mode == MxRecMode.ASC and not is_mpi_in_use():
            raise EnvironmentError(f"Hash table with ASC mode must use mpi to start task.")

        if not self.skip_emb_transfer and not self._optimizer_instance_list:
            raise ValueError("ASC with DDR mode should config optimizers before instantiating sparse table, "
                             "but nothing was configured.")
        if not self.skip_emb_transfer and self.use_dynamic_expansion:
            raise ValueError("DDR mode do not support embedding dynamic_expansion for now.")

        self._optimizer_instance_list = [] if self._optimizer_instance_list is None else self._optimizer_instance_list
        if isinstance(self._optimizer_instance_list, CustomizedOptimizer):
            self._optimizer_instance_list = [self._optimizer_instance_list]

        if not isinstance(self._optimizer_instance_list, (tuple, list)):
            raise ValueError(f"args optimizer list must be a list or an instance of CustomizedOptimizer.")
        self._optimizer_instance_list = list(self._optimizer_instance_list)

        self.check_optimizer_instance()

    def get_default_lookup_name(self):
        logging.debug(f"getting one default lookup name")
        self._default_name_count += 1
        return "sparse_lookup_%d" % self._default_name_count

    def set_using_feature_mapping(self):
        self._use_feature_mapping = True

    def set_emb_size(self):
        self.emb_size = self.embedding_size.as_list()[0]

    def set_ext_emb_size(self):
        self.ext_coefficient += len(self.optimizer_slot_info_list)
        if self.use_dynamic_expansion and len(self._optimizer_instance_list) != 0:
            self.ext_coefficient += self._slot_num.get(self.table_name)
        self.ext_emb_size = self.emb_size * self.ext_coefficient
        logging.debug(f"init table, ext_emb_size is set to be {self.ext_emb_size}")

    def set_slice_vocab_size(self):
        rank_size = get_rank_size()
        if rank_size == 0:
            raise ZeroDivisionError("Rank size cannot be zero.")
        if self.use_dynamic_expansion:
            self.slice_device_vocabulary_size = 1  # 动态扩容模式下，保留device侧variable，大小设置为 1
            self.slice_host_vocabulary_size = 0
        else:
            self.slice_device_vocabulary_size = math.ceil(self.device_vocabulary_size / rank_size)
            self.slice_host_vocabulary_size = math.ceil(self.host_vocabulary_size / rank_size)

    def register_anchor_attribute(self, anchor_ids, feature_spec, kwargs):
        SparseEmbedding.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.TABLE_INSTANCE] = self
        SparseEmbedding.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.IS_TRAINING] = kwargs.get("is_train")
        SparseEmbedding.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.FEATURE_SPEC] = feature_spec

    def check_mode(self, method_mode):
        if self.mode != method_mode:
            raise RuntimeError(f"Current sparse table was config in {self.mode.value} mode, but sparse lookup method "
                               f"for {method_mode} was in use.")

    def check_and_format_lookup_params(self, feature, send_count, is_training):
        logging.debug(f"sparse lookup for table {self.table_name} with is_training {is_training}")

        def check_params():
            if not isinstance(is_training, bool):
                raise ValueError("Arg is_train should be a boolean.")

            if isinstance(feature, FeatureSpec):
                if not feature.initialized:
                    raise ValueError(f"Feature Spec has not been initialized.")
                key_info = "{}_{}".format(feature.name, feature.index_key)
                if is_training not in feature.pipeline_mode:
                    raise ValueError(f"You have not config feature for is training mode '{is_training}', please config "
                                     f"feature with func sparse_lookup at first.")

            elif isinstance(feature, tf.Tensor):
                logging.debug("Input feature is a Tensor.")

            else:
                raise TypeError(f"Given feature must be a FeatureSpec or tf.Tensor.")

            if is_training not in self.lookup_info:
                self.lookup_info.add(is_training)

            if get_use_static():
                if isinstance(send_count, int) and send_count > 0:
                    if self._send_count and self._send_count != send_count:
                        logging.warning(f"A new send count {send_count} will be used to replace the old one"
                                        f"({self._send_count}).")

                    self._send_count = send_count
                else:
                    raise ValueError("Send count must be a integer which is larger than 0.")

        check_params()
        max_int32 = np.iinfo(np.int32).max
        if self.slice_host_vocabulary_size + self.slice_device_vocabulary_size > max_int32:
            raise ValueError(f"Given device_vocabulary_size and host_vocabulary_size was too big for table "
                             f"'{self.table_name}', in which slice_device_vocabulary_size was "
                             f"{self.slice_device_vocabulary_size} and slice_host_vocabulary_size was "
                             f"{self.slice_host_vocabulary_size} ")

        is_check_mode = self.mode == MxRecMode.ASC and not self.skip_emb_transfer and not self.use_dynamic_expansion
        if is_check_mode and self.slice_device_vocabulary_size < self.send_count * get_rank_size():
            raise ValueError(f"Given device_vocabulary_size was too small for table '{self.table_name}', in which "
                             f"slice_device_vocabulary_size was {self.slice_device_vocabulary_size} and "
                             f"send_count({self.send_count}) * rank_size({get_rank_size()}) was "
                             f"{self.send_count * get_rank_size()}")

        if is_check_mode and self.slice_host_vocabulary_size < self.send_count * get_rank_size():
            raise ValueError(f"Given host_vocabulary_size was too small for table '{self.table_name}', in which "
                             f"slice_host_vocabulary_size was {self.slice_host_vocabulary_size} and "
                             f"send_count({self.send_count}) * rank_size({get_rank_size()}) was "
                             f"{self.send_count * get_rank_size()}")

    def set_optimizer(self, key, state_dict):
        if key in self._optimizer:
            raise ValueError(f"Optimizer {key} has been set for hash table {self.table_name}")

        self._optimizer[key] = state_dict

    def set_channel_name(self, ids_channel_name, eval_mode):
        self.channel_name_list.append(ids_channel_name)
        if not eval_mode:
            self.channel_name_dict.get(True).insert(0, ids_channel_name)
        self.channel_name_dict.get(False).insert(0, ids_channel_name)

    def lookup_for_asc(self, ids: tf.Tensor, send_count, **kwargs):
        """

        Args:
            ids: Tensor to lookup from hashtable
            send_count: int, used to config all2all communication parameters
            kwargs:
                dim: not in use
                is_train:
                name: not in use
                modify_graph: if True, the original graph will be modified before building a Session instance

        Returns: Tensor for lookup result

        """
        logging.debug(f"Enter ASC Branch.")
        if not kwargs.get("modify_graph"):
            raise ValueError(f"modify_graph must be turn-on when lookup by ids(Tensor, not FeatureSpec).")

        self.check_mode(MxRecMode.ASC)
        is_training = kwargs.get("is_train")
        if is_asc_frozen() and is_training:
            raise RuntimeError(f"Cannot build new sparse forward graph after emb cache management was built.")

        feature_spec = get_feature_spec(self.table_name, kwargs.get("access_and_evict_config"))
        feature_spec.set_feat_attribute(ids, is_training)
        # 'clear_channel()' function needs to be executed after 'set_feat_attribute()' function
        if is_asc_frozen() and not is_training:
            clear_channel(is_train_channel=False)

        self.check_and_format_lookup_params(ids, send_count, is_training)
        anchor_ids = tf.identity(ids, name="ids")
        tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE, anchor_ids)
        self.register_anchor_attribute(anchor_ids, feature_spec, kwargs)

        use_dynamic_expansion = get_use_dynamic_expansion()
        use_static = get_use_static()
        use_hot = get_use_hot()
        eval_mode = not is_training and len(self.channel_name_dict.get(not is_training)) == 0
        ids_channel_name = ""
        # set in train mode, train and eval mode, eval mode
        if is_training or eval_mode:
            ids_channel_name = feature_spec.name + "_lookup_ids"
            self.set_channel_name(ids_channel_name, eval_mode)
            send_count = send_count if send_count is not None else 0
            self._send_count = send_count
            self.send_count_map[ids_channel_name] = send_count
        self.modify_graph = kwargs.get("modify_graph", True)
        logging.debug(f"In lookup_for_asc function, table name: {self.table_name}, anchor_ids: {anchor_ids}, "
                      f"ids_channel_name: {ids_channel_name}, use_dynamic_expansion: {use_dynamic_expansion}, "
                      f"use_static: {use_static}, use_hot: {use_hot}")

        rank_size = get_rank_size()
        id_offsets = tf.ones(shape=[send_count * rank_size if use_static else 1 * rank_size, ],
                             dtype=tf.int64 if get_use_dynamic_expansion() else tf.int32, name="id_offsets")
        id_offsets = tf.identity(id_offsets, name=ASCAnchorAttr.ID_OFFSETS.value)
        SparseEmbedding.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.ID_OFFSETS] = id_offsets
        local_embeddings = None
        if use_dynamic_expansion:
            local_embeddings = get_host_pipeline_ops().embedding_lookup_by_address(id_offsets,
                                                                                   embedding_dim=self.emb_size,
                                                                                   embedding_type=1)
        if is_training:
            tf.add_to_collection(ASCEND_SPARSE_LOOKUP_ID_OFFSET, id_offsets)
            tf.add_to_collection(ASCEND_SPARSE_LOOKUP_LOCAL_EMB, local_embeddings)

        @tf.custom_gradient
        def sparse_forward(table, feat_ids):
            logging.debug(f"fp rank size: {rank_size}")
            if feat_ids.shape.as_list()[0] is not None:
                restore_vector = tf.ones(shape=[np.prod(feat_ids.shape.as_list()), ], dtype=tf.int32,
                                         name="restore_vector")
            else:
                restore_vector = tf.ones(shape=[tf.math.reduce_prod(array_ops.shape(feat_ids)[0]), ], dtype=tf.int32,
                                         name="restore_vector")

            restore_vector = tf.identity(restore_vector, name=ASCAnchorAttr.RESTORE_VECTOR.value)
            tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_RESTORE_VECTOR, restore_vector)
            SparseEmbedding.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.RESTORE_VECTOR] = restore_vector

            all2all_matrix = None
            if not use_static:
                all2all_matrix = tf.ones(shape=[rank_size, rank_size], dtype=tf.int64, name="all2all_matrix")
                all2all_matrix = tf.identity(all2all_matrix, name=ASCAnchorAttr.ALL2ALL_MATRIX.value)
                tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_ALL2ALL_MATRIX, all2all_matrix)
                SparseEmbedding.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.ALL2ALL_MATRIX] = all2all_matrix

            hot_pos = None
            if use_hot:
                import mxrec_pybind
                hot_size = int(mxrec_pybind.get_ub_hot_size(get_device_id()) / self.emb_size)
                hot_pos = tf.ones(shape=[hot_size, ], dtype=tf.int32, name="hot_pos")
                hot_pos = tf.identity(hot_pos, name=ASCAnchorAttr.HOT_POS.value)
                tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_HOT_POS, hot_pos)
                SparseEmbedding.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.HOT_POS] = hot_pos

            if not use_dynamic_expansion:
                id_offsets_abs = tf.abs(id_offsets)
                local_emb = tf.gather(table, id_offsets_abs, axis=0, name="gather_for_id_offsets")
            else:
                local_emb = tf.identity(table, name="identity_local_emb")
            all2all_args = send_count if use_static else all2all_matrix
            unique_embeddings = get_own_emb(local_emb, all2all_args, self.scalar_emb_size, use_static)

            if hot_pos is not None:
                unique_embeddings = tf.concat([tf.gather(unique_embeddings, hot_pos, name="hot_pos"),
                                               unique_embeddings], axis=0)

            embeddings = tf.gather(unique_embeddings, restore_vector, axis=0, name="gather_for_restore_vector")
            if use_static:
                lookup_result = tf.reshape(embeddings, feat_ids.shape.as_list() + [self.scalar_emb_size])
            else:
                dest_shape = array_ops.concat([array_ops.shape(feat_ids), [self.scalar_emb_size]], 0)
                lookup_result = array_ops.reshape(embeddings, dest_shape)

            def grad(lookup_diff):
                embedding_diff = tf.reshape(lookup_diff, [-1, self.scalar_emb_size])
                logging.debug(f"bp rank size: {rank_size}")
                unique_embeddings_shape = unique_embeddings.shape.as_list() if use_static \
                    else tf.shape(unique_embeddings)
                unique_grads = tf.compat.v1.unsorted_segment_sum(embedding_diff,
                                                                 restore_vector,
                                                                 unique_embeddings_shape[0])
                bp_all2all_args = all2all_args if use_static else tf.transpose(all2all_args)
                if hot_pos is not None:
                    hot, cold = tf.split(unique_grads, [tf.shape(hot_pos)[0],
                                                        tf.shape(unique_grads)[0] - tf.shape(hot_pos)[0]], axis=0)
                    unique_grads = tf.tensor_scatter_nd_update(cold, tf.expand_dims(hot_pos, 1), hot)
                local_grad = get_own_emb(unique_grads, bp_all2all_args, self.scalar_emb_size, use_static)

                if use_dynamic_expansion:
                    return local_grad, feat_ids
                return ops.IndexedSlices(values=local_grad, indices=id_offsets, dense_shape=tf.shape(table)), feat_ids

            return lookup_result, grad

        if use_dynamic_expansion:
            return sparse_forward(local_embeddings, ids)
        return sparse_forward(self.variable, ids)

    def lookup_for_asc_with_feature_spec(self, feature_spec: FeatureSpec, send_count: int, **kwargs):
        """
        Args:
            feature_spec: an instance of FeatureSpec to lookup from hashtable
            send_count: int, used to config all2all communication parameters
            kwargs:
                dim: not in use
                is_train:
                name: not in use
                modify_graph: if True, the original graph will be modified before building a Session instance

        Returns: Tensor for lookup result

        """
        spec_name = feature_spec.name
        is_training = kwargs.get("is_train")
        if self.lookup_result is not None and spec_name in self.lookup_result \
                and is_training in self.lookup_result.get(spec_name):
            return self.lookup_result.get(spec_name).get(is_training)

        table_name = feature_spec.table_name
        same_table_feature_spec = ConfigInitializer.get_instance().table_name_to_feature_spec[table_name][is_training]
        if len(same_table_feature_spec) == 0:
            raise RuntimeError(f"spec_name {spec_name} not in table {table_name}")
        if len(same_table_feature_spec) == 1:
            lookup_result = self.lookup_for_asc_with_feature_spec_inner(feature_spec, send_count, **kwargs)
            self.lookup_result = {spec_name: {is_training: lookup_result}}
            return self.lookup_result.get(spec_name).get(is_training)
        else:
            same_table_feature_spec = sorted(same_table_feature_spec, key=lambda x: x.name)
            same_table_spec_count = len(same_table_feature_spec)
            feature_count = [x.feat_cnt * x.batch_size for x in same_table_feature_spec]
            total_feature_count = sum(feature_count)
            mock_feature_spec = FeatureSpec(f"mock_feature_spec_{table_name}",
                                            feat_count=total_feature_count, table_name=table_name)
            mock_feature_spec.batch_size = 1
            mock_feature_spec.dims = [1, total_feature_count]
            mock_feature_spec.initialized = True
            mock_feature_spec.pipeline_mode.add(True)
            mock_feature_spec.pipeline_mode.add(False)
            lookup_result = self.lookup_for_asc_with_feature_spec_inner(mock_feature_spec,
                                                                        send_count * same_table_spec_count, **kwargs)
            logging.debug(f"lookup table {table_name} via {feature_count}")
            lookup_result = tf.reshape(lookup_result, [-1, self.scalar_emb_size])
            split_size = [x.feat_cnt * x.batch_size for x in same_table_feature_spec]
            lookup_result_split = tf.split(lookup_result, split_size)
            self.lookup_result = {k.name: {is_training: tf.reshape(v, k.dims + [self.scalar_emb_size])}
                                  for k, v in zip(same_table_feature_spec, lookup_result_split)}
            return self.lookup_result.get(spec_name).get(is_training)

    def lookup_for_asc_with_feature_spec_inner(self, feature_spec: FeatureSpec, send_count: int, **kwargs):
        """
        Args:
            feature_spec: an instance of FeatureSpec to lookup from hashtable
            send_count: int, used to config all2all communication parameters
            kwargs:
                dim: not in use
                is_train:
                name: not in use
                modify_graph: if True, the original graph will be modified before building a Session instance

        Returns: Tensor for lookup result

        """
        logging.debug(f"Enter ASC Branch, looking up with FeatureSpec.")
        self.check_mode(MxRecMode.ASC)
        is_training = kwargs.get("is_train")
        self.check_and_format_lookup_params(feature_spec, send_count, is_training)
        rank_size = get_rank_size()
        device_id = get_device_id()
        use_hot = get_use_hot()
        use_dynamic_expansion = get_use_dynamic_expansion()

        # check training mode order and ensure channel id
        channel_id = get_training_mode_channel_id(is_training=is_training)
        logging.debug(f"get preprocessed tensor for asc for table {self.table_name} with skip emb transfer "
                      f"{self.skip_emb_transfer} is_training: {is_training}, channel_id: {channel_id} .")

        config = dict(batch_size=feature_spec.batch_size, feat_cnt=feature_spec.feat_cnt, send_count=send_count,
                      rank_size=rank_size, channel_id=channel_id, table_name=self.table_name,
                      skip_emb_transfer=self.skip_emb_transfer, ext_emb_size=self.ext_emb_size,
                      emb_size=self.emb_size, use_hot=use_hot, device_id=device_id,
                      use_dynamic_expansion=use_dynamic_expansion)

        if self.skip_emb_transfer:
            result = get_preprocessed_tensor_for_asc(self.variable, config)
        else:
            variable_list = [self.variable] + [slot_info.get("slot") for slot_info in self.optimizer_slot_info_list]
            result = get_preprocessed_tensor_for_asc(variable_list, config)
        restore_vector = result.get("restore_vector")
        hot_pos = result.get("hot_pos")
        id_offsets = result.get("id_offsets")
        swap_in = result.get("swap_in")
        all2all_matrix = result.get("all2all_matrix")
        control_ops = swap_in

        id_offsets = tf.identity(id_offsets, name="identity_addr")
        restore_vector = tf.identity(restore_vector, name="identity_restore")
        if is_training and use_dynamic_expansion:
            tf.add_to_collection(ASCEND_SPARSE_LOOKUP_ID_OFFSET, id_offsets)

        use_static = get_use_static()
        host_pipeline_ops = get_host_pipeline_ops()

        @tf.custom_gradient
        def sparse_forward(table):
            logging.debug(f"fp rank size: {rank_size}")
            if not use_dynamic_expansion:
                id_offsets_abs = tf.abs(id_offsets)
                local_embeddings = tf.gather(table, id_offsets_abs, axis=0, name="gather_for_id_offsets")
            else:
                local_embeddings = tf.identity(table, name="identity_local_emb")

            all2all_args = send_count if use_static else all2all_matrix
            unique_embeddings = get_own_emb(local_embeddings, all2all_args, self.scalar_emb_size, use_static)

            if hot_pos is not None:
                unique_embeddings = tf.concat([tf.gather(unique_embeddings, hot_pos, name="hot_pos"),
                                               unique_embeddings], axis=0)
            if use_static:
                unique_embeddings_shape = unique_embeddings.shape.as_list()
            else:
                unique_embeddings_shape = tf.shape(unique_embeddings)
            embeddings = tf.gather(unique_embeddings, restore_vector, axis=0, name="gather_for_restore_vector")

            if use_static:
                lookup_result = tf.reshape(embeddings, feature_spec.dims + [self.scalar_emb_size])
            else:
                lookup_result = tf.reshape(embeddings, [-1, self.scalar_emb_size])

            def grad(lookup_diff):
                embedding_diff = tf.reshape(lookup_diff, [-1, self.scalar_emb_size])
                logging.debug(f"bp rank size: {rank_size}")
                unique_grads = tf.compat.v1.unsorted_segment_sum(embedding_diff,
                                                                 restore_vector,
                                                                 unique_embeddings_shape[0])
                bp_all2all_args = all2all_args if use_static else tf.transpose(all2all_args)
                if hot_pos is not None:
                    hot, cold = tf.split(unique_grads, [tf.shape(hot_pos)[0],
                                                        tf.shape(unique_grads)[0] - tf.shape(hot_pos)[0]], axis=0)
                    unique_grads = tf.tensor_scatter_nd_update(cold, tf.expand_dims(hot_pos, 1), hot)
                local_grad = get_own_emb(unique_grads, bp_all2all_args, self.scalar_emb_size, use_static)
                if use_dynamic_expansion:
                    update_grad = local_grad
                else:
                    update_grad = ops.IndexedSlices(values=local_grad, indices=id_offsets,
                                                    dense_shape=tf.shape(table))
                return update_grad

            return lookup_result, grad

        with tf.control_dependencies(control_ops):
            if not use_dynamic_expansion:
                return sparse_forward(self.variable)

            local_embeddings = \
                host_pipeline_ops.embedding_lookup_by_address(id_offsets, embedding_dim=self.emb_size,
                                                              embedding_type=1)
            if is_training:
                tf.add_to_collection(ASCEND_SPARSE_LOOKUP_LOCAL_EMB, local_embeddings)

            return sparse_forward(local_embeddings)

    def _record(self):
        insert_table_instance(self.table_name, self.variable, self)
        logging.debug(f"Device vocabulary_size for table {self.table_name} is {self.device_vocabulary_size}.")
        logging.debug(f"Slice_device_vocabulary_size for table {self.table_name} is"
                      f" {self.slice_device_vocabulary_size}.")
        logging.debug(f"Host vocabulary size for table {self.table_name} is {self.host_vocabulary_size}.")
        logging.debug(f"Slice host vocabulary_size for table {self.table_name} is"
                      f" {self.slice_host_vocabulary_size}.")

    def _initialize_variables(self):
        initialized_tensor = self.emb_initializer(self.slice_device_vocabulary_size + self.embedding_size)
        self.variable = tf.compat.v1.get_variable(self.table_name, trainable=False, initializer=initialized_tensor)
        # make sure sparse table variable will not be saved and restored within tf checkpoint.
        remove_saving_var(self.variable)
        self._record()

        if self.use_dynamic_expansion:
            for sparse_optimizer_instance in self._optimizer_instance_list:
                self._slot_num[self.table_name] = sparse_optimizer_instance.slot_num
                logging.info(f"init emb, table name: {self.table_name}, slot_num: {sparse_optimizer_instance.slot_num}")

        if self.mode == MxRecMode.ASC and not self.skip_emb_transfer:
            # build optimizer states
            for sparse_optimizer_instance in self._optimizer_instance_list:
                slot_info_list = sparse_optimizer_instance.initialize_slots(self.variable, self)
                self.optimizer_slot_info_list.extend(slot_info_list)

            for slot_info in self.optimizer_slot_info_list:
                self.set_optimizer_slot(slot_info)


def get_own_ids(unique_ids, origin_id_lens, send_cnt, self):
    from mx_rec.util.tf_version_adapter import hccl_ops
    rank_size = get_rank_size()
    if rank_size > 1:
        ids_send_cnt = tf.constant([send_cnt] * rank_size, dtype=tf.int64)
        ids_send_offset = tf.constant([send_cnt * i for i in range(rank_size)], dtype=tf.int64)
        own_ids = hccl_ops.all_to_all_v(send_data=unique_ids,
                                        send_counts=ids_send_cnt,
                                        send_displacements=ids_send_offset,
                                        recv_counts=ids_send_cnt,
                                        recv_displacements=ids_send_offset)

        lens_sc = tf.constant([1] * rank_size, dtype=tf.int64)
        lens_sd = tf.constant([i for i in range(rank_size)], dtype=tf.int64)
        local_id_lens = hccl_ops.all_to_all_v(send_data=origin_id_lens,
                                              send_counts=lens_sc,
                                              send_displacements=lens_sd,
                                              recv_counts=lens_sc,
                                              recv_displacements=lens_sd)

    else:
        own_ids = unique_ids
        local_id_lens = origin_id_lens

    def feature_mapping():
        self.set_using_feature_mapping()
        id_offsets = SparseEmbedding.customized_ops.feature_mapping(own_ids, table_name=self.table_name)
        return id_offsets

    id_offsets = feature_mapping()
    id_offsets.set_shape([send_cnt * rank_size])

    return id_offsets, local_id_lens


def get_own_emb(emb, all2all_args, emb_size, use_static):
    '''
    obtain embedding of source data
    '''
    from mx_rec.util.tf_version_adapter import hccl_ops
    rank_size = get_rank_size()
    rank_id = get_rank_id()

    src_emb = emb

    reshape_info = [all2all_args * rank_size, emb_size] if use_static else [-1, emb_size]

    if rank_size == 1 and use_static:
        return tf.reshape(src_emb, reshape_info)

    if use_static:
        emb_send_cnt = tf.constant([all2all_args * emb_size] * rank_size, dtype=tf.int64)
        emb_send_offset = tf.constant([all2all_args * emb_size * i for i in range(rank_size)], dtype=tf.int64)
        src_emb = hccl_ops.all_to_all_v(send_data=emb,
                                        send_counts=emb_send_cnt,
                                        send_displacements=emb_send_offset,
                                        recv_counts=emb_send_cnt,
                                        recv_displacements=emb_send_offset)
    else:
        src_emb = hccl_ops.all_to_all_v_c(send_data=emb,
                                          send_count_matrix=all2all_args,
                                          rank=rank_id)

    return tf.reshape(src_emb, reshape_info)


class _EvictHook(tf.compat.v1.train.SessionRunHook):
    """Sets evict based on global step or time."""

    def __init__(self,
                 evict_enable=False,
                 evict_time_interval=DEFAULT_EVICT_TIME_INTERVAL,
                 evict_step_interval=None):
        self._evict_enable = evict_enable
        self._evict_time_interval = evict_time_interval
        self._evict_step_interval = evict_step_interval
        self._hash_table_instance = dict()
        self._start_time = time.time()
        self._global_step = 0
        self._evict_op = dict()
        self._global_step_tensor = None

        self.check_evict_init_params()
        logging.info(f"_EvictHook - > evict_time_interval: {self._evict_time_interval}, "
                     f"evict_step_interval: {self._evict_step_interval}")

    def begin(self):
        self._global_step_tensor = tf.compat.v1.train.get_or_create_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use _EvictHook.")
        self.check_name_and_get_hashtable()
        for name, instance in self._hash_table_instance.items():
            scope_name = "{0}//{1}".format(instance.table_name, "evict")
            with tf.compat.v1.variable_scope(scope_name):
                logging.debug(f'Channel {instance.table_name}_evict_{TRAIN_CHANNEL_ID} was built for op '
                              f'getnext')

                use_static = get_use_static()
                if use_static:
                    evict_pos = npu_ops.gen_npu_ops.get_next(
                        output_types=[tf.int32],
                        output_shapes=[instance.slice_device_vocabulary_size],
                        channel_name=f'{instance.table_name}_evict_{TRAIN_CHANNEL_ID}')[0]
                    initialized_tensor = instance.emb_initializer(
                        instance.slice_device_vocabulary_size + instance.embedding_size)
                else:
                    evict_pos = npu_ops.gen_npu_ops.get_next(
                        output_types=[tf.int32],
                        output_shapes=[None],
                        channel_name=f'{instance.table_name}_evict_{TRAIN_CHANNEL_ID}')[0]

                    initialized_tensor = instance.emb_initializer(
                        evict_pos.shape.as_list()[0] + instance.embedding_size)

                logging.debug(f'evict_pos output shape {evict_pos}, and slice_device_vocabulary_size '
                              f'{instance.slice_device_vocabulary_size}, '
                              f'initialized_tensor shape: {initialized_tensor}')

                nd_evict_pos = tf.expand_dims(evict_pos, 1)
                self._evict_op[name] = tf.compat.v1.scatter_nd_update(instance.variable, nd_evict_pos,
                                                                      initialized_tensor)

    def after_create_session(self, session, coord):
        self._global_step = session.run(self._global_step_tensor)
        logging.debug(f"_EvictHook - > after_create_session, step: {self._global_step}")

    def after_run(self, run_context, run_values):
        if not self._evict_enable:
            return

        self._global_step = run_context.session.run(self._global_step_tensor)
        cur_time = time.time()
        if cur_time - self._start_time > self._evict_time_interval or \
                (self._evict_step_interval is not None and self._global_step % self._evict_step_interval == 0):
            logging.info(f"_EvictHook - > evict switch on!!! after_run step: {self._global_step}")
            trigger_evict()
            self._start_time = cur_time
            for name in self._hash_table_instance.keys():
                run_context.session.run(self._evict_op.get(name))

    def check_name_and_get_hashtable(self):
        for _, feature_spec in export_feature_spec().items():
            if feature_spec.eviction_threshold:
                logging.debug(f"_EvictHook - > check and get instance: table_names {feature_spec.table_name}")
                self._hash_table_instance[feature_spec.table_name] = get_table_instance_by_name(feature_spec.table_name)

    def check_evict_init_params(self):
        def check_type(arg, n_type, param_name):
            if not isinstance(arg, n_type):
                raise TypeError(f"{param_name} should be type '{n_type}', whose value is {arg} with type "
                                f"'{type(arg)}' in fact.")
            if type(arg) == int and arg < 1:
                raise ValueError(f"{param_name} should be bigger than 0, whose value is {arg} in fact")

        check_type(self._evict_enable, bool, "evict_enable")
        if self._evict_time_interval is not None:
            check_type(self._evict_time_interval, int, "evict_time_interval")
        if self._evict_step_interval is not None:
            check_type(self._evict_step_interval, int, "evict_time_interval")
