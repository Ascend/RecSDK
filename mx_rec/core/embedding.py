#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
import math
import os
import re
from collections import defaultdict
from typing import Optional

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.init_ops import Initializer as InitializerV1
from tensorflow.python.ops.init_ops_v2 import Initializer as InitializerV2

from mx_rec.core.asc.build_graph import get_preprocessed_tensor_for_asc
from mx_rec.core.asc.feature_spec import FeatureSpec, get_feature_spec, set_temporary_feature_spec_attribute
from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.constants.constants import (ASCEND_SPARSE_LOOKUP_ENTRANCE, ASCEND_SPARSE_LOOKUP_ID_OFFSET,\
    ASCEND_SPARSE_LOOKUP_UNIQUE_KEYS, MxRecMode, ASCAnchorAttr, ASCEND_SPARSE_LOOKUP_LOCAL_EMB, MULTI_LOOKUP_TIMES,\
    ASCEND_TABLE_NAME_MUST_CONTAIN, MAX_INT32, All2allGradientsOp, ApplyGradientsStrategy, MAX_HOST_VOCABULARY_SIZE)
from mx_rec.util.initialize import get_rank_id, get_rank_size, is_mpi_in_use, is_asc_frozen, get_customized_ops, \
    insert_table_instance, get_training_mode_channel_id, get_use_static, get_name_to_var_dict, \
    clear_channel, get_use_hot, get_device_id, ConfigInitializer, get_ascend_global_hashtable_collection, \
    get_host_pipeline_ops, get_use_dynamic_expansion, set_modify_graph, insert_removing_var_list, get_bool_gauge_set, \
    get_table_instance_by_name
from mx_rec.validator.validator import ClassValidator, StringValidator
from mx_rec.util.tf_version_adapter import npu_ops
from mx_rec.util.normalization import fix_invalid_table_name


def check_ssd_relate_param(host_vocabulary_size, ssd_vocabulary_size, ssd_data_path):
    h_size = 0
    s_size = 0
    try:
        h_size = int(host_vocabulary_size)
        s_size = int(ssd_vocabulary_size)
    except ValueError:
        raise ValueError("host_vocabulary_size and ssd_vocabulary_size should be integer")
    if h_size == 0 and s_size != 0:
        raise ValueError("ssd_vocabulary_size value is invalid, it effected by host_vocabulary_size not zero")
    if h_size != 0 and s_size < 0:
        raise ValueError("ssd_vocabulary_size value is invalid, it need be greater than 0")
    invalid_ssd_data_path = []
    for tmp_path in ssd_data_path:
        if is_invalid_path(tmp_path):
            invalid_ssd_data_path.append(tmp_path)
    if invalid_ssd_data_path:
        raise ValueError("ssd_data_path value is invalid, detail:{}, the path need exist and is real path"
                         .format(", ".join(invalid_ssd_data_path)))


def is_invalid_path(tmp_path):
    return not os.path.exists(tmp_path) or not os.path.isdir(tmp_path) or os.path.islink(tmp_path) or ".." in tmp_path


def create_table(**kwargs):
    """
    Args:
        key_dtype: data type for feature id
        dim: embedding vector size
        name: hash table name
        emb_initializer: the initializer for embedding values
        device_vocabulary_size: embedding vector numbers on device
        host_vocabulary_size: embedding vector numbers on ddr
        ssd_vocabulary_size: embedding vector numbers on ssd
        ssd_data_path: ssd embedding data save and load path
        relation from feature to variable offset will be built
        optimizer_list: specify the optimizers to use for current hash table
        mode: specify which mode to run for current sparse table
        value_dtype: the type of the value tensors.
        shard_num: embedding partition number
        fusion_optimizer_var: fusion optimizer variable with embedding
        hashtable_threshold: choose to implement based on hash table or linear layer
        is_save: switch whether to store sparse table data.
        init_param: embedding init param-coefficient
        all2all_gradients_op: sum_grads (default) or sum_gradients_and_div_by_ranksize.
        apply_gradients_strategy: direct_apply (default) or sum_same_id_gradients_and_apply.

    """
    key_dtype = kwargs.get("key_dtype")
    dim = kwargs.get("dim")
    name = kwargs.get("name")
    emb_initializer = kwargs.get("emb_initializer")
    device_vocabulary_size = kwargs.get("device_vocabulary_size", 1)
    host_vocabulary_size = kwargs.get("host_vocabulary_size", 0)
    ssd_vocabulary_size = kwargs.get("ssd_vocabulary_size", 0)
    ssd_data_path = kwargs.get("ssd_data_path", [os.getcwd()])
    optimizer_list = kwargs.get("optimizer_list")
    mode = kwargs.get("mode", MxRecMode.ASC)
    value_dtype = kwargs.get("value_dtype", tf.float32)
    shard_num = kwargs.get("shard_num", 1)
    fusion_optimizer_var = kwargs.get("fusion_optimizer_var", True)
    hashtable_threshold = kwargs.get("hashtable_threshold", 0)
    is_save = kwargs.get("is_save", True)
    init_param = kwargs.get("init_param", 1.0)
    all2all_gradients_op = kwargs.get("all2all_gradients_op", All2allGradientsOp.SUM_GRADIENTS)
    apply_gradients_strategy = kwargs.get("apply_gradients_strategy", ApplyGradientsStrategy.DIRECT_APPLY)

    name = fix_invalid_table_name(name)
    check_create_table_params(key_dtype, dim, name, emb_initializer)
    check_ssd_relate_param(host_vocabulary_size, ssd_vocabulary_size, ssd_data_path)

    config = dict(key_dtype=key_dtype, embedding_size=dim, table_name=name, emb_initializer=emb_initializer,
                  device_vocabulary_size=device_vocabulary_size, host_vocabulary_size=host_vocabulary_size,
                  ssd_vocabulary_size=ssd_vocabulary_size, ssd_data_path=ssd_data_path,
                  optimizer_list=optimizer_list, mode=mode, value_dtype=value_dtype, shard_num=shard_num,
                  fusion_optimizer_var=fusion_optimizer_var, hashtable_threshold=hashtable_threshold,
                  init_param=init_param, is_save=is_save, all2all_gradients_op=all2all_gradients_op,
                  apply_gradients_strategy=apply_gradients_strategy)
    embedding = SparseEmbedding(config)
    return embedding


def sparse_lookup(hashtable, ids, send_count, is_train, **kwargs):
    """
    Args:
        hashtable: SparseEmbedding instance to be looked up
        ids: Tensor to lookup from hashtable
        send_count: used to config all2all communication parameters
        is_train: indicates whether the mode is train.
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
            raise TypeError("Given modify_graph must be a boolean.")

        if not isinstance(kwargs.get("is_train"), bool):
            raise TypeError("Given is_train must be a boolean.")

        if send_count is not None and not isinstance(send_count, int):
            raise TypeError("Given send_count must be an int.")

    def check_table_legality_for_feature_spec(table, feature_spec):
        # check whether the name of the table exists with FeatureSpec.
        if table.table_name != feature_spec.table_name:
            raise ValueError(f"The table name '{feature_spec.table_name}' specified by FeatureSpec is inconsistent with"
                             f" the SparseEmbedding table name '{table.table_name}'.")

    def check_modify_graph():
        if not kwargs.get("modify_graph"):
            raise ValueError(f"modify_graph must be turn-on when lookup by ids(Tensor, not FeatureSpec).")

    kwargs["is_train"] = is_train
    check_lookup_kwargs()
    scope_name = "{0}//{1}".format(hashtable.table_name, kwargs.get("name"))

    with tf.compat.v1.variable_scope(scope_name):
        if hashtable.mode != MxRecMode.ASC:
            raise EnvironmentError("Invalid MxRec Mode.")
        if not isinstance(ids, (FeatureSpec, tf.Tensor)):
            raise ValueError(f"Invalid ids type, it should be: `FeatureSpec` or `tf.Tensor`, but get `{type(ids)}`.")

        if isinstance(ids, FeatureSpec):
            check_table_legality_for_feature_spec(hashtable, ids)
            return hashtable.lookup_for_asc_with_feature_spec(ids, send_count, **kwargs)

        check_modify_graph()
        set_modify_graph(True)
        return hashtable.lookup_for_asc(ids, send_count, **kwargs)


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
        self.ssd_vocabulary_size = config.get("ssd_vocabulary_size")
        self.ssd_data_path = config.get("ssd_data_path")
        if self.host_vocabulary_size > MAX_HOST_VOCABULARY_SIZE:
            raise ValueError(f"host_vocabulary_size is larger than {MAX_HOST_VOCABULARY_SIZE}.")
        self.table_name = config.get("table_name")
        self.key_dtype = config.get("key_dtype")
        self._optimizer_instance_list = config.get("optimizer_list")
        self.emb_initializer = config.get("emb_initializer")
        self._mode = config.get("mode")
        self.is_save = config.get("is_save")
        self.optimizer_slot_info_list = []
        self._slot_num = dict()
        self._send_count = 0
        self.same_table_send_count = 0
        self._use_feature_mapping = False
        self.skip_emb_transfer = True if self.host_vocabulary_size <= 0 else False
        self._default_name_count = -1
        self.emb_size = None
        self.ext_emb_size = None
        self.ext_coefficient = 1
        self._optimizer = dict()
        self.slice_device_vocabulary_size = 0
        self.slice_host_vocabulary_size = 0
        self.slice_ssd_vocabulary_size = 0
        self.variable = None
        self.lookup_info = set()
        self.lookup_result = dict()
        self.use_dynamic_expansion = get_use_dynamic_expansion()
        self.lookup_name_list = []
        self.modify_graph = False
        self.init_param = config.get("init_param")
        self.all2all_gradients_op = All2allGradientsOp.mapping(config.get("all2all_gradients_op"))
        self.apply_gradients_strategy = ApplyGradientsStrategy.mapping(config.get("apply_gradients_strategy"))

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

    @staticmethod
    def _get_own_emb(emb, all2all_args, emb_size, use_static):
        """
        obtain embedding of source data
        :param emb: origin embeddding
        :param all2all_args: dynamic shape condition parameters
        :param emb_size: size of embedding table
        :param use_static: enable static shape training or not
        :return: local embedding after all2all
        """
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

    @staticmethod
    def get_emb_table_size(table_name: str) -> int:
        """
        For HBM or DDR mode, return the size of sparse embedding table
        :param table_name: the name of sparse embedding table
        :return: the size of the sparse embedding table
        """
        table_instance = get_table_instance_by_name(table_name)
        host_vocabulary_size = table_instance.host_vocabulary_size()
        device_vocabulary_size = table_instance.device_vocabulary_size
        if not host_vocabulary_size and not get_use_dynamic_expansion():
            embed_dim = table_instance.emb_size
            size = embed_dim * device_vocabulary_size
        elif not host_vocabulary_size and get_use_dynamic_expansion():
            embed_dim = table_instance.ext_emb_size
            size = embed_dim * device_vocabulary_size
        else:
            embed_dim = table_instance.ext_emb_size
            size = (device_vocabulary_size + host_vocabulary_size) * embed_dim
        return size

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
        self._default_name_count += 1
        default_name = "sparse_lookup_%d" % self._default_name_count
        logging.debug(f"getting one default lookup name {default_name}")
        return default_name

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
            self.slice_ssd_vocabulary_size = math.ceil(self.ssd_vocabulary_size / rank_size)

    def register_anchor_attribute(self, anchor_ids, feature_spec, kwargs):
        SparseEmbedding.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.TABLE_INSTANCE] = self
        SparseEmbedding.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.IS_TRAINING] = kwargs.get("is_train")
        SparseEmbedding.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.FEATURE_SPEC] = feature_spec

    def check_mode(self, method_mode):
        if self.mode != method_mode:
            raise RuntimeError(f"Current sparse table was config in {self.mode.value} mode, but sparse lookup method "
                               f"for {method_mode} was in use.")

    def check_multi_lookup_times(self):
        lookup_times = len(self.lookup_name_list) if self.modify_graph else len(self.lookup_result)
        if not self.modify_graph and get_training_mode_channel_id(True) is not None and \
                get_training_mode_channel_id(False) is not None:
            lookup_times = int(lookup_times / 2)
        if lookup_times > MULTI_LOOKUP_TIMES:
            run_mode = "Modify Graph" if self.modify_graph else "Feature Spec"
            raise RuntimeError(f"In '{run_mode}' mode, the number of multiple sparse lookup for a table"
                               f"({self.table_name}) is {MULTI_LOOKUP_TIMES}, and current times is {lookup_times}.")

    def check_and_format_lookup_params(self, feature, send_count, is_training):
        logging.debug(f"sparse lookup for table {self.table_name} with is_training {is_training}")

        def check_params():
            if not isinstance(is_training, bool):
                raise ValueError("Arg is_train should be a boolean.")

            if isinstance(feature, FeatureSpec):
                if not feature.initialized:
                    raise ValueError(f"Feature Spec has not been initialized.")
                if is_training not in feature.pipeline_mode:
                    raise ValueError(f"You have not config feature for is training mode '{is_training}', please config "
                                     f"feature with func sparse_lookup at first.")

            elif isinstance(feature, tf.Tensor):
                logging.debug("Input feature is a Tensor.")

            else:
                raise TypeError(f"Given feature must be a FeatureSpec or tf.Tensor.")

            if is_training not in self.lookup_info:
                self.lookup_info.add(is_training)

            if not isinstance(self.init_param, float):
                raise ValueError("Arg init_param should be a float.")

            if get_use_static():
                if isinstance(send_count, int) and send_count > 0:
                    if self._send_count and self._send_count != send_count:
                        logging.warning(f"A new send count {send_count} will be used to replace the old one"
                                        f"({self._send_count}).")

                    self._send_count = send_count
                else:
                    raise ValueError("Send count must be a integer which is larger than 0.")

        check_params()
        if self.slice_host_vocabulary_size + self.slice_device_vocabulary_size > MAX_INT32:
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
        # check params
        self.check_mode(MxRecMode.ASC)
        is_training = kwargs.get("is_train")
        self.check_and_format_lookup_params(ids, send_count, is_training)
        if is_asc_frozen() and is_training:
            raise RuntimeError(f"Cannot build new sparse forward graph after emb cache management was built.")

        # record send count
        eval_mode = not is_training and get_training_mode_channel_id(True) is None
        if is_training or eval_mode or "train_and_evaluate" in get_bool_gauge_set():
            self.same_table_send_count += send_count if send_count is not None else 0

        # create feature spec
        feature_spec = get_feature_spec(self.table_name, kwargs.get("access_and_evict_config"))
        feature_spec.set_feat_attribute(ids, is_training)
        # 'clear_channel()' function needs to be executed after 'set_feat_attribute()' function
        if is_asc_frozen() and not is_training:
            clear_channel(is_train_channel=False)

        # record anchor ids
        anchor_ids = tf.identity(ids, name="ids")
        tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE, anchor_ids)
        self.register_anchor_attribute(anchor_ids, feature_spec, kwargs)

        # record multi lookup info
        ids_lookup_name = feature_spec.name + "_lookup_ids"
        # set in train mode, train and eval mode, eval mode
        if is_training or eval_mode:
            self.lookup_name_list.append(ids_lookup_name)
        self.modify_graph = kwargs.get("modify_graph", True)
        self.check_multi_lookup_times()

        # return the stub tensor of the lookup result
        if not get_use_static():
            kwargs["ids"] = ids
        mock_lookup_result = self.lookup_for_asc_with_feature_spec_inner(feature_spec, send_count, **kwargs)
        mock_lookup_result = tf.identity(mock_lookup_result, name=ASCAnchorAttr.MOCK_LOOKUP_RESULT.value)
        SparseEmbedding.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.MOCK_LOOKUP_RESULT] = mock_lookup_result
        logging.debug("Return the stub tensor `%s` of the `%s` table.", mock_lookup_result, self.table_name)
        return mock_lookup_result

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
        if spec_name in self.lookup_result and is_training in self.lookup_result.get(spec_name):
            return self.lookup_result.get(spec_name).get(is_training)

        if not get_use_static() and not self.modify_graph and kwargs.get("batch") is None:
            raise RuntimeError("When the 'feature spec' mode and 'dynamic shape' are used, the 'batch' is required.")
        table_name = feature_spec.table_name
        same_table_feature_spec = ConfigInitializer.get_instance().table_name_to_feature_spec[table_name][is_training]
        same_table_spec_count = len(same_table_feature_spec)
        if same_table_spec_count == 0:
            raise RuntimeError(f"spec_name {spec_name} not in table {table_name}.")
        if same_table_spec_count == 1:
            lookup_result = self.lookup_for_asc_with_feature_spec_inner(feature_spec, send_count, **kwargs)
            if spec_name not in self.lookup_result:
                self.lookup_result[spec_name] = {}
            self.lookup_result[spec_name][is_training] = lookup_result
        else:
            def get_tensor_list() -> list:
                """
                Use 'feature spec' to find the corresponding tensor from batch.
                Returns: Tensor list in batch.
                """
                same_table_tensor_list = []
                for feat_spec in same_table_feature_spec:
                    feature_spec_tensor_dict = kwargs.get("batch")
                    modify_graph_tensor_dict = kwargs.get("feature_spec_name_ids_dict")
                    batch_tensor_dict = feature_spec_tensor_dict if not self.modify_graph else modify_graph_tensor_dict
                    if batch_tensor_dict is None:
                        raise KeyError(f"The tensor dict of batch does not exist in kwargs, and modify graph "
                                       f"is `{self.modify_graph}`.")

                    feature_spec_tensor = batch_tensor_dict.get(feat_spec.index_key)
                    modify_graph_tensor = batch_tensor_dict.get(feat_spec.name)
                    tensor = feature_spec_tensor if not self.modify_graph else modify_graph_tensor
                    if tensor is None:
                        tensor_key = feat_spec.index_key if not self.modify_graph else feat_spec.name
                        raise KeyError(f"Key `{tensor_key}` does not exist in batch_tensor_dict.")
                    same_table_tensor_list.append(tensor)
                return same_table_tensor_list

            # Ensure that tensors in the same table are sorted according to the lookup sequence (modify graph mode) or
            # the sequence in which feature specs are created (feature spec mode).
            same_table_feature_spec = sorted(same_table_feature_spec, key=lambda x: x.name)
            mock_feature_spec = FeatureSpec(f"mock_feature_spec_{table_name}", feat_count=1, table_name=table_name)

            if get_use_static():
                tensor_list = []
                tensor_split_list = [feat_spec.split for feat_spec in same_table_feature_spec]
                total_feature_count = sum(tensor_split_list)
            else:
                tensor_list = get_tensor_list()
                tensor_split_list = [tf.math.reduce_prod(array_ops.shape(tensor)) for tensor in tensor_list]
                total_feature_count = tf.add_n(tensor_split_list)
            set_temporary_feature_spec_attribute(mock_feature_spec, total_feature_count)

            kwargs["multi_lookup"] = True
            total_send_count = self.same_table_send_count if self.modify_graph else send_count * same_table_spec_count
            lookup_result = self.lookup_for_asc_with_feature_spec_inner(mock_feature_spec, total_send_count, **kwargs)
            logging.debug(f"lookup table {table_name} via {tensor_split_list}")
            self.split_lookup_result(same_table_feature_spec, tensor_split_list, tensor_list, lookup_result,
                                     is_training)

        if not self.modify_graph:
            self.check_multi_lookup_times()
        return self.lookup_result.get(spec_name).get(is_training)

    def split_lookup_result(self, same_table_feature_spec: list, tensor_split_list: list, tensor_list: list,
                            lookup_result: tf.Tensor, is_training: bool):
        """
        Splits the result of the merge sparse lookup.

        Args:
            same_table_feature_spec: a list of feature specs in a same table
            tensor_split_list: a list of tensor split in a same table
            tensor_list: a list of tensor in a same table
            lookup_result: results of the sparse lookup
            is_training: indicates whether the training mode is used.

        Returns: None

        """
        lookup_result_split = tf.split(lookup_result, tensor_split_list)
        if len(lookup_result_split) != len(same_table_feature_spec) or (
                not get_use_static() and len(same_table_feature_spec) != len(tensor_list)):
            raise RuntimeError(f"shape not match. len(lookup_result_split): {len(lookup_result_split)},"
                               f"len(same_table_feature_spec): {len(same_table_feature_spec)}"
                               f"len(tensor_list): {len(tensor_list)}")
        for idx, (one_feature_spec, one_result) in enumerate(zip(same_table_feature_spec, lookup_result_split)):
            if one_feature_spec.name not in self.lookup_result:
                self.lookup_result[one_feature_spec.name] = {}
            if get_use_static():
                dest_shape = one_feature_spec.dims + [self.scalar_emb_size]
            else:
                dest_shape = array_ops.concat([array_ops.shape(tensor_list[idx]), [self.scalar_emb_size]], 0)
            self.lookup_result[one_feature_spec.name][is_training] = array_ops.reshape(one_result, dest_shape)

    def generate_lookup_id_notify_hybrid(self, channel_id: int):

        """
        Args:
         channel_id: channel id 0 for train，1 for eval
        Returns: npu_ops.outfeed_enqueue_op notify preprocess step
        """
        channel_name = "d2h_notify_hybridmgmt_{}".format(channel_id)
        notify_hybridmgmt_op = tf.no_op(channel_name)
        return notify_hybridmgmt_op

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
                      use_dynamic_expansion=use_dynamic_expansion, gradients_strategy=self.apply_gradients_strategy)

        # 用于打桩的op节点，它的name用于标识此次的sparse lookup是train还是eval
        # 后续在session run的时候，通过图反向查找该子图中查找到此op
        # 最后通过名称判断session run是调用的哪个通道，并通知c++侧进行计数和唤醒操作
        notify_hybridmgmt_op = self.generate_lookup_id_notify_hybrid(channel_id)
        with tf.control_dependencies([notify_hybridmgmt_op]):
            if self.skip_emb_transfer:
                result = get_preprocessed_tensor_for_asc(self.variable, config)
            else:
                variable_list = [self.variable] + [slot_info.get("slot") for slot_info in self.optimizer_slot_info_list]
                result = get_preprocessed_tensor_for_asc(variable_list, config)
        restore_vector = result.get("restore_vector")
        restore_vector_second = result.get("restore_vector_second")
        hot_pos = result.get("hot_pos")
        id_offsets = result.get("id_offsets")
        unique_keys = result.get("unique_keys")
        swap_in = result.get("swap_in")
        all2all_matrix = result.get("all2all_args")
        control_ops = swap_in

        id_offsets = tf.identity(id_offsets, name="identity_addr")
        restore_vector = tf.identity(restore_vector, name="identity_restore")

        use_static = get_use_static()
        host_pipeline_ops = get_host_pipeline_ops()

        @tf.custom_gradient
        def sparse_forward(table):
            logging.debug(f"fp rank size: {rank_size}")
            if not use_dynamic_expansion:
                id_offsets_abs = tf.abs(id_offsets)
                local_embeddings = tf.gather(table, id_offsets_abs, axis=0, name="gather_for_id_offsets")
                local_embeddings = set_zero_for_non_valid_key(id_offsets, local_embeddings,
                                                              feature_spec.access_threshold)
            else:
                local_embeddings = tf.identity(table, name="identity_local_emb")

            all2all_args = send_count if use_static else all2all_matrix
            unique_embeddings = self._get_own_emb(local_embeddings, all2all_args, self.scalar_emb_size, use_static)

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
                if kwargs.get("multi_lookup"):
                    lookup_result = tf.reshape(embeddings, [-1, self.scalar_emb_size])
                else:
                    feature_spec_tensor = None
                    if not self.modify_graph:
                        feature_spec_tensor = kwargs.get("batch").get(feature_spec.index_key)
                    modify_graph_tensor = kwargs.get("ids")
                    tensor = feature_spec_tensor if not self.modify_graph else modify_graph_tensor
                    if tensor is None:
                        raise KeyError(f"key or ids does not exist in batch, now modify graph is {self.modify_graph}.")
                    dest_shape = array_ops.concat([array_ops.shape(tensor), [self.scalar_emb_size]], 0)
                    lookup_result = array_ops.reshape(embeddings, dest_shape)

            def grad(lookup_diff):
                logging.debug("Into lookup grad function, feature spec name: %s.", feature_spec.name)
                embedding_diff = tf.reshape(lookup_diff, [-1, self.scalar_emb_size])
                unique_grads = tf.compat.v1.unsorted_segment_sum(embedding_diff,
                                                                 restore_vector,
                                                                 unique_embeddings_shape[0])
                bp_all2all_args = all2all_args if use_static else tf.transpose(all2all_args)
                if hot_pos is not None:
                    hot, cold = tf.split(unique_grads, [tf.shape(hot_pos)[0],
                                                        tf.shape(unique_grads)[0] - tf.shape(hot_pos)[0]], axis=0)
                    unique_grads = tf.tensor_scatter_nd_add(cold, tf.expand_dims(hot_pos, 1), hot)
                local_grad = self._get_own_emb(unique_grads, bp_all2all_args, self.scalar_emb_size, use_static)
                if self.all2all_gradients_op == All2allGradientsOp.SUM_GRADIENTS_AND_DIV_BY_RANKSIZE:
                    try:
                        local_grad = local_grad / get_rank_size()
                    except ZeroDivisionError as exp:
                        raise ZeroDivisionError("Rank size cannot be zero.") from exp

                if use_dynamic_expansion:
                    if self.apply_gradients_strategy == ApplyGradientsStrategy.SUM_SAME_ID_GRADIENTS_AND_APPLY:
                        update_grad = tf.compat.v1.unsorted_segment_sum(local_grad,
                                                                        restore_vector_second,
                                                                        array_ops.shape(unique_keys)[0])
                    else:
                        update_grad = local_grad
                else:
                    if self.apply_gradients_strategy == ApplyGradientsStrategy.SUM_SAME_ID_GRADIENTS_AND_APPLY:
                        unique_local_grad = tf.compat.v1.unsorted_segment_sum(local_grad,
                                                                              restore_vector_second,
                                                                              array_ops.shape(unique_keys)[0])
                        update_grad = ops.IndexedSlices(values=unique_local_grad,
                                                        indices=unique_keys,
                                                        dense_shape=tf.shape(table))
                    else:
                        update_grad = ops.IndexedSlices(values=local_grad,
                                                        indices=id_offsets,
                                                        dense_shape=tf.shape(table))
                return update_grad

            return lookup_result, grad

        with tf.control_dependencies(control_ops):
            if not use_dynamic_expansion:
                return sparse_forward(self.variable)

            local_embeddings = \
                host_pipeline_ops.embedding_lookup_by_address(id_offsets, embedding_dim=self.emb_size,
                                                              embedding_type=1)

            is_table_name_valid = ASCEND_TABLE_NAME_MUST_CONTAIN is None or \
                                  ASCEND_TABLE_NAME_MUST_CONTAIN in self.table_name

            def add_to_collection():
                tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_LOCAL_EMB, local_embeddings)
                if self.apply_gradients_strategy == ApplyGradientsStrategy.SUM_SAME_ID_GRADIENTS_AND_APPLY:
                    tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_UNIQUE_KEYS, unique_keys)
                else:
                    tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_ID_OFFSET, id_offsets)

                logging.debug(f"feature spec mode, table_name: {self.table_name}, "
                              f"ASCEND_TABLE_NAME_MUST_CONTAIN: {ASCEND_TABLE_NAME_MUST_CONTAIN}")
            if is_training and is_table_name_valid:
                add_to_collection()

            return sparse_forward(local_embeddings)

    def _record(self):
        insert_table_instance(self.table_name, self.variable, self)
        logging.debug(f"Device vocabulary_size for table {self.table_name} is {self.device_vocabulary_size}.")
        logging.debug(f"Slice_device_vocabulary_size for table {self.table_name} is"
                      f" {self.slice_device_vocabulary_size}.")
        logging.debug(f"Host vocabulary size for table {self.table_name} is {self.host_vocabulary_size}.")
        logging.debug(f"Slice host vocabulary_size for table {self.table_name} is"
                      f" {self.slice_host_vocabulary_size}.")
        logging.debug(f"SSD vocabulary size for table {self.table_name} is {self.ssd_vocabulary_size}.")
        logging.debug(f"Slice ssd vocabulary_size for table {self.table_name} is"
                      f" {self.slice_ssd_vocabulary_size}.")

    def _initialize_variables(self):
        initialized_tensor = \
            self.emb_initializer(self.slice_device_vocabulary_size + self.embedding_size) * self.init_param

        self.variable = tf.compat.v1.get_variable(self.table_name, trainable=False, initializer=initialized_tensor)
        # make sure sparse table variable will not be saved and restored within tf checkpoint.
        insert_removing_var_list(self.variable.name)
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


def set_zero_for_non_valid_key(id_offsets: Optional[tf.Tensor], embeddings: Optional[tf.Tensor],
                               access_threshold: bool):
    """
    将key为-1的特征对应的emb置为0
    :param id_offsets: 特征索引
    :param embeddings: 稀疏表
    :param access_threshold: 准入阈值
    :return:
    """
    if access_threshold is None or access_threshold <= 0:
        return embeddings

    if tf.__version__.startswith("1"):
        id_offsets_expand = tf.math.greater_equal(id_offsets, 0)
        embeddings = tf.where(id_offsets_expand, embeddings, tf.zeros_like(embeddings))
        return embeddings

    id_offsets_expand = tf.compat.v1.expand_dims(id_offsets >= 0, axis=-1)
    embeddings = tf.where(id_offsets_expand, embeddings, tf.zeros_like(embeddings))
    return embeddings


def check_create_table_params(key_dtype, dim, name, emb_initializer):
    """
    校验create_table接口必选参数：key_dtype, dim, name, emb_initializer和optimizer_list（已有校验）
    :param key_dtype: data type for feature id, tf.int64 or tf.int32 or tf.string
    :param dim: embedding vector size, dim's type: int or tf.TensorShape
    :param name: hash table name, name's type: str
    :param emb_initializer: the initializer for embedding values
    :return:
    """
    # check key_dtype
    if key_dtype not in [tf.int64, tf.int32, tf.string]:
        raise ValueError(f"key_dtype: {key_dtype} not in [tf.int64, tf.int32, tf.string]")
    # check dim
    dim_validator = ClassValidator(value=dim, classes=(int, tf.TensorShape))
    dim_validator.check_isinstance()
    dim_validator.check()
    # check name
    name_validator = StringValidator(value=name, max_len=255)
    name_validator.check_string_length()
    name_validator.check_whitelist()
    name_validator.check()
    # check emb_initializer
    emb_initializer_validator = ClassValidator(value=emb_initializer, classes=(InitializerV1, InitializerV2))
    emb_initializer_validator.check_isinstance()
    emb_initializer_validator.check()
