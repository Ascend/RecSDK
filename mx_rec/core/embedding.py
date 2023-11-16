#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import math
import os
from collections import defaultdict
from typing import Optional, Union

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.init_ops import Initializer as InitializerV1
from tensorflow.python.ops.init_ops_v2 import Initializer as InitializerV2

from mx_rec.core.asc.build_graph import get_preprocessed_tensor_for_asc
from mx_rec.core.asc.feature_spec import FeatureSpec, get_feature_spec, set_temporary_feature_spec_attribute
from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.constants.constants import ASCEND_SPARSE_LOOKUP_ENTRANCE, ASCEND_SPARSE_LOOKUP_ID_OFFSET, \
    ASCEND_SPARSE_LOOKUP_UNIQUE_KEYS, ASCAnchorAttr, ASCEND_SPARSE_LOOKUP_LOCAL_EMB, MULTI_LOOKUP_TIMES, \
    ASCEND_TABLE_NAME_MUST_CONTAIN, MAX_INT32, All2allGradientsOp, ApplyGradientsStrategy, MAX_VOCABULARY_SIZE
from mx_rec.util.initialize import get_rank_id, get_rank_size, is_asc_frozen, get_customized_ops, \
    insert_table_instance, get_training_mode_channel_id, get_use_static, get_name_to_var_dict, \
    clear_channel, get_use_hot, get_device_id, ConfigInitializer, get_ascend_global_hashtable_collection, \
    get_host_pipeline_ops, get_use_dynamic_expansion, set_modify_graph, insert_removing_var_list, get_bool_gauge_set, \
    get_table_instance_by_name, get_asc_manager
from mx_rec.validator.validator import ClassValidator, StringValidator, SSDFeatureValidator, \
    para_checker_decorator, IntValidator, NumValidator, OptionValidator, OptionalIntValidator, OptionalStringValidator
from mx_rec.util.tf_version_adapter import npu_ops
from mx_rec.util.normalization import fix_invalid_table_name
from mx_rec.util.global_env_conf import global_env
from mx_rec.util.log import logger


@para_checker_decorator(check_option_list=[
    ("key_dtype", OptionValidator, {"options": (tf.int64, tf.int32, tf.string)}),
    ("dim", ClassValidator, {"classes": (int, tf.TensorShape)}),
    ("dim", NumValidator, {"min_value": 1, "max_value": 8192}, ["check_value"]),
    ("name", StringValidator, {"min_len": 1, "max_len": 255}, ["check_string_length", "check_whitelist"]),
    ("emb_initializer", ClassValidator, {"classes": (InitializerV1, InitializerV2)}),
    ("optimizer_list", ClassValidator, {"classes": (list, type(None))}),
    (["ssd_vocabulary_size", "ssd_data_path", "host_vocabulary_size"], SSDFeatureValidator),
    ("device_vocabulary_size", IntValidator, {"min_value": 1, "max_value": MAX_VOCABULARY_SIZE}, ["check_value"]),
    ("host_vocabulary_size", IntValidator, {"min_value": 0, "max_value": MAX_VOCABULARY_SIZE}, ["check_value"]),
    ("ssd_vocabulary_size", IntValidator, {"min_value": 0, "max_value": MAX_VOCABULARY_SIZE}, ["check_value"]),
    ("ssd_data_path", ClassValidator, {"classes": (list, tuple)}),
    ("is_save", ClassValidator, {"classes": (bool, )}),
    ("init_param", NumValidator, {"min_value": -10, "max_value": 10}, ["check_value"]),
    ("all2all_gradients_op", OptionValidator, {"options": [i.value for i in list(All2allGradientsOp)]}),
    ("value_dtype", OptionValidator, {"options": [tf.float32]}),
    ("shard_num", IntValidator, {"min_value": 1, "max_value": 8192}, ["check_value"]),
    ("fusion_optimizer_var", ClassValidator, {"classes": (bool, )}),
    ("hashtable_threshold", IntValidator, {"min_value": 0, "max_value": MAX_INT32}, ["check_value"])
])
def create_table(key_dtype, dim, name, emb_initializer,
                 optimizer_list: Optional[list] = None,
                 device_vocabulary_size=1,
                 host_vocabulary_size=0,
                 ssd_vocabulary_size=0,
                 ssd_data_path=(os.getcwd(), ),
                 is_save=True,
                 init_param=1.,
                 all2all_gradients_op=All2allGradientsOp.SUM_GRADIENTS.value,
                 value_dtype=tf.float32,
                 shard_num=1,
                 fusion_optimizer_var=True,
                 hashtable_threshold=0):
    """
    Args:
        key_dtype: data type for feature id
        dim: embedding vector size
        name: hash table name
        emb_initializer: the initializer for embedding values
        optimizer_list: specify the optimizers to use for current hash table
        device_vocabulary_size: embedding vector numbers on device
        host_vocabulary_size: embedding vector numbers on ddr
        ssd_vocabulary_size: embedding vector numbers on ssd
        ssd_data_path: ssd embedding data save and load path relation from feature to variable offset will be built
        is_save: switch whether to store sparse table data.
        init_param: embedding init param-coefficient
        all2all_gradients_op: sum_grads (default) or sum_gradients_and_div_by_ranksize.
        value_dtype: the type of the value tensors. only tf.float32 if supported for now.
        shard_num: embedding partition number
        fusion_optimizer_var: fusion optimizer variable with embedding
        hashtable_threshold: choose to implement based on hash table or linear layer
    """
    name = fix_invalid_table_name(name)

    config = dict(key_dtype=key_dtype, embedding_size=dim, table_name=name, emb_initializer=emb_initializer,
                  device_vocabulary_size=device_vocabulary_size, host_vocabulary_size=host_vocabulary_size,
                  ssd_vocabulary_size=ssd_vocabulary_size, ssd_data_path=ssd_data_path,
                  optimizer_list=optimizer_list, init_param=init_param, is_save=is_save,
                  all2all_gradients_op=all2all_gradients_op)
    embedding = SparseEmbedding(config)
    return embedding


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
        self.ssd_data_path = list(config.get("ssd_data_path"))
        self.table_name = config.get("table_name")
        self.key_dtype = config.get("key_dtype")
        self._optimizer_instance_list = config.get("optimizer_list")
        self.emb_initializer = config.get("emb_initializer")
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
        self.lookup_name_dict = {True: [], False: []}
        self.modify_graph = False
        self.init_param = config.get("init_param")
        self.all2all_gradients_op = All2allGradientsOp.mapping(config.get("all2all_gradients_op"))
        self.is_grad = False

        self.set_slice_vocab_size()
        self.set_emb_size()
        if is_asc_frozen() and self.table_name in get_name_to_var_dict():
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
    def send_count(self):
        return self._send_count

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def optimizer_instance_list(self):
        return self._optimizer_instance_list

    @staticmethod
    def generate_lookup_id_notify_hybrid(channel_id: int):

        """
        Args:
         channel_id: channel id 0 for train，1 for eval
        Returns: npu_ops.outfeed_enqueue_op notify preprocess step
        """
        channel_name = "d2h_notify_hybridmgmt_{}".format(channel_id)
        notify_hybridmgmt_op = tf.no_op(channel_name)
        return notify_hybridmgmt_op

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

    def size(self) -> int:
        """
        For HBM or DDR or SSD mode, return the size of sparse table
        """
        return get_asc_manager().get_table_size(self.table_name)

    def capacity(self) -> int:
        """
        For HBM or DDR or SSD mode, return the capacity of sparse table
        """
        if get_use_dynamic_expansion():
            return get_asc_manager().get_table_capacity(self.table_name)

        if not self.host_vocabulary_size and not self.ssd_vocabulary_size:
            return self.device_vocabulary_size
        if not self.ssd_vocabulary_size:
            return self.device_vocabulary_size + self.host_vocabulary_size
        return self.device_vocabulary_size + self.host_vocabulary_size + self.ssd_vocabulary_size

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
        if self.embedding_size.ndims != 1:
            raise ValueError("Parameter 'embedding_size' can only be one dim shape.")

        if is_asc_frozen():
            raise EnvironmentError(f"Emb cache management has been established, you cannot build new ASC hash table.")

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
        logger.debug("getting one default lookup name %s", default_name)
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
        logger.debug("init table, ext_emb_size is set to be %s", self.ext_emb_size)

    def set_slice_vocab_size(self):
        rank_size = get_rank_size()
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
        SparseEmbedding.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.IS_GRAD] = kwargs.get("is_grad")

    def check_multi_lookup_times(self, is_training):
        lookup_times = len(self.lookup_name_dict.get(is_training)) if self.modify_graph else len(self.lookup_result)
        if not self.modify_graph and get_training_mode_channel_id(True) is not None and \
                get_training_mode_channel_id(False) is not None:
            lookup_times = int(lookup_times / 2)
        if lookup_times > MULTI_LOOKUP_TIMES:
            run_mode = "Modify Graph" if self.modify_graph else "Feature Spec"
            raise RuntimeError(f"In '{run_mode}' mode, the number of multiple sparse lookup for a table"
                               f"({self.table_name}) is {MULTI_LOOKUP_TIMES}, and current times is {lookup_times}.")

    def check_and_format_lookup_params(self, feature, send_count, is_training):
        logger.debug("sparse lookup for table %s with is_training %s", self.table_name, is_training)

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
                logger.debug("Input feature is a Tensor.")

            else:
                raise TypeError(f"Given feature must be a FeatureSpec or tf.Tensor.")

            if is_training not in self.lookup_info:
                self.lookup_info.add(is_training)

            if not isinstance(self.init_param, float):
                raise ValueError("Arg init_param should be a float.")

            if get_use_static():
                if isinstance(send_count, int) and send_count > 0:
                    if self._send_count and self._send_count != send_count:
                        logger.warning("A new send count %s will be used to replace the old one (%s).",
                                       send_count, self._send_count)

                    self._send_count = send_count
                else:
                    raise ValueError("Send count must be a integer which is larger than 0.")

        check_params()
        if self.slice_host_vocabulary_size + self.slice_device_vocabulary_size > MAX_INT32:
            raise ValueError(f"Given device_vocabulary_size and host_vocabulary_size was too big for table "
                             f"'{self.table_name}', in which slice_device_vocabulary_size was "
                             f"{self.slice_device_vocabulary_size} and slice_host_vocabulary_size was "
                             f"{self.slice_host_vocabulary_size} ")

        is_check_mode = not self.skip_emb_transfer and not self.use_dynamic_expansion
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
        logger.debug(f"Enter ASC Branch.")
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
        if self.lookup_name_dict.get(is_training) is None:
            self.lookup_name_dict[is_training] = []
        self.lookup_name_dict.get(is_training).append(ids_lookup_name)
        self.modify_graph = kwargs.get("modify_graph", True)
        self.check_multi_lookup_times(is_training)

        # return the stub tensor of the lookup result
        if not get_use_static():
            kwargs["lookup_ids"] = ids
        mock_lookup_result = self.lookup_for_asc_with_feature_spec_inner(feature_spec, send_count, **kwargs)
        mock_lookup_result = tf.identity(mock_lookup_result, name=ASCAnchorAttr.MOCK_LOOKUP_RESULT.value)
        if not kwargs.get("is_grad"):
            mock_lookup_result = tf.stop_gradient(mock_lookup_result, name="mock_stop_grad_lookup_res")
        SparseEmbedding.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.MOCK_LOOKUP_RESULT] = mock_lookup_result
        logger.debug("Return the stub tensor `%s` of the `%s` table.", mock_lookup_result, self.table_name)
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
            if not kwargs.get("is_grad"):
                return tf.stop_gradient(self.lookup_result.get(spec_name).get(is_training), name="stop_grad_lookup_res")
            return self.lookup_result.get(spec_name).get(is_training)

        if not get_use_static() and not self.modify_graph and kwargs.get("batch") is None:
            raise RuntimeError("When the 'feature spec' mode and 'dynamic shape' are used, the 'batch' is required.")
        table_name = feature_spec.table_name
        same_table_feature_spec = ConfigInitializer.get_instance().table_name_to_feature_spec[table_name][is_training]
        logger.debug("The feature spec of the same table is %s, table name is %s.",
                     [fs.name for fs in same_table_feature_spec], self.table_name)
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

            # 改图模式下FeatureSpec是按照lookup顺序创建的，无需对ids进行排序；fs模式下手动创建FeatureSpec，不一定有序
            if not self.modify_graph:
                same_table_feature_spec = sorted(same_table_feature_spec, key=lambda x: x.name)
            mock_feature_spec = FeatureSpec(f"mock_feature_spec_{table_name}", table_name=table_name)

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
            logger.debug("multi lookup table %s via %s.", table_name, tensor_split_list)
            self.split_lookup_result(same_table_feature_spec, tensor_split_list, tensor_list, lookup_result,
                                     is_training)
            # 当一表多查完成后，将此表对应的feature specs列表清空，便于estimator模式下多轮eval时不会累加上轮eval的feature specs
            ConfigInitializer.get_instance().clear_same_table_feature_spec(self.table_name, is_training)

        if not self.modify_graph:
            self.check_multi_lookup_times(is_training)
        if not kwargs.get("is_grad"):
            return tf.stop_gradient(self.lookup_result.get(spec_name).get(is_training), name="stop_grad_lookup_res")
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
        logger.debug(f"Enter ASC Branch, looking up with FeatureSpec.")
        is_training = kwargs.get("is_train")
        self.check_and_format_lookup_params(feature_spec, send_count, is_training)
        rank_size = get_rank_size()
        device_id = get_device_id()
        use_hot = get_use_hot()
        use_dynamic_expansion = get_use_dynamic_expansion()

        # check training mode order and ensure channel id
        channel_id = get_training_mode_channel_id(is_training=is_training)
        logger.debug("get preprocessed tensor for asc for table %s with skip emb transfer %s is_training: %s, "
                     "channel_id: %s .", self.table_name, self.skip_emb_transfer, is_training, channel_id)
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
            logger.debug("fp rank size: %s", rank_size)
            if not use_dynamic_expansion:
                id_offsets_abs = tf.abs(id_offsets)
                local_embeddings = tf.gather(table, id_offsets_abs, axis=0, name="gather_for_id_offsets")
                local_embeddings = set_specific_value_for_non_valid_key(id_offsets,
                                                                        local_embeddings,
                                                                        feature_spec.access_threshold,
                                                                        kwargs.get("serving_default_value"),
                                                                        is_training=is_training)
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

            # 用于打桩的op节点，它的name用于标识此次的sparse lookup是train还是eval
            # 后续在session run的时候，通过图反向查找该子图中查找到此op
            # 最后通过名称判断session run是调用的哪个通道，并通知c++侧进行计数和唤醒操作
            notify_hybridmgmt_op = self.generate_lookup_id_notify_hybrid(channel_id)
            with tf.control_dependencies([notify_hybridmgmt_op]):
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
                    modify_graph_tensor = kwargs.get("lookup_ids")
                    tensor = feature_spec_tensor if not self.modify_graph else modify_graph_tensor
                    if tensor is None:
                        raise KeyError(f"key or ids does not exist in batch, now modify graph is {self.modify_graph}.")
                    dest_shape = array_ops.concat([array_ops.shape(tensor), [self.scalar_emb_size]], 0)
                    lookup_result = array_ops.reshape(embeddings, dest_shape)

            def grad(lookup_diff):
                logger.debug("Into lookup grad function, feature spec name: %s.", feature_spec.name)
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
                    if global_env.apply_gradients_strategy == \
                            ApplyGradientsStrategy.SUM_SAME_ID_GRADIENTS_AND_APPLY.value:
                        update_grad = tf.compat.v1.unsorted_segment_sum(local_grad,
                                                                        restore_vector_second,
                                                                        array_ops.shape(unique_keys)[0])
                    else:
                        update_grad = local_grad
                else:
                    if global_env.apply_gradients_strategy == \
                            ApplyGradientsStrategy.SUM_SAME_ID_GRADIENTS_AND_APPLY.value:
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
                if global_env.apply_gradients_strategy == ApplyGradientsStrategy.SUM_SAME_ID_GRADIENTS_AND_APPLY.value:
                    tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_UNIQUE_KEYS, unique_keys)
                else:
                    tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_ID_OFFSET, id_offsets)

                logger.debug("feature spec mode, table_name: %s, ASCEND_TABLE_NAME_MUST_CONTAIN: %s",
                             self.table_name, ASCEND_TABLE_NAME_MUST_CONTAIN)

            if is_training and is_table_name_valid:
                add_to_collection()

            return sparse_forward(local_embeddings)

    def _record(self):
        insert_table_instance(self.table_name, self.variable, self)
        logger.debug("Device vocabulary_size for table %s is %s.", self.table_name, self.device_vocabulary_size)
        logger.debug("Slice_device_vocabulary_size for table %s is %s.",
                     self.table_name, self.slice_device_vocabulary_size)
        logger.debug(f"Host vocabulary size for table %s is %s.", self.table_name, self.host_vocabulary_size)
        logger.debug(f"Slice host vocabulary_size for table %s is %s.",
                     self.table_name, self.slice_host_vocabulary_size)
        logger.debug(f"SSD vocabulary size for table %s is %s.", self.table_name, self.ssd_vocabulary_size)
        logger.debug(f"Slice ssd vocabulary_size for table %s is %s.", self.table_name, self.slice_ssd_vocabulary_size)

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
                logger.info("init emb, table name: %s, slot_num: %s",
                            self.table_name, sparse_optimizer_instance.slot_num)

        if not self.skip_emb_transfer:
            # build optimizer states
            for sparse_optimizer_instance in self._optimizer_instance_list:
                slot_info_list = sparse_optimizer_instance.initialize_slots(self.variable, self)
                self.optimizer_slot_info_list.extend(slot_info_list)

            for slot_info in self.optimizer_slot_info_list:
                self.set_optimizer_slot(slot_info)


@para_checker_decorator(check_option_list=[
    ("hashtable", ClassValidator, {"classes": (SparseEmbedding, )}),
    ("ids", ClassValidator, {"classes": (FeatureSpec, tf.Tensor)}),
    ("is_train", ClassValidator, {"classes": (bool, )}),
    ("send_count", ClassValidator, {"classes": (int, type(None))}),
    ("send_count", OptionalIntValidator, {"min_value": 1, "max_value": MAX_INT32}, ["check_value"]),
    ("name", ClassValidator, {"classes": (str, type(None))}),
    ("name", OptionalStringValidator, {"min_len": 1, "max_len": 255}, ["check_string_length"]),
    ("modify_graph", ClassValidator, {"classes": (bool, type(None))}),
    ("batch", ClassValidator, {"classes": (dict, type(None))}),
    ("access_and_evict_config", ClassValidator, {"classes": (dict, type(None))}),
    ("is_grad", ClassValidator, {"classes": (bool, )}),
    ("serving_default_value", ClassValidator, {"classes": (tf.Tensor, type(None))})
])
def sparse_lookup(hashtable: SparseEmbedding,
                  ids: Union[FeatureSpec, tf.Tensor],
                  send_count: Optional[int] = None,
                  is_train: bool = True,
                  name: Optional[str] = None,
                  modify_graph: bool = False,
                  batch: Optional[dict] = None,
                  access_and_evict_config: Optional[dict] = None,
                  is_grad: bool = True,
                  serving_default_value: Optional[tf.Tensor] = None,
                  **kwargs):
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
    # 一表多查时，只要有一次查询需要grad，那么这张表也需要grad；否则整张表都不需要gard，同时在全局unique情况下，C++也不需要send数据
    hashtable.is_grad |= is_grad
    kwargs["is_train"] = is_train
    kwargs["name"] = name if name is not None else hashtable.get_default_lookup_name()
    kwargs["modify_graph"] = modify_graph
    kwargs["batch"] = batch
    kwargs["access_and_evict_config"] = access_and_evict_config
    # 参数由内部创建，不使用外部入参，覆盖外部入参
    kwargs["feature_spec_name_ids_dict"] = None
    kwargs["multi_lookup"] = False
    kwargs["lookup_ids"] = None
    kwargs["serving_default_value"] = serving_default_value
    scope_name = "{0}//{1}".format(hashtable.table_name, kwargs.get("name"))
    logger.info("Lookup: The table name is %s, and the value of `is_grad` in this lookup (lookup name is %s) is %s.",
                hashtable.table_name, name, is_grad)

    with tf.compat.v1.variable_scope(scope_name):
        if isinstance(ids, FeatureSpec):
            # check whether the name of the table exists with FeatureSpec.
            if hashtable.table_name != ids.table_name:
                raise ValueError(f"The table name '{ids.table_name}' specified by FeatureSpec is inconsistent with"
                                 f" the SparseEmbedding table name '{hashtable.table_name}'.")

            return hashtable.lookup_for_asc_with_feature_spec(ids, send_count, **kwargs)

        if not modify_graph:
            raise ValueError("'ids' is type of tf.Tensor, 'modify_graph' should be set to True")

        set_modify_graph(modify_graph)
        return hashtable.lookup_for_asc(ids, send_count, **kwargs)


def set_specific_value_for_non_valid_key(id_offsets: Optional[tf.Tensor],
                                         embeddings: Optional[tf.Tensor],
                                         access_threshold: Optional[int],
                                         serving_default_value: Optional[tf.Tensor] = None,
                                         is_training: bool = True):
    """
    将key为-1(无效值)的特征对应的emb置为0或者指定值
    :param id_offsets: 特征索引
    :param embeddings: 稀疏表
    :param access_threshold: 准入阈值
    :param serving_default_value: 参考create_table接口描述
    :param is_training: 当前流程是训练还是推理
    :return:
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
