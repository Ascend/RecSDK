#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import abc
from collections import defaultdict
from typing import Optional, Union, Callable

import tensorflow as tf
from tensorflow.python.ops import array_ops

from mx_rec.constants.constants import All2allGradientsOp, ASCEND_SPARSE_LOOKUP_ENTRANCE, ASCAnchorAttr
from mx_rec.core.asc.feature_spec import set_temporary_feature_spec_attribute, get_feature_spec, FeatureSpec
from mx_rec.util.communication.hccl_ops import get_rank_size, get_rank_id, get_device_id
from mx_rec.util.tf_version_adapter import hccl_ops
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.util.log import logger
from mx_rec.validator.emb_validator import check_emb_init_params, check_emb_lookup_params


class BaseSparseEmbedding(metaclass=abc.ABCMeta):
    """
    稀疏表基类
    """
    # 自动改图使用的全局字典，以ids和待保存内容的字符串为key
    anchor_tensor_specs = defaultdict(dict)

    def __init__(self, config: dict):
        self._embedding_size = config.get("embedding_size")
        if isinstance(self._embedding_size, int):
            self._embedding_size = tf.TensorShape([self._embedding_size])
        self._table_name = config.get("table_name")
        self._key_dtype = config.get("key_dtype")
        self._emb_initializer = config.get("emb_initializer")
        self._is_save = config.get("is_save")
        self._init_param = config.get("init_param")
        self._is_hbm = True if config.get("host_vocabulary_size") <= 0 else False
        self._ssd_data_path = list(config.get("ssd_data_path"))
        self._send_count = 0
        self._slice_device_vocabulary_size = 0
        self._slice_host_vocabulary_size = 0
        self._slice_ssd_vocabulary_size = 0
        self._emb_size = self._embedding_size.as_list()[0]
        self._is_grad = False
        self._ext_emb_size = None
        self._variable = None
        self._multi_lookup_times = {True: 0, False: 0}

        self._all2all_gradients_op = All2allGradientsOp.mapping(config.get("all2all_gradients_op"))
        self._device_vocabulary_size = config.get("device_vocabulary_size")
        self._host_vocabulary_size = config.get("host_vocabulary_size")
        self._ssd_vocabulary_size = config.get("ssd_vocabulary_size")
        self._ext_coefficient = 1
        self._default_name_count = -1
        self._same_table_send_count = 0
        self._lookup_result = dict()
        self._modify_graph = False

        self._rank_size = get_rank_size()
        self._rank_id = get_rank_id()
        self._device_id = get_device_id()
        self._use_static = ConfigInitializer.get_instance().use_static

        # init variable
        self._set_slice_vocab_size()

        if ConfigInitializer.get_instance().hybrid_manager_config.freeze and \
                self._table_name in ConfigInitializer.get_instance().sparse_embed_config.name_to_var_dict:
            self._variable = tf.compat.v1.get_variable(self._table_name, trainable=False,
                                                       shape=(self._slice_device_vocabulary_size, self._emb_size))
            if not ConfigInitializer.get_instance().use_dynamic_expansion:
                self.__record(eval_flag=True)
                tf.compat.v1.add_to_collection(
                    ConfigInitializer.get_instance().train_params_config.ascend_global_hashtable_collection,
                    self._variable)

        else:
            check_emb_init_params(self._is_hbm, self._embedding_size)
            self.__initialize_variables()
            tf.compat.v1.add_to_collection(
                ConfigInitializer.get_instance().train_params_config.ascend_global_hashtable_collection, self._variable)
        self._set_ext_emb_size()

    @property
    def optimizer_instance_list(self):
        return []

    @property
    def optimizer(self):
        return dict()

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def table_name(self):
        return self._table_name

    @property
    def key_dtype(self):
        return self._key_dtype

    @property
    def emb_initializer(self):
        return self._emb_initializer

    @property
    def is_save(self):
        return self._is_save

    @property
    def init_param(self):
        return self._init_param

    @property
    def send_count(self):
        return self._send_count

    @property
    def slice_device_vocabulary_size(self):
        return self._slice_device_vocabulary_size

    @property
    def slice_host_vocabulary_size(self):
        return self._slice_host_vocabulary_size

    @property
    def slice_ssd_vocabulary_size(self):
        return self._slice_ssd_vocabulary_size

    @property
    def emb_size(self):
        return self._emb_size

    @property
    def is_grad(self):
        return self._is_grad

    @property
    def ext_emb_size(self):
        return self._ext_emb_size

    @property
    def variable(self):
        return self._variable

    @property
    def multi_lookup_times(self):
        return self._multi_lookup_times

    @property
    def ssd_data_path(self):
        return self._ssd_data_path

    @property
    def is_hbm(self):
        return self._is_hbm

    @send_count.setter
    def send_count(self, send_count: int):
        self._send_count = send_count

    @ext_emb_size.setter
    def ext_emb_size(self, ext_emb_size: int):
        self._ext_emb_size = ext_emb_size

    @is_grad.setter
    def is_grad(self, is_grad: bool):
        self._is_grad = is_grad

    @staticmethod
    def get_anchor_attribute(anchor: tf.Tensor, attr: ASCAnchorAttr) -> \
            Union['BaseSparseEmbedding', FeatureSpec, bool]:
        """
        获取anchor ids对应的属性.

        Args:
            anchor: lookup传入的ids
            attr: 待获取属性名称

        Returns: anchor_tensor_specs中key为attr的属性.
        """
        if not isinstance(anchor, tf.Tensor):
            raise TypeError("Anchor must be a Tensor.")

        if attr not in ASCAnchorAttr:
            raise ValueError("Given attr must be limited in Enum 'ASCAnchorAttr'.")

        specs = BaseSparseEmbedding.anchor_tensor_specs.get(anchor)
        if specs is None:
            raise KeyError(f"Given anchor '{anchor}' was not registered.")

        return specs.get(attr)

    @abc.abstractmethod
    def capacity(self) -> int:
        """
        获取稀疏表的容量.
        Returns: 稀疏表的容量
        """
        pass

    @abc.abstractmethod
    def set_optimizer(self, key: str, state_dict: dict):
        """
        设置optimizer state.

        Args:
            key: 优化器名字
            state_dict: optimizer state

        Returns: None
        """
        pass

    @abc.abstractmethod
    def _set_slice_vocab_size(self):
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

    @abc.abstractmethod
    def _get_update_grad(self, local_grad: tf.Tensor, result: dict,
                         table: Union[tf.compat.v1.Variable, tf.Tensor]) -> Union[tf.IndexedSlices, tf.Tensor]:
        pass

    @abc.abstractmethod
    def _get_local_embeddings(self, table: Union[tf.compat.v1.Variable, tf.Tensor], result: dict,
                              feature_spec: FeatureSpec, **kwargs) -> tf.Tensor:
        pass

    @abc.abstractmethod
    def _get_sparse_forward_result(self, sparse_forward_fn: Callable, table: Union[tf.compat.v1.Variable, tf.Tensor],
                                   result: dict, is_training: bool) -> tf.Tensor:
        pass

    def size(self) -> int:
        """
        获取稀疏表的大小.
        Returns: 稀疏表的大小
        """
        return ConfigInitializer.get_instance().hybrid_manager_config.asc_manager.get_table_size(self._table_name)

    def register_anchor_attribute(self, anchor_ids: tf.Tensor, feature_spec: FeatureSpec, kwargs: dict):
        """
        注册anchor ids的相关属性.

        Args:
            anchor_ids: lookup传入的ids
            feature_spec: 根据ids创建的FeatureSpec
            kwargs: lookup参数字典

        Returns: None
        """
        self.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.TABLE_INSTANCE] = self
        self.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.IS_TRAINING] = kwargs.get("is_train")
        self.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.FEATURE_SPEC] = feature_spec
        self.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.IS_GRAD] = kwargs.get("is_grad")

    def get_default_lookup_name(self) -> str:
        """
        获取该表此次lookup的默认名字.
        Returns: lookup的默认名字
        """
        self._default_name_count += 1
        default_name = "sparse_lookup_%d" % self._default_name_count
        logger.debug("getting one default lookup name %s.", default_name)
        return default_name

    def increase_multi_lookup_times(self, is_training: bool):
        """
        增加该表的一次查询次数，用于校验一表多查次数.

        Args:
            is_training: 当前流程是训练还是推理

        Returns: None
        """
        self._multi_lookup_times[is_training] = self._multi_lookup_times.get(is_training) + 1

    def lookup(self, ids: tf.Tensor, send_count: Optional[int], **kwargs) -> tf.Tensor:
        """
        稀疏表的lookup，自动改图模式.

        Args:
            ids: 此次lookup的tensor
            send_count: all2all通信参数
            **kwargs: lookup参数字典

        Returns: lookup结果
        """
        is_training = kwargs.get("is_train")
        if ConfigInitializer.get_instance().hybrid_manager_config.freeze and is_training:
            raise RuntimeError("Cannot build new sparse forward graph after emb cache management was built.")

        # record send count
        eval_mode = not is_training and \
                    ConfigInitializer.get_instance().train_params_config.get_training_mode_channel_id(True) is None
        if is_training or eval_mode or \
                "train_and_evaluate" in ConfigInitializer.get_instance().train_params_config.bool_gauge_set:
            self._same_table_send_count += send_count if send_count is not None else 0

        # create feature spec
        feature_spec = get_feature_spec(self._table_name, kwargs.get("access_and_evict_config"))
        feature_spec.set_feat_attribute(ids, is_training)

        # record anchor ids
        anchor_ids = tf.identity(ids, name="ids")
        tf.compat.v1.add_to_collection(ASCEND_SPARSE_LOOKUP_ENTRANCE, anchor_ids)
        self.register_anchor_attribute(anchor_ids, feature_spec, kwargs)

        # set modify graph
        self._modify_graph = kwargs.get("modify_graph", True)

        # return the stub tensor of the lookup result
        if not self._use_static:
            kwargs["lookup_ids"] = ids
        mock_lookup_result = self._lookup_forward(feature_spec, send_count, **kwargs)
        mock_lookup_result = tf.identity(mock_lookup_result, name=ASCAnchorAttr.MOCK_LOOKUP_RESULT.value)
        if not kwargs.get("is_grad"):
            mock_lookup_result = tf.stop_gradient(mock_lookup_result, name="mock_stop_grad_lookup_res")
        self.anchor_tensor_specs[anchor_ids][ASCAnchorAttr.MOCK_LOOKUP_RESULT] = mock_lookup_result
        logger.debug("Return the stub tensor `%s` of the `%s` table.", mock_lookup_result, self._table_name)
        return mock_lookup_result

    def lookup_for_feat_spec(self, feature_spec: FeatureSpec, send_count: Optional[int], **kwargs) -> tf.Tensor:
        """
        稀疏表的lookup，FeatureSpec模式.

        Args:
            feature_spec: 此次lookup的tensor的包装类
            send_count: all2all通信参数
            **kwargs: lookup参数字典

        Returns: lookup结果
        """
        spec_name = feature_spec.name
        is_training = kwargs.get("is_train")
        if spec_name in self._lookup_result and is_training in self._lookup_result.get(spec_name):
            lookup_result = self._lookup_result.get(spec_name).get(is_training)
            if not kwargs.get("is_grad"):
                return tf.stop_gradient(lookup_result, name="stop_grad_lookup_result")
            return lookup_result

        if not self._use_static and not self._modify_graph and kwargs.get("batch") is None:
            raise RuntimeError("When the 'feature spec' mode and 'dynamic shape' are used, the 'batch' is required.")
        table_name = feature_spec.table_name
        same_table_feature_spec = \
            ConfigInitializer.get_instance().feature_spec_config.table_name_to_feature_spec[table_name][is_training]
        logger.debug("The feature spec of the same table is %s, table name is %s.",
                     ([fs.name for fs in same_table_feature_spec],), self._table_name)

        same_table_spec_count = len(same_table_feature_spec)
        if same_table_spec_count == 0:
            raise RuntimeError(f"spec_name {spec_name} not in table {table_name}.")

        if same_table_spec_count == 1:
            lookup_result = self._lookup_forward(feature_spec, send_count, **kwargs)
            if spec_name not in self._lookup_result:
                self._lookup_result[spec_name] = {}
            if not kwargs.get("is_grad"):
                lookup_result = tf.stop_gradient(lookup_result, name="stop_grad_lookup_result")
            self._lookup_result[spec_name][is_training] = lookup_result
            return lookup_result

        # 改图模式下FeatureSpec是按照lookup顺序创建的，无需对ids进行排序；fs模式下手动创建FeatureSpec，不一定有序
        if not self._modify_graph:
            same_table_feature_spec = sorted(same_table_feature_spec, key=lambda x: x.name)
        mock_feature_spec = FeatureSpec(f"mock_feature_spec_{table_name}", table_name=table_name)

        if self._use_static:
            tensor_list = []
            tensor_split_list = [feat_spec.split for feat_spec in same_table_feature_spec]
            total_feature_count = sum(tensor_split_list)
        else:
            tensor_list = self.__get_tensor_list(same_table_feature_spec, **kwargs)
            tensor_split_list = [tf.math.reduce_prod(array_ops.shape(tensor)) for tensor in tensor_list]
            total_feature_count = tf.add_n(tensor_split_list)
        set_temporary_feature_spec_attribute(mock_feature_spec, total_feature_count)

        kwargs["multi_lookup"] = True
        total_send_count = send_count * same_table_spec_count
        lookup_result = self._lookup_forward(mock_feature_spec, total_send_count, **kwargs)
        logger.debug("multi lookup table %s via %s.", table_name, tensor_split_list)
        self.__split_lookup_result(same_table_feature_spec, tensor_split_list, tensor_list, lookup_result, is_training)

        # 当一表多查完成后，将此表对应的feature specs列表清空，便于estimator模式下多轮eval时不会累加上轮eval的feature specs
        ConfigInitializer.get_instance().feature_spec_config.clear_same_table_feature_spec(self.table_name, is_training)
        if not kwargs.get("is_grad"):
            return tf.stop_gradient(self._lookup_result.get(spec_name).get(is_training), name="stop_grad_lookup_res")
        return self._lookup_result.get(spec_name).get(is_training)

    def _lookup_forward(self, feature_spec: FeatureSpec, send_count: Optional[int], **kwargs) -> tf.Tensor:
        is_training = kwargs.get("is_train")
        hashtable_params = dict(slice_device_vocabulary_size=self._slice_device_vocabulary_size,
                                slice_host_vocabulary_size=self._slice_host_vocabulary_size, send_count=send_count,
                                table_name=self._table_name, is_hbm=self._is_hbm)
        check_emb_lookup_params(hashtable_params, feature_spec, send_count, is_training)
        if ConfigInitializer.get_instance().use_static:
            self._send_count = send_count
        result = self._get_preprocessed_tensor(feature_spec, is_training, send_count)

        @tf.custom_gradient
        def sparse_forward(table):
            def grad(lookup_grad):
                logger.debug("Into lookup grad function, feature spec name: %s.", feature_spec.name)
                embedding_grad = tf.reshape(lookup_grad, [-1, self._emb_size])
                unique_grads = tf.compat.v1.unsorted_segment_sum(embedding_grad,
                                                                 result.get("restore_vector"),
                                                                 unique_embeddings_shape[0])
                bp_all2all_args = all2all_args if self._use_static else tf.transpose(all2all_args)
                hot, cold = tf.split(unique_grads,
                                     [tf.shape(result.get("hot_pos"))[0],
                                      tf.shape(unique_grads)[0] - tf.shape(result.get("hot_pos"))[0]], axis=0)
                unique_grads = tf.tensor_scatter_nd_add(cold, tf.expand_dims(result.get("hot_pos"), 1), hot)
                local_grad = self.__get_own_emb(unique_grads, bp_all2all_args)

                if self._all2all_gradients_op == All2allGradientsOp.SUM_GRADIENTS_AND_DIV_BY_RANKSIZE:
                    try:
                        local_grad = local_grad / get_rank_size()
                    except ZeroDivisionError as exp:
                        raise ZeroDivisionError("Rank size cannot be zero.") from exp

                return self._get_update_grad(local_grad, result, table)

            logger.debug("fp rank size: %s", self._rank_size)
            local_embeddings = self._get_local_embeddings(table, result, feature_spec, **kwargs)
            all2all_args = send_count if self._use_static else result.get("all2all_args")

            unique_embeddings = self.__get_own_emb(local_embeddings, all2all_args)
            unique_embeddings = tf.concat([tf.gather(unique_embeddings, result.get("hot_pos"), name="hot_pos"),
                                           unique_embeddings], axis=0)

            if self._use_static:
                unique_embeddings_shape = unique_embeddings.shape.as_list()
            else:
                unique_embeddings_shape = tf.shape(unique_embeddings)

            notify_hybridmgmt_op = self.__generate_lookup_id_notify_hybrid(is_training)
            with tf.control_dependencies([notify_hybridmgmt_op]):
                embeddings = tf.gather(unique_embeddings, result.get("restore_vector"), axis=0,
                                       name="gather_for_restore_vector")

            if self._use_static:
                return tf.reshape(embeddings, feature_spec.dims + [self._emb_size]), grad

            if kwargs.get("multi_lookup"):
                return tf.reshape(embeddings, [-1, self._emb_size]), grad

            feature_spec_tensor = None
            if not self._modify_graph:
                feature_spec_tensor = kwargs.get("batch").get(feature_spec.index_key)
            modify_graph_tensor = kwargs.get("lookup_ids")
            tensor = feature_spec_tensor if not self._modify_graph else modify_graph_tensor
            if tensor is None:
                raise KeyError(f"key or ids does not exist in batch, now modify graph is {self._modify_graph}.")
            dest_shape = array_ops.concat([array_ops.shape(tensor), [self._emb_size]], 0)

            return array_ops.reshape(embeddings, dest_shape), grad

        with tf.control_dependencies(result.get("swap_in")):
            return self._get_sparse_forward_result(sparse_forward, self._variable, result, is_training)

    def __initialize_variables(self):
        initialized_tensor = self._emb_initializer(
            self._slice_device_vocabulary_size + self._embedding_size) * self._init_param
        self._variable = tf.compat.v1.get_variable(self._table_name, trainable=False, initializer=initialized_tensor)

        # make sure sparse table variable will not be saved and restored within tf checkpoint.
        ConfigInitializer.get_instance().sparse_embed_config.insert_removing_var_list(self._variable.name)

        self.__record()
        self._build_optimizer_states()

    def __record(self, eval_flag=False):
        ConfigInitializer.get_instance().sparse_embed_config.insert_table_instance(
            self._table_name, self._variable, self, eval_flag)
        logger.debug("Device vocabulary_size for table %s is %s.", self._table_name, self._device_vocabulary_size)
        logger.debug("Slice_device_vocabulary_size for table %s is %s.",
                     self._table_name, self._slice_device_vocabulary_size)
        logger.debug("Host vocabulary size for table %s is %s.", self._table_name, self._host_vocabulary_size)
        logger.debug("Slice host vocabulary_size for table %s is %s.",
                     self._table_name, self._slice_host_vocabulary_size)
        logger.debug("SSD vocabulary size for table %s is %s.", self._table_name, self._ssd_vocabulary_size)
        logger.debug("Slice ssd vocabulary_size for table %s is %s.",
                     self._table_name, self._slice_ssd_vocabulary_size)

    def __get_own_emb(self, emb: tf.Tensor, all2all_args: Union[int, tf.Tensor]) -> tf.Tensor:
        src_emb = emb
        reshape_info = [all2all_args * self._rank_size, self._emb_size] if self._use_static else \
            [-1, self._emb_size]

        if self._rank_size == 1 and self._use_static:
            return tf.reshape(src_emb, reshape_info)

        if self._use_static:
            emb_send_cnt = tf.constant([all2all_args * self._emb_size] * self._rank_size, dtype=tf.int64)
            emb_send_offset = tf.constant([all2all_args * self._emb_size * i for i in range(self._rank_size)],
                                          dtype=tf.int64)
            src_emb = hccl_ops.all_to_all_v(send_data=emb,
                                            send_counts=emb_send_cnt,
                                            send_displacements=emb_send_offset,
                                            recv_counts=emb_send_cnt,
                                            recv_displacements=emb_send_offset)
        else:
            src_emb = hccl_ops.all_to_all_v_c(send_data=emb,
                                              send_count_matrix=all2all_args,
                                              rank=self._rank_id)

        return tf.reshape(src_emb, reshape_info)

    def __get_tensor_list(self, same_table_feature_spec: list, **kwargs) -> list:
        same_table_tensor_list = []
        for feat_spec in same_table_feature_spec:
            feature_spec_tensor_dict = kwargs.get("batch")
            modify_graph_tensor_dict = kwargs.get("feature_spec_name_ids_dict")
            batch_tensor_dict = feature_spec_tensor_dict if not self._modify_graph else modify_graph_tensor_dict
            if batch_tensor_dict is None:
                raise KeyError(f"The tensor dict of batch does not exist in kwargs, and modify graph "
                               f"is `{self._modify_graph}`.")

            feature_spec_tensor = batch_tensor_dict.get(feat_spec.index_key)
            modify_graph_tensor = batch_tensor_dict.get(feat_spec.name)
            tensor = feature_spec_tensor if not self._modify_graph else modify_graph_tensor
            if tensor is None:
                tensor_key = feat_spec.index_key if not self._modify_graph else feat_spec.name
                raise KeyError(f"Key `{tensor_key}` does not exist in batch_tensor_dict.")
            same_table_tensor_list.append(tensor)
        return same_table_tensor_list

    def __split_lookup_result(self, same_table_feature_spec: list, tensor_split_list: list, tensor_list: list,
                              lookup_result: tf.Tensor, is_training: bool):
        lookup_result_split = tf.split(lookup_result, tensor_split_list)
        if len(lookup_result_split) != len(same_table_feature_spec) or (
                not self._use_static and len(same_table_feature_spec) != len(tensor_list)):
            raise RuntimeError(f"shape not match. len(lookup_result_split): {len(lookup_result_split)},"
                               f"len(same_table_feature_spec): {len(same_table_feature_spec)}"
                               f"len(tensor_list): {len(tensor_list)}")
        for idx, (one_feature_spec, one_result) in enumerate(zip(same_table_feature_spec, lookup_result_split)):
            if one_feature_spec.name not in self._lookup_result:
                self._lookup_result[one_feature_spec.name] = {}
            if self._use_static:
                dest_shape = one_feature_spec.dims + [self._emb_size]
            else:
                dest_shape = array_ops.concat([array_ops.shape(tensor_list[idx]), [self._emb_size]], 0)
            self._lookup_result[one_feature_spec.name][is_training] = array_ops.reshape(one_result, dest_shape)

    def __generate_lookup_id_notify_hybrid(self, is_training: bool):
        """
        用于打桩的op节点，它的name用于标识此次的sparse lookup是train还是eval，后续在session run的时候，
        通过图反向查找该子图中查找到此op，最后通过名称判断session run是调用的哪个通道，并通知c++侧进行计数和唤醒操作。

        Args:
            is_training: 当前流程是训练还是推理

        Returns: 指定名字的tf.no_op()
        """
        channel_id = ConfigInitializer.get_instance().train_params_config.get_training_mode_channel_id(is_training)
        channel_name = "d2h_notify_hybridmgmt_{}".format(channel_id)
        notify_hybridmgmt_op = tf.no_op(channel_name)
        logger.debug("The notify hybridmgmg op of table `%s` is `%s`.", self._table_name, notify_hybridmgmt_op.name)
        return notify_hybridmgmt_op
