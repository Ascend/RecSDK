#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import weakref
from typing import Any

import tensorflow as tf
import tensorflow_estimator as tensorflow_estimator_lib
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python.data.ops.dataset_ops import _VariantTracker
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator.training import EvalSpec
from tensorflow.python.eager.monitoring import BoolGauge, BoolGaugeCell
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.client.session import BaseSession

from mx_rec.constants import constants
from mx_rec.util.initialize import get_is_graph_modify_hook_running, get_modify_graph, insert_bool_gauge, \
    get_bool_gauge_set, terminate_config_initializer, get_run_times, set_is_last_round, get_asc_manager, \
    export_table_instances
from mx_rec.util.tf_version_adapter import NPUCheckpointSaverHook
from mx_rec.graph.merge_lookup import do_merge_lookup
from mx_rec.util.log import logger


def init_dataset(self, input_data):
    """
    input_data: A DT_VARIANT tensor that represents the dataset.
    """
    tf.compat.v1.add_to_collection("dataset_group", self)
    self._variant_tensor_attr = input_data
    # get obj
    dataset_obj = weakref.proxy(self)
    self._variant_tracker = self._track_trackable(
        _VariantTracker(self._variant_tensor, lambda: dataset_obj._trace_variant_creation()()), name="_variant_tracker")
    self._graph_attr = ops.get_default_graph()


def run(self, fetches, feed_dict=None, options=None, run_metadata=None):

    """
    Replace tensorflow's session run method with this method, this method will
     notify  the hybridMgmt side to wake up and count each time sess run is called.

    Args:
      fetches: A single graph element, a list of graph elements, or a dictionary
        whose values are graph elements or lists of graph elements (described
        above).
      feed_dict: A dictionary that maps graph elements to values (described
        above).
      options: A [`RunOptions`] protocol buffer
      run_metadata: A [`RunMetadata`] protocol buffer

    Returns:
      Either a single value if `fetches` is a single graph element, or
      a list of values if `fetches` is a list, or a dictionary with the
      same keys as `fetches` if that is a dictionary (described above).
      Order in which `fetches` operations are evaluated inside the call
      is undefined.

    Raises:
      RuntimeError: If this `Session` is in an invalid state (e.g. has been
        closed).
      TypeError: If `fetches` or `feed_dict` keys are of an inappropriate type.
      ValueError: If `fetches` or `feed_dict` keys are invalid or refer to a
        `Tensor` that doesn't exist.
    Returns:None
    """

    all_op = []

    def get_all_tensor(tensor_or_tensorlist):
        # 把所有的tensor和Operation取出来
        if isinstance(tensor_or_tensorlist, (list, tuple)) :
            for i in tensor_or_tensorlist:
                get_all_tensor(i)
        elif isinstance(tensor_or_tensorlist, dict):
            for k in tensor_or_tensorlist.keys():
                get_all_tensor(tensor_or_tensorlist.get(k))
        elif isinstance(tensor_or_tensorlist, (tf.Tensor, tf.Operation)):
            name = tensor_or_tensorlist.name
            if ":" in name:
                name = name[:name.find(":")]
            all_op.append(name)

    def get_channel_id_by_sub_graph(input_tensors, name2channel_cache):
        # 通过fetches需要运行的节点来找到 spase look up中的打桩tensor
        # 从而判断该session run运行的是train还是eval
        name_list_str_key = "_".join(input_tensors)
        if name_list_str_key in name2channel_cache.keys():
            return name2channel_cache.get(name_list_str_key)
        this_channel_id = -1
        graph_def = self.graph_def
        cut_graph_input = tf.compat.v1.graph_util.extract_sub_graph(graph_def, input_tensors)
        node_list_input = cut_graph_input.node
        for node in node_list_input:
            if "d2h_notify_hybridmgmt_" in node.name:
                this_channel_id = int(node.name[-1])
                break
        name2channel_cache[name_list_str_key] = this_channel_id
        return this_channel_id

    # patch的方式为图增加缓存属性
    name2channel_cache = self.get_mxrec_name2channel_cache()

    # 查找相应的channel_id
    get_all_tensor(fetches)
    try:
        channel_id = get_channel_id_by_sub_graph(all_op, name2channel_cache)
    except AssertionError:
        channel_id = -1

    if channel_id != -1:
        get_asc_manager().block_notify_wake(channel_id)

    if channel_id == constants.EVAL_CHANNEL_ID:
        # eval的时候不进行循环下沉
        steps = 1
    else:
        # patch的方式为session增加步数属性
        steps = self.get_mxrec_steps()

    # 调用tensorflow原生的方法
    result = self.old_run_method(fetches, feed_dict, options, run_metadata)
    if channel_id != -1:
        get_asc_manager().block_count_steps(channel_id, steps)
    return result


def patch_for_dataset():
    DatasetV2.__init__ = init_dataset


def patch_for_session():

    def get_mxrec_steps(self):
        try:
            # 不能在未调用非__init__函数之前调用非__init__中定义的实例化属性
            return self.steps
        except AttributeError:
            self.steps = 1
            for custom_optimizer in self.get_config().graph_options.rewrite_options.custom_optimizers:
                if custom_optimizer.name == "NpuOptimizer" \
                        and custom_optimizer.parameter_map["iterations_per_loop"].i != 0:
                    self.steps = custom_optimizer.parameter_map["iterations_per_loop"].i
                    break
            return self.steps

    def get_mxrec_name2channel_cache(self):
        try:
            # 不能在未调用非__init__函数之前调用非__init__中定义的实例化属性
            return self.name2channel_cache
        except AttributeError:
            self.name2channel_cache = {}
            return self.name2channel_cache

    def get_config(self):
        return getattr(self, '_config')

    BaseSession.old_run_method = BaseSession.run
    BaseSession.run = run
    BaseSession.get_mxrec_name2channel_cache = get_mxrec_name2channel_cache
    BaseSession.get_mxrec_steps = get_mxrec_steps
    BaseSession.get_config = get_config


def chief_session_creator_init(self, scaffold=None, master='', config=None, checkpoint_dir=None,
                               checkpoint_filename_with_path=None):
    """
    Initializes a chief session creator and check if 'GraphModifierHook' is configured.

    Args:
        self: An instance object of the class ChiefSessionCreator.
        scaffold: A `Scaffold` used for gathering or building supportive ops. If
            not specified a default one is created. It's used to finalize the graph.
        master: `String` representation of the TensorFlow master to use.
        config: `ConfigProto` proto used to configure the session.
        checkpoint_dir: A string. Optional path to a directory where to restore variables.
        checkpoint_filename_with_path: Full file name path to the checkpoint file.
    Returns:None
    """
    logger.debug("Enter the mxrec init function of Class 'monitored_session.ChiefSessionCreator'.")
    if get_modify_graph() and not get_is_graph_modify_hook_running():
        raise RuntimeError(
            f"When 'modify_graph' is True, 'GraphModifierHook' must be configured. Example: \n"
            f"\t from mx_rec.graph.modifier import GraphModifierHook \n"
            f"\t estimator.train(..., hooks=[GraphModifierHook()])")

    self._checkpoint_dir = checkpoint_dir
    self._checkpoint_filename_with_path = checkpoint_filename_with_path
    self._scaffold = scaffold or tf.compat.v1.train.Scaffold()
    self._session_manager = None
    self._master = master
    self._config = config


def patch_for_chief_session_creator():
    """
    The 'train, predict, train_and_evaluate' mode in the estimator mode ultimately creates the 'ChiefSessionCreator'
    class, so it can be determined whether 'GraphModifierHook' is configured in the init function of this class.
    Returns:None
    """
    tf.compat.v1.train.ChiefSessionCreator.__init__ = chief_session_creator_init
    logger.debug("__init__ in Class 'monitored_session.ChiefSessionCreator' has been patched.")


def get_cell(self: BoolGauge, *labels: Any) -> Any:
    """
    Retrieves the cell.
    Args:
        self: An `BoolGauge` instance.
        *labels: The label list of the new metric.

    Returns: Obtains the cell value set by the user.
    """

    logger.debug("Enter patch 'BoolGauge.get_cell'.")
    if len(labels) > 0:
        logger.debug("BoolGauge insert: %s.", labels[0])
        insert_bool_gauge(labels[0])
    return BoolGaugeCell(super(BoolGauge, self).get_cell(*labels))


def patch_for_bool_gauge():
    """Patch for 'BoolGauge.get_cell'."""

    BoolGauge.get_cell = get_cell
    logger.debug("Function 'get_cell' in Class 'BoolGauge' has been patched.")


def end(self: NPUCheckpointSaverHook, session: tf.compat.v1.Session):
    """
    Call at the end of session hook.

    Args:
        self: An `NPUCheckpointSaverHook` instance.
        session: A TensorFlow Session that will be soon closed.

    Returns: None

    """

    logger.debug("Enter patch 'NPUCheckpointSaverHook.end'.")
    logger.info("NPUCheckpointSaverHook end...")
    basic_session_run_hooks.CheckpointSaverHook.end(self, session)

    if 'train_and_evaluate' in get_bool_gauge_set() and get_run_times() == 1:
        set_is_last_round(True)
        return
    logger.debug("NPUCheckpointSaverHook call 'terminate_config_initializer'...")
    terminate_config_initializer()


def patch_for_end():
    """Patch for 'NPUCheckpointSaverHook.end'."""

    NPUCheckpointSaverHook.end = end
    logger.debug("Function 'end' in Class 'NPUCheckpointSaverHook' has been patched.")


def assert_eval_spec(eval_spec: EvalSpec):
    """
    Raise error if `eval_spec` is not of the right type.

    Args:
        eval_spec: A `TrainSpec` instance to specify the training specification.

    Returns: None

    """

    logger.debug("Enter patch 'tensorflow_estimator.python.estimator.training._assert_eval_spec'.")
    if not isinstance(eval_spec, EvalSpec):
        raise TypeError('`eval_spec` must have type `tf.estimator.EvalSpec`. Got: {}'.format(type(eval_spec)))

    if 'train_and_evaluate' not in get_bool_gauge_set():
        insert_bool_gauge('train_and_evaluate')
        logger.debug("assert_eval_spec: add 'train_and_evaluate' to BoolGaugeCell.")


def patch_for_assert_eval_spec():
    """Patch for 'tensorflow_estimator.python.estimator.training._assert_eval_spec'."""

    tensorflow_estimator_lib.python.estimator.training._assert_eval_spec = assert_eval_spec
    logger.debug("Function '_assert_eval_spec' in 'tensorflow_estimator.python.estimator.training' has been patched.")


def scale_loss(self: Optimizer, loss_value: tf.Tensor) -> tf.Tensor:
    """
    Multiply the loss value by a scalar factor.

    Args:
        self: self: An `Optimizer` instance.
        loss_value: A Tensor containing the value to minimize or a callable taking no arguments which returns the value
                    to minimize. When eager execution is enabled it must be a callable.

    Returns: loss_value

    """

    logger.debug("Enter patch 'Optimizer._scale_loss'.")
    # In train mode, merge lookup must be completed during compute gradients.
    # Ensure that the backward of graph is constructed and the gradient calculation is correct.
    do_merge_lookup(is_train=True)

    # 在训练情况下，至少要有一个variable参与反向，否则报错
    is_grad = False
    table_var_list = []
    for _, table_instance in export_table_instances().items():
        is_grad |= table_instance.is_grad
        table_var_list.append(table_instance.variable)
    if not is_grad:
        raise RuntimeError("No gradients provided for any variable: %s." % (table_var_list,))

    # origin code
    ops.get_default_graph()._is_loss_scaled_by_optimizer = False
    if distribute_lib.get_loss_reduction() == ds_reduce_util.ReduceOp.MEAN:
        # origin name is num_replicas
        loss_num_replicas = distribute_ctx.get_strategy().num_replicas_in_sync
        if loss_num_replicas > 1:
            loss_value *= (1. / loss_num_replicas)
            ops.get_default_graph()._is_loss_scaled_by_optimizer = True
    return loss_value


def patch_for_scale_loss():
    """Patch for 'Optimizer._scale_loss'."""

    Optimizer._scale_loss = scale_loss
    logger.debug("Function '_scale_loss' in Class 'Optimizer' has been patched.")
