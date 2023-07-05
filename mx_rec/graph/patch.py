#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import weakref
import logging
from typing import Any

import tensorflow as tf
import tensorflow_estimator as tensorflow_estimator_lib
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python.data.ops.dataset_ops import _VariantTracker
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator.training import EvalSpec
from tensorflow.python.eager.monitoring import BoolGauge, BoolGaugeCell

from mx_rec.util.initialize import get_is_graph_modify_hook_running, get_modify_graph, insert_bool_gauge, \
    get_bool_gauge_set, terminate_config_initializer, get_run_times, set_is_last_round
from mx_rec.util.tf_version_adapter import NPUCheckpointSaverHook


def init_dataset(self, input_data):
    """
    input_data: A DT_VARIANT tensor that represents the dataset.
    """
    # pylint: disable=W
    tf.compat.v1.add_to_collection("dataset_group", self)
    self._variant_tensor_attr = input_data
    # get obj
    dataset_obj = weakref.proxy(self)
    self._variant_tracker = self._track_trackable(
        _VariantTracker(self._variant_tensor, lambda: dataset_obj._trace_variant_creation()()), name="_variant_tracker")
    self._graph_attr = ops.get_default_graph()


def patch_for_dataset():
    DatasetV2.__init__ = init_dataset


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
    logging.debug("Enter the mxrec init function of Class 'monitored_session.ChiefSessionCreator'.")
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
    logging.debug("__init__ in Class 'monitored_session.ChiefSessionCreator' has been patched.")


def get_cell(self: BoolGauge, *labels: Any) -> Any:
    """
    Retrieves the cell.
    Args:
        self: An `BoolGauge` instance.
        *labels: The label list of the new metric.

    Returns: Obtains the cell value set by the user.
    """

    logging.debug(f"Enter patch 'BoolGauge.get_cell'.")
    if len(labels) > 0:
        insert_bool_gauge(labels[0])
    return BoolGaugeCell(super(BoolGauge, self).get_cell(*labels))


def patch_for_bool_gauge():
    """Patch for 'BoolGauge.get_cell'."""

    BoolGauge.get_cell = get_cell
    logging.debug(f"Function 'get_cell' in Class 'BoolGauge' has been patched.")


def end(self: NPUCheckpointSaverHook, session: tf.compat.v1.Session):
    """
    Call at the end of session hook.

    Args:
        self: An `NPUCheckpointSaverHook` instance.
        session: A TensorFlow Session that will be soon closed.

    Returns: None

    """

    logging.debug(f"Enter patch 'NPUCheckpointSaverHook.end'.")
    logging.info("NPUCheckpointSaverHook end...")
    basic_session_run_hooks.CheckpointSaverHook.end(self, session)

    if 'train_and_evaluate' in get_bool_gauge_set() and get_run_times() == 1:
        set_is_last_round(True)
        return
    logging.debug(f"NPUCheckpointSaverHook call 'terminate_config_initializer'...")
    terminate_config_initializer()


def patch_for_end():
    """Patch for 'NPUCheckpointSaverHook.end'."""

    NPUCheckpointSaverHook.end = end
    logging.debug(f"Function 'end' in Class 'NPUCheckpointSaverHook' has been patched.")


def assert_eval_spec(eval_spec: EvalSpec):
    """
    Raise error if `eval_spec` is not of the right type.

    Args:
        eval_spec: A `TrainSpec` instance to specify the training specification.

    Returns: None

    """

    logging.debug(f"Enter patch 'tensorflow_estimator.python.estimator.training._assert_eval_spec'.")
    if not isinstance(eval_spec, EvalSpec):
        raise TypeError('`eval_spec` must have type `tf.estimator.EvalSpec`. Got: {}'.format(type(eval_spec)))

    if 'train_and_evaluate' not in get_bool_gauge_set():
        insert_bool_gauge('train_and_evaluate')
        logging.debug("assert_eval_spec: add 'train_and_evaluate' to BoolGaugeCell.")


def patch_for_assert_eval_spec():
    """Patch for 'tensorflow_estimator.python.estimator.training._assert_eval_spec'."""

    tensorflow_estimator_lib.python.estimator.training._assert_eval_spec = assert_eval_spec
    logging.debug(f"Function '_assert_eval_spec' in 'tensorflow_estimator.python.estimator.training' has been patched.")
