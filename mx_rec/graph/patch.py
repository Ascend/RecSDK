#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import weakref
import logging

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python.data.ops.dataset_ops import _VariantTracker
from tensorflow.python.framework import ops

from mx_rec.util.initialize import get_is_graph_modify_hook_running, get_modify_graph


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
