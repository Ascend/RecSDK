# coding: UTF-8
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2025. All rights reserved.
# Description: build script.
# Author: MindX SDK

import weakref

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python.data.ops.dataset_ops import _VariantTracker
from tensorflow.python.framework import ops


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
