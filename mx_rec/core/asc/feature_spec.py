#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import logging
from functools import reduce

import tensorflow as tf

from mx_rec.util.atomic import AtomicInteger
from mx_rec.util.initialize import insert_feature_spec, insert_training_mode_channel_id, get_use_static

feature_spec_global_id = AtomicInteger()


class FeatureSpec:
    instance_count_train = 0
    instance_count_eval = 0
    use_timestamp_train = False
    use_timestamp_eval = False

    def __init__(self, name, **kwargs):
        feature_spec_global_id.increase()
        spec_name = name + f"_{feature_spec_global_id}"
        self.name = spec_name
        self._index_key = kwargs.get("index_key") if kwargs.get("index_key") else name
        self._table_name = kwargs.get("table_name") if kwargs.get("table_name") else name
        self._feat_cnt = kwargs.get("feat_count")
        self._access_threshold = kwargs.get("access_threshold")
        self._eviction_threshold = kwargs.get("eviction_threshold")
        self._is_timestamp = kwargs.get("is_timestamp")
        self.feat_pos_train = None
        self.feat_pos_eval = None
        self.dims = None
        self.rank = None
        self.batch_size = kwargs.get("batch_size")
        self.split = None  # usually split == batch_size * feature_count
        self.initialized = False
        self._pipeline_mode = set()
        self.check_params()

    @property
    def is_timestamp(self):
        return self._is_timestamp

    @property
    def access_threshold(self):
        return self._access_threshold

    @property
    def eviction_threshold(self):
        return self._eviction_threshold

    @property
    def index_key(self):
        return self._index_key

    @property
    def table_name(self):
        return self._table_name

    @property
    def feat_cnt(self):
        return self._feat_cnt

    @property
    def pipeline_mode(self):
        return self._pipeline_mode

    @staticmethod
    def include_timestamp(is_training):
        if is_training:
            if FeatureSpec.use_timestamp_train:
                raise EnvironmentError(f"Timestamp was set twice for training mode.")
            FeatureSpec.use_timestamp_train = True
        else:
            FeatureSpec.use_timestamp_eval = True

    @staticmethod
    def use_timestamp(is_training):
        return FeatureSpec.use_timestamp_train if is_training else FeatureSpec.use_timestamp_eval

    def check_params(self):
        def check_str(arg, param_name):
            if not isinstance(arg, str):
                raise TypeError(f"{param_name} should be a string, whose value is {arg} with type '{type(arg)}' "
                                f"in fact.")

        def check_natural_number(arg, param_name):
            if not isinstance(arg, int) or arg < 1:
                raise TypeError(f"{param_name} should be a natural number, whose value is {arg} with type "
                                f"'{type(arg)}' in fact.")

        def check_bool(arg, param_name):
            if not isinstance(arg, bool):
                raise TypeError(f"{param_name} should be a bool, whose value is {arg} with type "
                                f"'{type(arg)}' in fact.")

        check_str(self.name, "name")
        check_str(self._table_name, "table_name")

        if self._feat_cnt is not None:
            check_natural_number(self._feat_cnt, "feat_count")

        if self._access_threshold is not None:
            check_natural_number(self._access_threshold, "access_threshold")
        elif self._eviction_threshold is not None:
            raise ValueError(f"Access_threshold should be configured before eviction_threshold.")

        if self._eviction_threshold is not None:
            check_natural_number(self._eviction_threshold, "eviction_threshold")

        if self._is_timestamp is not None:
            check_bool(self._is_timestamp, "is_timestamp")

    def set_feat_pos(self, is_training):
        if is_training:
            self.feat_pos_train = FeatureSpec.instance_count_train
            FeatureSpec.instance_count_train += 1
        else:
            self.feat_pos_eval = FeatureSpec.instance_count_eval
            FeatureSpec.instance_count_eval += 1

    def insert_pipeline_mode(self, mode):
        if not isinstance(mode, bool):
            raise TypeError("Is training mode must be a boolean.")

        if mode and mode in self._pipeline_mode:
            logging.info(f"FeatureSpec{self.name}. Is training mode [{mode}] has been set.")
            return

        insert_training_mode_channel_id(is_training=mode)

        self._pipeline_mode.add(mode)

    def set_feat_attribute(self, tensor, is_training):
        self.insert_pipeline_mode(is_training)
        self.set_feat_pos(is_training)
        if not self.initialized:
            self.initialized = True

            if get_use_static():
                self.dims = tensor.shape.as_list()
                self.rank = tensor.shape.rank
                if self.rank < 1:
                    raise ValueError(f"Given tensor rank cannot be smaller than 1, which is {self.rank} now.")

                inferred_feat_cnt = 1 if self.rank == 1 else reduce(lambda x, y: x * y, self.dims[1:])
                logging.debug(f"update feature_spec[{self.name}] feature_count "
                              f"from {self._feat_cnt} to {inferred_feat_cnt} via {self.dims}")
                self.batch_size = self.dims[0]
                self._feat_cnt = inferred_feat_cnt
                self.split = self.batch_size * self._feat_cnt
            else:
                tensor = tf.reshape(tensor, [-1])
                self.dims = tf.shape(tensor)
                self.rank = 1
                self.split = tf.math.reduce_prod(tf.shape(tensor))
                self.batch_size = self.split
                self._feat_cnt = 1

        else:
            logging.debug(f"The initialized Feature Spec was set once again.")
            if get_use_static():
                if self.dims != tensor.shape.as_list():
                    raise ValueError(f"Given static Tensor shape mismatches with the last one, whose is_training mode "
                                     f"is not {is_training}. ")
            else:
                if self.dims.shape.as_list() != tf.shape(tf.reshape(tensor, [-1])).shape.as_list():
                    raise ValueError(f"Given dynamic Tensor shape mismatches with the last one, whose is_training mode "
                                     f"is not {is_training}. ")

        insert_feature_spec(self, is_training)
        reslt = {
            'tensor' : tensor,
            'table_name' : self.table_name,
            'feat_count' : self.feat_cnt,
            'split' : self.split,
        }
        return reslt


def get_feature_spec(table_name, access_and_evict_config):
    access_threshold = None
    eviction_threshold = None
    if access_and_evict_config:
        access_threshold = access_and_evict_config.get("access_threshold")
        eviction_threshold = access_and_evict_config.get("eviction_threshold")
    return FeatureSpec(table_name, access_threshold=access_threshold, eviction_threshold=eviction_threshold)
