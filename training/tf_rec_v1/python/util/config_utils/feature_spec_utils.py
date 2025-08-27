#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
from typing import Optional

from rec_sdk_common.log import logger


class FeatureSpecConfig:
    """
    Feature Spec configurations, including relationship between table name and feature spec
    """
    def __init__(self):
        self._table_name_to_feature_spec = dict()
        self._feature_spec_dict = dict()

    @property
    def table_name_to_feature_spec(self):
        return self._table_name_to_feature_spec

    @property
    def feature_spec_dict(self):
        return self._feature_spec_dict

    def clear_same_table_feature_spec(self, table_name: Optional[str], is_training: bool) -> None:
        if self.table_name_to_feature_spec.get(table_name) is None or \
                self.table_name_to_feature_spec.get(table_name).get(is_training) is None:
            raise KeyError("the table name `%s` does not exist in table_name_to_feature_spec, "
                           "please check whether the insert_feature_spec(...) is invoked.", table_name)
        self.table_name_to_feature_spec.get(table_name)[is_training] = []
        logger.debug("The feature spec of the table name `%s` has been cleared.", table_name)

    def insert_feature_spec(self, feature_spec: object, is_training: bool) -> None:
        self._feature_spec_dict[feature_spec.name] = feature_spec
        if feature_spec.table_name not in self._table_name_to_feature_spec:
            self._table_name_to_feature_spec[feature_spec.table_name] = {True: [], False: []}
        self._table_name_to_feature_spec[feature_spec.table_name][is_training].append(feature_spec)

    def get_feature_spec(self, feature_spec_name: Optional[str]) -> object:
        return self._feature_spec_dict.get(feature_spec_name)
