#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.


class OptimizerConfig:
    def __init__(self):
        self._optimizer_instance = None
        self._table_optimizer_dict = {}

    @property
    def optim_params_list(self):
        if not self._optimizer_instance:
            return []
        return self._optimizer_instance.optim_param_list

    @property
    def optimizer_instance(self):
        return self._optimizer_instance

    @optimizer_instance.setter
    def optimizer_instance(self, optimizer):
        self._optimizer_instance = optimizer

    def set_optimizer_for_table(self, table_name, optimizer_name, optimizer_dict):
        self._table_optimizer_dict[table_name] = {optimizer_name: optimizer_dict}

    def get_optimizer_by_table_name(self, table_name):
        return self._table_optimizer_dict.get(table_name)

