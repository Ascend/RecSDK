#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.


class OptimizerConfig:
    def __init__(self):
        self._optimizer_instance = None

    @property
    def optimizer_instance(self):
        return self._optimizer_instance

    @optimizer_instance.setter
    def optimizer_instance(self, optimizer):
        self._optimizer_instance = optimizer
