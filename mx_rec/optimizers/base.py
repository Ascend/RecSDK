#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict


class CustomizedOptimizer:

    name_counter = defaultdict(int)

    def __init__(self):
        self.unique_name = ""
        self.base_name = ""

    def _get_name(self, name="CustomizedOptimizer"):
        if name in CustomizedOptimizer.name_counter:
            CustomizedOptimizer.name_counter[name] += 1
            count = CustomizedOptimizer.name_counter.get(name)

        else:
            count = CustomizedOptimizer.name_counter[name]
        self.unique_name = name + "_" + str(count)
        self.base_name = name

    def initialize_slots(self, var, table_instance):
        raise NotImplementedError(f"Please define a specific realization on {self.__class__.__name__}")

    def insert_slot(self, slot, named_slots_key, slot_name):
        raise NotImplementedError(f"Please define a specific realization on {self.__class__.__name__}")

    def get_slot_init_values(self):
        raise NotImplementedError(f"Please define a specific realization on {self.__class__.__name__}")
