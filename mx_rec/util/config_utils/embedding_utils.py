#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
from typing import Optional

from tensorflow import Variable

from mx_rec.util.log import logger


class SparseEmbedConfig:
    """
    Sparse table related configurations.
    """
    def __init__(self):
        self._table_instance_dict = dict()
        self._dangling_table = []
        self._table_name_set = set()
        self._removing_var_list = []
        self._name_to_var_dict = dict()

    @property
    def table_instance_dict(self):
        return self._table_instance_dict

    @property
    def dangling_table(self):
        return self._dangling_table

    @property
    def table_name_set(self):
        return self._table_name_set

    @property
    def name_to_var_dict(self):
        return self._name_to_var_dict

    @property
    def removing_var_list(self):
        return self._removing_var_list

    def get_table_instance(self, key) -> object:
        if key not in self._table_instance_dict:
            raise KeyError(f"Given key does not exist.")

        return self._table_instance_dict.get(key)

    def get_table_instance_by_name(self, table_name: Optional[str]) -> object:
        if table_name not in self._name_to_var_dict:
            raise KeyError(f"Given table name does not exist.")

        key = self._name_to_var_dict.get(table_name)
        return self._table_instance_dict.get(key)

    def insert_dangling_table(self, table_name: Optional[str]) -> None:
        if table_name not in self._dangling_table:
            self._dangling_table.append(table_name)

    def insert_removing_var_list(self, var_name) -> None:
        if var_name not in self._removing_var_list:
            self._removing_var_list.append(var_name)

    def insert_table_instance(self, name: str, key: Variable, instance: object, eval_flag: bool) -> None:
        if eval_flag:
            name = name + ".eval_flag"
        if key in self._table_instance_dict:
            raise KeyError(f"Given key {key} has been used.")

        if name in self._table_name_set:
            raise ValueError(f"Duplicated hashtable name '{name}' was used.")

        logger.debug("Record one hash table, with name: %s, key: %s.", name, key)
        self._table_name_set.add(name)
        self._name_to_var_dict[name] = key
        self._table_instance_dict[key] = instance

    def export_table_num(self) -> int:
        return len(self.table_instance_dict) if self.table_instance_dict else 0
