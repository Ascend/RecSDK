#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
from typing import Optional, Any

from tensorflow.python.framework import ops
from tensorflow import Variable

from rec_sdk_common.log.log import LoggingProxy as logger


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
        self._tensor_to_table_instance_dict = dict()

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
        """
        Get table instance by key.

        Args:
            key: It's tf.Tensor in dynamic expansion mode and tf.Variable in normal mode(HBM/DDR/SSD).

        Returns: Table instance.

        """

        from mx_rec.util.initialize import ConfigInitializer

        # Dynamic expansion mode.
        if ConfigInitializer.get_instance().use_dynamic_expansion and isinstance(key, ops.Tensor):
            return self.get_table_instance_by_tensor(key)

        # Normal mode.
        if key not in self._table_instance_dict:
            raise KeyError("given key => '{}' does not exist".format(key))

        return self._table_instance_dict.get(key)

    def get_table_instance_by_tensor(self, tensor) -> object:
        if tensor not in self._tensor_to_table_instance_dict:
            raise KeyError("given tensor => '{}' does not exist".format(tensor))

        return self._tensor_to_table_instance_dict.get(tensor)

    def get_table_instance_by_name(self, table_name: Optional[str]) -> object:
        if table_name not in self._name_to_var_dict:
            raise KeyError("given table name => '{}' does not exist".format(table_name))

        key = self._name_to_var_dict.get(table_name)
        return self._table_instance_dict.get(key)

    def insert_dangling_table(self, table_name: Optional[str]) -> None:
        if table_name not in self._dangling_table:
            self._dangling_table.append(table_name)

    def insert_removing_var_list(self, var_name) -> None:
        if var_name not in self._removing_var_list:
            self._removing_var_list.append(var_name)

    def insert_table_instance(self, name: str, key: Variable, instance: object, eval_flag: bool) -> None:
        if key in self._table_instance_dict:
            raise KeyError(f"Given key {key} has been used.")
        if eval_flag:
            self._table_instance_dict[key] = instance
            return
        if name in self._table_name_set:
            raise ValueError(f"Duplicated hashtable name '{name}' was used.")

        logger.debug("Record one hash table, with name: %s, key: %s.", name, key)
        self._table_name_set.add(name)
        self._name_to_var_dict[name] = key
        self._table_instance_dict[key] = instance

    def insert_table_instance_to_tensor_dict(self, tensor: ops.Tensor, instance: object) -> None:
        if tensor in self._tensor_to_table_instance_dict:
            raise KeyError(f"Given tensor {tensor} has been used.")
        logger.debug("Record one hash table for expansion mode, with tensor: %s.", tensor)
        self._tensor_to_table_instance_dict[tensor] = instance

    def update_table_instance(self, table_name: str, emb_table: Any, old_var: Variable, new_var: Variable) -> None:
        self._name_to_var_dict[table_name] = new_var
        del self._table_instance_dict[old_var]
        self._table_instance_dict[new_var] = emb_table

    def export_table_num(self) -> int:
        return len(self.table_instance_dict) if self.table_instance_dict else 0
