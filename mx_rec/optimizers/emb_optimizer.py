#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from mx_rec.optimizers.base import CustomizedOptimizer
from mx_rec.util.tf_version_adapter import NPULossScaleOptimizer


class EmbOptimizer:
    """
    稀疏表的优化器.
    """

    def __init__(self, optimizer_list):
        self._optimizer_instance_list = optimizer_list
        self._optimizer_slot_info_list = []
        self._optimizer = dict()

    @property
    def optimizer_instance_list(self):
        return self._optimizer_instance_list

    @property
    def optimizer_slot_info_list(self):
        return self._optimizer_slot_info_list

    @property
    def optimizer(self):
        return self._optimizer

    @staticmethod
    def set_optimizer_slot(slot_info: dict):
        """
        设置稀疏表优化器的slot信息.

        Args:
            slot_info: 优化器slot信息

        Returns: None
        """
        slot = slot_info.get("slot")
        slot_name = slot_info.get("slot_name")
        optimizer = slot_info.get("optimizer")
        named_slot_key = slot_info.get("named_slot_key")

        optimizer.insert_slot(slot, named_slot_key, slot_name)

    def set_optimizer(self, key: str, state_dict: dict, table_name: str):
        """
        设置optimizer state.

        Args:
            key: 优化器名字
            state_dict: optimizer state
            table_name: 稀疏表名

        Returns: None
        """
        if key in self._optimizer:
            raise ValueError(f"optimizer {key} has been set for hash table {table_name}.")
        self._optimizer[key] = state_dict

    def check_optimizer_instance_list(self):
        """
        校验优化器实例列表.
        """
        if not self._optimizer_instance_list:
            raise ValueError("External storage mode should config optimizers before instantiating sparse table, "
                             "but nothing was configured.")

        for optimizer_instance in self._optimizer_instance_list:
            if isinstance(optimizer_instance, NPULossScaleOptimizer):
                optimizer_instance = getattr(optimizer_instance, '_opt')

            if not isinstance(optimizer_instance, CustomizedOptimizer):
                raise TypeError("the optimizer instance must be an instance of CustomizedOptimizer.")
