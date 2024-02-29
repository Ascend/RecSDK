#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
from typing import Optional

from mx_rec.util.log import logger


class HybridManagerConfig:
    def __init__(self):
        self._asc_manager = None
        self._is_freeze = False

    @property
    def asc_manager(self):
        return self._asc_manager

    @property
    def freeze(self):
        return self._is_freeze

    def set_asc_manager(self, manager) -> None:
        from mxrec_pybind import HybridMgmt
        if not isinstance(manager, HybridMgmt):
            raise ValueError(f"Given manager must be the instance of {HybridMgmt}, which is {type(manager)} "
                             f"type currently.")
        self._asc_manager = manager
        self._is_freeze = True

    def del_asc_manager(self) -> None:
        if self.asc_manager:
            self._asc_manager.destroy()
            self._asc_manager = None
            self._is_freeze = False
        logger.debug("ASC manager has been destroyed.")

    def trigger_evict(self) -> bool:
        if not self._asc_manager:
            raise RuntimeError("ASC manager does not exist.")

        if self.asc_manager.evict():
            logger.debug("Feature evict is triggered by ops.")
            return True
        logger.warning("Feature evict not success, skip this time!")
        return False

    def get_host_data(self, table_name: str) -> object:
        if self.asc_manager is None:
            raise RuntimeError("ASC manager does not exist.")
        logger.debug("start to get host data.")
        return self.asc_manager.send(table_name)

    def save_host_data(self, root_dir: Optional[str]) -> None:
        if self.asc_manager is None:
            raise RuntimeError("ASC manager does not exist.")

        self.asc_manager.save(root_dir)
        logger.debug("Data from host pipeline has been saved.")

    def restore_host_data(self, root_dir: Optional[str]) -> None:
        if self.asc_manager is None:
            raise RuntimeError("ASC manager does not exist.")

        if not self.asc_manager.load(root_dir):
            raise TypeError("Asc load data does not match usr setups, \
            please re-consider if you want to restore from this dir")
        logger.debug("Data from host pipeline has been restored.")
