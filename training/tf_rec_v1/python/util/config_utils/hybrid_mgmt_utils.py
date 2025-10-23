#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Optional

from rec_sdk_common.log.log import LoggingProxy as logger


class HybridManagerConfig:  # pragma: no cover
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
            raise ValueError(
                f"given manager must be the instance of {HybridMgmt}, which is {type(manager)} type currently"
            )
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

    def get_load_offset(self, table_name: str) -> object:
        if self.asc_manager is None:
            raise RuntimeError("ASC manager does not exist.")
        logger.debug("start to get load offset for loading embedding files.")
        return self.asc_manager.send_load_offset(table_name)

    def set_optim_info(self, table_name: str, optim_info):
        if self.asc_manager is None:
            raise RuntimeError("ASC manager does not exist.")
        logger.debug("start to send optimizer info.")
        self.asc_manager.set_optim_info(table_name, optim_info)

    def save_host_data(self, root_dir: Optional[str], save_delta: bool, is_save_l3_storage: bool = True) -> None:
        if self.asc_manager is None:
            raise RuntimeError("ASC manager does not exist.")

        self.asc_manager.save(root_dir, save_delta, is_save_l3_storage)
        logger.debug("Data from host pipeline has been saved.")

    def restore_host_data(self, root_dir: Optional[str], warm_start_tables=None) -> None:
        if self.asc_manager is None:
            raise RuntimeError("ASC manager does not exist.")
        if not warm_start_tables:
            warm_start_tables = []
        if not self.asc_manager.load(root_dir, warm_start_tables):
            raise TypeError(
                "load data does not match usr setups, please re-consider if you want to restore from this dir"
            )
        logger.debug("Data from host pipeline has been restored.")

    def fetch_device_emb(self):
        if self.asc_manager is None:
            raise RuntimeError("ASC manager not exist.")
        self.asc_manager.fetch_device_emb()
        logger.debug("request of fetching embedding from device to host for saving has been send")

    def start_sync_thread(self):
        if self.asc_manager is None:
            raise RuntimeError("ASC manager not exist.")
        self.asc_manager.start_sync_thread()
        logger.debug("Finish calling start_sync_thread.")
