#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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

import os
import unittest
from unittest import mock
from unittest.mock import mock_open, patch

import rec_sdk_common.util
from rec_sdk_common.constants.constants import CommonEnv, RankTableInfo
from rec_sdk_common.communication.hccl.hccl_mgmt import _get_rank_info_with_ranktable, _get_rank_info_without_ranktable


class HCCLMGMTTest(unittest.TestCase):
    def setUp(self):
        """
        准备步骤
        :return:无
        """
        self.rank_table_file = os.getenv(RankTableInfo.RANK_TABLE_FILE.value, "")
        os.environ[RankTableInfo.RANK_TABLE_FILE.value] = __file__

    def tearDown(self):
        """
        销毁步骤
        :return: 无
        """
        os.environ[RankTableInfo.RANK_TABLE_FILE.value] = self.rank_table_file

    def test_get_rank_info_with_ranktable_when_attribute_error(self):
        with patch(
            "builtins.open",
            mock_open(
                read_data="""{
          "server_count":"1",
          "status":"completed",
          "version":"1.0"
        }"""
            ),
        ) as mock_file:
            with self.assertRaises(AttributeError):
                rank_to_device_dict, local_rank_size = _get_rank_info_with_ranktable()

        with patch(
            "builtins.open",
            mock_open(
                read_data="""{
          "server_count":"1",
          "server_list":[
            {
              "server_id":"xxx.xxx.xx.xxx"
            }
          ],
          "status":"completed",
          "version":"1.0"
        }"""
            ),
        ) as mock_file:
            with self.assertRaises(AttributeError):
                rank_to_device_dict, local_rank_size = _get_rank_info_with_ranktable()

    def test_get_rank_info_with_ranktable_when_value_error(self):
        with patch(
            "builtins.open",
            mock_open(
                read_data="""{
          "server_count":"1",
          "server_list":[],
          "status":"completed",
          "version":"1.0"
        }"""
            ),
        ) as mock_file:
            with self.assertRaises(ValueError):
                rank_to_device_dict, local_rank_size = _get_rank_info_with_ranktable()
        with patch(
            "builtins.open",
            mock_open(
                read_data="""{
          "server_count":"1",
          "server_list":[
            {
              "device":[
                { "device_id":"0", "device_ip":"xxx.xxx.xx.xxx"}
              ],
              "server_id":"xxx.xxx.xx.xxx"
            }
          ],
          "status":"completed",
          "version":"1.0"
        }"""
            ),
        ) as mock_file:
            with self.assertRaises(ValueError):
                rank_to_device_dict, local_rank_size = _get_rank_info_with_ranktable()
        with patch(
            "builtins.open",
            mock_open(
                read_data="""{
          "server_count":"1",
          "server_list":[
            {
              "device":[
                {"device_ip":"xxx.xxx.xx.xxx", "rank_id":"0" }
              ],
              "server_id":"xxx.xxx.xx.xxx"
            }
          ],
          "status":"completed",
          "version":"1.0"
        }"""
            ),
        ) as mock_file:
            with self.assertRaises(ValueError):
                rank_to_device_dict, local_rank_size = _get_rank_info_with_ranktable()


class TestSetHcclInfoWithoutRanktable(unittest.TestCase):
    @mock.patch("os.environ", {CommonEnv.CM_WORKER_SIZE.value: "1", CommonEnv.CM_CHIEF_DEVICE.value: "0"})
    @mock.patch.multiple(
        "rec_sdk_common.communication.hccl.hccl_mgmt",
        get_device_list=mock.MagicMock(return_value=[1]),
    )
    def test_value_err(self):
        with self.assertRaises(ValueError):
            _get_rank_info_without_ranktable()

    @mock.patch("os.environ", {CommonEnv.CM_WORKER_SIZE.value: "1", CommonEnv.CM_CHIEF_DEVICE.value: "0"})
    @mock.patch.multiple(
        "rec_sdk_common.communication.hccl.hccl_mgmt",
        get_device_list=mock.MagicMock(return_value=[0]),
    )
    def test_ok(self):
        self.assertIsNotNone(_get_rank_info_without_ranktable())


if __name__ == "__main__":
    unittest.main()
