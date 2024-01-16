#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

import sys
import unittest
from unittest import mock
from unittest.mock import mock_open, patch

from mx_rec.util.communication.hccl_mgmt import parse_hccl_json
from mx_rec.util.communication.hccl_mgmt import set_hccl_info_without_json
from mx_rec.util.global_env_conf import global_env


class HCCLMGMTTest(unittest.TestCase):
    def setUp(self):
        """
        准备步骤
        :return:无
        """
        self.rank_table_file = global_env.rank_table_file
        global_env.rank_table_file = __file__

    def tearDown(self):
        """
        销毁步骤
        :return: 无
        """
        global_env.rank_table_file = self.rank_table_file

    @patch.multiple("mx_rec.util.communication.hccl_mgmt",
                    get_logic_id=mock.MagicMock(return_value=1))
    def test_parse_hccl_json_when_success(self):
        with patch("builtins.open", mock_open(read_data="""{
              "server_count":"1",
              "server_list":[
                {
                  "device":[
                    { "device_id":"0", "device_ip":"xxx.xxx.xx.xxx", "rank_id":"0" }
                  ],
                  "server_id":"xxx.xxx.xx.xxx"
                }
              ],
              "status":"completed",
              "version":"1.0"
            }""")) as mock_file:
            rank_to_device_dict, local_rank_size = parse_hccl_json()
        self.assertEqual(1, local_rank_size)

    def test_parse_hccl_json_when_attribute_error(self):
        with patch("builtins.open", mock_open(read_data="""{
          "server_count":"1",
          "status":"completed",
          "version":"1.0"
        }""")) as mock_file:
            with self.assertRaises(AttributeError):
                rank_to_device_dict, local_rank_size = parse_hccl_json()

        with patch("builtins.open", mock_open(read_data="""{
          "server_count":"1",
          "server_list":[
            {
              "server_id":"xxx.xxx.xx.xxx"
            }
          ],
          "status":"completed",
          "version":"1.0"
        }""")) as mock_file:
            with self.assertRaises(AttributeError):
                rank_to_device_dict, local_rank_size = parse_hccl_json()

    def test_parse_hccl_json_when_value_error(self):
        with patch("builtins.open", mock_open(read_data="""{
          "server_count":"1",
          "server_list":[],
          "status":"completed",
          "version":"1.0"
        }""")) as mock_file:
            with self.assertRaises(ValueError):
                rank_to_device_dict, local_rank_size = parse_hccl_json()
        with patch("builtins.open", mock_open(read_data="""{
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
        }""")) as mock_file:
            with self.assertRaises(ValueError):
                rank_to_device_dict, local_rank_size = parse_hccl_json()
        with patch("builtins.open", mock_open(read_data="""{
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
        }""")) as mock_file:
            with self.assertRaises(ValueError):
                rank_to_device_dict, local_rank_size = parse_hccl_json()

    def test_set_hccl_info_without_json(self):
        rank_to_device_dict, local_rank_size = set_hccl_info_without_json("0-7", "8", "0")
        self.assertEqual(8, local_rank_size)
        rank_to_device_dict, local_rank_size = set_hccl_info_without_json("0,1", "2", "0")
        self.assertEqual(2, local_rank_size)
        rank_to_device_dict, local_rank_size = set_hccl_info_without_json("0", "1", "0")
        self.assertEqual(1, local_rank_size)
        with self.assertRaises(ValueError):
            rank_to_device_dict, local_rank_size = set_hccl_info_without_json("0", "8", "0")
        with self.assertRaises(ValueError):
            rank_to_device_dict, local_rank_size = set_hccl_info_without_json("0-2", "8", "3")
        with self.assertRaises(ValueError):
            rank_to_device_dict, local_rank_size = set_hccl_info_without_json("17", "1", "1")


if __name__ == '__main__':
    unittest.main()
