#!/usr/bin/env python3
# coding: UTF-8
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

import unittest
from unittest import mock

from rec_sdk_common.communication.hccl.hccl_info import get_rank_id, get_rank_size, \
    get_local_rank_size, get_device_id, get_min_device_id
from rec_sdk_common.constants.constants import MPIParams, RankTableInfo


class TestGetRankId(unittest.TestCase):
    @mock.patch("os.environ", {MPIParams.OMPI_COMM_WORLD_RANK.value: "0"})
    def test_get_rank_id_ok(self):
        self.assertEqual(get_rank_id(), 0)

    @mock.patch("os.environ", {MPIParams.OMPI_COMM_WORLD_RANK.value: None})
    def test_get_rank_id_none_err(self):
        with self.assertRaises(RuntimeError) as e:
            get_rank_id()

        self.assertIn("Environment variable RANK_ID has not been exported", str(e.exception))

    @mock.patch("os.environ", {MPIParams.OMPI_COMM_WORLD_RANK.value: "xx"})
    def test_get_rank_id_value_err(self):
        with self.assertRaises(ValueError) as e:
            get_rank_id()

        self.assertIn("the value should be number", str(e.exception))


class TestGetRankSize(unittest.TestCase):
    @mock.patch("os.environ", {MPIParams.OMPI_COMM_WORLD_SIZE.value: "1"})
    def test_get_rank_size_ok(self):
        self.assertEqual(get_rank_size(), 1)

    @mock.patch("os.environ", {MPIParams.OMPI_COMM_WORLD_SIZE.value: "0"})
    def test_get_rank_size_zero_err(self):
        with self.assertRaises(ValueError) as e:
            get_rank_size()

        self.assertIn("the value must be greater than or equal to 1", str(e.exception))

    @mock.patch("os.environ", {MPIParams.OMPI_COMM_WORLD_SIZE.value: None})
    def test_get_rank_size_none_err(self):
        with self.assertRaises(RuntimeError) as e:
            get_rank_size()

        self.assertIn("Environment variable RANK_SIZE has not been exported", str(e.exception))

    @mock.patch("os.environ", {MPIParams.OMPI_COMM_WORLD_SIZE.value: "xx"})
    def test_get_rank_size_value_err(self):
        with self.assertRaises(ValueError) as e:
            get_rank_size()

        self.assertIn("the value should be number", str(e.exception))


class TestGetLocalRankSize(unittest.TestCase):
    @mock.patch("os.environ", {MPIParams.OMPI_COMM_WORLD_LOCAL_SIZE.value: "1"})
    def test_get_local_rank_size_ok(self):
        self.assertEqual(get_local_rank_size(), 1)

    @mock.patch("os.environ", {MPIParams.OMPI_COMM_WORLD_LOCAL_SIZE.value: "0"})
    def test_get_local_rank_size_zero_err(self):
        with self.assertRaises(ValueError) as e:
            get_local_rank_size()

        self.assertIn("the value must be greater than or equal to 1", str(e.exception))

    @mock.patch("os.environ", {MPIParams.OMPI_COMM_WORLD_LOCAL_SIZE.value: None})
    def test_get_local_rank_size_none_err(self):
        with self.assertRaises(RuntimeError) as e:
            get_local_rank_size()

        self.assertIn("Environment variable LOCAL_RANK_SIZE has not been exported", str(e.exception))

    @mock.patch("os.environ", {MPIParams.OMPI_COMM_WORLD_LOCAL_SIZE.value: "xx"})
    def test_get_local_rank_size_value_err(self):
        with self.assertRaises(ValueError) as e:
            get_local_rank_size()

        self.assertIn("the value should be number", str(e.exception))


class TestGetDeviceId(unittest.TestCase):
    @mock.patch("os.environ", {MPIParams.OMPI_COMM_WORLD_RANK.value: "0", RankTableInfo.RANK_TABLE_FILE.value: "file"})
    @mock.patch.multiple(
        "rec_sdk_common.communication.hccl.hccl_info",
        _get_rank_info_with_ranktable=mock.MagicMock(return_value={0: 0}),
        get_rank_id=mock.MagicMock(return_value=0),
        get_local_rank_size=mock.MagicMock(return_value=1),
    )
    def test_get_device_id_with_rank_table_ok(self):
        self.assertEqual(get_device_id(), 0)

    @mock.patch("os.environ", {MPIParams.OMPI_COMM_WORLD_RANK.value: "0", RankTableInfo.RANK_TABLE_FILE.value: ""})
    @mock.patch.multiple(
        "rec_sdk_common.communication.hccl.hccl_info",
        _get_rank_info_without_ranktable=mock.MagicMock(return_value={0: 0}),
        get_rank_id=mock.MagicMock(return_value=0),
        get_local_rank_size=mock.MagicMock(return_value=1),
    )
    def test_get_device_id_without_rank_table_ok(self):
        self.assertEqual(get_device_id(), 0)

    @mock.patch("os.environ", {MPIParams.OMPI_COMM_WORLD_RANK.value: "0", RankTableInfo.RANK_TABLE_FILE.value: ""})
    @mock.patch.multiple(
        "rec_sdk_common.communication.hccl.hccl_info",
        _get_rank_info_without_ranktable=mock.MagicMock(return_value={1: 0}),
        get_rank_id=mock.MagicMock(return_value=0),
        get_local_rank_size=mock.MagicMock(return_value=1),
    )
    def test_get_device_id_none_err(self):
        with self.assertRaises(RuntimeError) as e:
            get_device_id()

        self.assertIn("Environment variable DEVICE_ID has not been exported", str(e.exception))

    @mock.patch("os.environ", {MPIParams.OMPI_COMM_WORLD_RANK.value: "0", RankTableInfo.RANK_TABLE_FILE.value: ""})
    @mock.patch.multiple(
        "rec_sdk_common.communication.hccl.hccl_info",
        _get_rank_info_without_ranktable=mock.MagicMock(return_value={0: "xxx"}),
        get_rank_id=mock.MagicMock(return_value=0),
        get_local_rank_size=mock.MagicMock(return_value=1),
    )
    def test_get_device_id_value_err(self):
        with self.assertRaises(ValueError) as e:
            get_device_id()

        self.assertIn("Environment variable DEVICE_ID should be number", str(e.exception))
