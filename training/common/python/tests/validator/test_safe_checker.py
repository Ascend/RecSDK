#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
import shutil
import pytest

from rec_sdk_common.validator.safe_checker import (
    str_safe_check,
    int_safe_check,
    class_safe_check,
    dir_safe_check,
    float_safe_check,
)
from rec_sdk_common.constants.constants import NumCheckValueMethod
from rec_sdk_common.communication.hccl import get_rank_id


class TestStrSafeCheck:
    """Test for 'mxrec.python.util.validator.safe_checker.str_safe_check'."""

    @staticmethod
    def test_ok():
        try:
            str_safe_check("table_name", "user_table", min_len=0, max_len=255)
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_type_err():
        with pytest.raises(ValueError) as excinfo:
            str_safe_check("table_name", 1)
        assert "is not str" in str(excinfo.value)

    @staticmethod
    def test_len_less_err():
        with pytest.raises(ValueError) as excinfo:
            str_safe_check("table_name", "user_table", min_len=20)
        assert "'table_name' length is less than" in str(excinfo.value)

    @staticmethod
    def test_len_greater_err():
        with pytest.raises(ValueError) as excinfo:
            str_safe_check("table_name", "user_table", max_len=5)
        assert "'table_name' length is bigger than" in str(excinfo.value)

    @staticmethod
    def test_whitelist_err():
        with pytest.raises(ValueError) as excinfo:
            str_safe_check("table_name", "user_table*")
        assert "The string 'table_name' is invalid" in str(excinfo.value)

    @staticmethod
    def test_black_element_err():
        with pytest.raises(ValueError) as excinfo:
            str_safe_check("table_name", "user_table", black_element="user")
        assert "contain black element" in str(excinfo.value)


class TestIntSafeCheck:
    """Test for 'mxrec.python.util.validator.safe_checker.int_safe_check'."""

    @staticmethod
    def test_ok():
        try:
            int_safe_check("embedding_dim", 8, min_value=0, max_value=8192)
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_type_err():
        with pytest.raises(ValueError) as excinfo:
            int_safe_check("embedding_dim", "8")
        assert "is not int" in str(excinfo.value)

    @staticmethod
    def test_max_value_err():
        with pytest.raises(ValueError) as excinfo:
            int_safe_check("embedding_dim", 8, max_value=4)
        assert "is bigger than" in str(excinfo.value)

    @staticmethod
    def test_min_value_err():
        with pytest.raises(ValueError) as excinfo:
            int_safe_check("embedding_dim", 8, min_value=16)
        assert "is less than" in str(excinfo.value)


class TestClassSafeCheck:
    """Test for 'mxrec.python.util.validator.safe_checker.class_safe_check'."""

    @staticmethod
    def test_ok():
        try:
            class_safe_check("embedding_dim", 8, int)
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_type_err():
        with pytest.raises(ValueError) as excinfo:
            class_safe_check("embedding_dim", 8, str)
        assert "not in <class 'str'>" in str(excinfo.value)


class TestDirSafeCheck:
    """Test for 'mxrec.python.util.validator.safe_checker.dir_safe_check'."""

    @staticmethod
    def test_ok():
        path = "test_path"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        try:
            dir_safe_check("path", path)
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")
        shutil.rmtree(path)


class TestFloatSafeCheck:
    """Test for 'mxrec.python.util.validator.safe_checker.float_safe_check'."""

    @staticmethod
    def test_default_method():
        try:
            float_safe_check(
                "embedding_dim",
                8.0,
                min_value=0.0,
                max_value=8.0,
                method=NumCheckValueMethod.DEFAULT.value,
            )
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_open_interval_method():
        with pytest.raises(ValueError) as excinfo:
            float_safe_check(
                "embedding_dim",
                8.0,
                min_value=0.0,
                max_value=8.0,
                method=NumCheckValueMethod.OPEN_INTERVAL.value,
            )
        assert "is bigger than or equal" in str(excinfo.value)

    @staticmethod
    def test_left_open_interval_method():
        with pytest.raises(ValueError) as excinfo:
            float_safe_check(
                "embedding_dim",
                0.0,
                min_value=0.0,
                max_value=8.0,
                method=NumCheckValueMethod.LEFT_OPEN_INTERVAL.value,
            )
        assert "is less than or equal" in str(excinfo.value)

    @staticmethod
    def test_right_open_interval_method():
        with pytest.raises(ValueError) as excinfo:
            float_safe_check(
                "embedding_dim",
                8.0,
                min_value=0.0,
                max_value=8.0,
                method=NumCheckValueMethod.RIGHT_OPEN_INTERVAL.value,
            )
        assert "is bigger than or equal" in str(excinfo.value)

    @staticmethod
    def test_method_not_support():
        with pytest.raises(ValueError) as excinfo:
            float_safe_check(
                "embedding_dim",
                8.0,
                min_value=0.0,
                max_value=8.0,
                method="xxx",
            )
        assert "the check method supports" in str(excinfo.value)
