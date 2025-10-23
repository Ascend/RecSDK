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

import pytest

from rec_sdk_common.log.log import LoggingProxy
from rec_sdk_common.constants.constants import LogLevel


class TestLoggingProxy:
    """Test for 'mxrec.python.util.log.LoggingProxy'."""

    @staticmethod
    def teardown_method():
        LoggingProxy._instance = None

    @staticmethod
    def test_twice_init_err():
        with pytest.raises(RuntimeError) as excinfo:
            LoggingProxy.set_instance(LogLevel.INFO.value)
            LoggingProxy.set_instance(LogLevel.INFO.value)
        assert "LoggingProxy has been initialized once, twice initialization was forbidden" in str(excinfo.value)

    @staticmethod
    def test_info_log_ok():
        LoggingProxy.set_instance(LogLevel.INFO.value)
        try:
            LoggingProxy.info("Test log, we have a %s, log level is %s.", "interesting problem", "info")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_debug_log_ok():
        LoggingProxy.set_instance(LogLevel.DEBUG.value)
        try:
            LoggingProxy.debug("Test log, we have a %s, log level is %s.", "thorny problem", "debug")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_warning_log_ok():
        LoggingProxy.set_instance(LogLevel.WARNING.value)
        try:
            LoggingProxy.warning("Test log, we have a %s, log level is %s.", "bit of a problem", "warning")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_error_log_ok():
        LoggingProxy.set_instance(LogLevel.ERROR.value)
        try:
            LoggingProxy.error("Test log, we have a %s, log level is %s.", "major problem", "error")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_exception_log_ok():
        LoggingProxy.set_instance(LogLevel.ERROR.value)
        try:
            LoggingProxy.exception("Test log, we have a %s, log level is %s.", "exception problem", "exception")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")

    @staticmethod
    def test_critical_log_ok():
        LoggingProxy.set_instance(LogLevel.CRITICAL.value)
        try:
            LoggingProxy.critical("Test log, we have a %s, log level is %s.", "major disaster", "critical")
        except Exception as e:
            pytest.fail(f"unexpected exception raised: {e}")
