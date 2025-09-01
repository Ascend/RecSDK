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

from typing import Optional, Any, Union, Tuple

from rec_sdk_common.constants.constants import ValidatorParams, NumCheckValueMethod
from rec_sdk_common.validator.validator import (
    StringValidator,
    IntValidator,
    FloatValidator,
    ClassValidator,
    DirectoryValidator,
)
from rec_sdk_common.log.log import LoggingProxy


def str_safe_check(
        name: str,
        value: str,
        min_len: int = ValidatorParams.STR_MIN_LENGTH.value,
        max_len: int = ValidatorParams.STR_MAX_LENGTH.value,
        black_element: Optional[str] = None,
):
    if black_element is None:
        LoggingProxy.warning("str_safe_check has black_element == None")
    validator = StringValidator(name, value, max_len, min_len, black_element)
    validator.check_whitelist().check_not_contain_black_element().check_string_length().check()


def int_safe_check(
        name: str,
        value: int,
        min_value: int = ValidatorParams.MIN_INT32.value,
        max_value: int = ValidatorParams.MAX_INT32.value
):
    validator = IntValidator(name, value, min_value, max_value)
    validator.check_value().check()


def float_safe_check(
        name: str,
        value: float,
        min_value: float = ValidatorParams.MIN_FLOAT32.value,
        max_value: float = ValidatorParams.MAX_FLOAT32.value,
        method: str = NumCheckValueMethod.DEFAULT.value,
):
    validator = FloatValidator(name, value, min_value, max_value)
    if method == NumCheckValueMethod.DEFAULT.value:
        validator.check_value().check()
    elif method == NumCheckValueMethod.OPEN_INTERVAL.value:
        validator.check_value_for_open_interval().check()
    elif method == NumCheckValueMethod.LEFT_OPEN_INTERVAL.value:
        validator.check_value_for_left_open_interval().check()
    elif method == NumCheckValueMethod.RIGHT_OPEN_INTERVAL.value:
        validator.check_value_for_right_open_interval().check()
    else:
        support_method = {n.value for n in NumCheckValueMethod}
        raise ValueError(f"the check method supports {support_method}, but got {method}")


def class_safe_check(name: str, value: Any, classes: Union[Any, Tuple[Any]]):
    validator = ClassValidator(name, value, classes)
    validator.check()


def dir_safe_check(name: str, directory: str) -> None:
    """
    Directory safety check
    :param name: a string which represents the parameter
    :param directory: the directory which need to check
    :return: None
    """
    validator = DirectoryValidator(name, directory, max_len=ValidatorParams.MAX_FILE_PATH_LENGTH.value)
    validator.check_is_not_none().path_should_exist(
        is_file=False
    ).check_not_soft_link().should_not_contains_sensitive_words().check_string_length().check()
