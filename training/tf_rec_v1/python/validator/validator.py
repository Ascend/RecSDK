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
from typing import List, Tuple, Any, Callable, Dict, Optional, Union, Type
import re

import os
import inspect
import functools
import stat

import tensorflow as tf

from rec_sdk_common.validator.validator import Validator, IntValidator, FloatValidator, StringValidator
from rec_sdk_common.constants.constants import FileParams
from rec_sdk_common.log.log import LoggingProxy


class TensorShapeValidator(Validator):
    def __init__(self, name: str, value: tf.TensorShape, int_checker_args: dict = None, msg: str = ""):
        super().__init__(name, value)
        self.int_checker_args = int_checker_args if int_checker_args else {}
        self.msg = msg
        self.register()

    def register(self):
        def check_tensor_shape():
            if isinstance(self.value, tf.TensorShape) and self.value.ndims == 1:
                value = self.value.as_list()[0]
            else:
                return False

            int_checker = IntValidator(
                name=self.name,
                value=value,
                **self.int_checker_args,
            )
            int_checker.check_value().check()
            return True

        self.register_checker(check_tensor_shape,
                              self.msg if self.msg else f"type of '{self.name}' is not TensorShape or ndims is not 1")


# Ensure that the passed parameter is a constant tensor or a tf.PlaceHolder that feeds a constant value. Otherwise,
# an exception may occur.
class LearningRateValidator(FloatValidator):
    def __init__(self, name: str, value: Union[tf.Tensor, float], min_value: float, max_value: float):
        if isinstance(value, tf.Tensor):
            sess = tf.Session() if tf.__version__.startswith("1.") else tf.compat.v1.Session()
            try:
                value = sess.run(value).item()
            except Exception as e:
                # 当前仅支持数值类型Tensor和feed数值类型的tf.PlaceHolder，其它tensor可能会导致程序异常
                LoggingProxy.warning("[Validator] Parameter %s is passed, and an exception occurred while getting the value "
                                     "in the tensor: \n%s\n. Ensure that the passed parameter is a constant tensor or "
                                     "a tf.PlaceHolder that feeds a constant value. Otherwise, an exception may occur.",
                                     value, e)
                value = 0.0 if min_value is None else float(min_value)

        super().__init__(name, value, min_value=min_value, max_value=max_value)


class FileValidator(StringValidator):
    """
    Check if file is valid.
    """

    def __init__(self, name, value):
        """
        @param value: the file path, should not be emtpy string, should not contain double dot(../)
        """
        super(FileValidator, self).__init__(name, value)
        self.register_checker(lambda: isinstance(self.value, str), "parameter value's type is not str")

    def check_file_size(self, max_size=FileParams.MAX_SIZE.value, min_size=FileParams.MIN_SIZE.value):
        file_stat = tf.io.gfile.stat(self.value)
        self.register_checker(lambda: min_size < file_stat.length <= max_size,
                              f"file size: {file_stat.length} is invalid, not in ({min_size}, {max_size}]")
        return self

    def check_not_soft_link(self):
        self.register_checker(lambda: os.path.abspath(self.value) == os.path.realpath(self.value),
                              f"soft link or relative path: {self.value} should not be in the path parameter")
        return self

    def check_user_group(self):
        process_uid = os.geteuid()
        process_gid = os.getegid()
        stat_info = os.stat(self.value)
        file_uid = stat_info.st_uid
        file_gid = stat_info.st_gid
        self.register_checker(lambda: process_uid == file_uid or process_gid == file_gid,
                              "Invalid log file user or group.")
        return self

    def check_file_mode(self, unsupported_mode=0o022):
        stat_info = os.stat(self.value)
        mode = stat.S_IMODE(stat_info.st_mode)
        self.register_checker(lambda: mode & unsupported_mode == 0,
                              f"Current file mode {oct(mode)} is unsupported")
        return self