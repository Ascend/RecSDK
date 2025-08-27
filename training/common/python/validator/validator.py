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
from typing import List, Tuple, Any, Callable, Dict, Optional, Union, Type
import re

import os
import inspect
import functools
import stat

import tensorflow as tf

from rec_sdk_common.constants.constants import EnvOptionCommon, LogLevel, FileParams, RankTableParams, ValidatorParams
from rec_sdk_common.log.log import LoggingProxy


class Validator:
    """
    A validator to check the input parameters
    """

    def __init__(self, name: Union[List[str], str], value: Union[List[Any], Any], msg="value is invalid"):
        """
        :param msg: default error msg
        """
        self.name = name
        self.value = value
        self.msg = msg
        self.checkers = []
        self.is_valid_state = None

    def register_checker(self, checker: Callable[[], bool], msg: str = None):
        self.checkers.append((checker, msg if msg else self.msg))

    def check(self):
        if self.is_valid_state is None:
            self.is_valid_state = True
        for checker, msg in self.checkers:
            if not checker():
                self.msg = msg
                raise ValueError(self.msg)
        if self.is_valid_state:
            self.msg = None
        return self

    def is_valid(self):
        if self.is_valid_state is None:
            self.check()
        return self.is_valid_state


def build_validator(option: Tuple[Union[str], Type[Validator], Optional[Dict], Optional[List[str]]], value):
    optional_check_list = None
    validator_kwargs = {}

    # 解包每个检查项的：待检查参数名，检查器，检查器参数，特定的检查方法
    option_num = len(option)
    if option_num == 2:
        para_list_to_be_check, validator = option
    elif option_num == 3:
        para_list_to_be_check, validator, validator_kwargs = option
    else:
        para_list_to_be_check, validator, validator_kwargs, optional_check_list = option

    # 更新检查器的参数
    validator_kwargs.update(
        {
            "name": para_list_to_be_check,
            "value": value
        }
    )

    validator_instance = validator(**validator_kwargs)

    # 添加检查器特定的检查方法
    if optional_check_list and len(optional_check_list) != 0:
        for optional_check in optional_check_list:
            getattr(validator_instance, optional_check)()

    # 执行检查
    return validator_instance


def para_checker_decorator(check_option_list: List[Tuple[Union[List[str], str],
                                                         Type[Validator],
                                                         Optional[Dict],
                                                         Optional[List[str]]]], output_log=True):
    """
    函数参数校验装饰器
    :param output_log: 是否打印日志
    :param check_option_list:
    需要校验的参数及其相关校验器[“需要检验的参数或参数组合”, "使用的校验器", "校验器的参数", "校验器需要执行的方法（添加指定校验）"]
    :return:
    """
    def para_checker(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signature = inspect.signature(func)
            # 获取实际传入的参数，包括默认值参数
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            actual_args = dict(bound_args.arguments)

            temp_kwargs = dict()
            if "kwargs" in actual_args:
                temp_kwargs = actual_args["kwargs"]
                del actual_args["kwargs"]
            actual_args.update(temp_kwargs)

            func_spec = inspect.getfullargspec(func)
            # 将函数有默认值的参数加入kwargs
            args_with_default = set()
            if func_spec.defaults is not None:
                arg_with_default_num = len(func_spec.defaults)
                for arg, default in zip(func_spec.args[-arg_with_default_num:], func_spec.defaults):
                    if arg in kwargs:
                        continue
                    args_with_default.add(arg)
                    kwargs.update({arg: default})
            if output_log:
                LoggingProxy.debug("[checker wrapper]func %s kwargs: %s", func.__name__, actual_args)
            # 执行每一个检查项
            for option in check_option_list:
                _handle_single_check_option(option, actual_args, func.__name__)

            for arg in args_with_default:
                del kwargs[arg]
            return func(*args, **kwargs)

        return wrapper

    def _handle_single_check_option(option, actual_args, func_name):
        """
        处理单个检查项，完全复刻原逻辑。
        """
        option_num = len(option)
        optional_check_list = None
        validator_kwargs = {}

        if option_num == 2:
            para_list_to_be_check, validator = option
        elif option_num == 3:
            para_list_to_be_check, validator, validator_kwargs = option
        else:
            para_list_to_be_check, validator, validator_kwargs, optional_check_list = option

        if not isinstance(para_list_to_be_check, list):
            para_list_to_be_check = [para_list_to_be_check]

        # 确认当前检查项需要检查的参数是否在函数参数中
        paras = []
        for para_to_be_check in para_list_to_be_check:
            if para_to_be_check not in actual_args:
                LoggingProxy.debug("[checker wrapper]invalid para '%s' to be checked, "
                            "not passed to the function '%s'", para_to_be_check, func_name)
                continue
            paras.append(actual_args.get(para_to_be_check))

        # 如果检查的参数不在传参中，跳过该检查项
        if not paras:
            return

        # 更新检查器的参数
        validator_kwargs.update(
            {
                "name": para_list_to_be_check[0] if len(para_list_to_be_check) == 1 else para_list_to_be_check,
                "value": paras[0] if len(paras) == 1 else paras
            }
        )

        validator_instance = validator(**validator_kwargs)

        # 添加检查器特定的检查方法
        if optional_check_list and len(optional_check_list) != 0:
            for optional_check in optional_check_list:
                getattr(validator_instance, optional_check)()

        # 执行检查
        validator_instance.check()

    return para_checker


class ClassValidator(Validator):
    """
    Check class validator.
    """

    def __init__(self, name, value, classes):
        super(ClassValidator, self).__init__(name, value)
        self.classes = classes
        self.register()

    def register(self):
        """Check arg isinstance of classes"""
        self.register_checker(lambda: isinstance(self.value, self.classes),
                              f"Invalid parameter type of para '{self.name}', "
                              f"not in {self.classes}, but: '{type(self.value)}'")
        return self


class ListValidator(Validator):
    def __init__(self, name, value: Union[List, Tuple], sub_checker: type, optional_check_list: List = None,
                 list_max_length: int = ValidatorParams.MAX_INT32.value, list_min_length: int = 1, sub_args: dict = None):
        super(ListValidator, self).__init__(name, value)
        if sub_args is None:
            sub_args = {}
        self.checker = sub_checker
        self.optional_check_list = optional_check_list
        self.sub_args = sub_args if sub_args else {}
        self.list_min_length = list_min_length
        self.list_max_length = list_max_length
        self.register()

    def register(self):
        def check_func():
            if not isinstance(self.value, (List, Tuple)):
                return False

            if not issubclass(self.checker, Validator):
                return False

            for elem in self.value:
                checker = self.checker("element of " + self.name, elem, **self.sub_args)
                if self.optional_check_list and len(self.optional_check_list) != 0:
                    for optional_check in self.optional_check_list:
                        getattr(checker, optional_check)()
                checker.check()

            return True

        self.register_checker(check_func, f"Invalid List '{self.name}'")

    def check_list_length(self):
        self.register_checker(lambda: self.list_min_length <= len(self.value) <= self.list_max_length,
                              f"Invalid length of list '{self.name}', "
                              f"should between '{self.list_min_length}' and '{self.list_max_length}'")
        return self


class OrValidator(Validator):
    def __init__(self, name: str, value: any, options):
        super().__init__(name, value)
        self.options = options
        self.register()

    def register(self):
        def or_check():
            validators = []
            for option in self.options:
                option = (self.name, *option)
                validators.append(build_validator(option, self.value))

            passed = False
            wrong_msg = []
            for validator in validators:
                try:
                    validator.check()
                except ValueError as exp:
                    wrong_msg.append(str(exp))
                    continue
                else:
                    passed = True
                    break

            if not passed:
                raise ValueError(f"Or validator of '{self.name}' check failed, due to {wrong_msg}")

            return True


        self.register_checker(or_check)


class AndValidator(Validator):
    def __init__(self, name: str, value: any, options):
        super().__init__(name, value)
        self.options = options
        self.register()

    def register(self):
        def and_check():
            validators = []
            for option in self.options:
                option = (self.name, *option)
                validators.append(build_validator(option, self.value))

            for validator in validators:
                try:
                    validator.check()
                except ValueError as exp:
                    exp.args = (f"And validator of '{self.name}' check failed, due to {str(exp)}", )
                    raise

            return True

        self.register_checker(and_check)


class OptionValidator(Validator):
    """
    Check class validator.
    """

    def __init__(self, name, value, options):
        super(OptionValidator, self).__init__(name, value)
        self.options = options
        self.register()

    def register(self):
        """Check arg isinstance of classes"""
        self.register_checker(lambda: self.value in self.options,
                              f"Invalid option of '{self.name}', "
                              f"should be one of '{self.options}', but: '{self.value}'")
        return self


class ValueCompareValidator(Validator):
    """
    Check value validator. Whether value equals to target value.
    """
    def __init__(self, name: Union[List[str], str], value: Union[List[Any], Any], target: Any):
        super(ValueCompareValidator, self).__init__(name, value)
        self.name = name if isinstance(name, list) else [name]
        self.value = value if isinstance(value, list) else [value]
        self.target = target

    def check_at_least_one_not_equal_to_target(self):
        """
        至少一个值不为目标值
        Returns:

        """
        self.register_checker(lambda: not all([v == self.target for v in self.value]),
                              f"at least one of '{','.join(self.name)}' should not be equal to {self.target}")
        return self

    def check_at_least_one_equal_to_target(self):
        """
        至少一个值为目标值
        Returns:

        """
        self.register_checker(lambda: any([v == self.target for v in self.value]),
                              f"at least one of '{','.join(self.name)}' should be equal to {self.target}")
        return self

    def check_all_not_equal_to_target(self):
        """
        所有值都不为目标值
        Returns:

        """
        self.register_checker(lambda: all([v != self.target for v in self.value]),
                              f" all of '{','.join(self.name)}' should not be equal to {self.target}")
        return self


class StringValidator(Validator):
    """
    String type validator.
    """

    def __init__(self, name, value, max_len: Optional[int] = None, min_len: Optional[int] = 0,
                 element: Optional[str] = None, msg=""):
        super(StringValidator, self).__init__(name, value)
        self.max_len = max_len
        self.min_len = min_len
        self.whitelist = "^[0-9A-Za-z_.]+$"
        self.element = element
        msg = msg if msg else f"type of '{name}' is not str, '{value}' is '{type(value)}'"
        self.register_checker(lambda: isinstance(value, str), msg)

    def check_string_length(self):
        if self.min_len is not None:
            self.register_checker(lambda: len(self.value) >= self.min_len,
                                  f"'{self.name}' length is less than {self.min_len}")
        if self.max_len is not None:
            self.register_checker(lambda: len(self.value) <= self.max_len,
                                  f"'{self.name}' length is bigger than {self.max_len}")
        return self

    def check_not_contain_black_element(self):
        if self.value is not None and self.element is not None and self.element != "":
            self.register_checker(lambda: self.value.find(self.element) == -1,
                                  f"'{self.name}' contain black element '{self.element}'")
        return self

    def check_whitelist(self):
        """Perform whitelist verification on the input string"""
        self.register_checker(lambda: self.value is not None and re.match(self.whitelist, self.value) is not None,
                              f"The string '{self.name}' is invalid, please check the input string. "
                              "Note: It should be a string consisting of numbers, letters, and underscores.")
        return self

    def can_be_transformed2int(self, min_value: int = None, max_value: int = None):
        if min_value is None:
            min_value = RankTableParams.MIN_RANK_SIZE.value
        if max_value is None:
            max_value = RankTableParams.MAX_RANK_SIZE.value

        can_transformed = self.value.isdigit()
        try:
            if can_transformed and (min_value > int(self.value) or max_value < int(self.value)):
                can_transformed = False
        except ValueError:
            can_transformed = False
        finally:
            if self.is_valid_state is not None:
                self.is_valid_state &= can_transformed
            else:
                self.is_valid_state = can_transformed
        return self


class OptionalStringValidator(StringValidator):
    """
    String type validator if value is not None
    """

    def __init__(self, name, value, max_len=None, min_len=0, element: Optional[str] = None, msg=""):
        if not isinstance(value, str):
            super(OptionalStringValidator, self).__init__(name, "", None, None, None, msg)
        elif isinstance(value, str):
            super(OptionalStringValidator, self).__init__(name, value, max_len, min_len, element, msg)


class SSDFeatureValidator(Validator):
    """
    Check SSD related parameters
    """

    def __init__(self, name, value):
        super(SSDFeatureValidator, self).__init__(name, value)
        self.register()

    def register(self):
        """Check ssd related parameters"""
        s_size, ssd_data_path, h_size = self.value
        self.register_checker(lambda: isinstance(s_size, int),
                              f"'{self.name[0]}', not int, but '{type(s_size)}'")
        self.register_checker(lambda: isinstance(h_size, int),
                              f"'{self.name[2]}', not int, but '{type(h_size)}'")

        if s_size != 0:
            self.register_checker(lambda: not (h_size == 0 and s_size > 0),
                                  f"'{self.name[2]}' should be greater than 0 when enabling ssd feature")

            self.register_checker(lambda: not (h_size != 0 and s_size < 0),
                                  f"'{self.name[0]}' should be greater than 0 when enabling ssd feature")

            self.register_checker(lambda: isinstance(ssd_data_path, (list, tuple)) and len(ssd_data_path) != 0,
                                  f"'{self.name[1]}' should be type of list and not empty")

            self.register_checker(lambda: len([p for p in ssd_data_path if self._is_invalid_path(p)]) == 0,
                                  f"'{self.name[1]}' contains invalid path")

        return self

    def _is_invalid_path(self, path: str):
        path_exists = os.path.exists(path)
        path_is_dir = os.path.isdir(path)
        path_contains_softlink = os.path.abspath(path) != os.path.realpath(path)
        return not path_exists or not path_is_dir or path_contains_softlink or ".." in path


class NumValidator(Validator):
    """
    number validator float or int
    """

    def __init__(self, name: str, value: Union[int, float], min_value: Union[int, float] = None,
                 max_value: Union[int, float] = None, invalid_options: List = None,
                 constrained_options: List = None, msg: str = ""):
        super(NumValidator, self).__init__(name, value)

        self.min_value = min_value
        self.max_value = max_value
        self.invalid_options = invalid_options
        self.constrained_options = constrained_options

    def check_value(self):
        if self.min_value is not None:
            self.register_checker(lambda: self.value >= self.min_value, f"'{self.name}' is less than {self.min_value}")
        if self.max_value is not None:
            self.register_checker(lambda: self.value <= self.max_value,
                                  f"'{self.name}' is bigger than {self.max_value}")
        if self.invalid_options is not None:
            self.register_checker(lambda: self.value not in self.invalid_options,
                                  f"'{self.name}' is invalid,  num in '{self.invalid_options}' is forbidden")

        if self.constrained_options is not None:
            self.register_checker(lambda: self.value in self.constrained_options,
                                  f"'{self.name}' is invalid,  only num in '{self.constrained_options}' is allowed")

        return self

    def check_value_for_open_interval(self):
        if self.min_value is not None:
            self.register_checker(lambda: self.value > self.min_value,
                                  f"'{self.name}' is less than or equal {self.min_value}")
        if self.max_value is not None:
            self.register_checker(lambda: self.value < self.max_value,
                                  f"'{self.name}' is bigger than or equal {self.max_value}")
        return self

    def check_value_for_left_open_interval(self):
        if self.min_value is not None:
            self.register_checker(lambda: self.value > self.min_value,
                                  f"'{self.name}' is less than or equal {self.min_value}")
        if self.max_value is not None:
            self.register_checker(lambda: self.value <= self.max_value,
                                  f"'{self.name}' is bigger than {self.max_value}")
        return self

    def check_value_for_right_open_interval(self):
        if self.min_value is not None:
            self.register_checker(lambda: self.value >= self.min_value, f"'{self.name}' is less than {self.min_value}")
        if self.max_value is not None:
            self.register_checker(lambda: self.value < self.max_value,
                                  f"'{self.name}' is bigger than or equal {self.max_value}")
        return self


class FloatValidator(NumValidator):
    """
    float type data validator
    """

    def __init__(self, name: str, value: float, min_value: float = None, max_value: float = None,
                 invalid_options: List = None, constrained_options: List = None, msg: str = ""):
        super(FloatValidator, self).__init__(name, value, min_value, max_value, invalid_options, constrained_options,
                                             msg)
        self.register_checker(lambda: isinstance(self.value, float), msg if msg else f"type of '{name}' is not float")


class IntValidator(NumValidator):
    """
    Int type validator
    """

    def __init__(self, name: str, value: int, min_value: int = None, max_value: int = None,
                 invalid_options: List = None, constrained_options: List = None, msg: str = ""):
        super(IntValidator, self).__init__(name, value, min_value, max_value, invalid_options, constrained_options, msg)

        def check_type():
            if isinstance(self.value, bool):
                # bool is subclass of int
                return False
            return isinstance(self.value, int)

        self.register_checker(check_type, msg if msg else f"type of '{name}' is not int")

class OptionalIntValidator(IntValidator):
    """
    Int type validator if value is not None
    """

    def __init__(self, name: str, value: int, min_value: int = None, max_value: int = None,
                 invalid_options: List = None, constrained_options: List = None, msg: str = ""):
        if not isinstance(value, int):
            super(OptionalIntValidator, self).__init__(name, 0, None, None, None, None, msg)
        else:
            super(OptionalIntValidator, self).__init__(name, value, min_value, max_value,
                                                       invalid_options, constrained_options, msg)


class OptionalFloatValidator(FloatValidator):
    """
    Float type validator if value is not None
    """

    def __init__(self, name: str, value: float, min_value: float = None, max_value: float = None,
                 invalid_options: List = None, constrained_options: List = None, msg: str = ""):
        if not isinstance(value, float):
            super(OptionalFloatValidator, self).__init__(name, 0.0, None, None, None, None, msg)
        else:
            super(OptionalFloatValidator, self).__init__(name, value, min_value, max_value,
                                                       invalid_options, constrained_options, msg)


class Convert2intValidator(IntValidator):
    """
    check whether a variable can be converted to int or not.
    """
    def __init__(self, name: str, value: int, min_value: int = None, max_value: int = None,
                 invalid_options: List = None, constrained_options: List = None, msg: str = ""):
        convertable = True
        int_value = None
        try:
            int_value = int(value)
        except TypeError:
            convertable = False
        if convertable:
            super(Convert2intValidator, self).__init__(name, int_value, min_value, max_value, invalid_options,
                                                       constrained_options, msg)
        else:
            super(Convert2intValidator, self).__init__(name,
                                                       value,
                                                       min_value,
                                                       max_value,
                                                       invalid_options,
                                                       constrained_options, f"'{name}' cannot be converted to int")


class DirectoryValidator(StringValidator):
    def __init__(self, name, value, max_len=None, min_len=1):
        """
        @param value: the path, should not be emtpy string, should not contain double dot(../)
        """
        super(DirectoryValidator, self).__init__(name, value, max_len, min_len)
        self.register_checker(lambda: isinstance(value, str), "type is not str")

    @staticmethod
    def remove_prefix(string: Optional[str], prefix: Optional[str]) -> Tuple[bool, Optional[str]]:
        if string is None or prefix is None or len(string) < len(prefix):
            return False, string
        if string.startswith(prefix):
            return True, string[len(prefix):]
        else:
            return False, string

    @staticmethod
    def check_is_children_path(path_: str, target_: str):
        if not target_:
            return False

        try:
            realpath_ = os.path.realpath(path_)
        except (TypeError, ValueError, OSError):
            return False

        try:
            realpath_target = os.path.realpath(target_)
        except (TypeError, ValueError, OSError):
            return False

        is_prefix, rest_part = DirectoryValidator.remove_prefix(realpath_target, realpath_)

        if rest_part.startswith(os.path.sep):
            rest_part = rest_part.lstrip(os.path.sep)
        if is_prefix:
            joint_path = os.path.join(realpath_, rest_part)
            return os.path.realpath(joint_path) == realpath_target
        else:
            return False

    @staticmethod
    def __check_with_sensitive_words(path: str, words: List):
        _, name = os.path.split(path)
        if name:
            return not any(map(lambda x: x in path, words))
        else:
            return True

    def check_is_not_none(self):
        self.register_checker(lambda: self.value is not None and len(self.value) > 0,
                              "Invalid directory parameter")
        return self

    def check_not_soft_link(self):
        self.register_checker(lambda: os.path.abspath(self.value) == os.path.realpath(self.value),
                              f"soft link or relative path: {self.value} should not be in the path parameter")
        return self

    def path_should_exist(self, is_file=True, msg=None):
        self.register_checker(lambda: os.path.exists(self.value),
                              msg if msg else "path parameter does not exist")
        if is_file:
            self.register_checker(lambda: os.path.isfile(self.value),
                                  msg if msg else "path parameter is not a file")
        return self

    def check_exists_if_not_empty(self):
        if self.value:
            self.register_checker(lambda: os.path.exists(os.path.realpath(self.value)), f"'{self.value}' not exists")

    def path_should_not_exist(self):
        self.register_checker(lambda: not os.path.exists(self.value), "path parameter does not exist")
        return self

    def with_blacklist(self, lst: List = None, exact_compare: bool = True, msg: str = None):
        if lst is None:
            lst = ["/usr/bin", "/usr/sbin", "/etc", "/usr/lib", "/usr/lib64", "/usr/local"]
        if len(lst) == 0:
            return self
        if msg is None:
            msg = "path should not in blacklist"
        if exact_compare:
            self.register_checker(lambda: self.value not in [os.path.realpath(each) for each in lst], msg)
        else:
            self.register_checker(
                lambda: not any([DirectoryValidator.check_is_children_path(each, self.value) for each in lst]), msg
            )
        return self

    def should_not_contains_sensitive_words(self, words: List = None, msg=None):
        if words is None:
            words = ["Key", "password", "privatekey"]
        self.register_checker(lambda: DirectoryValidator.__check_with_sensitive_words(self.value, words), msg)
        return self
