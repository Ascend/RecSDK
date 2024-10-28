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

import os
import sys
import tempfile
import unittest
from unittest import mock

import tensorflow as tf
from tensorflow.python.client.session import BaseSession

from mx_rec.validator.validator import ClassValidator, Convert2intValidator, DirectoryValidator, IntValidator, \
    NumValidator, OptionalIntValidator, OptionalStringValidator, OptionValidator, para_checker_decorator, \
    StringValidator, ValueCompareValidator, FloatValidator, SSDFeatureValidator, ListValidator, LearningRateValidator

sys.modules['mxrec_pybind'] = __import__('os')

os.environ[
    "HOST_PIPELINE_OPS_LIB_PATH"] = f"{os.getenv('so_path')}/libasc/libasc_ops.so"


class ParameterCheckerTest(unittest.TestCase):
    def setUp(self):
        """
        准备步骤
        :return:无
        """
        super().setUp()

    def tearDown(self):
        """
        销毁步骤
        :return: 无
        """
        super().tearDown()

    def test_length_list_validator(self):
        self.assertTrue(ListValidator("val", [123, 456], IntValidator, list_max_length=2).
                        check_list_length().check().is_valid())

        try:
            (ListValidator("val", [123, 456, 789], IntValidator, list_max_length=2).
             check_list_length().check().is_valid())
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

        try:
            (ListValidator("val", [123], IntValidator, list_min_length=2, list_max_length=3).
             check_list_length().check().is_valid())
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

    def test_elem_of_list_validator(self):
        self.assertTrue(ListValidator(
            name="whatever",
            value=[123, 1, 2],
            sub_checker=IntValidator,
            optional_check_list=["check_value"],
            list_max_length=3,
            sub_args={
                "min_value":1,
                "max_value":324
            }).check_list_length().check().is_valid())

        try:
            self.assertTrue(ListValidator(
                name="whatever",
                value=[123, 1, 2],
                sub_checker=IntValidator,
                optional_check_list=["check_value"],
                list_max_length=3,
                sub_args={
                    "min_value": 2,
                    "max_value": 324
                }).check_list_length().check().is_valid())
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

    def test_mutil_layer_list_validator(self):
        self.assertTrue(ListValidator(
            name="whatever",
            value=[[123, 1, 2]],
            sub_checker=ListValidator,
            list_max_length=3,
            optional_check_list=["check_list_length"],
            sub_args={
                "sub_checker": IntValidator,
                "optional_check_list": ["check_value"],
                "list_max_length": 3,
                "sub_args": {
                    "min_value": 1,
                    "max_value": 324
                }
            }).check_list_length().check().is_valid())

        try:
            self.assertTrue(ListValidator(
                name="whatever",
                value=[[123, 1, 2]],
                sub_checker=ListValidator,
                list_max_length=3,
                optional_check_list=["check_list_length"],
                sub_args={
                    "sub_checker": IntValidator,
                    "optional_check_list": ["check_value"],
                    "list_max_length": 3,
                    "sub_args": {
                        "min_value": 2,
                        "max_value": 324
                    }
                }).check_list_length().check().is_valid())
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

    def test_learning_rate_validator(self):
        if hasattr(BaseSession, "old_run_method"):
            BaseSession.run = BaseSession.old_run_method

        self.assertTrue(LearningRateValidator(
            name="whatever",
            value=tf.constant([1.0]),
            min_value=0.0,
            max_value=10.0
            ).check_value().check().is_valid())

        try:
            self.assertTrue(LearningRateValidator(
                name="whatever",
                value=tf.constant([11.0]),
                min_value=0.0,
                max_value=10.0
            ).check_value().check().is_valid())

        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

        try:
            self.assertTrue(LearningRateValidator(
                name="whatever",
                value=tf.constant([1.0, 2.0]),
                min_value=0.0,
                max_value=10.0
            ).check_value_for_left_open_interval().check().is_valid())

        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

    def test_string_validator_max_len_parameter(self):
        try:
            StringValidator("val", 'aa.1245', max_len=3).check_string_length().check().is_valid()
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

        self.assertTrue(StringValidator("val", 'aa.1245', max_len=30).check().is_valid())
        # default infinity
        self.assertTrue(StringValidator("val", 'aa.124512132456').check().is_valid())

    def test_string_validator_min_len_parameter(self):
        try:
            StringValidator("val", 'aa', min_len=3).check_string_length().check().is_valid()
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

        self.assertTrue(StringValidator("val", 'aa.', min_len=3).check().is_valid())
        # default 0
        self.assertTrue(StringValidator("val", '1').check().is_valid())

    def test_string_validator_can_be_transformed2int(self):
        self.assertFalse(StringValidator("val", '9' * 20).can_be_transformed2int().check().is_valid())
        self.assertFalse(StringValidator("val", '1,2').can_be_transformed2int().check().is_valid())
        self.assertTrue(StringValidator("val", '12').can_be_transformed2int().check().is_valid())
        self.assertFalse(
            StringValidator("val", '12').can_be_transformed2int(min_value=100, max_value=200).check().is_valid())

    def test_directory_black_list(self):
        try:
            DirectoryValidator("val", '/abc/d/e').with_blacklist(lst=['/abc/d/e']).check().is_valid()
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

        self.assertTrue(DirectoryValidator("val", '/abc/d/e').with_blacklist(['/abc/d/']).check().is_valid())
        self.assertTrue(DirectoryValidator("val", '/abc/d/e').with_blacklist(['/abc/d/'], exact_compare=True).check()
                        .is_valid())
        # if not exact compare, the /abc/d/e is children path of /abc/d/, so it is invalid
        try:
            self.assertFalse(DirectoryValidator("val", '/abc/d/e').with_blacklist(['/abc/d/'], exact_compare=False)
                             .check().is_valid())
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")
        self.assertTrue(DirectoryValidator("val", '/usr/bin/bash').with_blacklist().check().is_valid())

        try:
            DirectoryValidator("val", '/usr/bin/bash').with_blacklist(exact_compare=False).check().is_valid()
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

    def test_remove_prefix(self):
        self.assertEqual(DirectoryValidator.remove_prefix('/usr/bin', None)[1], '/usr/bin')
        self.assertEqual(DirectoryValidator.remove_prefix('/usr/bin', '')[1], '/usr/bin')
        self.assertIsNone(DirectoryValidator.remove_prefix(None, 'abc')[1])
        self.assertEqual(DirectoryValidator.remove_prefix('/usr/bin/python', '/usr/bin')[1], "/python")

    def test_directory_white_list(self):
        self.assertTrue(DirectoryValidator.check_is_children_path('/abc/d', '/abc/d/e'))
        self.assertTrue(DirectoryValidator.check_is_children_path('/abc/d', '/abc/d/'))
        self.assertFalse(DirectoryValidator.check_is_children_path('/abc/d', '/abc/de'))
        self.assertTrue(DirectoryValidator.check_is_children_path('/usr/bin/', '/usr/bin/bash'))

    def test_directory_soft_link(self):
        tmp = tempfile.NamedTemporaryFile(delete=True)
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, 'link.ink')
        # make a soft link
        os.symlink(tmp.name, path)

        try:
            # do stuff with temp
            tmp.write(b'stuff')
            DirectoryValidator("val", path).check_not_soft_link().check().is_valid()
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")
        finally:
            tmp.close()  # close means remove
            os.remove(path)
            os.removedirs(temp_dir)

    def test_decorator(self):
        @para_checker_decorator(check_option_list=[
            ("class_arg", ClassValidator, {"classes": (bool,)}),
            ("options_arg", OptionValidator, {"options": (1, 2, 3)}),
            (["options_arg", "int_arg"], ValueCompareValidator, {"target": -1}, ["check_all_not_equal_to_target"]),
            ("string_arg", OptionalStringValidator, {"max_len": 255}, ["check_string_length"]),
            ("int_arg", IntValidator, {"min_value": 1, "max_value": 100}, ["check_value"]),
            ("int_arg", OptionalIntValidator, {"min_value": 1, "max_value": 100}, ["check_value"]),
            ("int_arg", NumValidator, {"min_value": 1, "max_value": 100}, ["check_value"]),
            ("string_arg", Convert2intValidator, {"min_value": 1, "max_value": 100}, ["check_value"]),
        ])
        def demo_func(class_arg: bool, options_arg: int, string_arg: str, int_arg: int):
            return True

        try:
            result = demo_func(class_arg=True, options_arg=1, string_arg="72", int_arg=10)
        except ValueError:
            result = False
        self.assertTrue(result)

    def test_ssd_feature_validator_when_size_0(self):
        @para_checker_decorator(check_option_list=[
            ("name", OptionalStringValidator, {"min_len": 1, "max_len": 255},
             ["check_string_length", "check_whitelist"]),
            (["ssd_vocabulary_size", "ssd_data_path", "host_vocabulary_size"], SSDFeatureValidator)])
        def demo_func(name, host_vocabulary_size=0,
                      ssd_vocabulary_size=0,
                      ssd_data_path="./"):
            return True

        try:
            result = demo_func(name="host", host_vocabulary_size=0,
                               ssd_vocabulary_size=0,
                               ssd_data_path="./")
        except ValueError:
            result = False
        self.assertTrue(result)

    def test_ssd_feature_validator_when_size_not_0(self):
        @para_checker_decorator(check_option_list=[
            ("name", OptionalStringValidator, {"min_len": 1, "max_len": 255},
             ["check_string_length", "check_whitelist"]),
            (["ssd_vocabulary_size", "ssd_data_path", "host_vocabulary_size"], SSDFeatureValidator)])
        def demo_func(name, host_vocabulary_size=0,
                      ssd_vocabulary_size=0,
                      ssd_data_path="./"):
            return True

        try:
            result = demo_func(name="host", host_vocabulary_size=0,
                               ssd_vocabulary_size=1,
                               ssd_data_path="./")
        except ValueError:
            result = False
        self.assertFalse(result)

    def test_check_value_for_open_interval(self):
        @para_checker_decorator(check_option_list=[
            ("beta1", FloatValidator, {"min_value": 0, "max_value": 1},
             ["check_value_for_open_interval", "check_value_for_right_open_interval",
              "check_value_for_left_open_interval"])])
        def demo_func(beta1):
            return True

        try:
            result = demo_func(beta1=0.5)
        except ValueError:
            result = False
        self.assertTrue(result)

    def test_is_valid(self):
        self.assertTrue(StringValidator("val", 'aa.1245', max_len=30).is_valid())


if __name__ == '__main__':
    unittest.main()
