#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
import os
import sys
import tempfile
import unittest

from mx_rec.validator.validator import Validator, StringValidator, DirectoryValidator

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

    def test_validator_should_return_default_if_invalid(self):
        validation = Validator('aa')
        validation.register_checker(lambda x: len(x) < 5, 'length of string should be less than 5')
        self.assertTrue(validation.is_valid())
        try:
            validation = Validator('123456')
            validation.register_checker(lambda x: len(x) < 5, 'length of string should be less than 5')
            validation.is_valid()
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

    def test_string_validator_max_len_parameter(self):
        try:
            StringValidator('aa.1245', max_len=3).check_string_length().check().is_valid()
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

        self.assertTrue(StringValidator('aa.1245', max_len=30).check().is_valid())
        # default infinity
        self.assertTrue(StringValidator('aa.124512132456').check().is_valid())

    def test_string_validator_min_len_parameter(self):
        try:
            StringValidator('aa', min_len=3).check_string_length().check().is_valid()
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

        self.assertTrue(StringValidator('aa.', min_len=3).check().is_valid())
        # default 0
        self.assertTrue(StringValidator('1').check().is_valid())

    def test_string_validator_can_be_transformed2int(self):
        self.assertFalse(StringValidator('9' * 20).can_be_transformed2int().check().is_valid())
        self.assertFalse(StringValidator('1,2').can_be_transformed2int().check().is_valid())
        self.assertTrue(StringValidator('12').can_be_transformed2int().check().is_valid())
        self.assertFalse(StringValidator('12').can_be_transformed2int(min_value=100, max_value=200).check().is_valid())

    def test_directory_black_list(self):
        try:
            DirectoryValidator('/abc/d/e').with_blacklist(lst=['/abc/d/e']).check().is_valid()
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

        self.assertTrue(DirectoryValidator('/abc/d/e').with_blacklist(['/abc/d/']).check().is_valid())
        self.assertTrue(DirectoryValidator('/abc/d/e').with_blacklist(['/abc/d/'], exact_compare=True).check()
                        .is_valid())
        # if not exact compare, the /abc/d/e is children path of /abc/d/, so it is invalid
        try:
            self.assertFalse(DirectoryValidator('/abc/d/e').with_blacklist(['/abc/d/'], exact_compare=False)
                             .check().is_valid())
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")
        self.assertTrue(DirectoryValidator('/usr/bin/bash').with_blacklist().check().is_valid())

        try:
            DirectoryValidator('/usr/bin/bash').with_blacklist(exact_compare=False).check().is_valid()
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
            DirectoryValidator(path).check_not_soft_link().check().is_valid()
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")
        finally:
            tmp.close()  # close means remove
            os.remove(path)
            os.removedirs(temp_dir)

    def test_directory_check(self):

        try:
            DirectoryValidator('a/b/.././c/a.txt').check_not_soft_link().check().is_valid()
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

        try:
            DirectoryValidator("").check_is_not_none().check().is_valid()
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")

        try:
            DirectoryValidator(None).check_is_not_none().check().is_valid()
        except ValueError as exp:
            self.assertEqual(type(exp), ValueError)
        else:
            self.fail("ValueError not raised.")
        self.assertTrue(DirectoryValidator("a/bc/d").check().is_valid())


if __name__ == '__main__':
    unittest.main()
