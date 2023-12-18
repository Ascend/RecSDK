#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import unittest

from mx_rec.util.normalization import fix_invalid_table_name


class NormalizationTest(unittest.TestCase):

    def test_fix_invalid_table_name(self):
        self.assertEqual("user112", fix_invalid_table_name("user1#12"))
        with self.assertRaises(ValueError):
            fix_invalid_table_name("####@@@")


if __name__ == '__main__':
    unittest.main()
