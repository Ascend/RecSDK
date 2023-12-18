#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import unittest

from mx_rec.util.perf import performance


class PerfTest(unittest.TestCase):

    def test_performance(self):
        def func():
            return 9
        derec = performance("func")
        self.assertEqual(9, derec(func)())


if __name__ == '__main__':
    unittest.main()
