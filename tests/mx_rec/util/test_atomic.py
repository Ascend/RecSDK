#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import unittest
from threading import Thread

from mx_rec.util.atomic import AtomicInteger


class AtomicTest(unittest.TestCase):
    def setUp(self):
        """
        准备步骤
        :return:无
        """
        self.iter_times = 100_000
        self.num_thread = 2

    def test_str(self):
        num = AtomicInteger(1)
        self.assertEqual("1", str(num))

    def test_increase(self):
        num = AtomicInteger(0)
        thread_pool = []

        def runnable():
            for _ in range(self.iter_times):
                num.increase()

        for _ in range(self.num_thread):
            thread = Thread(target=runnable)
            thread.start()
            thread_pool.append(thread)
        for thread in thread_pool:
            thread.join()
        self.assertEqual(200_000, num.value())

    def test_decrease(self):
        num = AtomicInteger(200_000)
        thread_pool = []

        def runnable():
            for _ in range(self.iter_times):
                num.decrease()

        for _ in range(self.num_thread):
            thread = Thread(target=runnable)
            thread.start()
            thread_pool.append(thread)
        for thread in thread_pool:
            thread.join()
        self.assertEqual(0, num.value())


if __name__ == '__main__':
    unittest.main()
