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
