#!/usr/bin/env python3
# coding: UTF-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

import os
import sys
import unittest
from mx_rec.util.cpu import *


class TestCPUBind(unittest.TestCase):

    def test_pcie(self):
        for root, dirs, _ in os.walk("/sys/bus/pci/devices/"):
            for d in dirs:
                dev = os.path.join(root, d)
                numa = get_numa_by_pcie(d)
                with open(os.path.join(dev, "numa_node")) as f:
                    self.assertEqual(numa, int(f.read().strip()))

    def test_bind_failed(self):
        self.assertFalse(bind_cpu_by_device_logic_id(20))


if __name__ == '__main__':
    unittest.main()
