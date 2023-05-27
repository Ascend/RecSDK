#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import threading


class AtomicInteger():
    def __init__(self, value=0):
        self._value = int(value)
        self._lock = threading.Lock()

    def __str__(self):
        return str(self.value())

    def increase(self, num=1):
        with self._lock:
            self._value += int(num)
            return self._value

    def decrease(self, num=1):
        return self.inc(-num)

    def value(self):
        with self._lock:
            return self._value
