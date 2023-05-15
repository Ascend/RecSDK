#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2021-2023 Huawei Technologies Co., Ltd

import threading


class AtomicInteger():
    def __init__(self, value=0):
        self._value = int(value)
        self._lock = threading.Lock()
        
    def increase(self, num=1):
        with self._lock:
            self._value += int(num)
            return self._value

    def decrease(self, num=1):
        return self.inc(-num)    

    def value(self):
        with self._lock:
            return self._value

    def __str__(self):
        return str(self.value())
    