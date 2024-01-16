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

import threading


class AtomicInteger:
    """
    counter atomic increment/decrement
    """
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
        return self.increase(-num)

    def value(self):
        with self._lock:
            return self._value
