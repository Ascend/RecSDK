#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import time
import logging


def performance(method_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            span = time.perf_counter() - start
            logging.debug(f"{method_name} method consume {span:.6f}s.")
            return result
        return wrapper
    return decorator
