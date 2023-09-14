#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.

import time

from mx_rec.util.log import logger


def performance(method_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            span = time.perf_counter() - start
            logger.debug(f"%s method consume %s (s).", method_name, round(span, 6))
            return result
        return wrapper
    return decorator
