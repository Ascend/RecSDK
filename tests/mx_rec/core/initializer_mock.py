#!/usr/bin/env python3
# coding: UTF-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import os


class InitializerMock:
    """
    initializer mock module
    """
    @staticmethod
    def get_use_static():
        return os.getenv("use_static", True)
