#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
from enum import Enum, unique

from .ascend_native_backend import AscendNative
from .pytorch_native_backend import PytorchNative
from .ascend_fuse_backend import AscendFuse


@unique
class KernelBackend(Enum):
    ASCEND_NATIVE = "ASCEND_NATIVE"
    ASCEND_FUSE = "ASCEND_FUSE"
    PYTORCH_NATIVE = "PYTORCH_NATIVE"
    CUDA_NATIVE = "CUDA_NATIVE"
    CUDA_FUSE = "CUDA_FUSE"
    TRITON = "TRITON"


def create_hstu_atten_backend(kernel_backend: KernelBackend, **kwargs):
    if kernel_backend == KernelBackend.ASCEND_NATIVE:
        return AscendNative()
    elif kernel_backend == KernelBackend.PYTORCH_NATIVE:
        return PytorchNative()
    elif kernel_backend == KernelBackend.ASCEND_FUSE:
        return AscendFuse(**kwargs)
    else:
        raise ValueError(f"unknown kernel backend : {kernel_backend}")
