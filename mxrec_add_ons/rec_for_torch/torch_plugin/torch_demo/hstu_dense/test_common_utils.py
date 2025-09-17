#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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
import sysconfig
import torch

device_id: int = 0
mask_tril: int = 0
mask_triu: int = 1
mask_none: int = 2
mask_custom: int = 3


def init():
    torch.npu.config.allow_internal_format = False
    torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

    torch.npu.set_device(device_id)


def get_chip():
    return False


def allclose(tensor, other, atol, ratio):
    assert tensor.shape == other.shape
    diff = (torch.abs(tensor - other) > atol)
    diff_count = torch.sum(diff)
    return diff_count / tensor.numel() < ratio
