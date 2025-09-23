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
import pytest
import torch
import torch_npu
import fbgemm_gpu
import numpy as np

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def get_result(t_in):
    return torch.ops.fbgemm.asynchronous_complete_cumsum(t_in)


def get_ops_result(t_in, is_mxrec):
    if is_mxrec:
        return torch.ops.mxrec.asynchronous_complete_cumsum(t_in).cpu()
    else:
        return torch.ops.fbgemm.asynchronous_complete_cumsum(t_in).cpu()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("device", ["npu:0", "npu:5"])
@pytest.mark.parametrize("length", [1, 10, 100, 1000, 10000])
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_asynchronous_complete_cumsum(dtype, device, length, is_mxrec):
    t_int = torch.randint(0, 100, (length,), dtype=dtype)
    golden = get_result(t_int)
    result = get_ops_result(t_int.to(device), is_mxrec)
    assert torch.allclose(result, golden)
