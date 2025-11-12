#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

from pathlib import Path
import pytest
import torch
import torch_npu
import numpy as np

torch.npu.config.allow_internal_format = False
CURR_DIR = Path(__file__).resolve().parent
torch.ops.load_library(str(CURR_DIR.parent.parent.parent /
    "framework/torch_plugin/torch_library/gather_for_rank1/build/libgather_for_rank1.so"))


def get_golden(x, index):
    result = torch.index_select(x, dim=0, index=index)
    return result.cpu().detach().numpy()


def get_op(x, index, device):
    torch.npu.set_device(device)
    x = x.to(device)
    index = index.to(device)

    result = torch.ops.mxrec.gather_for_rank1(x, index)
    return result.cpu().detach().numpy()


@pytest.mark.parametrize("x_dim", [10, 20, 100, 1000, 10000])
@pytest.mark.parametrize("x_type", [torch.float16, torch.float32])
@pytest.mark.parametrize("index_dim", [1, 10, 100, 1000])
@pytest.mark.parametrize("device", ["npu:0"])
def test_gather_for_rank1_v200(x_dim, x_type, index_dim, device):
    x = torch.randn(x_dim).to(x_type)
    index = torch.randint(0, x_dim, (index_dim, )).to(torch.int32)

    y_golden = get_golden(x, index)
    torch.npu.synchronize()

    y_op = get_op(x, index, device)
    torch.npu.synchronize()

    loss = 1e-3
    if x_type == torch.float32:
        loss = 1e-4

    assert np.allclose(y_golden, y_op, loss)