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
import itertools
import logging
import sysconfig

from pathlib import Path
import pytest
import fbgemm_gpu
import numpy as np
import torch_npu
import torch

torch.npu.config.allow_internal_format = False
CURR_DIR = Path(__file__).resolve().parent
torch.ops.load_library(str(CURR_DIR.parent.parent.parent / 
                           "cust_op/framework/torch_plugin/torch_library/token_mixing/build"
                           "/libtoken_mixing.so"))


def get_global_loss(x):
    input_tensor = torch.from_numpy(x)
    transpose_tensor = torch.from_numpy(x).permute(0, 2, 1)
    add = torch.add(input_tensor, transpose_tensor)
    layer_norm = torch.nn.LayerNorm(add.size()[2:], eps=1e-7)
    result = layer_norm(add)
    return result.cpu().detach().numpy()
    

def get_fused_loss_op(x, gamma, beta, device):
    torch.npu.set_device(device)
    
    result = torch.ops.mxrec.token_mixing(
        torch.from_numpy(x).to(device), torch.from_numpy(gamma).to(device), torch.from_numpy(beta).to(device)
    )
    return result.cpu().detach().numpy()


@pytest.mark.parametrize("B", [1, 45, 256, 512, 1024])
@pytest.mark.parametrize("S", [128, 256, 338, 507, 512, 1024])
@pytest.mark.parametrize("device", ["npu:0"])
def test_token_mixing(B, S, device):
    tensor = np.random.randn(B, S, S).astype(np.float32)
    gamma = np.ones(S, dtype=np.float32)
    beta = np.zeros(S, dtype=np.float32)
    result = get_global_loss(tensor)
    result1 = get_fused_loss_op(tensor, gamma, beta, device)
    assert np.allclose(result, result1, atol=1e-6)