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

import logging
import sysconfig
import pytest
import fbgemm_gpu
import numpy as np
import torch_npu
import torch
logging.getLogger().setLevel(logging.INFO)
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def get_golden_result(device, denses, offsets, dense_datatype, offset_datatype):
    dense_torch = torch.from_numpy(denses).to(dense_datatype).to(device)

    offsets_torch = torch.from_numpy(offsets).to(offset_datatype).to(device)

    jagged_id_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(offsets_torch)

    output_size = jagged_id_offset[-1]

    jagged_embedding = torch.ops.fbgemm.dense_to_jagged(dense_torch, [jagged_id_offset], output_size)[0]

    return jagged_embedding.cpu()


def get_result(device, denses, offsets, dense_datatype, offset_datatype):
    dense_torch = torch.from_numpy(denses).to(dense_datatype).to(device)

    offsets_torch = torch.from_numpy(offsets).to(offset_datatype).to(device)

    jagged_id_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(offsets_torch)

    output_size = jagged_id_offset[-1]

    jagged_embedding = torch.ops.mxrec.dense_to_jagged(dense_torch, [jagged_id_offset], output_size)[0]

    return jagged_embedding.cpu()


@pytest.mark.parametrize("dense_dim0", [128, 40])
@pytest.mark.parametrize("dense_dim1", [210])
@pytest.mark.parametrize("dense_dim2", [1])
@pytest.mark.parametrize("dense_datatype", [torch.float32, torch.int64])
@pytest.mark.parametrize("offset_datatype", [torch.int32, torch.int64])
def test_dense_to_jagged(dense_dim0, dense_dim1, dense_dim2, dense_datatype, offset_datatype):
    denses = np.random.randn(dense_dim0, dense_dim1, dense_dim2).astype(np.float32)
    offsets = np.random.randint(0, dense_dim1, dense_dim0)
    gloden = get_golden_result(torch.device("cpu"), denses, offsets, dense_datatype, offset_datatype)
    npu_result = get_result(torch.device("npu"), denses, offsets, dense_datatype, offset_datatype)
    result_forward = torch.abs(gloden[0] - npu_result[0]) < 1e-4
    logging.info(result_forward.all().item())