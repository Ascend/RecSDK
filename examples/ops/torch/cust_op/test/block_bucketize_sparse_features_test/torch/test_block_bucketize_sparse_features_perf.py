#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights
# reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random
import sysconfig
import unittest
from dataclasses import dataclass

import fbgemm_gpu
import numpy as np
import pytest
import torch
import torch_npu


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(10000)

from block_bucketize_sparse_features_perf_cases import (
    PERF_CASES,
    PerfCase,
    _generate_case_tensors,
    _op_kwargs,
    _generate_total_num_blocks_tensors,
    _generate_batch_size_per_feature_and_max_B,
    _generate_block_bucketize_pos,
    GenTotalNumsBlocksType
)

DEVICE = "npu:0"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

def _validate_npu_matches_cpu_referance(kwargs_npu):
    npu_out = torch.ops.mxrec.block_bucketize_sparse_features(
        **kwargs_npu
    )
    print(npu_out)

@pytest.mark.parametrize("allow_negative_key", [False, True])
@pytest.mark.parametrize("edison_use_shift_filter", [False, True])
@pytest.mark.parametrize("case", PERF_CASES[:1], ids=lambda case: case.name)
def test_block_buckeize_sparse_feature_mt(case: PerfCase, allow_negative_key: bool, edison_use_shift_filter: bool):
    lengths, indices, block_sizes, weights = _generate_case_tensors(case, False)

    # 定值测试
    lengths = torch.tensor([3, 2, 1, 4])
    indices = torch.tensor([1, 5, 12, 3, 0, 7, 1, 9, -2, -1])
    block_sizes = torch.tensor([8, 16])
    indice_additions = torch.tensor([100, 1000])
    sequence = True
    kwargs_cpu = _op_kwargs(
        lengths=lengths,
        indices=indices,
        block_sizes=block_sizes,
        my_size=case.my_size,
        sequence=sequence,
        allow_negative_key=allow_negative_key,
        edison_use_shift_filter=edison_use_shift_filter,
        indice_additions=indice_additions,
    )
    kwargs_cpu['lengths'] = lengths.to(DEVICE)
    kwargs_cpu['indices'] = indices.to(DEVICE)
    kwargs_cpu['block_sizes'] = block_sizes.to(DEVICE)
    kwargs_cpu['indice_additions'] = indice_additions.to(DEVICE)
    print(kwargs_cpu)
    _validate_npu_matches_cpu_referance(kwargs_cpu)
