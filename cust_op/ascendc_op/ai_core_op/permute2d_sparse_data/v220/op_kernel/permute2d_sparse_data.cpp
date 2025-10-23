/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include "permute2d_sparse_data_kernel.h"
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void permute2d_sparse_data(GM_ADDR permute, GM_ADDR lengths, GM_ADDR values,
        GM_ADDR weights, GM_ADDR out_lengths, GM_ADDR out_indices, GM_ADDR out_weights, GM_ADDR workspace,
        GM_ADDR tiling)
{
    Permute2dSparseData::Args args{permute, lengths, values, weights, out_lengths,
        out_indices, out_weights, workspace, tiling};
    Permute2dSparseData::Permute2dSparseDataKernel<DTYPE_LENGTHS, DTYPE_VALUES> kernel(args);
    kernel.Compute();
}