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
        GM_ADDR weights, GM_ADDR totalOffset, GM_ADDR lengthsOffset, GM_ADDR permutedLengthsOffset,
        GM_ADDR outLengths, GM_ADDR outValues, GM_ADDR outWeights, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    Permute2dSparseData::Args args{permute, lengths, values, weights, totalOffset, lengthsOffset,
                                   permutedLengthsOffset, outLengths, outValues, outWeights, tiling};
    Permute2dSparseData::Permute2dSparseDataKernel<DTYPE_PERMUTE, DTYPE_LENGTHS, DTYPE_VALUES, DTYPE_WEIGHTS> kernel(
        args, &pipe);
    if (TILING_KEY_IS(1)) {
        kernel.ComputeAll();
    } else if (TILING_KEY_IS(2)) {
        kernel.ComputeData();
    }
}
