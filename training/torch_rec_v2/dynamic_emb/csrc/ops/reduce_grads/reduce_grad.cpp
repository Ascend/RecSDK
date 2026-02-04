/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <type_traits>

#include "reduce_grad_kernel.h"

extern "C" __global__ __aicore__ void reduce_grad_op(GM_ADDR grad, GM_ADDR inverse,
    int len, int dim, int baseBlk, int remainBlk, GM_ADDR output)
{
    int coreId = AscendC::GetBlockIdx();
    int blkCnt = (coreId < remainBlk) ? (baseBlk + 1) : baseBlk;
    int blkStart = coreId * baseBlk + ((coreId < remainBlk) ? coreId : remainBlk);

    __gm__ const float* pGrad = reinterpret_cast<__gm__ const float*>(grad);
    __gm__ const int64_t* pInverse = reinterpret_cast<__gm__ const int64_t*>(inverse);
    __gm__ float* pOutput = reinterpret_cast<__gm__ float*>(output);

    AscendC::Simt::VF_CALL<DynamicEmbeddingReduceGradOPSimt::ReduceGradCompute>(
        AscendC::Simt::Dim3{DynamicEmbeddingReduceGradOPSimt::MAX_THREADS_PER_BLOCK, 1, 1},
        pGrad, pInverse, len, dim, blkStart, blkCnt, pOutput);
}