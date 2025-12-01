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

#ifndef Invert_Permute_H
#define Invert_Permute_H

#include "kernel_operator.h"

namespace InvertPermute {
constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;

template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtCompute(__gm__ T* y, __gm__ T* x,
                                                    int32_t totalLength, uint32_t threadsPerBlock)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockNum = AscendC::Simt::GetBlockNum();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t globalThreadIdx = blockIdx * threadsPerBlock + threadIdx;

    for (int32_t i = globalThreadIdx; i < totalLength; i += blockNum * threadsPerBlock) {
        T src_val = x[i];
        if (src_val >= 0 && src_val < totalLength) {
            y[src_val] = static_cast<T>(i);
        }
    }
}

} // namespace InvertPermute
#endif // Invert_Permute_H