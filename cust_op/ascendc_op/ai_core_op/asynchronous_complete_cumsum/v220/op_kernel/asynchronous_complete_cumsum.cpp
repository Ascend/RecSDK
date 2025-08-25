/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "kernel_operator.h"

using namespace AscendC;

namespace {
    constexpr int32_t EMBEDDING_TYPE_INT64 = 0;
    constexpr int32_t EMBEDDING_TYPE_INT32 = 1;
}

extern "C" __global__ __aicore__ void asynchronous_complete_cumsum(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                                   GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    int64_t totalLen = tiling_data.totalLength;
    uint32_t inputType = tiling_data.inputType;

    switch (inputType)
    {
        case EMBEDDING_TYPE_INT64:
        {
            __gm__ int64_t* xPtr = (__gm__ int64_t*) x;
            __gm__ int64_t* yPtr = (__gm__ int64_t*) y;
            *(yPtr) = 0;
            for (int i=1; i<totalLen+1; i++) {
                *(yPtr+i) = *(xPtr+i-1) + *(yPtr+i-1);
            }
            break;
        }

        case EMBEDDING_TYPE_INT32:
        {
            __gm__ int32_t* xPtr = (__gm__ int32_t*) x;
            __gm__ int32_t* yPtr = (__gm__ int32_t*) y;
            *(yPtr) = 0;
            for (int i=1; i<totalLen+1; i++) {
                *(yPtr+i) = *(xPtr+i-1) + *(yPtr+i-1);
            }
            break;
        }

        default:
        {
            break;
        }
    }

    AscendC::GlobalTensor<int32_t> global;
    global.SetGlobalBuffer((__gm__ int32_t*)y, totalLen);
#ifdef SUPPORT_V200
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(global);
#else
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
        AscendC::DcciDst::CACHELINE_OUT>(global);
#endif
}