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

#include <type_traits>
#include "asynchronous_complete_cumsum_kernel.h"
#include "kernel_operator.h"

constexpr int32_t BLOCK_THREADS = AsynchronousCompleteCumsumSimt::MAX_THREADS_PER_BLOCK;
constexpr int32_t ELEMS_PER_THREAD = AsynchronousCompleteCumsumSimt::MAX_ELEMENTS_PER_THREAD;
constexpr int32_t MAX_WARPS = AsynchronousCompleteCumsumSimt::MAX_WARPS;
constexpr int32_t CACHE_ALIGN = AsynchronousCompleteCumsumSimt::CACHE_ALIGN;

// dtype 分发：isInt32=true 用 int32_t，否则用 int64_t
#define DTYPE_DISPATCH(isInt32, DTYPE, BODY) \
    do { \
        if (isInt32) { using DTYPE = int32_t; BODY; } \
        else { using DTYPE = int64_t; BODY; } \
    } while (0)

extern "C" __global__ __aicore__ void asynchronous_complete_cumsum(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    // 提取公共的tiling数据
    int32_t totalLength = tilingData.totalLength;
    int32_t totalBlocks = tilingData.totalBlocks;
    int32_t blocksPerCore = tilingData.blocksPerCore;
    int32_t remainderBlocks = tilingData.remainderBlocks;
    int32_t coreId = GetBlockIdx();

    bool isInt32 = tilingData.isInt32;
    bool isSmall = tilingData.isSmall;

    auto user_workspace = GetUserWorkspace(workspace);

    DTYPE_DISPATCH(isInt32, DTYPE_X, {
        __gm__ DTYPE_X* input = reinterpret_cast<__gm__ DTYPE_X*>(x);
        __gm__ DTYPE_X* output = reinterpret_cast<__gm__ DTYPE_X*>(y);
        __gm__ DTYPE_X* ws = reinterpret_cast<__gm__ DTYPE_X*>(user_workspace);

        TPipe pipe;
        TBuf<TPosition::VECCALC> sharedMem;
        pipe.InitBuffer(sharedMem, MAX_WARPS * sizeof(DTYPE_X));
        LocalTensor<DTYPE_X> sharedTensor = sharedMem.Get<DTYPE_X>();
        __ubuf__ DTYPE_X* sharedMemory =  reinterpret_cast<__ubuf__ DTYPE_X*>(sharedTensor.GetPhyAddr());

        if (isSmall) {
            Simt::VF_CALL<AsynchronousCompleteCumsumSimt::SimtSmallDataCompute<DTYPE_X>>(
                Simt::Dim3{BLOCK_THREADS, 1, 1},
                input, output, ws, sharedMemory, totalLength);

            if (totalBlocks > 1) {
                SyncAll();
                Simt::VF_CALL<AsynchronousCompleteCumsumSimt::SimtSmallDataUpdate<DTYPE_X>>(
                    Simt::Dim3{BLOCK_THREADS, 1, 1},
                    output, ws, totalLength);
            }
        } else {
            int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
            int32_t blockStartIdx = coreId * blocksPerCore + ((coreId < remainderBlocks) ? coreId : remainderBlocks);

            Simt::VF_CALL<AsynchronousCompleteCumsumSimt::SimtLargeDataCompute<DTYPE_X>>(
                Simt::Dim3{BLOCK_THREADS, 1, 1},
                input, output, ws, sharedMemory, totalLength, totalBlocks, blockStartIdx, curBlocksCount);

            SyncAll();
            Simt::VF_CALL<AsynchronousCompleteCumsumSimt::SimtLargeDataUpdate<DTYPE_X>>(
                Simt::Dim3{BLOCK_THREADS, 1, 1},
                output, ws, totalLength, totalBlocks, blockStartIdx, curBlocksCount);
        }
    });
}
