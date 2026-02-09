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

#include "asynchronous_complete_cumsum_kernel.h"

extern "C" __global__ __aicore__ void asynchronous_complete_cumsum(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    // 提取公共的tiling数据
    int64_t totalLength = tilingData.totalLength;
    int64_t totalBlocks = tilingData.totalBlocks;
    int64_t blocksPerCore = tilingData.blocksPerCore;
    int64_t remainderBlocks = tilingData.remainderBlocks;
    int64_t coreId = GetBlockIdx();

    bool isSmall = tilingData.isSmall;

    auto user_workspace = GetUserWorkspace(workspace);

    __gm__ DTYPE_X* input = reinterpret_cast<__gm__ DTYPE_X*>(x);
    __gm__ DTYPE_X* output = reinterpret_cast<__gm__ DTYPE_X*>(y);
    __gm__ DTYPE_X* ws = reinterpret_cast<__gm__ DTYPE_X*>(user_workspace);

    TPipe pipe;
    TBuf<TPosition::VECCALC> sharedMem;
    pipe.InitBuffer(sharedMem, MAX_WARPS * sizeof(DTYPE_X));
    LocalTensor<DTYPE_X> sharedTensor = sharedMem.Get<DTYPE_X>();
    __ubuf__ DTYPE_X* sharedMemory =  reinterpret_cast<__ubuf__ DTYPE_X*>(sharedTensor.GetPhyAddr());

    if (isSmall) {
        asc_vf_call<AsynchronousCompleteCumsumSimt::SimtSmallDataCompute<DTYPE_X>>(
            dim3{MAX_THREADS_PER_BLOCK, 1, 1},
            input,
            output,
            ws,
            sharedMemory,
            totalLength);

        if (totalBlocks > 1) {
            SyncAll();
            asc_vf_call<AsynchronousCompleteCumsumSimt::SimtSmallDataUpdate<DTYPE_X>>(
                dim3{MAX_THREADS_PER_BLOCK, 1, 1},
                output,
                ws,
                totalLength);
        }
    } else {
        int64_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
        int64_t blockStartIdx = coreId * blocksPerCore + ((coreId < remainderBlocks) ? coreId : remainderBlocks);

        asc_vf_call<AsynchronousCompleteCumsumSimt::SimtLargeDataCompute<DTYPE_X>>(
            dim3{MAX_THREADS_PER_BLOCK, 1, 1},
            input,
            output,
            ws,
            sharedMemory,
            totalLength,
            totalBlocks,
            blockStartIdx,
            curBlocksCount);

        SyncAll();
        asc_vf_call<AsynchronousCompleteCumsumSimt::SimtLargeDataUpdate<DTYPE_X>>(
            dim3{MAX_THREADS_PER_BLOCK, 1, 1},
            output,
            ws,
            totalLength,
            totalBlocks,
            blockStartIdx,
            curBlocksCount);
    }
}