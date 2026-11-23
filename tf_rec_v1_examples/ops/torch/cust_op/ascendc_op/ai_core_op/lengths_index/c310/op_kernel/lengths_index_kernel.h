/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#ifndef LENGTHS_INDEX_KERNEL_H
#define LENGTHS_INDEX_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

namespace LengthsIndex {

// SIMT VF 单次启动的线程数上界（LAUNCH_BOUND），须 >= asc_vf_call 的 dim3.x。
// Host 上 threadNum = vectorSize * rowsPerBlock 恒为 512；取 1024 与同仓库其它 VF 一致并留余量。
constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;

struct Args {
    GM_ADDR offsets;
    GM_ADDR output;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

// SIMT 子核：由 asc_vf_call 按 dim3{threadNum,1,1} 启动；tid 拆成块内行号与行内车道，
// 步长 vectorSize 交错写满 [rowStart, rowEnd)。
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void IndexRowsSimt(__gm__ T *offsetsPtr,
                                                                                      __gm__ T volatile *outputPtr,
                                                                                      int64_t numSeq,
                                                                                      int64_t outputSize,
                                                                                      int64_t startRow,
                                                                                      int64_t rowsPerBlock,
                                                                                      int64_t vectorSize)
{
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    int32_t rowOffset = tid / static_cast<int32_t>(vectorSize);
    int32_t colOffset = tid % static_cast<int32_t>(vectorSize);

    int64_t rowIdx = startRow + rowOffset;
    if (rowIdx >= numSeq) {
        return;
    }

    T rowStart = offsetsPtr[rowIdx];
    T rowEnd = (rowIdx < numSeq - 1) ? offsetsPtr[rowIdx + 1] : static_cast<T>(outputSize);
    for (T i = rowStart + static_cast<T>(colOffset); i < rowEnd; i += static_cast<T>(vectorSize)) {
        outputPtr[i] = static_cast<T>(rowIdx);
    }
}

template <typename T>
class LengthsIndexKernel {
public:
    __aicore__ inline LengthsIndexKernel(Args &args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        numSeq = tilingData.numSeq;
        outputSize = tilingData.outputSize;
        totalBlocks = tilingData.totalBlocks;
        blocksPerCore = tilingData.blocksPerCore;
        remainderBlocks = tilingData.remainderBlocks;
        vectorSize = tilingData.vectorSize;
        rowsPerBlock = tilingData.rowsPerBlock;
        offsetsPtr = reinterpret_cast<__gm__ T *>(args.offsets);
        outputPtr = reinterpret_cast<__gm__ T volatile *>(args.output);
    }

    __aicore__ inline void Compute()
    {
        if (numSeq <= 0 || totalBlocks <= 0) {
            return;
        }

        // 与 Host tiling 一致：前 remainderBlocks 个核多处理 1 个 logicalBlock
        int64_t coreIdx = GetBlockIdx();
        int64_t blockCount;
        int64_t blockStart;
        if (coreIdx < remainderBlocks) {
            blockCount = blocksPerCore + 1;
            blockStart = coreIdx * blockCount;
        } else {
            blockCount = blocksPerCore;
            blockStart = remainderBlocks * (blocksPerCore + 1) + (coreIdx - remainderBlocks) * blocksPerCore;
        }
        if (blockCount <= 0) {
            return;
        }

        uint32_t threadNum = static_cast<uint32_t>(vectorSize * rowsPerBlock);
        for (int64_t logicalBlock = blockStart; logicalBlock < blockStart + blockCount; ++logicalBlock) {
            int64_t startRow = logicalBlock * rowsPerBlock;
            asc_vf_call<IndexRowsSimt<T>>(dim3{threadNum, 1, 1},
                                            offsetsPtr,
                                            outputPtr,
                                            numSeq,
                                            outputSize,
                                            startRow,
                                            rowsPerBlock,
                                            vectorSize);
        }
    }

private:
    __gm__ T *offsetsPtr;
    __gm__ T volatile *outputPtr;

    int64_t numSeq;
    int64_t outputSize;
    int64_t totalBlocks;
    int64_t blocksPerCore;
    int32_t remainderBlocks;
    int64_t vectorSize;
    int64_t rowsPerBlock;
};

}  // namespace LengthsIndex

#endif  // LENGTHS_INDEX_KERNEL_H
