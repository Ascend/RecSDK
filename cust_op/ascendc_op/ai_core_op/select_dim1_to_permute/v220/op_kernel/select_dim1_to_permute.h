/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef SELECT_DIM1_TO_PERMUTE_H
#define SELECT_DIM1_TO_PERMUTE_H

#include "kernel_common_utils.h"
#include "kernel_operator.h"

namespace SelectDim1ToPermute {

using namespace AscendC;

constexpr int USE_QUEUE_NUM = 2;
constexpr int USE_BUFFER_NUM = 2;
constexpr int INT32_ALIGNMENT = 8;
struct Args {
    GM_ADDR indices;
    GM_ADDR permute;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename indicesDType>
class SelectDim1ToPermuteKernel {
public:
    __aicore__ inline SelectDim1ToPermuteKernel(Args& args, TPipe* pipePtr)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        InitTilingParams(tilingData);
        int64_t coreIdx = GetBlockIdx();
        if (coreIdx < tailSplitIndex) {
            loopCount = splitBaseLen + 1;
            offsetOfThisCore = coreIdx * loopCount * indicesLength;
        } else {
            loopCount = splitBaseLen;
            offsetOfThisCore = tailSplitIndex * (splitBaseLen + 1) * indicesLength +
                               (coreIdx - tailSplitIndex) * splitBaseLen * indicesLength;
        }
        baseTableIdx = offsetOfThisCore;
        baseAddValue = static_cast<indicesDType>((offsetOfThisCore / indicesLength) * batchSize);
        InitGmParams(args, pipePtr);
    }

    __aicore__ inline void Process(Args& args)
    {
        ProcessTables(args);
    }

private:
    __aicore__ inline void InitTilingParams(const SelectDim1ToPermuteTilingData& tilingData)
    {
        batchSize = tilingData.batchSize;
        batchNum = tilingData.batchNum;
        indicesLength = tilingData.indicesLength;
        indicesLengthWithPadding = tilingData.indicesLengthWithPadding;
        splitBaseLen = tilingData.splitBaseLen;
        tailSplitIndex = tilingData.tailSplitIndex;
        ubCanUsed = tilingData.ubCanUsed;
        blockLen = tilingData.blockLen;
    }

    __aicore__ inline void InitGmParams(Args& args, TPipe* pipePtr)
    {
        indicesGT.SetGlobalBuffer(reinterpret_cast<__gm__ indicesDType*>(args.indices),
                                  indicesLength * sizeof(indicesDType));
        permuteGT.SetGlobalBuffer(reinterpret_cast<__gm__ indicesDType*>(args.permute),
                                  indicesLength * batchNum * sizeof(indicesDType));
        pipe = pipePtr;
        pipe->InitBuffer(indicesQueue, USE_QUEUE_NUM, blockLen * sizeof(indicesDType));
        pipe->InitBuffer(permuteQueue, USE_QUEUE_NUM, blockLen * sizeof(indicesDType));
    }

    __aicore__ inline void CopyIn(int64_t offset, int64_t len)
    {
        LocalTensor<indicesDType> indicesLocal = indicesQueue.AllocTensor<indicesDType>();
        CpGm2Local<indicesDType>(indicesLocal, indicesGT[offset], len);
        AscendC::PipeBarrier<PIPE_ALL>();
        indicesQueue.EnQue(indicesLocal);
    }

    __aicore__ inline void Compute(int32_t i, int64_t tableIdx, int64_t offset, int64_t len)
    {
        LocalTensor<indicesDType> indicesLocal = indicesQueue.DeQue<indicesDType>();
        LocalTensor<indicesDType> permuteLocal = permuteQueue.AllocTensor<indicesDType>();
        AscendC::Adds(permuteLocal, indicesLocal, baseAddValue + static_cast<indicesDType>(i * batchSize),
                      static_cast<int32_t>(len));
        AscendC::PipeBarrier<PIPE_V>();
        permuteQueue.EnQue<indicesDType>(permuteLocal);
        indicesQueue.FreeTensor(indicesLocal);
    }

    __aicore__ inline void CopyOut(int32_t i, int64_t tableIdx, int64_t offset, int64_t len)
    {
        LocalTensor<indicesDType> permuteLocal = permuteQueue.DeQue<indicesDType>();
        CpLocal2Gm<indicesDType>(permuteGT[tableIdx + offset], permuteLocal, len);
        permuteQueue.FreeTensor(permuteLocal);
    }

    __aicore__ inline void ProcessTables(Args& args)
    {
        for (int32_t i = 0; i < loopCount; ++i) {
            int64_t tableIdx = offsetOfThisCore + i * indicesLength;
            int64_t offset = 0;
            while (offset < indicesLength) {
                int64_t remain = indicesLength - offset;
                int64_t len = remain < blockLen ? remain : blockLen;
                CopyIn(offset, len);
                Compute(i, tableIdx, offset, len);
                CopyOut(i, tableIdx, offset, len);
                offset += blockLen;
            }
        }
    }

    GlobalTensor<indicesDType> indicesGT;
    GlobalTensor<indicesDType> permuteGT;
    int64_t batchSize;
    int64_t indicesLength;
    int64_t indicesLengthWithPadding;
    int64_t batchNum;
    int32_t splitBaseLen;
    int64_t tailSplitIndex;
    int64_t ubCanUsed;
    int64_t blockLen;

    // ThisCoreLen for T
    int64_t offsetOfThisCore = 0;
    int64_t loopCount = 0;
    int64_t baseTableIdx = 0;
    indicesDType baseAddValue = 0;

    // Tpipe;
    TPipe* pipe;
    TQue<TPosition::VECIN, 1> indicesQueue;
    TQue<TPosition::VECOUT, 1> permuteQueue;
};

}  // namespace SelectDim1ToPermute

#endif