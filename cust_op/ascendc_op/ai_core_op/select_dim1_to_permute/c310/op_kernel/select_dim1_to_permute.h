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
#include "simt_api/asc_simt.h"

namespace SelectDim1ToPermute {

using namespace AscendC;

constexpr int USE_QUEUE_NUM = 2;
constexpr int USE_BUFFER_NUM = 2;
constexpr int INT32_ALIGNMENT = 8;
constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
struct Args {
    GM_ADDR indices;
    GM_ADDR lengths;
    GM_ADDR permute;
    GM_ADDR outputLengths;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

/**
 * @brief 对input按照indices进行选择，同时将indices加上offset后写入permute
 * @param input [in] 输入张量
 * @param indices [in] 索引张量
 * @param output [out] 输出张量
 * @param permute [out] 索引偏移后的输出张量
 * @param indicesSize 索引张量大小
 * @param offset 偏移量
 */
template <typename indicesDType, typename lengthsDType>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void IndexSelectSimt(
    __local_mem__ lengthsDType* input, __local_mem__ indicesDType* indices, __gm__ lengthsDType volatile* output,
    __gm__ indicesDType volatile* permute, int64_t indicesSize, int64_t offset)
{
    auto threadNum = AscendC::Simt::GetThreadNum<0>();
    auto threadIdx = AscendC::Simt::GetThreadIdx<0>();
    for (int64_t i = threadIdx; i < indicesSize; i += threadNum) {
        output[i] = input[indices[i]];
        permute[i] = indices[i] + offset;
    }
}

template <typename indicesDType, typename lengthsDType>
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
            lengthsOffsetOfThisCore = coreIdx * loopCount * batchSize;
        } else {
            loopCount = splitBaseLen;
            offsetOfThisCore = tailSplitIndex * (splitBaseLen + 1) * indicesLength +
                               (coreIdx - tailSplitIndex) * splitBaseLen * indicesLength;
            lengthsOffsetOfThisCore =
                tailSplitIndex * (splitBaseLen + 1) * batchSize + (coreIdx - tailSplitIndex) * splitBaseLen * batchSize;
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
        lengthsSize = tilingData.lengthsSize;
        batchSize = tilingData.batchSize;
        batchNum = tilingData.batchNum;
        indicesLength = tilingData.indicesLength;
        splitBaseLen = tilingData.splitBaseLen;
        tailSplitIndex = tilingData.tailSplitIndex;
        blockLen = tilingData.blockLen;
    }

    __aicore__ inline void InitGmParams(Args& args, TPipe* pipePtr)
    {
        indicesGT.SetGlobalBuffer(reinterpret_cast<__gm__ indicesDType*>(args.indices),
                                  indicesLength * sizeof(indicesDType));
        lengthsGT.SetGlobalBuffer(reinterpret_cast<__gm__ lengthsDType*>(args.lengths),
                                  lengthsSize * sizeof(lengthsDType));
        permuteGT.SetGlobalBuffer(reinterpret_cast<__gm__ indicesDType*>(args.permute),
                                  indicesLength * batchNum * sizeof(indicesDType));
        outputLengthsGT.SetGlobalBuffer(reinterpret_cast<__gm__ lengthsDType*>(args.outputLengths),
                                        indicesLength * batchNum * sizeof(lengthsDType));
        pipe = pipePtr;
        pipe->InitBuffer(indicesQueue, USE_QUEUE_NUM, blockLen * sizeof(indicesDType));
        pipe->InitBuffer(lengthsQueue, USE_QUEUE_NUM, batchSize * sizeof(lengthsDType));
    }

    __aicore__ inline void CopyIn(int32_t i, int64_t offset, int64_t len)
    {
        LocalTensor<indicesDType> indicesLocal = indicesQueue.AllocTensor<indicesDType>();
        LocalTensor<lengthsDType> lengthsLocal = lengthsQueue.AllocTensor<lengthsDType>();
        CpGm2Local<indicesDType>(indicesLocal, indicesGT[offset], len);
        CpGm2Local<lengthsDType>(lengthsLocal, lengthsGT[lengthsOffsetOfThisCore + i * batchSize], batchSize);
        AscendC::PipeBarrier<PIPE_ALL>();
        indicesQueue.EnQue(indicesLocal);
        lengthsQueue.EnQue(lengthsLocal);
    }

    __aicore__ inline void Compute(int32_t i, int64_t tableIdx, int64_t offset, int64_t len)
    {
        LocalTensor<indicesDType> indicesLocal = indicesQueue.DeQue<indicesDType>();
        LocalTensor<lengthsDType> lengthsLocal = lengthsQueue.DeQue<lengthsDType>();
        __local_mem__ indicesDType* indicesPtr = (__local_mem__ indicesDType*)indicesLocal.GetPhyAddr();
        __local_mem__ lengthsDType* lengthsPtr = (__local_mem__ lengthsDType*)lengthsLocal.GetPhyAddr();
        __gm__ lengthsDType* outputLengthsPtr = (__gm__ lengthsDType*)outputLengthsGT[tableIdx + offset].GetPhyAddr();
        __gm__ indicesDType* permutePtr = (__gm__ indicesDType*)permuteGT[tableIdx + offset].GetPhyAddr();
        int64_t permuteOffset = baseAddValue + static_cast<indicesDType>(i * batchSize);
        uint32_t threadNum = len > MAX_THREADS_PER_BLOCK ? MAX_THREADS_PER_BLOCK : len;
        asc_vf_call<IndexSelectSimt<indicesDType, lengthsDType>>(dim3{threadNum, 1, 1}, lengthsPtr, indicesPtr,
                                                                 outputLengthsPtr, permutePtr, len, permuteOffset);
        indicesQueue.FreeTensor(indicesLocal);
        lengthsQueue.FreeTensor(lengthsLocal);
    }

    __aicore__ inline void ProcessTables(Args& args)
    {
        for (int32_t i = 0; i < loopCount; ++i) {
            int64_t tableIdx = offsetOfThisCore + i * indicesLength;
            int64_t offset = 0;
            while (offset < indicesLength) {
                int64_t remain = indicesLength - offset;
                int64_t len = remain < blockLen ? remain : blockLen;
                CopyIn(i, offset, len);
                Compute(i, tableIdx, offset, len);
                offset += blockLen;
            }
        }
    }

    GlobalTensor<indicesDType> indicesGT;
    GlobalTensor<lengthsDType> lengthsGT;
    GlobalTensor<indicesDType> permuteGT;
    GlobalTensor<lengthsDType> outputLengthsGT;
    int64_t lengthsSize;
    int64_t batchSize;
    int64_t indicesLength;
    int64_t batchNum;
    int32_t splitBaseLen;
    int64_t tailSplitIndex;
    int64_t blockLen;

    // ThisCoreLen for T
    int64_t offsetOfThisCore = 0;
    int64_t loopCount = 0;
    int64_t baseTableIdx = 0;
    indicesDType baseAddValue = 0;
    // lengths
    int64_t lengthsOffsetOfThisCore = 0;

    // Tpipe;
    TPipe* pipe;
    TQue<TPosition::VECIN, 1> indicesQueue;
    TQue<TPosition::VECIN, 1> lengthsQueue;
};

}  // namespace SelectDim1ToPermute

#endif