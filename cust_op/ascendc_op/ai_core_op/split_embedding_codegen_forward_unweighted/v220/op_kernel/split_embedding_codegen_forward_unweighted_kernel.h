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

#ifndef SPLIT_EMBEDDING_CODEGEN_FORWARD_UNWEIGHTED_KERNEL_FUN_H
#define SPLIT_EMBEDDING_CODEGEN_FORWARD_UNWEIGHTED_KERNEL_FUN_H

#include <cstdint>

#include "common.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace SplitEmbeddingCodegenForwardUnweighted {

class SplitEmbeddingCodegenForwardUnweightedKernel {
public:
    __aicore__ inline SplitEmbeddingCodegenForwardUnweightedKernel(Args& args, TPipe* pipeIn)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        pipe = pipeIn;
        InitAddr(args);

        // Shape
        devWeightsDim0 = tilingData.devWeightsDim0;
        weightsOffsetsDim0 = tilingData.weightsOffsetsDim0;
        dOffsetsDim0 = tilingData.dOffsetsDim0;
        indicesDim0 = tilingData.indicesDim0;
        offsetsDim0 = tilingData.offsetsDim0;
        outDim0 = tilingData.outDim0;
        outDim1 = tilingData.outDim1;
        maxD = tilingData.maxD;
        alignMaxD = (maxD / FLOAT_ALIGNMENT + 1) * FLOAT_ALIGNMENT;
        enableHash = tilingData.enableHash;
        batchs = (offsetsDim0 - 1) / weightsOffsetsDim0;

        // DataType
        bytesOfDataType = sizeof(float);
        offsetDataType = DATA_TYPE_INT64;

        // Tiling
        splitBaseLen = tilingData.splitBaseLen;
        tailSplitIndex = tilingData.tailSplitIndex;

        // func
        poolMode = tilingData.poolMode;
        // ThisCoreLen
        if (GetBlockIdx() >= tailSplitIndex) {
            lenOfThisCore = splitBaseLen;
            offsetOfThisCore = tailSplitIndex * (splitBaseLen + 1) + (GetBlockIdx() - tailSplitIndex) * splitBaseLen;
        } else {
            lenOfThisCore = splitBaseLen + 1;
            offsetOfThisCore = GetBlockIdx() * (splitBaseLen + 1);
        }
        // Ub
        ubCanUsed = tilingData.ubCanUsed - offsetDataType * MAX_INDICES_ONE_BLOCK;
        blockLen = ubCanUsed / USE_QUEUE_NUM / bytesOfDataType;
        blockLen = blockLen / FLOAT_ALIGNMENT * FLOAT_ALIGNMENT;
        
        // Init globalbuffer
        devWeightsGT.SetGlobalBuffer((__gm__ float*)devWeights, devWeightsDim0);
        if (enableHash) {
            indicesGT.SetGlobalBuffer((__gm__ int64_t*)hashIndices, indicesDim0);
        } else {
            indicesGT.SetGlobalBuffer((__gm__ int64_t*)indices, indicesDim0);
        }
        offsetGT.SetGlobalBuffer((__gm__ int64_t*)offsets, offsetsDim0);
        dOffsetGT.SetGlobalBuffer((__gm__ int32_t*)dOffsets, dOffsetsDim0);
        weightOffsetGT.SetGlobalBuffer((__gm__ int64_t*)weightsOffsets, weightsOffsetsDim0);
        offsetPerKeyGT.SetGlobalBuffer((__gm__ int64_t*)offsetPerKey, indicesDim0);

        outGT.SetGlobalBuffer((__gm__ float*)out, outDim0 * outDim1);

        ASCENDC_ASSERT(offsetGT.GetValue(offsetsDim0 - 1) == indicesDim0,
                       "The last element in offsets must be equal to indices size");
        // Init pipe
        pipe->InitBuffer(queIn, 1, blockLen * sizeof(float));
        pipe->InitBuffer(queOut, 1, blockLen * sizeof(float));
        pipe->InitBuffer(queIndices, 1, MAX_INDICES_ONE_BLOCK * sizeof(int64_t));
    }

    __aicore__ inline void InitAddr(const Args &args)
    {
        // ADDR
        devWeights = args.devWeights;
        weightsPlacements = args.weightsPlacements;
        weightsOffsets = args.weightsOffsets;
        dOffsets = args.dOffsets;
        indices = args.indices;
        offsets = args.offsets;
        hashIndices = args.hashIndices;
        offsetPerKey = args.offsetPerKey;
        out = args.out;
        workspace = args.workspace;
    }

    template <typename T>
    __aicore__ inline void CpGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, int64_t len)
    {
        uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(T) - alignLen;

        DataCopy(lt, gt, alignLen / sizeof(T));
        if (unAlignLen != 0) {
            const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            const DataCopyPadExtParams<T> dataCopyPadExtParams{false, 0, 0, 0};
            DataCopyPad(lt[alignLen / sizeof(T)], gt[alignLen / sizeof(T)], dataCopyExtParams, dataCopyPadExtParams);
        }
    }

    template <typename T>
    __aicore__ inline void CpLocal2Gm(const GlobalTensor<T>& gt, const LocalTensor<T>& lt, int64_t len)
    {
        uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(T) - alignLen;
        DataCopy(gt, lt, alignLen / sizeof(T));
        if (unAlignLen != 0) {
            const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            DataCopyPad(gt[alignLen / sizeof(T)], lt[alignLen / sizeof(T)], dataCopyExtParams);
        }
    }

    __aicore__ inline void CopyInNormal(int64_t startIndices, int64_t thisLen, int64_t embedDim,
        int64_t thisWeightOffset)
    {
        LocalTensor<int64_t> indicesLt = queIndices.AllocTensor<int64_t>();
        LocalTensor<float> inputLt = queIn.AllocTensor<float>();
        CpGm2Local(indicesLt, indicesGT[startIndices], thisLen);
        queIndices.EnQue(indicesLt);
        indicesLt = queIndices.DeQue<int64_t>();
        for (int64_t i = 0; i < thisLen; ++i) {
            int64_t thisIndForThisTable = indicesLt.GetValue(i);
            int64_t indWeightOffset = thisIndForThisTable * embedDim + thisWeightOffset;
            CpGm2Local(inputLt[i * alignMaxD], devWeightsGT[indWeightOffset], embedDim);
        }
        queIndices.FreeTensor(indicesLt);
        queIn.EnQue(inputLt);
    }

    __aicore__ inline void CopyOutNoPooling(int64_t thisLen, int64_t startIndices)
    {
        LocalTensor<float> inputLt = queIn.DeQue<float>();
        LocalTensor<float> outLt = queOut.AllocTensor<float>();

        int64_t allLen = thisLen * maxD;
        DataCopy(outLt, inputLt, allLen);

        queOut.EnQue(outLt);
        outLt = queOut.DeQue<float>();

        CpLocal2Gm(outGT[startIndices * maxD], outLt, allLen);

        queIn.FreeTensor(inputLt);
        queOut.FreeTensor(outLt);
    }

    __aicore__ inline void CopyOutNoPoolingPad(int64_t thisLen, int64_t startIndices)
    {
        LocalTensor<float> inputLt = queIn.DeQue<float>();
        LocalTensor<float> outLt = queOut.AllocTensor<float>();

        int64_t allLen = thisLen * alignMaxD;
        DataCopy(outLt, inputLt, allLen); // datacopy len should align to 32B

        queOut.EnQue(outLt);
        outLt = queOut.DeQue<float>();

        for (int i = 0; i < thisLen; i++) {
            CpLocal2Gm(outGT[(startIndices + i) * maxD], outLt[i * alignMaxD], maxD);
        }
        queIn.FreeTensor(inputLt);
        queOut.FreeTensor(outLt);
    }

    __aicore__ inline void CopyOutWithPooling(int64_t outOffset, int64_t embedDim)
    {
        auto outLt = queOut.DeQue<float>();
        SetAtomicAdd<float>();
        CpLocal2Gm(outGT[outOffset], outLt, embedDim);
        SetAtomicNone();
        queOut.FreeTensor(outLt);
    }

    __aicore__ inline void Pooling(float meanLen, int64_t thisLen, int64_t embedDim)
    {
        LocalTensor<float> outLt = queOut.AllocTensor<float>();
        LocalTensor<float> inputLt = queIn.DeQue<float>();

        Duplicate<float>(outLt, 0, alignMaxD);
        for (int64_t i = 0; i < thisLen; i++) {
            Add(outLt, outLt, inputLt[i * alignMaxD], embedDim);
        }

        if (poolMode == static_cast<int64_t>(PoolingMode::MEAN)) {
            Muls<float>(outLt, outLt, meanLen, embedDim);
        }
        queIn.FreeTensor(inputLt);
        queOut.EnQue(outLt);
    }

    __aicore__ inline void ProcessWithPooling(int64_t remain, int64_t startIndices, int64_t embedDim,
                                              int64_t thisWeightOffset, int64_t outOffset)
    {
        float meanLen = static_cast<float>(1) / static_cast<float>(remain);
        int64_t thisLen = remain;
        while (remain > 0) {
            if (thisLen > indicesNumOneBlock) {
                thisLen = indicesNumOneBlock;
            }
            remain -= thisLen;

            // copyIn
            CopyInNormal(startIndices, thisLen, embedDim, thisWeightOffset);
            // compute
            Pooling(meanLen, thisLen, embedDim);
            // copyout
            CopyOutWithPooling(outOffset, embedDim);

            startIndices = startIndices + thisLen;
            thisLen = remain;
        }
    }

    __aicore__ inline void ProcessNoPooling(int64_t remain, int64_t startIndices, int64_t thisWeightOffset)
    {
        int64_t thisLen = remain;
        while (remain > 0) {
            if (thisLen > indicesNumOneBlock) {
                thisLen = indicesNumOneBlock;
            }
            remain -= thisLen;
            CopyInNormal(startIndices, thisLen, maxD, thisWeightOffset);
            if (alignMaxD == maxD) {
                CopyOutNoPooling(thisLen, startIndices);
            } else {
                CopyOutNoPoolingPad(thisLen, startIndices);
            }

            startIndices = startIndices + thisLen;
            thisLen = remain;
        }
    }

    __aicore__ inline void Scheduler(const int64_t &totalLen, int64_t &offsetLen, int64_t &calcLen)
    {
        splitBaseLen = totalLen / GetBlockNum();
        tailSplitIndex = totalLen % GetBlockNum();
        if (GetBlockIdx() >= tailSplitIndex) {
            calcLen = splitBaseLen;
            offsetLen = tailSplitIndex * (splitBaseLen + 1) + (GetBlockIdx() -  tailSplitIndex) * splitBaseLen;
        } else {
            calcLen = splitBaseLen + 1;
            offsetLen = GetBlockIdx() * (splitBaseLen + 1);
        }
    }

    __aicore__ inline void ComputeNoPooling()
    {
        int64_t lastIndices = 0;
        int64_t thisTableLen = 0;
        for (int64_t i = 1; i <= weightsOffsetsDim0; i++) {
            if (offsetPerKeyGT.GetValue(i) != lastIndices) {
                Scheduler(offsetPerKeyGT.GetValue(i) - lastIndices, offsetOfThisCore, thisTableLen);
                if (thisTableLen > 0) {
                    int64_t thisTableOffset = offsetOfThisCore + lastIndices;
                    int64_t thisWeightOffset = weightOffsetGT.GetValue(i - 1);
                    ProcessNoPooling(thisTableLen, thisTableOffset, thisWeightOffset);
                }
                lastIndices = offsetPerKeyGT.GetValue(i);
            }
        }
    }

    __aicore__ inline void ComputeWithPooling()
    {
        if (lenOfThisCore == 0) {
            return;
        }

        for (int64_t loop = 0; loop < lenOfThisCore; loop++) {
            int64_t i = (offsetOfThisCore + loop) / weightsOffsetsDim0;
            int64_t j = (offsetOfThisCore + loop) % weightsOffsetsDim0;
            int64_t thisOffsetIndex = j * batchs + i;
            int64_t startIndices = offsetGT.GetValue(thisOffsetIndex);
            int64_t endIndices = offsetGT.GetValue(thisOffsetIndex + 1);
            int32_t thisLen = endIndices - startIndices;

            if (thisLen <= 0) {
                continue;
            }

            // dataCopy In params
            int64_t tableIndex = thisOffsetIndex / batchs;
            int64_t thisWeightOffset = weightOffsetGT.GetValue(tableIndex);
            // dataCopy Out params
            int64_t outBatchInd = thisOffsetIndex % outDim0;
            int64_t outEmbedOffset = dOffsetGT.GetValue(tableIndex);
            int64_t outOffset = outBatchInd * outDim1 + outEmbedOffset;
            int64_t embedDim = dOffsetGT.GetValue(tableIndex + 1) - dOffsetGT.GetValue(tableIndex);
            ProcessWithPooling(thisLen, startIndices, embedDim, thisWeightOffset, outOffset);
        }
    }

    __aicore__ inline void Compute()
    {
        indicesNumOneBlock = blockLen / alignMaxD;
        if (indicesNumOneBlock >= MAX_INDICES_ONE_BLOCK) {
            indicesNumOneBlock = MAX_INDICES_ONE_BLOCK;
        }
        if (poolMode == static_cast<int64_t>(PoolingMode::NONE)) {
            ComputeNoPooling();
        } else {
            ComputeWithPooling();
        }
    }

private:
    // // GM_ADDR
    GM_ADDR devWeights;
    GM_ADDR weightsPlacements;
    GM_ADDR weightsOffsets;
    GM_ADDR dOffsets;
    GM_ADDR indices;
    GM_ADDR offsets;
    GM_ADDR hashIndices;
    GM_ADDR offsetPerKey;
    GM_ADDR out;
    GM_ADDR workspace;

    // Shape
    int64_t devWeightsDim0;
    int64_t weightsOffsetsDim0;
    int64_t dOffsetsDim0;
    int64_t indicesDim0;
    int64_t offsetsDim0;
    int64_t outDim0;
    int64_t outDim1;
    int64_t maxD;
    int64_t alignMaxD;
    int64_t batchs;
    bool enableHash;

    // // DataType
    int64_t bytesOfDataType;
    int64_t offsetDataType;

    // Tiling
    int64_t splitBaseLen;
    int64_t tailSplitIndex;
    int32_t blockDim;
    int64_t indicesNumOneBlock;

    // Ub
    int64_t ubCanUsed;
    int64_t blockLen;

    // func change
    int64_t poolMode;

    // ThisCoreLen
    int64_t lenOfThisCore;
    int64_t offsetOfThisCore;

    // dynamic
    int64_t blockEmbNum;
    bool isDynamic;

    // Tpipe
    TPipe* pipe;
    TQue<TPosition::VECIN, 1> queIn;
    TQue<TPosition::VECOUT, 1> queOut;
    TQue<TPosition::VECIN, 1> queIndices;
    
    // ThisCoreAddr
    GlobalTensor<float> devWeightsGT;
    GlobalTensor<float> outGT;
    GlobalTensor<int64_t> indicesGT;
    GlobalTensor<int64_t> offsetGT;
    GlobalTensor<int32_t> dOffsetGT;
    GlobalTensor<int64_t> weightOffsetGT;
    GlobalTensor<int64_t> offsetPerKeyGT;
};
}  // namespace SplitEmbeddingCodegenForwardUnweighted
#endif
