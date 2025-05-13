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

#ifndef PERMUTE2D_SPARSE_DATA_H
#define PERMUTE2D_SPARSE_DATA_H

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;

namespace Permute2dSparseData {

constexpr int USE_QUEUE_NUM = 2;
constexpr int QUEUE_SIZE = 64;
constexpr int UB_ALIGN = 8;
constexpr int DATA_TYPE_INT64 = 8;
constexpr int DATA_ALIGN_BYTES = 32;

struct Args {
    GM_ADDR permute;
    GM_ADDR lengths;
    GM_ADDR values;
    GM_ADDR weights;
    GM_ADDR out_lengths;
    GM_ADDR out_indices;
    GM_ADDR out_weights;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

class Permute2dSparseDataKernel {
public:
    __aicore__ inline Permute2dSparseDataKernel(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        coreNum = tilingData.coreNum;

        permuteDim0 = tilingData.permuteDim0;
        lengthsT = tilingData.lengthsT;
        lengthsB = tilingData.lengthsB;
        valuesDim = tilingData.valuesDim;

        valueDataType = tilingData.valueDataType;
        permuteDataType = tilingData.permuteDataType;
        lengthsDataType = tilingData.lengthsDataType;

        totalBatch = tilingData.totalBatch;
        baseBatchLen = tilingData.baseBatchLen;
        tailSplitIndex = tilingData.tailSplitIndex;

        ubCanUsed = tilingData.ubCanUsed;

        permute = args.permute;
        lengths = args.lengths;
        values = args.values;
        weights = args.weights;

        outLengths = args.out_lengths;
        outIndices = args.out_indices;
        outWeights = args.out_weights;
        workspace = args.workspace;

        // Calculate current core's tOffset.
        if (GetBlockIdx() < tailSplitIndex) {
            lenOfThisCore = baseBatchLen + 1;
            tOffsetOfThisCore = GetBlockIdx() * (baseBatchLen + 1);
        } else {
            lenOfThisCore = baseBatchLen;
            tOffsetOfThisCore = tailSplitIndex * (baseBatchLen + 1) + (GetBlockIdx() - tailSplitIndex) * baseBatchLen;
        }

        permuteGT.SetGlobalBuffer(permute, permuteDim0 * permuteDataType);
        lengthsGT.SetGlobalBuffer(lengths, lengthsT * lengthsB * lengthsDataType);
        valuesGT.SetGlobalBuffer(values, valuesDim * valueDataType);

        outLengthsGT.SetGlobalBuffer(outLengths, lengthsT * lengthsB * lengthsDataType);
        outIndicesGT.SetGlobalBuffer(outIndices, valuesDim * valueDataType);

        // Init pipe
        pipe.InitBuffer(inQueueX, USE_QUEUE_NUM, ubCanUsed / USE_QUEUE_NUM);
        blockLen = ubCanUsed / USE_QUEUE_NUM;

        pipe.InitBuffer(queueIn, 1, QUEUE_SIZE);
        pipe.InitBuffer(queueOut, 1, QUEUE_SIZE);
    }

    template <typename T>
    __aicore__ inline void CpGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, int64_t len)
    {
        uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(T) - alignLen;

        GlobalTensor<uint8_t> uint8Gt;
        uint8Gt.SetGlobalBuffer((__gm__ uint8_t*)gt.GetPhyAddr(), len * sizeof(T));
        LocalTensor<uint8_t> uint8Lt = lt.template ReinterpretCast<uint8_t>();

        DataCopy(uint8Lt, uint8Gt, alignLen);
        if (unAlignLen != 0) {
            const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            const DataCopyPadExtParams<uint8_t> dataCopyPadExtParams{false, 0, 0, 0};
            DataCopyPad(uint8Lt[alignLen], uint8Gt[alignLen], dataCopyExtParams, dataCopyPadExtParams);
        }
    }

    template <typename T>
    __aicore__ inline void CpLocal2Gm(const GlobalTensor<T>& gt, const LocalTensor<T>& lt, int64_t len)
    {
        uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(T) - alignLen;

        GlobalTensor<uint8_t> uint8Gt;
        uint8Gt.SetGlobalBuffer((__gm__ uint8_t*)gt.GetPhyAddr(), len * sizeof(T));
        LocalTensor<uint8_t> uint8Lt = lt.template ReinterpretCast<uint8_t>();

        DataCopy(uint8Gt, uint8Lt, alignLen);
        if (unAlignLen != 0) {
            const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            DataCopyPad(uint8Gt[alignLen], uint8Lt[alignLen], dataCopyExtParams);
        }
    }

    __aicore__ void CalculateOffsets()
    {
        offsetPtr = (__gm__ int64_t*)workspace;
        GlobalTensor<int64_t> offsetGT;
        offsetGT.SetGlobalBuffer((__gm__ int64_t*)offsetPtr, (lengthsT + 1) * UB_ALIGN * UB_ALIGN);
        if (lengthsDataType == DATA_TYPE_INT64) {
            __gm__ int64_t* lengthsPtr = (__gm__ int64_t*)lengths;
            for (int64_t i = tOffsetOfThisCore; i < lenOfThisCore + tOffsetOfThisCore; i++) {
                int64_t offsetT = 0;
                for (int64_t j = 0; j < lengthsB; j++) {
                    offsetT += *(lengthsPtr  + i * lengthsB + j);
                }
                offsetGT.SetValue(i * UB_ALIGN, offsetT);
            }
        } else {
            __gm__ int32_t* lengthsPtr = (__gm__ int32_t*)lengths;
            for (int64_t i = tOffsetOfThisCore; i < lenOfThisCore + tOffsetOfThisCore; i++) {
                int64_t offsetT = 0;
                for (int64_t j = 0; j < lengthsB; j++) {
                    offsetT += *(lengthsPtr  + i * lengthsB + j);
                }
                offsetGT.SetValue(i * UB_ALIGN, offsetT);
            }
        }
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                AscendC::DcciDst::CACHELINE_OUT>(offsetGT);
    }

    __aicore__ void PermuteLengths()
    {
        permutePtr = (__gm__ int32_t*)permute;
        for (int64_t i = tOffsetOfThisCore; i < lenOfThisCore + tOffsetOfThisCore; i++) {
            int64_t ToffsetThisIndex = *(permutePtr + i);
            int64_t ToffsetNextIndex = *(permutePtr + i) + 1;

            int64_t lengthsStartIndex = ToffsetThisIndex * lengthsB * lengthsDataType;
            int64_t lengthsEndIndex = ToffsetNextIndex * lengthsB * lengthsDataType;

            int64_t outStartIndex = i * lengthsB * lengthsDataType;
            int64_t outEndIndex = (i + 1) * lengthsB * lengthsDataType;
            int64_t totalLen = lengthsEndIndex - lengthsStartIndex;
            int64_t remainLen = totalLen;
            while (remainLen > 0) {
                int64_t thisLen = blockLen;
                if (remainLen < blockLen) {
                    thisLen = remainLen;
                }
                LocalTensor<uint8_t> inputTensor = inQueueX.AllocTensor<uint8_t>();

                CpGm2Local(inputTensor, lengthsGT[lengthsStartIndex], thisLen);
                inQueueX.EnQue(inputTensor);
                LocalTensor<uint8_t> outPutTensor = inQueueX.DeQue<uint8_t>();

                CpLocal2Gm(outLengthsGT[outStartIndex], outPutTensor, thisLen);

                outStartIndex += thisLen;
                lengthsStartIndex += thisLen;
                inQueueX.FreeTensor(outPutTensor);
                remainLen = remainLen - thisLen;
            }
        }
    }

    __aicore__ void PermuteValues()
    {
        int64_t outValueOffset = 0;
        int64_t currentT = 0;
        for (int64_t i = 0; i < permuteDim0; i++) {
            currentT = *(permutePtr + i);
            int64_t tLen = *(totalOffsetPtr + (currentT + 1) * UB_ALIGN) - *(totalOffsetPtr + currentT * UB_ALIGN);
            int64_t baseCoreLen = tLen / coreNum;
            int64_t tailLen = tLen % coreNum;

            // calculate current core permute values offset
            if (GetBlockIdx() < tailLen) {
                valueLenOfThisCore = baseCoreLen + 1;
                offsetOfThisCore = GetBlockIdx() * (baseCoreLen + 1);
            } else {
                valueLenOfThisCore = baseCoreLen;
                offsetOfThisCore = tailLen * (baseCoreLen + 1) + (GetBlockIdx() - tailLen) * baseCoreLen;
            }

            int64_t startIndex = *(totalOffsetPtr + currentT * UB_ALIGN);
            int64_t endIndex = *(totalOffsetPtr + (currentT + 1) * UB_ALIGN);

            int64_t valuesStartIndex = (startIndex + offsetOfThisCore) * valueDataType;
            int64_t outValueStartIndex = (outValueOffset + offsetOfThisCore) * valueDataType;

            int64_t remainLen =  valueLenOfThisCore * valueDataType;
            while (remainLen > 0) {
                int64_t thisLen = blockLen;
                if (remainLen < blockLen) {
                    thisLen = remainLen;
                }
                LocalTensor<uint8_t> inputTensor = inQueueX.AllocTensor<uint8_t>();

                CpGm2Local(inputTensor, valuesGT[valuesStartIndex], thisLen);
                inQueueX.EnQue(inputTensor);
                LocalTensor<uint8_t> outPutTensor = inQueueX.DeQue<uint8_t>();

                CpLocal2Gm(outIndicesGT[outValueStartIndex], outPutTensor, thisLen);

                outValueStartIndex += thisLen;
                valuesStartIndex += thisLen;
                inQueueX.FreeTensor(outPutTensor);
                remainLen = remainLen - thisLen;
            }
            outValueOffset+=tLen;
        }
    }

    __aicore__ void Compute()
    {
        CalculateOffsets();
        pipe_barrier(PIPE_ALL);
        SyncAll();
        totalOffsetPtr = (__gm__ int64_t*)workspace + (lengthsT + 1) * UB_ALIGN +
            GetBlockIdx() * (lengthsT + 1) * UB_ALIGN;
        *(totalOffsetPtr) = 0;
        GlobalTensor<int64_t> offsetGt;
        offsetGt.SetGlobalBuffer((__gm__ int64_t*)offsetPtr, (lengthsT + 1) * UB_ALIGN * UB_ALIGN);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                AscendC::DcciDst::CACHELINE_OUT>(offsetGt);

        for (int64_t i = 1; i < lengthsT + 1; i++) {
            *(totalOffsetPtr + i * UB_ALIGN) = *(totalOffsetPtr + (i - 1) * UB_ALIGN) +
                offsetGt.GetValue((i - 1) * UB_ALIGN);
        }
        PermuteLengths();
        PermuteValues();
    }

private:
    // GM_ADDR
    GM_ADDR permute;
    GM_ADDR lengths;
    GM_ADDR values;
    GM_ADDR weights;
    GM_ADDR outLengths;
    GM_ADDR outIndices;
    GM_ADDR outWeights;
    GM_ADDR workspace;

    // Shape
    int64_t permuteDim0;
    int64_t lengthsT;
    int64_t lengthsB;
    int64_t valuesDim;

    // DataType
    int64_t valueDataType;
    int64_t permuteDataType;
    int64_t lengthsDataType;

    // Tiling
    int64_t totalBatch;
    int64_t baseBatchLen;
    int64_t tailSplitIndex;
    size_t coreNum;

    // Ub
    int64_t ubCanUsed;
    int64_t blockLen;

    // ThisCoreLen for T
    int64_t lenOfThisCore;
    int64_t tOffsetOfThisCore;

    // ThisCoreLen for B
    int64_t valueLenOfThisCore;
    int64_t offsetOfThisCore;

    // Tpipe
    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, USE_QUEUE_NUM> inQueueX;

    // ThisCoreAddr
    GlobalTensor<uint8_t> permuteGT;
    GlobalTensor<uint8_t> lengthsGT;
    GlobalTensor<uint8_t> valuesGT;
    GlobalTensor<uint8_t> outLengthsGT;
    GlobalTensor<uint8_t> outIndicesGT;

    __gm__ int64_t* offsetPtr;
    __gm__ int32_t* permutePtr;
    __gm__ int64_t* totalOffsetPtr;

    TQue<TPosition::VECIN, 1> queueIn;
    TQue<TPosition::VECOUT, 1> queueOut;
};
}  // namespace Permute2dSparseData
#endif