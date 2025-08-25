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

template <typename LType, typename VType>
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
        valuesOutDim = tilingData.valuesOutDim;

        totalBatch = tilingData.totalBatch;
        baseBatchLen = tilingData.baseBatchLen;
        tailSplitIndex = tilingData.tailSplitIndex;

        ubCanUsed = tilingData.ubCanUsed;

        permute = args.permute;
        lengths = args.lengths;
        values = args.values;
        weights = args.weights;

        enableWeights = tilingData.enableWeights;

        outLengths = args.out_lengths;
        outIndices = args.out_indices;
        outWeights = args.out_weights;
        workspace = args.workspace;

        // 计算分核
        if (GetBlockIdx() < tailSplitIndex) {
            lenOfThisCore = baseBatchLen + 1;
            tOffsetOfThisCore = GetBlockIdx() * (baseBatchLen + 1);
        } else {
            lenOfThisCore = baseBatchLen;
            tOffsetOfThisCore = tailSplitIndex * (baseBatchLen + 1) + (GetBlockIdx() - tailSplitIndex) * baseBatchLen;
        }

        permuteGT.SetGlobalBuffer(permute, permuteDim0 * sizeof(int32_t));
        lengthsGT.SetGlobalBuffer(lengths, lengthsT * lengthsB * sizeof(LType));
        valuesGT.SetGlobalBuffer(values, valuesDim * sizeof(VType));

        outLengthsGT.SetGlobalBuffer(outLengths, lengthsT * lengthsB * sizeof(LType));
        outIndicesGT.SetGlobalBuffer(outIndices, valuesOutDim * sizeof(VType));

        if (enableWeights) {
            weightsGT.SetGlobalBuffer(weights, valuesDim * sizeof(float));
            outWeightsGT.SetGlobalBuffer(outWeights, valuesOutDim * sizeof(float));
        }

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

    __aicore__ void CalculateLengthSum()
    {
        offsetPtr = (__gm__ int64_t*)workspace;
        GlobalTensor<int64_t> offsetGT;
        // 创建[T+1, UB_ALIGN]大小的workspace
        offsetGT.SetGlobalBuffer((__gm__ int64_t*)offsetPtr, (lengthsT + 1) * UB_ALIGN * sizeof(int64_t));
        __gm__ LType* lengthsPtr = (__gm__ LType*)lengths;
        // 计算分核信息, 当前core计算lengths[T, B]中的哪几行之和
        int64_t rows;
        int64_t start;
        int64_t tailIndex = (lengthsT % coreNum);

        if (GetBlockIdx() < tailIndex) {
            rows = lengthsT / coreNum + 1;
            start = GetBlockIdx() * rows;
        } else {
            rows = lengthsT / coreNum;
            start = tailIndex * (rows + 1) + (GetBlockIdx() - tailIndex) * rows;
        }
        // 暂不考虑lengthsT过长情况, 默认UB可以装下lengthsT * sizeof(int64_t)
        for (int64_t i = start; i < start + rows; i++) {
            int64_t lineSum = 0;
            int64_t offset = i * lengthsB;
            for (int64_t j = 0; j < lengthsB; j++) {
                lineSum += *(lengthsPtr + offset + j);
            }
            // 竖着写入,保证GT首地址32位对齐。否则DataCacheCleanAndInvalid同步失效
            offsetGT.SetValue(i * UB_ALIGN, lineSum);
        }
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(offsetGT);
        pipe_barrier(PIPE_ALL);
        SyncAll();
    }

    __aicore__ void CalculateOffsets()
    {
        totalOffsetPtr = (__gm__ int64_t*)workspace + (1 + GetBlockIdx()) * (lengthsT + 1) * UB_ALIGN;
        *(totalOffsetPtr) = 0;
        GlobalTensor<int64_t> offsetGt;
        offsetGt.SetGlobalBuffer((__gm__ int64_t*)offsetPtr, (lengthsT + 1) * UB_ALIGN * UB_ALIGN);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(offsetGt);

        for (int64_t i = 1; i < lengthsT + 1; i++) {
            *(totalOffsetPtr + i * UB_ALIGN) = *(totalOffsetPtr + (i - 1) * UB_ALIGN) +
                                               offsetGt.GetValue((i - 1) * UB_ALIGN);
        }
    }

    __aicore__ void PermuteLengths()
    {
        permutePtr = (__gm__ int32_t*)permute;
        int64_t totalLen = lengthsB * sizeof(LType);

        for (int64_t i = tOffsetOfThisCore; i < lenOfThisCore + tOffsetOfThisCore; i++) {
            int64_t ToffsetThisIndex = *(permutePtr + i);
            int64_t lengthsStartIndex = ToffsetThisIndex * lengthsB * sizeof(LType);
            int64_t outStartIndex = i * lengthsB * sizeof(LType);

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

    __aicore__ void PermuteData(GlobalTensor<uint8_t> dstGT, GlobalTensor<uint8_t> srcGT, uint8_t datasize)
    {
        int64_t outValueOffset = 0;
        int64_t currentT = 0;
        for (int64_t i = 0; i < permuteDim0; i++) {
            currentT = *(permutePtr + i);
            int64_t startIndex = *(totalOffsetPtr + currentT * UB_ALIGN);
            int64_t endIndex = *(totalOffsetPtr + (currentT + 1) * UB_ALIGN);

            int64_t tLen = endIndex - startIndex;
            int64_t baseCoreLen = tLen / coreNum;
            int64_t tailLen = tLen % coreNum;

            // 计算当前核上处理的values起始位置、处理量
            if (GetBlockIdx() < tailLen) {
                valueLenOfThisCore = baseCoreLen + 1;
                offsetOfThisCore = GetBlockIdx() * (baseCoreLen + 1);
            } else {
                valueLenOfThisCore = baseCoreLen;
                offsetOfThisCore = tailLen * (baseCoreLen + 1) + (GetBlockIdx() - tailLen) * baseCoreLen;
            }

            int64_t valuesStartIndex = (startIndex + offsetOfThisCore) * datasize;
            int64_t outValueStartIndex = (outValueOffset + offsetOfThisCore) * datasize;

            int64_t remainLen = valueLenOfThisCore * datasize;
            while (remainLen > 0) {
                int64_t thisLen = blockLen;
                if (remainLen < blockLen) {
                    thisLen = remainLen;
                }
                LocalTensor<uint8_t> inputTensor = inQueueX.AllocTensor<uint8_t>();

                CpGm2Local(inputTensor, srcGT[valuesStartIndex], thisLen);
                inQueueX.EnQue(inputTensor);
                LocalTensor<uint8_t> outPutTensor = inQueueX.DeQue<uint8_t>();

                CpLocal2Gm(dstGT[outValueStartIndex], outPutTensor, thisLen);

                outValueStartIndex += thisLen;
                valuesStartIndex += thisLen;
                inQueueX.FreeTensor(outPutTensor);
                remainLen = remainLen - thisLen;
            }
            outValueOffset += tLen;
        }
    }

    __aicore__ void Compute()
    {
        CalculateLengthSum();
        CalculateOffsets();
        PermuteLengths();
        PermuteData(outIndicesGT, valuesGT, sizeof(VType));
        if (enableWeights) {
            PermuteData(outWeightsGT, weightsGT, sizeof(float));
        }
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
    int64_t valuesOutDim;
    bool enableWeights;

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
    int64_t weightLenOfThisCore;
    int64_t offsetOfThisCore;

    // Tpipe
    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, USE_QUEUE_NUM> inQueueX;

    // ThisCoreAddr
    GlobalTensor<uint8_t> permuteGT;
    GlobalTensor<uint8_t> lengthsGT;
    GlobalTensor<uint8_t> valuesGT;
    GlobalTensor<uint8_t> weightsGT;
    GlobalTensor<int64_t> offsetGT;
    GlobalTensor<uint8_t> outLengthsGT;
    GlobalTensor<uint8_t> outIndicesGT;
    GlobalTensor<uint8_t> outWeightsGT;

    __gm__ int64_t* offsetPtr;
    __gm__ int32_t* permutePtr;
    __gm__ int64_t* totalOffsetPtr;

    TQue<TPosition::VECIN, 1> queueIn;
    TQue<TPosition::VECOUT, 1> queueOut;
};
}  // namespace Permute2dSparseData
#endif