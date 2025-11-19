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
constexpr int DATA_ALIGN_BYTES = 32;

struct Args {
    GM_ADDR permute;
    GM_ADDR lengths;
    GM_ADDR values;
    GM_ADDR weights;
    GM_ADDR totalOffset;
    GM_ADDR lengthsOffset;
    GM_ADDR permutedLengthsOffset;
    GM_ADDR outLengths;
    GM_ADDR outValues;
    GM_ADDR outWeights;
    GM_ADDR workspace;
    GM_ADDR tiling;
};


template <typename LType, typename VType>
class Permute2dSparseDataKernel {
public:
    __aicore__ inline Permute2dSparseDataKernel(Args& args, TPipe* pipePtr)
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
        totalOffset = args.totalOffset;
        lengthsOffset = args.lengthsOffset;
        permutedLengthsOffset = args.permutedLengthsOffset;
        enableWeights = tilingData.enableWeights;
        enableTotalOffset = tilingData.enableTotalOffset;

        outLengths = args.outLengths;
        outValues = args.outValues;
        outWeights = args.outWeights;
        workspace = args.workspace;

        // 计算分核
        if (GetBlockIdx() < tailSplitIndex) {
            lenOfThisCore = baseBatchLen + 1;
            tOffsetOfThisCore = GetBlockIdx() * (baseBatchLen + 1);
        } else {
            lenOfThisCore = baseBatchLen;
            tOffsetOfThisCore = tailSplitIndex * (baseBatchLen + 1) + (GetBlockIdx() - tailSplitIndex) * baseBatchLen;
        }

        lengthsGT.SetGlobalBuffer(lengths, lengthsT * lengthsB * sizeof(LType));
        valuesGT.SetGlobalBuffer(values, valuesDim * sizeof(VType));

        outLengthsGT.SetGlobalBuffer(outLengths, permuteDim0 * lengthsB * sizeof(LType));
        outValuesGT.SetGlobalBuffer(outValues, valuesOutDim * sizeof(VType));

        if (enableWeights) {
            weightsGT.SetGlobalBuffer(weights, valuesDim * sizeof(float));
            outWeightsGT.SetGlobalBuffer(outWeights, valuesOutDim * sizeof(float));
        }
        if (enableTotalOffset) {
            totalOffsetGT.SetGlobalBuffer(totalOffset, lengthsT * sizeof(LType));
        } else {
            lengthsOffsetGT.SetGlobalBuffer(lengthsOffset, lengthsT * lengthsB * sizeof(LType));
            permutedLengthsOffsetGT.SetGlobalBuffer(permutedLengthsOffset, permuteDim0 * lengthsB * sizeof(LType));
        }
        pipe = pipePtr;
        pipe->InitBuffer(inQueueX, USE_QUEUE_NUM, ubCanUsed / USE_QUEUE_NUM);
        blockLen = ubCanUsed / USE_QUEUE_NUM;
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
        totalOffsetPtr = (__gm__ LType*)totalOffset;
        for (int64_t i = 0; i < permuteDim0; i++) {
            currentT = *(permutePtr + i);
            int64_t startIndex = static_cast<int64_t>(*(totalOffsetPtr + currentT));
            int64_t endIndex = static_cast<int64_t>(*(totalOffsetPtr + currentT + 1));
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

    // 在PermuteDataLine函数中，不再有外层循环，直接处理分配给当前core的行
    __aicore__ void PermuteDataLine(GlobalTensor<uint8_t> dstGT, GlobalTensor<uint8_t> srcGT, uint8_t datasize)
    {
        // 初始化指针
        permutePtr = (__gm__ int32_t*)permute;
        lengthsOffsetPtr = (__gm__ LType*)lengthsOffset;
        permutedLengthsOffsetPtr = (__gm__ LType*)permutedLengthsOffset;
        // lenOfThisCore 和 tOffsetOfThisCore 已在构造函数中计算好（165-172行）
        // lenOfThisCore: 当前core负责的行数
        // tOffsetOfThisCore: 当前core负责的第一行在输出中的索引
        
        // 遍历当前core负责的每一行
        for (int64_t i = tOffsetOfThisCore; i < lenOfThisCore + tOffsetOfThisCore; i++) {
            // 通过permute数组找到对应的输入T维度
            int64_t ToffsetThisIndex = *(permutePtr + i);
            // 计算输入数据的起始和结束位置
            int64_t startIndex = static_cast<int64_t>(*(lengthsOffsetPtr + ToffsetThisIndex));
            int64_t endIndex = static_cast<int64_t>(*(lengthsOffsetPtr + ToffsetThisIndex + 1));
            int64_t tLen = endIndex - startIndex;
            
            // 将数据长度转换为字节数
            int64_t valuesStartIndex = startIndex * datasize;
            int64_t outValueStartIndex = *(permutedLengthsOffsetPtr + i) * datasize;
            int64_t remainLen = tLen * datasize;

            // 分块拷贝数据（因为数据可能大于UB空间）
            while (remainLen > 0) {
                // 每次拷贝的数据量不超过blockLen
                int64_t copyLen = (remainLen < blockLen) ? remainLen : blockLen;
                
                // 分配UB空间
                LocalTensor<uint8_t> inputTensor = inQueueX.AllocTensor<uint8_t>();
                
                // GM -> UB: 从输入地址拷贝数据到UB
                CpGm2Local(inputTensor, srcGT[valuesStartIndex], copyLen);
                inQueueX.EnQue(inputTensor);
                
                // 从队列中取出数据（双缓冲机制）
                LocalTensor<uint8_t> outputTensor = inQueueX.DeQue<uint8_t>();
                
                // UB -> GM: 从UB拷贝数据到输出地址
                CpLocal2Gm(dstGT[outValueStartIndex], outputTensor, copyLen);
                
                // 更新偏移量
                valuesStartIndex += copyLen;
                outValueStartIndex += copyLen;
                remainLen -= copyLen;
                
                // 释放UB空间
                inQueueX.FreeTensor(outputTensor);
            }
        }
    }

    __aicore__ void ComputeAll()
    {
        PermuteLengths();
        PermuteData(outValuesGT, valuesGT, sizeof(VType));
        if (enableWeights) {
            PermuteData(outWeightsGT, weightsGT, sizeof(float));
        }
    }

    __aicore__ inline void ComputeData()
    {
        PermuteDataLine(outValuesGT, valuesGT, sizeof(VType));
        if (enableWeights) {
            PermuteDataLine(outWeightsGT, weightsGT, sizeof(float));
        }
    }

private:
    // GM_ADDR
    GM_ADDR permute;
    GM_ADDR lengths;
    GM_ADDR values;
    GM_ADDR weights;
    GM_ADDR totalOffset;
    GM_ADDR lengthsOffset;
    GM_ADDR permutedLengthsOffset;
    GM_ADDR outLengths;
    GM_ADDR outValues;
    GM_ADDR outWeights;
    GM_ADDR workspace;

    // Shape
    int64_t permuteDim0 = 0;
    int64_t lengthsT = 0;
    int64_t lengthsB = 0;
    int64_t valuesDim = 0;
    int64_t valuesOutDim = 0;
    bool enableWeights = false;
    bool enableTotalOffset = false;

    // Tiling
    int64_t totalBatch = 0;
    int64_t baseBatchLen = 0;
    int64_t tailSplitIndex = 0;
    size_t coreNum = 0;

    // Ub
    int64_t ubCanUsed = 0;
    int64_t blockLen = 0;

    // ThisCoreLen for T
    int64_t lenOfThisCore = 0;
    int64_t tOffsetOfThisCore = 0;

    // ThisCoreLen for B
    int64_t valueLenOfThisCore = 0;
    int64_t offsetOfThisCore = 0;

    // Tpipe
    TPipe* pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, USE_QUEUE_NUM> inQueueX;

    // ThisCoreAddr
    GlobalTensor<uint8_t> lengthsGT;
    GlobalTensor<uint8_t> valuesGT;
    GlobalTensor<uint8_t> weightsGT;
    GlobalTensor<uint8_t> totalOffsetGT;
    GlobalTensor<uint8_t> lengthsOffsetGT;
    GlobalTensor<uint8_t> permutedLengthsOffsetGT;
    GlobalTensor<uint8_t> outLengthsGT;
    GlobalTensor<uint8_t> outValuesGT;
    GlobalTensor<uint8_t> outWeightsGT;

    __gm__ int32_t* permutePtr;
    __gm__ LType* totalOffsetPtr;
    __gm__ LType* lengthsOffsetPtr;
    __gm__ LType* permutedLengthsOffsetPtr;
};
}  // namespace Permute2dSparseData
#endif
