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

#ifndef INT_NBIT_SPLIT_EMBEDDING_NOBAG_KERNEL_H
#define INT_NBIT_SPLIT_EMBEDDING_NOBAG_KERNEL_H

#include <algorithm>
#include <type_traits>
#include "common.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace IntNBitSplitEmbeddingNobag {

// Nobag模式Kernel（支持FP8+unweighted）
// 注意：nobag模式下CUDA强制使用int32_t，因此这里也固定使用int32_t
template <typename OutputType>
class IntNBitSplitEmbeddingNobagKernel {
public:
    __aicore__ inline IntNBitSplitEmbeddingNobagKernel(Args& args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        // 数据准备：初始化所有参数
        InitShapeParams(tilingData);
        InitTiling(tilingData);
        InitGmParams(args);
        InitUbParams(tilingData);
    }

    __aicore__ inline void Compute()
    {
        // 步骤1：预加载offsetPerKey到UB（减少GM访问）
        LocalTensor<int32_t> offsetPerKeyLt = offsetPerKeyQue.AllocTensor<int32_t>();
        CpGm2Local(offsetPerKeyLt, offsetPerKeyGT, weightsOffsetsDim0 + 1);
        offsetPerKeyQue.EnQue(offsetPerKeyLt);
        offsetPerKeyLt = offsetPerKeyQue.DeQue<int32_t>();

        // Nobag模式：对每张表串行处理，对每张表内的所有indices分核
        int64_t lastIndices = 0;
        int64_t thisTableLen = 0;

        // 遍历所有表（通过offsetPerKey确定每张表的indices范围）
        for (int64_t i = 1; i <= weightsOffsetsDim0; i++) {
            int64_t thisOffsetPerKey = static_cast<int64_t>(offsetPerKeyLt.GetValue(i));
            if (thisOffsetPerKey == lastIndices) {
                continue;  // 跳过空表
            }

            // 计算当前表的分核参数
            int64_t totalIndicesForThisTable = thisOffsetPerKey - lastIndices;
            Scheduler(totalIndicesForThisTable, offsetOfThisCore, thisTableLen);

            if (thisTableLen > 0) {
                // 当前core需要处理当前表的部分indices
                int64_t thisTableOffset = offsetOfThisCore + lastIndices;
                int64_t thisWeightOffset = weightOffsetGT.GetValue(i - 1);
                int64_t thisTableIndex = i - 1;

                // 处理当前表的indices（nobag模式下，所有表的dim相同，使用maxD）
                Process(thisTableLen, thisTableOffset, maxD, thisWeightOffset, thisTableIndex);
            }

            lastIndices = thisOffsetPerKey;
        }

        offsetPerKeyQue.FreeTensor(offsetPerKeyLt);
    }

private:
    // ========== 数据准备：初始化参数 ==========

    __aicore__ inline void InitShapeParams(const IntNbitSplitEmbeddingCodegenLookupFunctionTilingData& tilingData)
    {
        devWeightsDim0 = tilingData.devWeightsDim0;
        weightsOffsetsDim0 = tilingData.weightsOffsetsDim0;
        dOffsetsDim0 = tilingData.dOffsetsDim0;
        indicesDim0 = tilingData.indicesDim0;
        offsetsDim0 = tilingData.offsetsDim0;
        outDim0 = tilingData.outDim0;
        outDim1 = tilingData.outDim1;
        maxD = tilingData.maxD;

        // FP8相关参数
        fp8ExponentBits = tilingData.fp8ExponentBits;
        fp8ExponentBias = tilingData.fp8ExponentBias;
        rowAlignment = tilingData.rowAlignment;

        // Nobag模式下所有表的dim都相同（都是maxD），预先计算对齐参数
        rowBytes = (maxD + rowAlignment - 1) / rowAlignment * rowAlignment;
        alignedRowBytes = AlignTo32(rowBytes);
        alignedEmbedDimBytes = alignedRowBytes * sizeof(float);

        // 初始化FP8参数（使用公共函数）
        InitFp8Params(fp8ExponentBits, fp8ExponentBias, fp8BodyShift, fp8Multiplier);
    }

    __aicore__ inline void InitTiling(const IntNbitSplitEmbeddingCodegenLookupFunctionTilingData& tilingData)
    {
        splitBaseLen = tilingData.splitBaseLen;
        tailSplitIndex = tilingData.tailSplitIndex;

        // 注意：nobag模式下的分核是基于indices数量，而不是bag数量
        // 这里先初始化，实际分核在Scheduler中根据每张表的indices数量重新计算
        offsetOfThisCore = 0;
        lenOfThisCore = 0;
    }

    __aicore__ inline void InitGmParams(const Args& args)
    {
        // 权重使用uint8_t类型（量化数据）
        devWeightsGT.SetGlobalBuffer((__gm__ uint8_t*)args.devWeights, devWeightsDim0);
        indicesGT.SetGlobalBuffer((__gm__ int32_t*)args.indices, indicesDim0);
        offsetGT.SetGlobalBuffer((__gm__ int32_t*)args.offsets, offsetsDim0);
        dOffsetGT.SetGlobalBuffer((__gm__ int32_t*)args.dOffsets, dOffsetsDim0);
        weightOffsetGT.SetGlobalBuffer((__gm__ int64_t*)args.weightsOffsets, weightsOffsetsDim0);
        weightsTysGT.SetGlobalBuffer((__gm__ uint8_t*)args.weightsTys, weightsOffsetsDim0);
        // offsetPerKey的维度是表数+1（最后一个是总长度）
        offsetPerKeyGT.SetGlobalBuffer((__gm__ int32_t*)args.offsetPerKey, weightsOffsetsDim0 + 1);
        outGT.SetGlobalBuffer((__gm__ OutputType*)args.out, outDim0 * outDim1);
    }

    __aicore__ inline void InitUbParams(const IntNbitSplitEmbeddingCodegenLookupFunctionTilingData& tilingData)
    {
        ubCanUsed = tilingData.ubCanUsed;

        // 步骤1：先分配queIndices（固定大小，类型为int32_t）
        int64_t queIndicesBytes = MAX_INDICES_ONE_BLOCK * sizeof(int32_t);
        int64_t remainingUb1 = ubCanUsed - queIndicesBytes;

        // 步骤2：分配offsetPerKeyQue的UB缓冲区（减少GM访问）
        // offsetPerKey的维度是表数+1（最后一个是总长度）
        int64_t offsetPerKeyBytes = (weightsOffsetsDim0 + 1) * sizeof(int32_t);
        offsetPerKeyBytes = AlignTo32(offsetPerKeyBytes);
        int64_t remainingUb2 = remainingUb1 - offsetPerKeyBytes;

        // 步骤3：计算maxRows（这些缓冲区对应的元素个数一致）
        int64_t bytesPerRow = alignedEmbedDimBytes + alignedRowBytes + alignedRowBytes + alignedRowBytes
                              + alignedRowBytes * sizeof(uint32_t);
        int64_t newqueOutBytes = 0;
        if constexpr (!std::is_same_v<OutputType, float>) {
            bytesPerRow += maxD * sizeof(OutputType);
        }
        int64_t maxRows = remainingUb2 / bytesPerRow;
        maxRows = Std::min(maxRows, MAX_INDICES_ONE_BLOCK);
        maxRows = maxRows <= 0 ? 1 : maxRows;

        // 步骤4：根据maxRows计算各个缓冲区的bytes
        queOutBytes = maxRows * alignedEmbedDimBytes;
        queInBytes = maxRows * alignedRowBytes;
        int64_t fp8MaskBytes = maxRows * alignedRowBytes;
        int64_t fp8TmpByteBytes = maxRows * alignedRowBytes;
        int64_t fp8SignUint32Bytes = maxRows * alignedRowBytes * sizeof(uint32_t);
        if constexpr (!std::is_same_v<OutputType, float>) {
            newqueOutBytes = maxRows * maxD * sizeof(OutputType);
        }

        // 步骤5：分配内存
        pipe.InitBuffer(queIndices, 1, queIndicesBytes);
        pipe.InitBuffer(queOut, 1, queOutBytes);
        pipe.InitBuffer(queIn, 1, queInBytes);
        pipe.InitBuffer(offsetPerKeyQue, 1, offsetPerKeyBytes);
        pipe.InitBuffer(fp8MaskBuf, AlignTo32(fp8MaskBytes));
        pipe.InitBuffer(fp8TmpByteBuf, AlignTo32(fp8TmpByteBytes));
        pipe.InitBuffer(fp8SignUint32Buf, AlignTo32(fp8SignUint32Bytes));
        if constexpr (!std::is_same_v<OutputType, float>) {
            pipe.InitBuffer(newqueOut, 1, newqueOutBytes);
        }

        // 计算实际能处理的indices数量（基于maxRows）
        indicesNumOneBlock = maxRows;
    }

    // ========== 分核调度 ==========

    __aicore__ inline void Scheduler(const int64_t& totalLen, int64_t& offsetLen, int64_t& calcLen)
    {
        // 根据当前表的indices总数重新计算分核
        int64_t coreNum = GetBlockNum();
        splitBaseLen = totalLen / coreNum;
        tailSplitIndex = totalLen % coreNum;

        if (GetBlockIdx() >= tailSplitIndex) {
            calcLen = splitBaseLen;
            offsetLen = tailSplitIndex * (splitBaseLen + 1) + (GetBlockIdx() - tailSplitIndex) * splitBaseLen;
        } else {
            calcLen = splitBaseLen + 1;
            offsetLen = GetBlockIdx() * (splitBaseLen + 1);
        }
    }

    // ========== CopyIn：读取权重并反量化 ==========

    __aicore__ inline void CopyInWithDequantize(
        int64_t startIndices,
        int64_t thisLen,
        int64_t embedDim,
        int64_t thisWeightOffset,
        int64_t tableIndex)
    {
        // 获取当前表的量化类型
        uint8_t weightType = weightsTysGT.GetValue(tableIndex);
        if (weightType == static_cast<uint8_t>(SparseType::FP8)) {
            CopyInFP8(startIndices, thisLen, embedDim, thisWeightOffset);
        } else {
            // 保留接口: 其他类型（INT8/INT4/INT2）的反量化
            ASCENDC_ASSERT(false, "Unsupported weight type");
        }
    }

    __aicore__ inline void CopyInFP8(
        int64_t startIndices,
        int64_t thisLen,
        int64_t embedDim,
        int64_t thisWeightOffset)
    {
        // 步骤1：先读取indices到queIndices（单独分配，不与其他缓冲区复用）
        LocalTensor<int32_t> indicesLt = queIndices.AllocTensor<int32_t>();
        CpGm2Local(indicesLt, indicesGT[startIndices], thisLen);
        queIndices.EnQue(indicesLt);
        indicesLt = queIndices.DeQue<int32_t>();

        // 步骤2：DataCopy量化数据到queIn（每行对齐，从GM读取rowBytes字节）
        LocalTensor<uint8_t> quantizedLt = queIn.AllocTensor<uint8_t>();
        Duplicate<uint8_t>(quantizedLt, 0, thisLen * alignedRowBytes);
        for (int64_t i = 0; i < thisLen; ++i) {
            int64_t thisInd = static_cast<int64_t>(indicesLt.GetValue(i));
            int64_t indWeightOffset = thisInd * rowBytes + thisWeightOffset;
            // 从GM读取rowBytes字节（实际存储长度），写入UB的alignedRowBytes对齐位置
            CpGm2Local(quantizedLt[i * alignedRowBytes], devWeightsGT[indWeightOffset], rowBytes);
        }
        queIn.EnQue(quantizedLt);
        quantizedLt = queIn.DeQue<uint8_t>();

        // 步骤3：手写FP8 → FP32转换，直接写入queOut（消除冗余拷贝）
        LocalTensor<float> floatLt = queOut.AllocTensor<float>();
        int64_t totalElements = thisLen * alignedRowBytes;  // 包含padding
        FP8U8ToFP32Bitpack(quantizedLt, floatLt, totalElements, fp8BodyShift,
                           fp8Multiplier, fp8MaskBuf, fp8TmpByteBuf, fp8SignUint32Buf);

        queOut.EnQue(floatLt);
        queIn.FreeTensor(quantizedLt);
        queIndices.FreeTensor(indicesLt);
    }

    // ========== CopyOut：输出结果 ==========

    template <bool isPad>
    __aicore__ inline void CopyOut(int64_t thisLen, int64_t startIndices, int64_t embedDim)
    {
        // 直接从queOut读取（已经包含反量化后的结果）
        LocalTensor<float> outLt = queOut.DeQue<float>();
        if constexpr (std::is_same_v<OutputType, float>) {
            if constexpr (isPad) {
                for (int64_t i = 0; i < thisLen; i++) {
                    int64_t outOffset = (startIndices + i) * embedDim;
                    CpLocal2Gm(outGT[outOffset], outLt[i * alignedRowBytes], embedDim);
                }
            } else {
                int64_t allLen = thisLen * embedDim;
                CpLocal2Gm(outGT[startIndices * embedDim], outLt, allLen);
            }
        } else if constexpr (std::is_same_v<OutputType, half> || std::is_same_v<OutputType, bfloat16_t>) {
            if (!isPad) {
                LocalTensor<OutputType> outLtDst = newqueOut.AllocTensor<OutputType>();
                Cast(outLtDst, outLt, RoundMode::CAST_RINT, thisLen * embedDim);
                newqueOut.EnQue(outLtDst);
                outLtDst = newqueOut.DeQue<OutputType>();
                CpLocal2Gm(outGT[startIndices * embedDim], outLtDst, thisLen * embedDim);
                newqueOut.FreeTensor(outLtDst);
            } else {
                for (int64_t i = 0; i < thisLen; i++) {
                    LocalTensor<OutputType> rowDst = newqueOut.AllocTensor<OutputType>();
                    Cast(rowDst, outLt[i * alignedRowBytes], RoundMode::CAST_RINT, embedDim);
                    newqueOut.EnQue(rowDst);
                    rowDst = newqueOut.DeQue<OutputType>();
                    int64_t outOffset = (startIndices + i) * embedDim;
                    CpLocal2Gm(outGT[outOffset], rowDst, embedDim);
                    newqueOut.FreeTensor(rowDst);
                }
            }
        } else if constexpr (std::is_same_v<OutputType, uint8_t>) {
            ASCENDC_ASSERT(false, "Unsupported output type");
        }

        queOut.FreeTensor(outLt);
    }

    // ========== Process：处理一张表的indices（可能分批） ==========

    __aicore__ inline void Process(
        int64_t remain,
        int64_t startIndices,
        int64_t embedDim,
        int64_t thisWeightOffset,
        int64_t tableIndex)
    {
        int64_t thisLen = remain;

        // Nobag模式下所有表的dim都相同（都是maxD），直接使用预计算的对齐参数
        // 计算当前表能处理的indices数量（基于queOut和queIn）
        int64_t maxIndicesByQueOut = queOutBytes / alignedEmbedDimBytes;
        int64_t maxIndicesByQueIn = queInBytes / alignedRowBytes;
        int64_t maxIndicesForThisTable = Std::min(Std::min(maxIndicesByQueOut, maxIndicesByQueIn), indicesNumOneBlock);

        while (remain > 0) {
            if (thisLen > maxIndicesForThisTable) {
                thisLen = maxIndicesForThisTable;
            }
            remain -= thisLen;

            // copyIn（包含反量化）
            CopyInWithDequantize(startIndices, thisLen, embedDim, thisWeightOffset, tableIndex);

            // copyOut（nobag模式：直接输出，无需累加）
            if (alignedRowBytes == embedDim) {
                // 刚好满足对齐，一次性拷出
                CopyOut<false>(thisLen, startIndices, embedDim);
            } else {
                // 否则一行一行地拷出
                CopyOut<true>(thisLen, startIndices, embedDim);
            }

            startIndices = startIndices + thisLen;
            thisLen = remain;
        }
    }

    // ========== 成员变量 ==========

    // Shape参数
    int64_t devWeightsDim0;
    int64_t weightsOffsetsDim0;
    int64_t dOffsetsDim0;
    int64_t indicesDim0;
    int64_t offsetsDim0;
    int64_t outDim0;
    int64_t outDim1;
    int64_t maxD;

    // FP8参数
    int64_t fp8ExponentBits;
    int64_t fp8ExponentBias;
    int64_t rowAlignment;
    uint32_t fp8BodyShift;
    float fp8Multiplier;

    // 预计算的对齐参数
    int64_t rowBytes;
    int64_t alignedRowBytes;
    int64_t alignedEmbedDimBytes;

    // Tiling参数
    int64_t splitBaseLen;
    int64_t tailSplitIndex;
    int64_t lenOfThisCore;
    int64_t offsetOfThisCore;

    // UB内存参数
    int64_t ubCanUsed;
    int64_t queOutBytes;
    int64_t queInBytes;
    int64_t indicesNumOneBlock;

    // TPipe
    TPipe pipe;
    TQue<TPosition::VECIN, 1> queIndices;
    TQue<TPosition::VECIN, 1> queIn;
    TQue<TPosition::VECIN, 1> offsetPerKeyQue;
    TQue<TPosition::VECOUT, 1> queOut;
    TQue<TPosition::VECOUT, 1> newqueOut;
    TBuf<TPosition::VECCALC> fp8MaskBuf;
    TBuf<TPosition::VECCALC> fp8TmpByteBuf;
    TBuf<TPosition::VECCALC> fp8SignUint32Buf;

    // GlobalTensor
    GlobalTensor<uint8_t> devWeightsGT;
    GlobalTensor<int32_t> indicesGT;
    GlobalTensor<int32_t> offsetGT;
    GlobalTensor<int32_t> dOffsetGT;
    GlobalTensor<int64_t> weightOffsetGT;
    GlobalTensor<uint8_t> weightsTysGT;
    GlobalTensor<int32_t> offsetPerKeyGT;
    GlobalTensor<OutputType> outGT;
};

}  // namespace IntNBitSplitEmbeddingNobag

#endif  // INT_NBIT_SPLIT_EMBEDDING_NOBAG_KERNEL_H
