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

#ifndef PERMUTE_POOLED_EMBS_H
#define PERMUTE_POOLED_EMBS_H

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;

namespace PermutePooledEmbs {

constexpr int USE_QUEUE_NUM = 2;
constexpr int MAX_BLOCK_LEN = 4095;
constexpr int UB_ALIGN = 32;

struct Args {
    GM_ADDR pooled_embs;
    GM_ADDR offset_dim_list;
    GM_ADDR permute_list;
    GM_ADDR inv_offset_dim_list;
    GM_ADDR output;
    GM_ADDR tiling;
};

template <typename POOLED_EMBS_TYPE>
class PermutePooledEmbsKernel {
public:
    __aicore__ inline PermutePooledEmbsKernel(Args& args, TPipe* pipePtr)
    {
        // 加载tiling信息
        GET_TILING_DATA(tilingData, args.tiling);
        
        batchSize = tilingData.batchSize;
        totalFeatureNum = tilingData.totalFeatureNum;
        totalDim = tilingData.totalDim;

        baseBatchLen = tilingData.baseBatchLen;
        tailSplitIndex = tilingData.tailSplitIndex;

        ubCanUsed = tilingData.ubCanUsed;

        // 分配核处理范围
        if (GetBlockIdx() < tailSplitIndex) {
            lenOfThisCore = baseBatchLen + 1;
            tOffsetOfThisCore = GetBlockIdx() * (baseBatchLen + 1);
        } else {
            lenOfThisCore = baseBatchLen;
            tOffsetOfThisCore = tailSplitIndex * (baseBatchLen + 1) + (GetBlockIdx() - tailSplitIndex) * baseBatchLen;
        }

        pooledEmbsGT.SetGlobalBuffer(args.pooled_embs, batchSize * totalDim * sizeof(POOLED_EMBS_TYPE));
        outputGT.SetGlobalBuffer(args.output, batchSize * totalDim * sizeof(POOLED_EMBS_TYPE));

        offsetDimListPtr = reinterpret_cast<__gm__ int64_t*>(args.offset_dim_list);
        permuteListPtr = reinterpret_cast<__gm__ int64_t*>(args.permute_list);
        invOffsetDimListPtr = reinterpret_cast<__gm__ int64_t*>(args.inv_offset_dim_list);

        pipe = pipePtr;
        pipe->InitBuffer(inQueueX, USE_QUEUE_NUM, ubCanUsed / USE_QUEUE_NUM);
        blockLen = ubCanUsed / USE_QUEUE_NUM;
    }

    // 按列分块搬运：从GM拷贝到UB
    template <typename scalar_t>
    __aicore__ inline void CpGm2LocalColumnBlock(
        const LocalTensor<scalar_t>& lt,
        const GlobalTensor<scalar_t>& gt,
        int64_t startRow, int64_t numRows,
        int64_t startCol, int64_t numCols,
        int64_t totalDim)
    {
        if (IsColumnBlockAligned(numCols, startCol, totalDim)) {
            // 对齐情况：使用DataCopyParams进行优化拷贝
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = numRows;                    // 行数
            dataCopyParams.blockLen = numCols / DATA_BLOCK_ELEM;      // 列数转换为块数
            dataCopyParams.srcStride = (totalDim - numCols) / DATA_BLOCK_ELEM;  // 行间跨距
            dataCopyParams.dstStride = 0;                           // UB中连续存储

            DataCopy(lt, gt[startCol * DATA_SIZE], dataCopyParams);
        } else {
            // 非对齐情况：使用DataCopyExtParams
            DataCopyExtParams dataCopyExtParams;
            dataCopyExtParams.blockCount = numRows;                 // 行数
            dataCopyExtParams.blockLen = numCols * DATA_SIZE;       // 字节数
            dataCopyExtParams.srcStride = (totalDim - numCols) * DATA_SIZE;  // 行间跨距（字节）
            dataCopyExtParams.dstStride = 0;                        // UB中连续存储

            DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
            DataCopyPad(lt, gt[startCol * DATA_SIZE], dataCopyExtParams, padParams);
        }
    }

    // 按列分块搬运：从UB拷贝到GM
    template <typename scalar_t>
    __aicore__ inline void CpLocal2GmColumnBlock(
        const GlobalTensor<scalar_t>& gt,
        const LocalTensor<scalar_t>& lt,
        int64_t startRow, int64_t numRows,
        int64_t startCol, int64_t numCols,
        int64_t totalDim)
    {
        if (IsColumnBlockAligned(numCols, startCol, totalDim)) {
            // 对齐情况：使用DataCopyParams
            DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = numRows;                    // 行数
            dataCopyParams.blockLen = numCols / DATA_BLOCK_ELEM;      // 列数转换为块数
            dataCopyParams.srcStride = 0;                           // UB中连续存储
            dataCopyParams.dstStride = (totalDim - numCols) / DATA_BLOCK_ELEM;  // 行间跨距

            DataCopy(gt[startCol * DATA_SIZE], lt, dataCopyParams);
        } else {
            // 非对齐情况：使用DataCopyExtParams
            DataCopyExtParams dataCopyExtParams;
            dataCopyExtParams.blockCount = numRows;                 // 行数
            dataCopyExtParams.blockLen = numCols * DATA_SIZE;       // 字节数
            dataCopyExtParams.srcStride = 0;                        // UB中连续存储
            dataCopyExtParams.dstStride = (totalDim - numCols) * DATA_SIZE;  // 行间跨距（字节）

            DataCopyPad(gt[startCol * DATA_SIZE], lt, dataCopyExtParams);
        }
    }

    // copy单列块
    __aicore__ void copyColumn(int64_t srcStartCol, int64_t dstStartCol, int64_t featureDim)
    {
        int64_t currentRow = 0;

        // 计算每次拷贝的行数（根据UB大小限制）
        int64_t maxRowsPerCopy = blockLen / (featureDim * DATA_SIZE + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
        maxRowsPerCopy = (maxRowsPerCopy < MAX_BLOCK_LEN) ? maxRowsPerCopy : MAX_BLOCK_LEN;
        if (maxRowsPerCopy == 0) maxRowsPerCopy = 1;

        while (currentRow < batchSize) {
            int64_t copyRows = (batchSize - currentRow < maxRowsPerCopy) ? batchSize - currentRow : maxRowsPerCopy;

            LocalTensor<uint8_t> ubTensor = inQueueX.AllocTensor<uint8_t>();

            // GM -> UB: 拷贝源数据块
            CpGm2LocalColumnBlock(ubTensor, pooledEmbsGT[currentRow * totalDim * DATA_SIZE],
                currentRow, copyRows, srcStartCol, featureDim, totalDim);

            inQueueX.EnQue(ubTensor);

            LocalTensor<uint8_t> dequeuedTensor = inQueueX.DeQue<uint8_t>();

            // UB -> GM: 拷贝到目标位置
            CpLocal2GmColumnBlock(outputGT[currentRow * totalDim * DATA_SIZE], dequeuedTensor,
                currentRow, copyRows, dstStartCol, featureDim, totalDim);

            inQueueX.FreeTensor(dequeuedTensor);

            currentRow += copyRows;
        }
    }

    // PermuteColumns: 对每个core负责的列区间，完成列搬运
    __aicore__ void PermuteColumns()
    {
        int64_t colStart = tOffsetOfThisCore;
        int64_t colEnd = tOffsetOfThisCore + lenOfThisCore;

        // 查找colStart属于哪个segment
        int64_t segStart = 0;
        for (; segStart < totalFeatureNum; ++segStart) {
            if (colStart >= invOffsetDimListPtr[segStart] && colStart < invOffsetDimListPtr[segStart + 1]) {
                break;
            }
        }

        if (segStart == totalFeatureNum) {
            return;
        }

        // 对segments遍历，直到超出colEnd
        int64_t seg = segStart;
        while (seg < totalFeatureNum) {
            int64_t invOffsetSeg = invOffsetDimListPtr[seg];
            if (invOffsetSeg >= colEnd) {
                break;
            }
            int64_t invOffsetSegNext = invOffsetDimListPtr[seg + 1];
            int64_t segColStart = invOffsetSeg < colStart ? colStart : invOffsetSeg;
            int64_t segColEnd = invOffsetSegNext < colEnd ? invOffsetSegNext : colEnd;
            int64_t copyLen = segColEnd - segColStart;
            if (copyLen <= 0) {
                continue;
            }
            int64_t srcBase = offsetDimListPtr[permuteListPtr[seg]];
            int64_t srcOffset = segColStart - invOffsetSeg;
            int64_t srcCol = srcBase + srcOffset;
            int64_t dstCol = segColStart;
            copyColumn(srcCol, dstCol, copyLen);
            seg++;
        }
    }

private:
    // Tiling data
    static constexpr uint8_t DATA_SIZE = sizeof(POOLED_EMBS_TYPE);
    static constexpr uint8_t DATA_BLOCK_ELEM = UB_ALIGN / DATA_SIZE;

    int64_t batchSize = 0;
    int64_t totalFeatureNum = 0;
    int64_t totalDim = 0;

    int64_t baseBatchLen = 0;
    int64_t tailSplitIndex = 0;

    int64_t ubCanUsed = 0;
    int64_t blockLen = 0;

    // Core-specific data
    int64_t lenOfThisCore = 0;
    int64_t tOffsetOfThisCore = 0;

    // GM addresses
    GM_ADDR pooled_embs;
    GM_ADDR offset_dim_list;
    GM_ADDR permute_list;
    GM_ADDR inv_offset_dim_list;
    GM_ADDR output;

    // Pointers
    __gm__ int64_t* offsetDimListPtr = nullptr;
    __gm__ int64_t* permuteListPtr = nullptr;
    __gm__ int64_t* invOffsetDimListPtr = nullptr;

    // Global tensors
    GlobalTensor<uint8_t> pooledEmbsGT;
    GlobalTensor<uint8_t> outputGT;

    // TPipe
    TPipe* pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, USE_QUEUE_NUM> inQueueX;

    __aicore__ static bool IsColumnBlockAligned(int64_t numCols, int64_t startCol, int64_t totalDim)
    {
        return (numCols % DATA_BLOCK_ELEM == 0) &&
               (startCol % DATA_BLOCK_ELEM == 0) &&
               (totalDim % DATA_BLOCK_ELEM == 0);
    }
};

}  // namespace PermutePooledEmbs

#endif
