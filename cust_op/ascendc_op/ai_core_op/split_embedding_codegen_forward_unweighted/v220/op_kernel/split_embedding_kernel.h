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

#ifndef MXREC_SPLIT_EMBEDDING_KERNEL_H
#define MXREC_SPLIT_EMBEDDING_KERNEL_H
#include "split_embedding_kernel_pattern.h"

namespace SplitEmbeddingCodegenForwardUnweighted {

template <PoolingMode mode>
class CachedTableProcessor {
public:
    __aicore__ inline CachedTableProcessor(TQue<TPosition::VECIN, 1>& queIn,
                                           TQue<TPosition::VECOUT, 1>& queOut,
                                           GlobalTensor<float>& devWeightsGT,
                                           GlobalTensor<int64_t>& indicesGT,
                                           GlobalTensor<float>& outGT,
                                           int64_t indicesNumOneBlock)
    {
        this->queIn = queIn;
        this->queOut = queOut;
        this->devWeightsGT = devWeightsGT;
        this->indicesGT = indicesGT;
        this->outGT = outGT;
        this->indicesNumOneBlock = indicesNumOneBlock;
    }

    __aicore__ inline void ProcessSmallTable(int64_t tableIndex,
                                             int64_t tableRows,
                                             int64_t tableDim,
                                             int64_t tableOffset,
                                             int64_t indicesOffset,
                                             int64_t indicesLen,
                                             int64_t outOffset)
    {
        const bool cacheHit = (tableIndex == cachedTableIndex);
        if (!cacheHit) {
            ReleaseCachedTable();
            CacheThisTable(tableIndex, tableRows, tableDim, tableOffset);
        }

        float meanLen = static_cast<float>(1) / static_cast<float>(indicesLen);
        int64_t remain = indicesLen;
        int64_t thisLen = indicesLen;
        while (remain > 0) {
            if (thisLen > indicesNumOneBlock) {
                thisLen = indicesNumOneBlock;
            }
            remain -= thisLen;
            LookupTable2Ub(thisLen, tableDim, indicesOffset, meanLen);
            CopyOut(outOffset, tableDim);

            indicesOffset = indicesOffset + thisLen;
            thisLen = remain;
        }
    }

    __aicore__ inline void ReleaseCachedTable()
    {
        if (cachedTableIndex == CACHE_MISS_MARK) {
            return ;
        }
        LocalTensor<float> inputLt = queIn.DeQue<float>();
        queIn.FreeTensor(inputLt);
        cachedTableIndex = CACHE_MISS_MARK;
    }

protected:
    __aicore__ inline void CacheThisTable(int64_t tableIndex,
                                          int64_t tableRows,
                                          int64_t tableDim,
                                          int64_t tableOffset)
    {
        cachedTableLen = tableRows * tableDim;

        LocalTensor<float> inputLt = queIn.AllocTensor<float>();
        CpGm2Local(inputLt, devWeightsGT[tableOffset], cachedTableLen);
        queIn.EnQue(inputLt);

        cachedTableIndex = tableIndex;
    }

    __aicore__ inline void LookupTable2Ub(int64_t thisLen, int64_t tableDim, int64_t indicesOffset, float meanScale)
    {
        LocalTensor<float> inputLt = queIn.DeQue<float>();
        LocalTensor<float> outLt = queOut.AllocTensor<float>();

        Duplicate<float>(outLt, 0, tableDim);
        for (int64_t i = 0; i < thisLen; i++) {
            int64_t ind = indicesGT.GetValue(indicesOffset + i);
            Add(outLt, outLt, inputLt[ind * tableDim], tableDim);
        }
        if constexpr (mode == PoolingMode::MEAN) {
            Muls<float>(outLt, outLt, meanScale, tableDim);
        }
        queIn.EnQue(inputLt);  // 供下一次同表的查询使用
        queOut.EnQue(outLt);
    }

    __aicore__ inline void CopyOut(int64_t outOffset, int64_t tableDim)
    {
        auto outLt = queOut.DeQue<float>();
        SetAtomicAdd<float>();
        CpLocal2Gm(outGT[outOffset], outLt, tableDim);
        SetAtomicNone();
        queOut.FreeTensor(outLt);
    }

private:
    int64_t cachedTableIndex = CACHE_MISS_MARK;
    int64_t cachedTableLen = 0;

    int64_t indicesNumOneBlock;  // 一个批次处理的emb条数

    TQue<TPosition::VECIN, 1> queIn;
    TQue<TPosition::VECOUT, 1> queOut;

    GlobalTensor<float> devWeightsGT;
    GlobalTensor<int64_t> indicesGT;
    GlobalTensor<float> outGT;
};

template <PoolingMode mode>
class SplitEmbeddingKernel : public SplitEmbeddingKernelPattern {
public:
    __aicore__ inline SplitEmbeddingKernel(Args& args, TPipe* pipe) : SplitEmbeddingKernelPattern(args, pipe) {}

    __aicore__ inline void Compute()
    {
        if (lenOfThisCore == 0) {
            return;
        }
        indicesNumOneBlock = blockLen / alignMaxD;
        if (indicesNumOneBlock >= MAX_INDICES_ONE_BLOCK) {
            indicesNumOneBlock = MAX_INDICES_ONE_BLOCK;
        }

        CachedTableProcessor<mode> smallTableProcessor(queIn, queOut, devWeightsGT, indicesGT, outGT,
                                                       indicesNumOneBlock);
        int64_t tableIndex = offsetOfThisCore / batchs;
        int64_t batchIndex = offsetOfThisCore % batchs;
        int64_t thisOffsetIndex = tableIndex * batchs + batchIndex;
        int64_t startIndices = offsetGT.GetValue(thisOffsetIndex);

        for (int64_t loop = 0; loop < lenOfThisCore; loop++) {
            int64_t endIndices = offsetGT.GetValue(thisOffsetIndex + 1);
            int32_t thisLen = endIndices - startIndices;

            if (thisLen <= 0) {
                startIndices = endIndices;
                thisOffsetIndex++;
                continue;
            }

            // dataCopy In params
            tableIndex = thisOffsetIndex / batchs;
            int64_t thisWeightOffset = weightOffsetGT.GetValue(tableIndex);
            // dataCopy Out params
            int64_t outBatchInd = thisOffsetIndex % outDim0;
            int64_t outEmbedOffset = dOffsetGT.GetValue(tableIndex);
            int64_t outOffset = outBatchInd * outDim1 + outEmbedOffset;
            int64_t embedDim = dOffsetGT.GetValue(tableIndex + 1) - outEmbedOffset;
            int64_t embedLen = GetTableRows(tableIndex);
            int64_t embedSize = embedLen * embedDim;
            bool unalignTable = (embedDim % FLOAT_ALIGNMENT != 0);
            if (embedSize >= SMALL_TABLE_THRESHOLD or unalignTable) {
                smallTableProcessor.ReleaseCachedTable();
                Process(thisLen, startIndices, embedDim, thisWeightOffset, outOffset);
            } else {
                smallTableProcessor.ProcessSmallTable(tableIndex, embedLen, embedDim, thisWeightOffset, startIndices,
                                                      thisLen, outOffset);
            }
            startIndices = endIndices;
            thisOffsetIndex++;
        }
        smallTableProcessor.ReleaseCachedTable();
    }

private:
    __aicore__ inline void Process(int64_t remain, int64_t startIndices, int64_t embedDim, int64_t thisWeightOffset,
                                   int64_t outOffset)
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
            CopyOut(outOffset, embedDim);

            startIndices = startIndices + thisLen;
            thisLen = remain;
        }
    }

    __aicore__ inline void Pooling(float meanLen, int64_t thisLen, int64_t embedDim)
    {
        LocalTensor<float> outLt = queOut.AllocTensor<float>();
        LocalTensor<float> inputLt = queIn.DeQue<float>();

        Duplicate<float>(outLt, 0, alignMaxD);
        for (int64_t i = 0; i < thisLen; i++) {
            Add(outLt, outLt, inputLt[i * alignMaxD], embedDim);
        }

        if constexpr (mode == PoolingMode::MEAN) {
            Muls<float>(outLt, outLt, meanLen, embedDim);
        }
        queIn.FreeTensor(inputLt);
        queOut.EnQue(outLt);
    }

    __aicore__ inline void CopyOut(int64_t outOffset, int64_t embedDim)
    {
        auto outLt = queOut.DeQue<float>();
        SetAtomicAdd<float>();
        CpLocal2Gm(outGT[outOffset], outLt, embedDim);
        SetAtomicNone();
        queOut.FreeTensor(outLt);
    }

    __aicore__ inline int64_t GetTableRows(int64_t tableIndex)
    {
        if (enableRowsPerTable) {
            return rowsPerTableGT.GetValue(tableIndex);
        }
        return SMALL_TABLE_THRESHOLD + 1;
    }
};
}  // namespace SplitEmbeddingCodegenForwardUnweighted
#endif  // MXREC_SPLIT_EMBEDDING_KERNEL_H
