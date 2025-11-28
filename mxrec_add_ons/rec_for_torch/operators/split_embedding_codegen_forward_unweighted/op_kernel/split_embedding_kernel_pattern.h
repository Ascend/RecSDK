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

#ifndef MXREC_SPLIT_EMBEDDING_KERNEL_PATTERN_H
#define MXREC_SPLIT_EMBEDDING_KERNEL_PATTERN_H
#include "common.h"

namespace SplitEmbeddingCodegenForwardUnweighted {

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

class SplitEmbeddingKernelPattern {
public:
    __aicore__ inline SplitEmbeddingKernelPattern(Args& args, TPipe* pipeIn)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        pipe = pipeIn;
        InitShapeParams(tilingData);
        InitTiling(tilingData);
        InitGmParams(args);
        InitUbParams(tilingData);
    }

protected:
    __aicore__ inline void InitShapeParams(const SplitEmbeddingCodegenForwardUnweightedTilingData &tilingData)
    {
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
        enableRowsPerTable = tilingData.enableRowsPerTable;
        batchs = (offsetsDim0 - 1) / weightsOffsetsDim0;
    }

    __aicore__ inline void InitTiling(const SplitEmbeddingCodegenForwardUnweightedTilingData &tilingData)
    {
        splitBaseLen = tilingData.splitBaseLen;
        tailSplitIndex = tilingData.tailSplitIndex;

        if (GetBlockIdx() >= tailSplitIndex) {
            lenOfThisCore = splitBaseLen;
            offsetOfThisCore = tailSplitIndex * (splitBaseLen + 1) + (GetBlockIdx() - tailSplitIndex) * splitBaseLen;
        } else {
            lenOfThisCore = splitBaseLen + 1;
            offsetOfThisCore = GetBlockIdx() * (splitBaseLen + 1);
        }
    }

    __aicore__ inline void InitGmParams(const Args &args)
    {
        devWeightsGT.SetGlobalBuffer((__gm__ float*)args.devWeights, devWeightsDim0);
        if (enableHash) {
            indicesGT.SetGlobalBuffer((__gm__ int64_t*)args.hashIndices, indicesDim0);
        } else {
            indicesGT.SetGlobalBuffer((__gm__ int64_t*)args.indices, indicesDim0);
        }
        offsetGT.SetGlobalBuffer((__gm__ int64_t*)args.offsets, offsetsDim0);
        dOffsetGT.SetGlobalBuffer((__gm__ int32_t*)args.dOffsets, dOffsetsDim0);
        weightOffsetGT.SetGlobalBuffer((__gm__ int64_t*)args.weightsOffsets, weightsOffsetsDim0);
        offsetPerKeyGT.SetGlobalBuffer((__gm__ int64_t*)args.offsetPerKey, indicesDim0);
        if (enableRowsPerTable) {
            rowsPerTableGT.SetGlobalBuffer((__gm__ int64_t*)args.rowsPerTable, indicesDim0);
        }
        outGT.SetGlobalBuffer((__gm__ float*)args.out, outDim0 * outDim1);

        ASCENDC_ASSERT(offsetGT.GetValue(offsetsDim0 - 1) == indicesDim0,
                       "The last element in offsets must be equal to indices size");
    }

    __aicore__ inline void InitUbParams(const SplitEmbeddingCodegenForwardUnweightedTilingData &tilingData)
    {
        ubCanUsed = tilingData.ubCanUsed;
        blockLen = ubCanUsed / USE_QUEUE_NUM / sizeof(float);
        blockLen = blockLen / FLOAT_ALIGNMENT * FLOAT_ALIGNMENT;

        pipe->InitBuffer(queIn, 1, blockLen * sizeof(float));
        pipe->InitBuffer(queOut, 1, blockLen * sizeof(float));
    }

    __aicore__ inline void CopyInNormal(int64_t startIndices, int64_t thisLen, int64_t embedDim,
                                        int64_t thisWeightOffset)
    {
        LocalTensor<float> inputLt = queIn.AllocTensor<float>();
        for (int64_t i = 0; i < thisLen; ++i) {
            int64_t thisIndForThisTable = indicesGT.GetValue(startIndices + i);
            int64_t indWeightOffset = thisIndForThisTable * embedDim + thisWeightOffset;
            CpGm2Local(inputLt[i * alignMaxD], devWeightsGT[indWeightOffset], embedDim);
        }
        queIn.EnQue(inputLt);
    }

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
    bool enableRowsPerTable;

    // DataType
    int64_t offsetDataType = DATA_TYPE_INT64;

    // Tiling
    int64_t splitBaseLen;
    int64_t tailSplitIndex;
    int32_t blockDim;
    int64_t indicesNumOneBlock;

    // Ub
    int64_t ubCanUsed;
    int64_t blockLen;

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

    // ThisCoreAddr
    GlobalTensor<float> devWeightsGT;
    GlobalTensor<float> outGT;
    GlobalTensor<int64_t> indicesGT;
    GlobalTensor<int64_t> offsetGT;
    GlobalTensor<int32_t> dOffsetGT;
    GlobalTensor<int64_t> weightOffsetGT;
    GlobalTensor<int64_t> offsetPerKeyGT;
    GlobalTensor<int64_t> rowsPerTableGT;
};
}  // namespace SplitEmbeddingCodegenForwardUnweighted

#endif  // MXREC_SPLIT_EMBEDDING_KERNEL_PATTERN_H
