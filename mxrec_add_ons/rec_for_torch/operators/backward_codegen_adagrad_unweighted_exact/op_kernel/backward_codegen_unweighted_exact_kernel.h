/**
 * @file backward_codegen_unweighted_exact_kernel.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef BACKWARD_CODEGEN_UNWEIGHTED_EXACT_KERNEL_KERNEL_FUN_H
#define BACKWARD_CODEGEN_UNWEIGHTED_EXACT_KERNEL_KERNEL_FUN_H

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;

namespace BackwardCodegenUnweightedExact {

constexpr int USE_QUEUE_NUM = 2;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int DATA_TYPE_INT64 = 1;
constexpr int FLOAT_ALIGNMENT = 8;
constexpr int INT_ALIGNMENT = 8;
constexpr int DATA_TYPE_FLOAT32 = 0;
constexpr int SUM_POOL = 0;
constexpr int MEAN_POOL = 1;
constexpr int NONE_POOL = 2;
constexpr int8_t NEED_UPDATE = 33;
constexpr int MAX_ARGS_PIPE_LEN = 300;
constexpr int FLAG_LEN = DATA_ALIGN_BYTES / sizeof(int8_t);

struct Args {
    GM_ADDR gradOutput;
    GM_ADDR devWeights;
    GM_ADDR weightsPlacements;
    GM_ADDR weightsOffsets;
    GM_ADDR dOffsets;
    GM_ADDR hashSizeCumsum;
    GM_ADDR indices;
    GM_ADDR offsets;
    GM_ADDR momentum1Dev;
    GM_ADDR momentum2Dev;
    GM_ADDR hashIndices;
    GM_ADDR uniqueId;
    GM_ADDR uniqueHashSize;
    GM_ADDR uniqueInverse;

    GM_ADDR out;
    GM_ADDR momentum1DevOut;
    GM_ADDR momentum2DevOut;
    GM_ADDR weightsDevOut;

    GM_ADDR workspace;
    GM_ADDR tiling;
};

struct ComputeArgs {
    int64_t offsetIndex;
    int64_t embedDim;
    int64_t inputOffset;
    int64_t indWeightOffset;
};

struct UpdateArgs {
    int64_t inputOffset;
    int64_t embedDim;
    int64_t thisOutOffset;
};

__aicore__ inline int64_t GetOffset(GM_ADDR offsetAddr, int64_t index, int64_t datType)
{
    if (datType == DATA_TYPE_INT64) {
        __gm__ int64_t* offsetPtr = (__gm__ int64_t*)offsetAddr;
        return *(offsetPtr + index);
    } else {
        return -1;
    }
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


class BackwardCodegenUnweightedExactKernel {
public:
    __aicore__ inline BackwardCodegenUnweightedExactKernel() {}

    __aicore__ inline void Init(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        // ADDR
        gradOutput = args.gradOutput;
        devWeights = args.devWeights;
        weightsPlacements = args.weightsPlacements;
        weightsOffsets = args.weightsOffsets;
        dOffsets = args.dOffsets;
        hashSizeCumsum = args.hashSizeCumsum;
        indices = args.indices;
        hashIndices = args.hashIndices;
        offsets = args.offsets;
        momentum1Dev = args.momentum1Dev;
        workspace = args.workspace;

        out = args.out;
        momentum1DevOut = args.momentum1DevOut;
        weightsDevOut = args.weightsDevOut;

        gradOutputDim0 = tilingData.gradOutputDim0;
        gradOutputDim1 = tilingData.gradOutputDim1;
        devWeightsDim0 = tilingData.devWeightsDim0;
        weightsOffsetsDim0 = tilingData.weightsOffsetsDim0;
        dOffsetsDim0 = tilingData.dOffsetsDim0;
        indicesDim0 = tilingData.indicesDim0;
        offsetsDim0 = tilingData.offsetsDim0;
        outDim0 = tilingData.outDim0;
        maxD = tilingData.maxD;
        enableHash = tilingData.enableHash;

        // DataType
        bytesOfDataType = sizeof(float);
        offsetDataType = DATA_TYPE_INT64;

        // Tiling
        offsetsSplitLen = tilingData.splitBaseLen;
        offsetsSplitIndex = tilingData.tailSplitIndex;

        // Ub
        ubCanUsed = tilingData.ubCanUsed;
        blockLen = ubCanUsed / USE_QUEUE_NUM / bytesOfDataType;
        blockLen = blockLen / FLOAT_ALIGNMENT * FLOAT_ALIGNMENT;

        // func
        poolMode = tilingData.poolMode;
        eps = tilingData.eps;
        learning_rate = tilingData.learningRate;
        // ThisCoreLen
        if (GetBlockIdx() >= offsetsSplitIndex) {
            lenOfThisCore = offsetsSplitLen;
            offsetOfThisCore =
                offsetsSplitIndex * (offsetsSplitLen + 1) + (GetBlockIdx() - offsetsSplitIndex) * offsetsSplitLen;
        } else {
            lenOfThisCore = offsetsSplitLen + 1;
            offsetOfThisCore = GetBlockIdx() * (offsetsSplitLen + 1);
        }

        gradOutputGT.SetGlobalBuffer((__gm__ float*)gradOutput, gradOutputDim0 * gradOutputDim1);
        devWeightsGT.SetGlobalBuffer((__gm__ float*)devWeights, devWeightsDim0);
        momentum1DevGT.SetGlobalBuffer((__gm__ float*)momentum1Dev, outDim0);

        outGT.SetGlobalBuffer((__gm__ float*)out, outDim0); // InitGlobalMemory
        momentum1DevOutGT.SetGlobalBuffer((__gm__ float*)momentum1DevOut, outDim0);
        weightsDevOutGT.SetGlobalBuffer((__gm__ float*)weightsDevOut, outDim0);
        hashSizeCumsumGT.SetGlobalBuffer((__gm__ int64_t*)hashSizeCumsum, weightsOffsetsDim0 + 1);

        totalHashSize = hashSizeCumsumGT.GetValue(weightsOffsetsDim0);
        workspaceGT.SetGlobalBuffer((__gm__ int8_t*)workspace, totalHashSize);

        // Init pipe
        pipe.InitBuffer(queIn, 1, blockLen * sizeof(float));
        pipe.InitBuffer(queOut, 1, blockLen * sizeof(float));

        pipe.InitBuffer(queFlagIn, 1, DATA_ALIGN_BYTES);
        pipe.InitBuffer(queFlagOut, 1, DATA_ALIGN_BYTES);
    }

    __aicore__ inline void SetTheFlag(const GlobalTensor<int8_t>& IndexGt, int8_t flagValue)
    {
        LocalTensor<int8_t> flagOutLt = queFlagOut.AllocTensor<int8_t>();
        LocalTensor<int32_t> clearLt = flagOutLt.ReinterpretCast<int32_t>();
        Duplicate<int32_t>(clearLt, (int32_t)0, FLAG_LEN);
        flagOutLt.SetValue(0, flagValue);

        queFlagOut.EnQue(flagOutLt);
        LocalTensor<int8_t> newFlagOutLt = queFlagOut.DeQue<int8_t>();

        SetAtomicMax<int8_t>();
        DataCopy(IndexGt, newFlagOutLt, FLAG_LEN);
        SetAtomicNone();
        queFlagOut.FreeTensor(newFlagOutLt);
    }

    template <typename T>
    __aicore__ inline void ClearGT(const GlobalTensor<T>& clearGt, int64_t clearSize)
    {
        int64_t baseLen = clearSize / GetBlockNum();
        int64_t tailSplit = clearSize % GetBlockNum();

        int64_t outLenThisCore;
        int64_t outOffset;
        if (GetBlockIdx() >= tailSplit) {
            outLenThisCore = baseLen;
            outOffset = tailSplit * (baseLen + 1) + (GetBlockIdx() - tailSplit) * baseLen;
        } else {
            outLenThisCore = baseLen + 1;
            outOffset = GetBlockIdx() * (baseLen + 1);
        }

        int64_t total = outLenThisCore;
        int64_t remain = total;
        int thisAlignment = DATA_ALIGN_BYTES / sizeof(T);
        while (remain > 0) {
            int64_t thisLen = blockLen;
            if (remain < thisLen) {
                thisLen = (remain + thisAlignment - 1) / thisAlignment * thisAlignment;
            }

            LocalTensor<T> outLt = queOut.AllocTensor<T>();
            LocalTensor<int32_t> clearLt = outLt.template ReinterpretCast<int32_t>();
            Duplicate<int32_t>(clearLt, (int32_t)0, thisLen);
            queOut.EnQue(outLt);

            int thisOffset = total - remain;
            LocalTensor<T> newOutLt = queOut.DeQue<T>();
            DataCopy(clearGt[outOffset + thisOffset], newOutLt, thisLen);
            queOut.FreeTensor(newOutLt);
            remain = remain - thisLen;
        }
    }

    __aicore__ inline void ClearGrad()
    {
        __gm__ int32_t* dOffsetsPtr = (__gm__ int32_t*)dOffsets;
        __gm__ int64_t* weightsOffsetsPtr = (__gm__ int64_t*)weightsOffsets;
        __gm__ int64_t* offsetsPtr = (__gm__ int64_t*)offsets;
        __gm__ float* x = (__gm__ float*)out;
        int64_t thisOffsetIndex = 0;
        for (int64_t i = offsetsDim0 - 1; i >= 0; i--) {
            if (offsetOfThisCore >= *(offsetsPtr + i)) {
                thisOffsetIndex = i;
                break;
            }
        }

        int64_t total = lenOfThisCore;
        int64_t remain = total;
        int64_t indicesNumOneBlock = blockLen / maxD;
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
        ComputeArgs argsArry[MAX_ARGS_PIPE_LEN];
        while (remain > 0) {
            int64_t thisLen = 0;
            while (thisLen < indicesNumOneBlock && remain > 0) {
                int64_t indicesInd = offsetOfThisCore + total - remain;
                remain = remain - 1;
                while (indicesInd < *(offsetsPtr + thisOffsetIndex) ||
                       indicesInd >= *(offsetsPtr + thisOffsetIndex + 1)) {
                    thisOffsetIndex = thisOffsetIndex + 1;
                }
                // Which Table Used, and the table embedDim
                int64_t batchSize = (offsetsDim0 - 1) / (dOffsetsDim0 - 1);
                int64_t tableIndex = thisOffsetIndex / batchSize;

                int64_t embedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
                int64_t thisWeightOffset = *(weightsOffsetsPtr + tableIndex);
                int64_t thisIndForThisTable = 0;
                if (enableHash) {
                    thisIndForThisTable = GetOffset(hashIndices, indicesInd, offsetDataType);
                } else {
                    thisIndForThisTable = GetOffset(indices, indicesInd, offsetDataType);
                }
                int64_t thisIndForTotalTable = hashSizeCumsumGT.GetValue(tableIndex) + thisIndForThisTable;
                // Out offset
                int64_t thisOutOffset = thisWeightOffset + thisIndForThisTable * embedDim;
                int64_t inputBatchInd = thisOffsetIndex % batchSize;
                int64_t inputEmbedOffset = *(dOffsetsPtr + tableIndex);
                int64_t inputOffset = inputBatchInd * gradOutputDim1 + inputEmbedOffset;

                argsArry[thisLen] = {thisOffsetIndex, embedDim, inputOffset, thisOutOffset};
                thisLen += 1;
            }
            LocalTensor<float> outLt = queOut.AllocTensor<float>();
            Duplicate<float>(outLt, 0.0, blockLen);
            queOut.EnQue(outLt);
            LocalTensor<float> newOutLt = queOut.DeQue<float>();
            for (int64_t i = 0; i < thisLen; i++) {
                ComputeArgs theArgs = argsArry[i];
                CpLocal2Gm(outGT[theArgs.indWeightOffset], newOutLt[i * maxD], theArgs.embedDim);
            }
            queOut.FreeTensor(newOutLt);
        }
    }

    __aicore__ inline void ComputeGrad()
    {
        __gm__ int32_t* dOffsetsPtr = (__gm__ int32_t*)dOffsets;
        __gm__ int64_t* weightsOffsetsPtr = (__gm__ int64_t*)weightsOffsets;
        __gm__ int64_t* offsetsPtr = (__gm__ int64_t*)offsets;
        __gm__ float* x = (__gm__ float*)out;
        int64_t thisOffsetIndex = 0;
        for (int64_t i = offsetsDim0 - 1; i >= 0; i--) {
            if (offsetOfThisCore >= *(offsetsPtr + i)) {
                thisOffsetIndex = i;
                break;
            }
        }

        int64_t total = lenOfThisCore;
        int64_t remain = total;
        int64_t indicesNumOneBlock = blockLen / maxD;
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
        ComputeArgs argsArry[MAX_ARGS_PIPE_LEN];
        while (remain > 0) {
            int64_t thisLen = 0;
            while (thisLen < indicesNumOneBlock && remain > 0) {
                int64_t indicesInd = offsetOfThisCore + total - remain;
                remain = remain - 1;
                while (indicesInd < *(offsetsPtr + thisOffsetIndex) ||
                       indicesInd >= *(offsetsPtr + thisOffsetIndex + 1)) {
                    thisOffsetIndex = thisOffsetIndex + 1;
                }
                // Which Table Used, and the table embedDim
                int64_t batchSize = (offsetsDim0 - 1) / (dOffsetsDim0 - 1);
                int64_t tableIndex = thisOffsetIndex / batchSize;
                int64_t embedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
                int64_t thisWeightOffset = *(weightsOffsetsPtr + tableIndex);
                int64_t thisIndForThisTable = 0;
                if (enableHash) {
                    thisIndForThisTable = GetOffset(hashIndices, indicesInd, offsetDataType);
                } else {
                    thisIndForThisTable = GetOffset(indices, indicesInd, offsetDataType);
                }
                int64_t thisIndForTotalTable = hashSizeCumsumGT.GetValue(tableIndex) + thisIndForThisTable;
                SetTheFlag(workspaceGT[thisIndForTotalTable], NEED_UPDATE);
                // Out offset
                int64_t thisOutOffset = thisWeightOffset + thisIndForThisTable * embedDim;

                int64_t inputBatchInd = thisOffsetIndex % gradOutputDim0;
                int64_t inputEmbedOffset = *(dOffsetsPtr + tableIndex);

                int64_t inputOffset;
                if (poolMode == NONE_POOL) {
                    inputOffset = indicesInd * gradOutputDim1;
                } else {
                    inputOffset = inputBatchInd * gradOutputDim1 + inputEmbedOffset;
                }

                ComputeArgs& theArgs = argsArry[thisLen];
                theArgs.offsetIndex = thisOffsetIndex;
                theArgs.embedDim = embedDim;
                theArgs.indWeightOffset = thisOutOffset;
                theArgs.inputOffset = inputOffset;
                thisLen += 1;
            }
            // copy in
            LocalTensor<float> inputLt = queIn.AllocTensor<float>();
            for (int64_t i = 0; i < thisLen; i++) {
                ComputeArgs theArgs = argsArry[i];
                CpGm2Local(inputLt[i * maxD], gradOutputGT[theArgs.inputOffset], theArgs.embedDim);
            }
            queIn.EnQue(inputLt);

            LocalTensor<float> newInputLt = queIn.DeQue<float>();
            LocalTensor<float> outLt = queOut.AllocTensor<float>();

            if (poolMode == MEAN_POOL) {
                for (int64_t i = 0; i < thisLen; i++) {
                    ComputeArgs theArgs = argsArry[i];
                    int64_t thisBagLen = *(offsetsPtr + theArgs.offsetIndex + 1) - *(offsetsPtr + theArgs.offsetIndex);
                    float meanLen = (float)1 / thisBagLen;
                    Muls<float>(outLt[i * maxD], newInputLt[i * maxD], meanLen, maxD);
                }
            } else {
                DataCopy(outLt, newInputLt, blockLen);
            }

            queOut.EnQue(outLt);
            queIn.FreeTensor(newInputLt);

            LocalTensor<float> newOutLt = queOut.DeQue<float>();
            SetAtomicAdd<float>();
            for (int64_t i = 0; i < thisLen; i++) {
                ComputeArgs theArgs = argsArry[i];
                CpLocal2Gm(outGT[theArgs.indWeightOffset], newOutLt[i * maxD], theArgs.embedDim);
            }
            SetAtomicNone();
            queOut.FreeTensor(newOutLt);
        }
    }

    // GM_ADDR
    GM_ADDR gradOutput;
    GM_ADDR devWeights;
    GM_ADDR weightsPlacements;
    GM_ADDR weightsOffsets;
    GM_ADDR dOffsets;
    GM_ADDR hashSizeCumsum;
    GM_ADDR indices;
    GM_ADDR momentum1Dev;
    GM_ADDR offsets;
    GM_ADDR hashIndices;
    GM_ADDR workspace;

    GM_ADDR out;
    GM_ADDR momentum1DevOut;
    GM_ADDR weightsDevOut;

    // Shape
    int64_t gradOutputDim0;
    int64_t gradOutputDim1;
    int64_t devWeightsDim0;
    int64_t weightsOffsetsDim0;
    int64_t dOffsetsDim0;
    int64_t indicesDim0;
    int64_t offsetsDim0;
    int64_t outDim0;
    int64_t totalHashSize;

    // // DataType
    int64_t bytesOfDataType;
    int64_t offsetDataType;

    // Tiling
    int64_t offsetsSplitLen;
    int64_t offsetsSplitIndex;

    // Ub
    int64_t ubCanUsed;
    int64_t blockLen;

    // func
    int64_t poolMode;
    int64_t maxD;
    float eps;
    float learning_rate;
    bool enableHash;

    // ThisCoreLen
    int64_t lenOfThisCore;
    int64_t offsetOfThisCore;

    // Tpipe
    TPipe pipe;
    TQue<TPosition::VECIN, 1> queIn;
    TQue<TPosition::VECOUT, 1> queOut;

    TQue<TPosition::VECIN, 1> queFlagIn;
    TQue<TPosition::VECOUT, 1> queFlagOut;

    // ThisCoreAddr
    GlobalTensor<float> devWeightsGT;
    GlobalTensor<float> outGT;
    GlobalTensor<float> gradOutputGT;
    GlobalTensor<float> momentum1DevGT;
    GlobalTensor<int8_t> workspaceGT;
    GlobalTensor<float> momentum1DevOutGT;
    GlobalTensor<float> weightsDevOutGT;
    GlobalTensor<int64_t> hashSizeCumsumGT;
};
}  // namespace BackwardCodegenUnweightedExact
#endif