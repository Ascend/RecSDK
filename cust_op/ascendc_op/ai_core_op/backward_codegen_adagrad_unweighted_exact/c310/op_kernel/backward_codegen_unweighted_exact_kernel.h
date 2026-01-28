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

#ifndef BACKWARD_CODEGEN_UNWEIGHTED_EXACT_KERNEL_KERNEL_FUN_H
#define BACKWARD_CODEGEN_UNWEIGHTED_EXACT_KERNEL_KERNEL_FUN_H

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;

namespace BackwardCodegenUnweightedExact {

constexpr int USE_QUEUE_NUM = 2;
constexpr int USE_BUFFER_NUM = 2;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int DATA_TYPE_INT64 = 1;
constexpr int FLOAT_ALIGNMENT = 8;
constexpr int INT_ALIGNMENT = 8;
constexpr int DATA_TYPE_FLOAT32 = 0;
constexpr int SUM_POOL = 0;
constexpr int MEAN_POOL = 1;
constexpr int NONE_POOL = 2;

enum class UpdateState : uint32_t {
    CLEAR = 0,         // 初始状态
    NEED_UPDATE = 1,   // 需要更新状态
    COMPUTE_GRAD = 2   // 计算梯度状态
};

enum class TriadIndex : int64_t {
    UNIQUE_INDEX = 0,      // 唯一索引在三元组中的位置
    WEIGHT_OFFSET = 1,     // 权重偏移在三元组中的位置
    EMBED_DIM = 2,         // 嵌入维度在三元组中的位置
    ACCESS_STRIDE = 3      // 访问三元组元素的步长
};

constexpr int MAX_ARGS_PIPE_LEN = 300;
constexpr int FLAG_LEN = DATA_ALIGN_BYTES / sizeof(uint32_t);
constexpr uint32_t MAX_THREADS_PER_BLOCK = 1024;

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
    GM_ADDR indiceSizeCumsum;

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
    int64_t thisIndForTotalTable;
};

struct UpdateArgs {
    int64_t inputOffset;
    int64_t embedDim;
    int64_t thisOutOffset;
    int64_t thisMomentumOffset;
};

__aicore__ inline int64_t GetOffset(GM_ADDR offsetAddr, int64_t index)
{
    __gm__ int64_t* offsetPtr = (__gm__ int64_t*)offsetAddr;
    return *(offsetPtr + index);
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

__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtDedupIndices(
    __gm__ int32_t* dOffsets,        // shape: [num_tables + 1], embed dim boundaries
    __gm__ int64_t* weightsOffsets,  // shape: [num_tables], weight offset per table (in elements)
    __gm__ int64_t* indices, __gm__ uint64_t* hashIndices, __gm__ int64_t* offsets, __gm__ int64_t* hashSizeCumsumGT,
    __gm__ uint32_t* workspace, __gm__ volatile int64_t* indicesUniq, __ubuf__ uint32_t* validListLenPtr,
    int64_t offsetOfThisCore, int64_t lenOfThisCore, bool enableHash, int64_t offsetsDim0, int64_t batchSize,
    int64_t indicesDim0, int64_t realTotalHashSize)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t numThreads = AscendC::Simt::GetThreadNum<0>();

    // === Step 1: 分配本线程负责的连续 local 范围 ===
    if (lenOfThisCore == 0) {
        return;
    }

    int64_t elementsPerThread = lenOfThisCore / numThreads;
    int64_t remainder = lenOfThisCore % numThreads;

    int64_t start, end;
    if (threadIdx < remainder) {
        start = threadIdx * (elementsPerThread + 1);
        end = start + elementsPerThread + 1;
    } else {
        start = threadIdx * elementsPerThread + remainder;
        end = start + elementsPerThread;
    }

    if (start >= lenOfThisCore) {
        return;
    }

    // === Step 2: 定位第一个 globalIndexPos 所属的 offsets 段（仅一次二分）===
    int64_t globalFirst = offsetOfThisCore + start;
    int64_t currentOffsetIndex = 0;

    // 二分查找：找到最大的 k 满足 offsets[k] <= globalFirst
    int64_t low = 0, high = offsetsDim0 - 2;  // 注意：比较到 offsets[k+1]，所以 k 最大为 offsetsDim0-2
    while (low <= high) {
        int64_t mid = (low + high) >> 1;
        if (offsets[mid] <= globalFirst) {
            currentOffsetIndex = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    // === Step 3: 遍历本线程负责的连续段 ===
    for (int64_t posOffset = start; posOffset < end; ++posOffset) {
        int64_t globalIndexPos = offsetOfThisCore + posOffset;

        // 线性推进 currentOffsetIndex 直到覆盖当前 globalIndexPos
        // 条件：globalIndexPos >= offsets[currentOffsetIndex + 1]
        while (currentOffsetIndex + 1 < offsetsDim0 && globalIndexPos >= offsets[currentOffsetIndex + 1]) {
            currentOffsetIndex++;
        }

        // 计算 table index
        int64_t tableIndex = currentOffsetIndex / batchSize;

        // 获取原始 index 值
        uint64_t rawIndex = enableHash ? hashIndices[globalIndexPos] : indices[globalIndexPos];

        // 计算全表偏移
        int64_t hashStart = hashSizeCumsumGT[tableIndex];
        uint64_t thisIndForTotalTable = hashStart + rawIndex;

        // 原子去重
        uint32_t oldFlag = AscendC::Simt::AtomicCas(workspace + thisIndForTotalTable,
                                                   static_cast<uint32_t>(UpdateState::CLEAR),
                                                   static_cast<uint32_t>(UpdateState::NEED_UPDATE));

        if (oldFlag == static_cast<uint32_t>(UpdateState::CLEAR)) {
            uint32_t uniqPos = AscendC::Simt::AtomicAdd(validListLenPtr, static_cast<uint32_t>(1));

            // 计算 base index for the triplet
            uint64_t baseIdx = (static_cast<uint64_t>(offsetOfThisCore) + uniqPos) *
                               static_cast<int64_t>(TriadIndex::ACCESS_STRIDE);

            // Calculate embedDim and weightOffset using tableIndex
            int64_t embedDim = static_cast<int64_t>(dOffsets[tableIndex + 1] - dOffsets[tableIndex]);
            int64_t weightBase = weightsOffsets[tableIndex];
            int64_t actualWeightOffset = weightBase + rawIndex * embedDim;

            // Store the triplet
            indicesUniq[baseIdx + static_cast<int64_t>(TriadIndex::UNIQUE_INDEX)] =
                static_cast<int64_t>(thisIndForTotalTable);
            indicesUniq[baseIdx + static_cast<int64_t>(TriadIndex::WEIGHT_OFFSET)] =
                static_cast<int64_t>(actualWeightOffset);
            indicesUniq[baseIdx + static_cast<int64_t>(TriadIndex::EMBED_DIM)] = static_cast<int64_t>(embedDim);
        }
    }
    AscendC::Simt::ThreadBarrier();
}

class BackwardCodegenUnweightedExactKernel {
public:
    __aicore__ inline BackwardCodegenUnweightedExactKernel() {}

    __aicore__ inline void InitAddr(Args& args, BackwardCodegenAdagradUnweightedExactTilingData& tilingData)
    {
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
        momentumDim0 = tilingData.momentumDim0;
        totalHashSize = tilingData.totalHashSize;
    }

    __aicore__ inline void InitDataType()
    {
        bytesOfDataType = sizeof(float);
    }

    __aicore__ inline void InitTiling(BackwardCodegenAdagradUnweightedExactTilingData& tilingData)
    {
        // Tiling
        offsetsSplitLen = tilingData.splitBaseLen;
        offsetsSplitIndex = tilingData.tailSplitIndex;
    }

    __aicore__ inline void InitUb(BackwardCodegenAdagradUnweightedExactTilingData& tilingData)
    {
        // ub
        ubCanUsed = tilingData.ubCanUsed;
        blockLen = ubCanUsed / USE_QUEUE_NUM / bytesOfDataType / USE_BUFFER_NUM;
        blockLen = blockLen / FLOAT_ALIGNMENT * FLOAT_ALIGNMENT;
    }

    __aicore__ inline void InitFunc(BackwardCodegenAdagradUnweightedExactTilingData& tilingData)
    {
        // func
        poolMode = tilingData.poolMode;
        eps = tilingData.eps;
        learning_rate = tilingData.learningRate;
        useRegBase = tilingData.useRegBase;
    }

    __aicore__ inline void InitTensor()
    {
        // tensor
        gradOutputGT.SetGlobalBuffer((__gm__ float*)gradOutput, gradOutputDim0 * gradOutputDim1);
        devWeightsGT.SetGlobalBuffer((__gm__ float*)devWeights, devWeightsDim0);
        momentum1DevGT.SetGlobalBuffer((__gm__ float*)momentum1Dev, momentumDim0);

        outGT.SetGlobalBuffer((__gm__ float*)out, outDim0);  // InitGlobalMemory
        momentum1DevOutGT.SetGlobalBuffer((__gm__ float*)momentum1DevOut, momentumDim0);
        weightsDevOutGT.SetGlobalBuffer((__gm__ float*)weightsDevOut, outDim0);
        hashSizeCumsumGT.SetGlobalBuffer((__gm__ int64_t*)hashSizeCumsum, weightsOffsetsDim0 + 1);

        realTotalHashSize = hashSizeCumsumGT.GetValue(weightsOffsetsDim0);
        workspaceGT.SetGlobalBuffer((__gm__ uint32_t*)workspace, totalHashSize);
        indicesUniqGT.SetGlobalBuffer((__gm__ int64_t*)((__gm__ uint32_t*)workspace + totalHashSize),
                                      indicesDim0 * static_cast<int64_t>(TriadIndex::ACCESS_STRIDE));
        validListLenPtrLt = tbuf.Get<uint32_t>();
    }

    __aicore__ inline void InitOffset()
    {
        // ThisCoreLen
        if (GetBlockIdx() >= offsetsSplitIndex) {
            lenOfThisCore = offsetsSplitLen;
            offsetOfThisCore =
                offsetsSplitIndex * (offsetsSplitLen + 1) + (GetBlockIdx() - offsetsSplitIndex) * offsetsSplitLen;
        } else {
            lenOfThisCore = offsetsSplitLen + 1;
            offsetOfThisCore = GetBlockIdx() * (offsetsSplitLen + 1);
        }
    }

    __aicore__ inline void InitPipe()
    {
        // Init pipe
        pipe.InitBuffer(queIn, USE_BUFFER_NUM, blockLen * sizeof(float));
        pipe.InitBuffer(queOut, USE_BUFFER_NUM, blockLen * sizeof(float));
        pipe.InitBuffer(tbuf, DATA_ALIGN_BYTES);
    }

    __aicore__ inline void Init(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        InitAddr(args, tilingData);
        InitDataType();
        InitTiling(tilingData);
        InitUb(tilingData);
        InitFunc(tilingData);
        InitOffset();
        InitPipe();
        InitTensor();
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
        LocalTensor<T> outLt = queOut.AllocTensor<T>();
        LocalTensor<int32_t> clearLt = outLt.template ReinterpretCast<int32_t>();
        Duplicate<int32_t>(clearLt, (int32_t)0, blockLen);
        queOut.EnQue(outLt);
        LocalTensor<T> newOutLt = queOut.DeQue<T>();
        while (remain > 0) {
            int64_t thisLen = blockLen;
            if (remain < thisLen) {
                thisLen = (remain + thisAlignment - 1);
            }
            thisLen = thisLen / thisAlignment * thisAlignment;
            int thisOffset = total - remain;
            DataCopy(clearGt[outOffset + thisOffset], newOutLt, thisLen);
            remain = remain - thisLen;
        }
        queOut.FreeTensor(newOutLt);
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

        int64_t remain = lenOfThisCore;
        // 限制indicesNumOneBlock在MAX_ARGS_PIPE_LEN内
        int64_t indicesNumOneBlock = (blockLen / maxD) >= MAX_ARGS_PIPE_LEN ? MAX_ARGS_PIPE_LEN : (blockLen / maxD);
        ComputeArgs argsArry[MAX_ARGS_PIPE_LEN];
        int64_t batchSize = (offsetsDim0 - 1) / (dOffsetsDim0 - 1);
        int64_t cachedTableIndex = -1;
        int64_t cachedEmbedDim = 0;
        int64_t cachedWeightOffset = 0;
        int64_t cachedInputEmbedOffset = 0;
        int64_t cachedHashStart = 0;
        while (remain > 0) {
            int64_t thisLen = 0;
            while (thisLen < indicesNumOneBlock && remain > 0) {
                int64_t indicesInd = offsetOfThisCore + lenOfThisCore - remain;
                remain = remain - 1;
                while (indicesInd < *(offsetsPtr + thisOffsetIndex) ||
                       indicesInd >= *(offsetsPtr + thisOffsetIndex + 1)) {
                    thisOffsetIndex = thisOffsetIndex + 1;
                }
                // Which Table Used, and the table embedDim
                int64_t tableIndex = thisOffsetIndex / batchSize;
                if (tableIndex != cachedTableIndex) {
                    cachedTableIndex = tableIndex;
                    cachedEmbedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
                    cachedWeightOffset = *(weightsOffsetsPtr + tableIndex);
                    cachedInputEmbedOffset = *(dOffsetsPtr + tableIndex);
                    cachedHashStart = hashSizeCumsumGT.GetValue(tableIndex);
                }

                int64_t embedDim = cachedEmbedDim;
                int64_t thisWeightOffset = cachedWeightOffset;
                int64_t thisIndForThisTable =
                    enableHash ? GetOffset(hashIndices, indicesInd) : GetOffset(indices, indicesInd);
                // Out offset
                int64_t thisOutOffset = thisWeightOffset + thisIndForThisTable * embedDim;
                int64_t inputBatchInd = thisOffsetIndex % batchSize;
                int64_t inputOffset = inputBatchInd * gradOutputDim1 + cachedInputEmbedOffset;
                int64_t thisIndForTotalTable = cachedHashStart + thisIndForThisTable;
                argsArry[thisLen] = {thisOffsetIndex, embedDim, inputOffset, thisOutOffset, thisIndForTotalTable};
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
        int64_t batchSize = (offsetsDim0 - 1) / (dOffsetsDim0 - 1);
        int64_t cachedTableIndex = -1;
        int64_t cachedEmbedDim = 0;
        int64_t cachedWeightOffset = 0;
        int64_t cachedHashStart = 0;
        int64_t cachedInputEmbedOffset = 0;

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
                int64_t tableIndex = thisOffsetIndex / batchSize;
                if (tableIndex != cachedTableIndex) {
                    cachedTableIndex = tableIndex;
                    cachedEmbedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
                    cachedWeightOffset = *(weightsOffsetsPtr + tableIndex);
                    cachedHashStart = hashSizeCumsumGT.GetValue(tableIndex);
                    cachedInputEmbedOffset = *(dOffsetsPtr + tableIndex);
                }
                int64_t embedDim = cachedEmbedDim;
                int64_t thisWeightOffset = cachedWeightOffset;
                int64_t thisIndForThisTable = 0;
                if (enableHash) {
                    thisIndForThisTable = GetOffset(hashIndices, indicesInd);
                } else {
                    thisIndForThisTable = GetOffset(indices, indicesInd);
                }
                int64_t thisIndForTotalTable = cachedHashStart + thisIndForThisTable;

                // Out offset
                int64_t thisOutOffset = thisWeightOffset + thisIndForThisTable * embedDim;
                int64_t inputBatchInd = thisOffsetIndex % gradOutputDim0;

                int64_t inputOffset;
                if (poolMode == NONE_POOL) {
                    inputOffset = indicesInd * gradOutputDim1;
                } else {
                    inputOffset = inputBatchInd * gradOutputDim1 + cachedInputEmbedOffset;
                }

                ComputeArgs& theArgs = argsArry[thisLen];
                theArgs.offsetIndex = thisOffsetIndex;
                theArgs.embedDim = embedDim;
                theArgs.indWeightOffset = thisOutOffset;
                theArgs.inputOffset = inputOffset;
                theArgs.thisIndForTotalTable = thisIndForTotalTable;
                thisLen += 1;
            }
            // copy in
            LocalTensor<float> inputLt = queIn.AllocTensor<float>();  // 同一个bag拷贝时同一个的地址，可以跳过
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

    __aicore__ inline void UniqIndices()
    {
        validListLenPtrLt.SetValue(0, 0);
        validListLen = 0;
        uint32_t numThreadOfPerCore;
        if (lenOfThisCore < static_cast<int64_t>(MAX_THREADS_PER_BLOCK)) {
            numThreadOfPerCore = lenOfThisCore;
        } else {
            numThreadOfPerCore = MAX_THREADS_PER_BLOCK;
        }

        int64_t lenOfPerThreadPerCore = lenOfThisCore / numThreadOfPerCore;
        int64_t leftCoreLen = lenOfThisCore % numThreadOfPerCore;
        int64_t batchSize = (offsetsDim0 - 1) / (dOffsetsDim0 - 1);

        AscendC::Simt::VF_CALL<SimtDedupIndices>(
            AscendC::Simt::Dim3{
                numThreadOfPerCore, 1, 1
            },
            (__gm__ int32_t*)dOffsets, (__gm__ int64_t*)weightsOffsets,
            (__gm__ int64_t*)indices, (__gm__ uint64_t*)hashIndices, (__gm__ int64_t*)offsets,
            (__gm__ int64_t*)hashSizeCumsum, (__gm__ uint32_t*)workspace, (__gm__ int64_t*)indicesUniqGT.GetPhyAddr(),
            (__ubuf__ uint32_t*)validListLenPtrLt.GetPhyAddr(), offsetOfThisCore, lenOfThisCore, enableHash,
            offsetsDim0, batchSize, indicesDim0, realTotalHashSize
        );
        pipe_barrier(PIPE_ALL);
        SyncAll();

        validListLen = validListLenPtrLt.GetValue(0);
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
    int64_t realTotalHashSize;
    int64_t momentumDim0;

    // DataType
    int64_t bytesOfDataType;

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
    bool useRegBase;

    // ThisCoreLen
    int64_t lenOfThisCore;
    int64_t offsetOfThisCore;

    // Tpipe
    TPipe pipe;
    TQue<TPosition::VECIN, 1> queIn;
    TQue<TPosition::VECOUT, 1> queOut;
    TBuf<TPosition::VECCALC> tbuf;

    // ThisCoreAddr
    GlobalTensor<float> devWeightsGT;
    GlobalTensor<float> outGT;
    GlobalTensor<float> gradOutputGT;
    GlobalTensor<float> momentum1DevGT;
    GlobalTensor<uint32_t> workspaceGT;

    GlobalTensor<float> momentum1DevOutGT;
    GlobalTensor<float> weightsDevOutGT;
    GlobalTensor<int64_t> hashSizeCumsumGT;

    // do indices uniq
    GlobalTensor<int64_t> indicesUniqGT;
    LocalTensor<uint32_t> validListLenPtrLt;
    int64_t validListLen;
};
}  // namespace BackwardCodegenUnweightedExact
#endif