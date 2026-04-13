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

#ifndef SPLIT_EMBEDDING_KERNEL_SIMT_H
#define SPLIT_EMBEDDING_KERNEL_SIMT_H

#include <cstdint>

#include "kernel_operator.h"
#include "common.h"

namespace SplitEmbeddingCodegenForwardUnweightedSimt {

constexpr int32_t warpSize = 32;
constexpr int ALIGN = 32;

constexpr int SIMT_LAUNCH_BOUND = 32 * 16;
constexpr int MAX_CHUNK = 8;

constexpr int SUM_POOL = 0;
constexpr int MEAN_POOL = 1;
constexpr int NONE_POOL = 2;

template <typename T>
__aicore__ inline void CpGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, int64_t len)
{
    uint32_t alignLen = len * sizeof(T) / ALIGN * ALIGN;
    uint32_t unAlignLen = len * sizeof(T) - alignLen;

    DataCopy(lt, gt, alignLen / sizeof(T));
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
        const DataCopyPadExtParams<T> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyPad(lt[alignLen / sizeof(T)], gt[alignLen / sizeof(T)], dataCopyExtParams, dataCopyPadExtParams);
    }
}

__simt_callee__ inline void AccumulateFromGlobalVec(const __gm__ float* src, int64_t len, float* dst)
{
    if (len == 8) {
        float4 vec0 = reinterpret_cast<const __gm__ float4*>(src)[0];
        float4 vec1 = reinterpret_cast<const __gm__ float4*>(src)[1];
        const float* vals0 = reinterpret_cast<const float*>(&vec0);
        const float* vals1 = reinterpret_cast<const float*>(&vec1);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            dst[i] += vals0[i];
            dst[i + 4] += vals1[i];
        }
        return;
    }
    if (len == 4) {
        float4 vec = reinterpret_cast<const __gm__ float4*>(src)[0];
        const float* vals = reinterpret_cast<const float*>(&vec);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            dst[i] += vals[i];
        }
        return;
    }
    for (int64_t i = 0; i < len; ++i) {
        dst[i] += src[i];
    }
}

__simt_callee__ inline void StoreToGlobalVec(__gm__ float* dst, const float* src, int64_t len)
{
    if (len == 8) {
        reinterpret_cast<__gm__ float4*>(dst)[0] = reinterpret_cast<const float4*>(src)[0];
        reinterpret_cast<__gm__ float4*>(dst)[1] = reinterpret_cast<const float4*>(src)[1];
        return;
    }
    if (len == 4) {
        reinterpret_cast<__gm__ float4*>(dst)[0] = reinterpret_cast<const float4*>(src)[0];
        return;
    }
    for (int64_t i = 0; i < len; ++i) {
        dst[i] = src[i];
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_LAUNCH_BOUND)
    inline void PoolingSimt(__gm__ float* devWeights,
                            __gm__ int64_t* weightsOffsets,
                            __gm__ int32_t* dOffsets,
                            __gm__ int64_t* indices,
                            __gm__ int64_t* offsets,
                            __gm__ float* out,
                            int64_t weightsOffsetsDim0,
                            int64_t totalBags,
                            int64_t dOffsetsDim0,
                            int64_t batchs,
                            int64_t warpPerBag,
                            int64_t outDim1,
                            int64_t poolMode)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t warpId = threadIdx / warpSize;
    int32_t laneId = threadIdx % warpSize;
    int32_t warpsPerBlock = static_cast<int32_t>(AscendC::Simt::GetThreadNum<0>()) / warpSize;
    if (warpId >= warpsPerBlock) {
        return;
    }

    int64_t totalTasks = static_cast<int64_t>(totalBags) * warpPerBag;
    int64_t totalWarps = static_cast<int64_t>(AscendC::Simt::GetBlockNum()) * warpsPerBlock;
    int64_t taskIndex = static_cast<int64_t>(AscendC::Simt::GetBlockIdx()) * warpsPerBlock + warpId;

    for (; taskIndex < totalTasks; taskIndex += totalWarps) {
        int64_t bagIndex = taskIndex / warpPerBag;
        int32_t chunkIdx = static_cast<int32_t>(taskIndex % warpPerBag);

        int64_t tableIdx = bagIndex / batchs;
        int64_t batchIdx = bagIndex % batchs;

        int64_t startIndex = offsets[bagIndex];
        int64_t bagLen = offsets[bagIndex + 1] - startIndex;
        int64_t startD = static_cast<int64_t>(dOffsets[tableIdx]);
        int64_t embedDim = static_cast<int64_t>(dOffsets[tableIdx + 1]) - startD;

        int64_t chunkSize = (embedDim + warpPerBag - 1) / warpPerBag;
        int64_t chunkStart = chunkIdx * chunkSize;
        if (chunkStart >= embedDim) {
            continue;
        }

        int64_t chunkLen = chunkSize;
        if (chunkStart + chunkLen > embedDim) {
            chunkLen = embedDim - chunkStart;
        }

        int64_t weightBase = weightsOffsets[tableIdx];
        int64_t outBase = batchIdx * outDim1 + startD + chunkStart;

        if (bagLen == 0) {
            if (laneId == 0) {
                __gm__ float4* outPtrZero = reinterpret_cast<__gm__ float4*>(out + outBase);
                float4 zeroVec = {0.f, 0.f, 0.f, 0.f};
                int64_t vecCnt = chunkLen / 4;
                for (int64_t v = 0; v < vecCnt; ++v) {
                    outPtrZero[v] = zeroVec;
                }
                for (int64_t tail = vecCnt * 4; tail < chunkLen; ++tail) {
                    out[outBase + tail] = 0.f;
                }
            }
            continue;
        }

        alignas(16) float local[MAX_CHUNK] = {0};
#pragma unroll 1
        for (int64_t idx = laneId; idx < bagLen; idx += warpSize) {
            int64_t indexVal = indices[startIndex + idx];
            int64_t readBase = weightBase + indexVal * embedDim + chunkStart;
            const __gm__ float* rowPtr = devWeights + readBase;
            AccumulateFromGlobalVec(rowPtr, chunkLen, local);
        }

        float bagLenFloat = static_cast<float>(bagLen);
        __gm__ float* outPtr = out + outBase;

        alignas(16) float outVals[MAX_CHUNK];
#pragma unroll 1
        for (int64_t dim = 0; dim < chunkLen; ++dim) {
            float sum = AscendC::Simt::WarpReduceAddSync(local[dim]);
            outVals[dim] = (poolMode == MEAN_POOL) ? (sum / static_cast<float>(bagLen)) : sum;
        }
        if (laneId == 0) {
            StoreToGlobalVec(outPtr, outVals, chunkLen);
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_LAUNCH_BOUND)
    inline void NonePoolSimt(__gm__ float* devWeights,
                             __gm__ int64_t* weightsOffsets,
                             __gm__ int64_t* indices,
                             __gm__ float* out,
                             int64_t totalIndices,
                             int64_t outDim1,
                             __local_mem__ int64_t* offsetPrefix,
                             int64_t numTables,
                             int32_t warpGroup,
                             int64_t embedDim)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockThreads = AscendC::Simt::GetThreadNum<0>();

    int32_t groupsPerBlock = blockThreads / warpGroup;
    if (groupsPerBlock <= 0) {
        return;
    }
    int32_t activeThreads = groupsPerBlock * warpGroup;
    if (threadIdx >= activeThreads) {
        return;
    }

    int32_t groupId = threadIdx / warpGroup;
    int32_t laneInGroup = threadIdx % warpGroup;

    int64_t globalGroup = static_cast<int64_t>(AscendC::Simt::GetBlockIdx()) * groupsPerBlock + groupId;
    int64_t totalGroups = static_cast<int64_t>(AscendC::Simt::GetBlockNum()) * groupsPerBlock;

    for (int64_t idx = globalGroup; idx < totalIndices; idx += totalGroups) {
        int64_t left = 0;
        int64_t right = numTables;
        while (left < right) {
            int64_t mid = (left + right) / 2;
            if (offsetPrefix[mid + 1] <= idx) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        int64_t tableIdx = left;

        int64_t indexVal = indices[idx];
        int64_t weightBase = weightsOffsets[tableIdx] + (indexVal * embedDim);
        const __gm__ float* src = devWeights + weightBase;
        __gm__ float* dst = out + idx * embedDim;

        int64_t chunkSize = (embedDim + warpGroup - 1) / warpGroup;
        int64_t chunkStart = static_cast<int64_t>(laneInGroup) * chunkSize;
        if (chunkStart >= embedDim) {
            continue;
        }
        int64_t chunkLen = chunkSize;
        if (chunkStart + chunkLen > embedDim) {
            chunkLen = embedDim - chunkStart;
        }

        const __gm__ float* srcChunk = src + chunkStart;
        __gm__ float* dstChunk = dst + chunkStart;

        int64_t vecCnt = chunkLen / 4;
        for (int64_t v = 0; v < vecCnt; ++v) {
            reinterpret_cast<__gm__ float4*>(dstChunk)[v] = reinterpret_cast<const __gm__ float4*>(srcChunk)[v];
        }
        for (int64_t tail = vecCnt * 4; tail < chunkLen; ++tail) {
            dstChunk[tail] = srcChunk[tail];
        }
    }
}

__aicore__ inline void LaunchSimtKernel(Args &args)
{
    GET_TILING_DATA(tilingData, args.tiling)
    int64_t simtThreadNum = tilingData.simtBlockDim;

    __gm__ float* devWeightsPtr = reinterpret_cast<__gm__ float*>(args.devWeights);
    __gm__ int64_t* weightsOffsetsPtr = reinterpret_cast<__gm__ int64_t*>(args.weightsOffsets);
    __gm__ int32_t* dOffsetsPtr = reinterpret_cast<__gm__ int32_t*>(args.dOffsets);
    __gm__ int64_t* offsetsPtr = reinterpret_cast<__gm__ int64_t*>(args.offsets);
    __gm__ float* outPtr = reinterpret_cast<__gm__ float*>(args.out);
    __gm__ int64_t* indicesPtr = (tilingData.enableHash) ?
                                                         reinterpret_cast<__gm__ int64_t*>(args.hashIndices) :
                                                         reinterpret_cast<__gm__ int64_t*>(args.indices);

    AscendC::Simt::Dim3 simtDim{static_cast<uint32_t>(simtThreadNum), 1, 1};
    int64_t weightsOffsetsDim0 = tilingData.weightsOffsetsDim0;
    int64_t offsetsDim0 = tilingData.offsetsDim0;
    int64_t totalBags = offsetsDim0 - 1;
    int64_t batchs = (offsetsDim0 - 1) / weightsOffsetsDim0;
    int64_t warpPerBag = (tilingData.maxD + MAX_CHUNK - 1) / MAX_CHUNK;

    if (tilingData.poolMode == NONE_POOL) {
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> prefixIn;

        int64_t prefixBytes = (weightsOffsetsDim0 + 1) * static_cast<int64_t>(sizeof(int64_t));
        int64_t prefixBytesAligned = ((prefixBytes + ALIGN - 1) / ALIGN) * ALIGN;
        pipe.InitBuffer(prefixIn, 1, static_cast<uint32_t>(prefixBytesAligned));
        LocalTensor<int64_t> prefixTensor = prefixIn.AllocTensor<int64_t>();

        GlobalTensor<int64_t> offsetPerKeyGT;
        offsetPerKeyGT.SetGlobalBuffer((__gm__ int64_t*)args.offsetPerKey, weightsOffsetsDim0 + 1);

        CpGm2Local(prefixTensor, offsetPerKeyGT, weightsOffsetsDim0 + 1);
        prefixIn.EnQue(prefixTensor);
        prefixTensor = prefixIn.DeQue<int64_t>();
        __local_mem__ int64_t* offsetPrefix = reinterpret_cast<__local_mem__ int64_t*>(prefixTensor.GetPhyAddr());

        AscendC::Simt::VF_CALL<NonePoolSimt>(
            simtDim,
            devWeightsPtr,
            weightsOffsetsPtr,
            indicesPtr,
            outPtr,
            tilingData.indicesDim0,
            tilingData.maxD,
            offsetPrefix,
            weightsOffsetsDim0,
            static_cast<int32_t>(warpPerBag),
            tilingData.outDim1);
        prefixIn.FreeTensor(prefixTensor);
    } else {
        AscendC::Simt::VF_CALL<PoolingSimt>(
            simtDim,
            devWeightsPtr,
            weightsOffsetsPtr,
            dOffsetsPtr,
            indicesPtr,
            offsetsPtr,
            outPtr,
            weightsOffsetsDim0,
            totalBags,
            tilingData.dOffsetsDim0,
            batchs,
            warpPerBag,
            tilingData.outDim1,
            tilingData.poolMode);
    }
}

}  // namespace SplitEmbeddingCodegenForwardUnweightedSimt

#endif
