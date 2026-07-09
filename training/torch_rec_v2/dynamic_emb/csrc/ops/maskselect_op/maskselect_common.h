/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MASKSELECT_COMMON_H
#define MASKSELECT_COMMON_H

#include "kernel_operator.h"
#include "../ops_utils.h"

using namespace AscendC;

constexpr int32_t MASKSELECT_BUFFER_NUM = 2;
constexpr int32_t MASKSELECT_MAX_THREADS = 1024;
constexpr int32_t MASKSELECT_WARP_SIZE = 32;
constexpr int32_t MASKSELECT_MAX_ELEMENTS_PER_THREAD = 4;
constexpr int32_t MASKSELECT_MAX_WARPS = MASKSELECT_MAX_THREADS / MASKSELECT_WARP_SIZE;
constexpr int32_t MASKSELECT_DATA_ALIGN_BYTES = 32;
constexpr int32_t MASKSELECT_CACHE_ALIGN = 64;

template <typename T>
__aicore__ inline void CpGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, int64_t len)
{
    uint32_t alignLen = len * sizeof(T) / MASKSELECT_DATA_ALIGN_BYTES * MASKSELECT_DATA_ALIGN_BYTES;
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
    uint32_t alignLen = len * sizeof(T) / MASKSELECT_DATA_ALIGN_BYTES * MASKSELECT_DATA_ALIGN_BYTES;
    uint32_t unAlignLen = len * sizeof(T) - alignLen;
    DataCopy(gt, lt, alignLen / sizeof(T));
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
        DataCopyPad(gt[alignLen / sizeof(T)], lt[alignLen / sizeof(T)], dataCopyExtParams);
    }
}

__aicore__ inline void MaskSelectPublishInt64GmCache(__gm__ int64_t* gmBase, int32_t len)
{
    if (len <= 0) {
        return;
    }
    constexpr int32_t elemsPerCacheLine = MASKSELECT_CACHE_ALIGN / static_cast<int32_t>(sizeof(int64_t));
    GlobalTensor<int64_t> gt;
    gt.SetGlobalBuffer(gmBase, static_cast<uint64_t>(len));
    for (int32_t i = 0; i < len; i += elemsPerCacheLine) {
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(gt[i]);
    }
}

__aicore__ inline void MaskSelectInvalidateInt64GmCache(__gm__ int64_t* gmBase, int32_t len)
{
    if (len <= 0) {
        return;
    }
    GlobalTensor<int64_t> gt;
    gt.SetGlobalBuffer(gmBase, static_cast<uint64_t>(len));
    AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        gt[0]);
}

__aicore__ inline void MaskSelectFlushBlockSumGm(__gm__ int64_t* blockSums, int64_t blockIdx)
{
    GlobalTensor<int64_t> blockSumGt;
    blockSumGt.SetGlobalBuffer(blockSums + blockIdx, 1);
    AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        blockSumGt[0]);
}

__aicore__ inline void CpLocal2GmInt64AfterCompute(__gm__ int64_t* gmBase, const LocalTensor<int64_t>& lt, int32_t len)
{
    if (len <= 0) {
        return;
    }
    ops_utils::SyncVMte3();
    GlobalTensor<int64_t> gt;
    gt.SetGlobalBuffer(gmBase, static_cast<uint64_t>(len));
    CpLocal2Gm(gt, lt, len);
    ops_utils::SyncMte3Mte2();
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    MaskSelectPublishInt64GmCache(gmBase, len);
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline int64_t ReadBlockSumGm(__gm__ int64_t* blockSums, int64_t blockIdx, LocalTensor<int64_t>& scratchLt)
{
    GlobalTensor<int64_t> blockSumGt;
    blockSumGt.SetGlobalBuffer(blockSums + blockIdx, 1);
    CpGm2Local(scratchLt, blockSumGt, 1);
    ops_utils::SyncMte2V();
    return scratchLt.GetValue(0);
}

#endif  // MASKSELECT_COMMON_H
