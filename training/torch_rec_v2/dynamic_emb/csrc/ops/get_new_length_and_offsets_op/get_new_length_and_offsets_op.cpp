/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
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

#include <type_traits>

#include "get_new_length_and_offsets_op_kernel.h"

#define DTYPE_DISPATCH(isInt32, DTYPE, ...) \
    do {                                    \
        if (isInt32) {                      \
            using DTYPE = int32_t;          \
            __VA_ARGS__;                    \
        } else {                            \
            using DTYPE = int64_t;          \
            __VA_ARGS__;                    \
        }                                   \
    } while (0)

constexpr int32_t DATA_ALIGNED_BYTES = 32;

template <typename T>
__aicore__ inline void CpGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, int32_t numElements)
{
    uint32_t alignLen = numElements * sizeof(T) / DATA_ALIGNED_BYTES * DATA_ALIGNED_BYTES;
    uint32_t unAlignLen = numElements * sizeof(T) - alignLen;
    uint32_t alignDataCount = alignLen / sizeof(T);
    DataCopy(lt, gt, alignDataCount);
    if (unAlignLen > 0) {
        const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
        const DataCopyPadExtParams<T> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyPad(lt[alignDataCount], gt[alignDataCount], dataCopyExtParams, dataCopyPadExtParams);
    }
}

extern "C" __global__ __aicore__ void get_new_length_and_offsets_op(GM_ADDR dUniqueOffsets,
                                                                    GM_ADDR dTableOffsetsInFeature, GM_ADDR newOffsets,
                                                                    GM_ADDR newLenghths, int tableNum,
                                                                    int64_t newLengthsSize, int localBatchSize,
                                                                    int32_t isInt32)
{
    DTYPE_DISPATCH(isInt32 > 0, DTYPE_X, {
        using namespace AscendC;
        using namespace DynamicEmbeddingGetNewLengthAndOffsetsOPSimt;

        __gm__ uint64_t* dUniqueOffsetsGm = reinterpret_cast<__gm__ uint64_t*>(dUniqueOffsets);
        __gm__ int64_t* dTableOffsetsInFeatureGm = reinterpret_cast<__gm__ int64_t*>(dTableOffsetsInFeature);
        __gm__ DTYPE_X* newOffsetsGm = reinterpret_cast<__gm__ DTYPE_X*>(newOffsets);
        __gm__ DTYPE_X* newLengthsGm = reinterpret_cast<__gm__ DTYPE_X*>(newLenghths);

        uint32_t calCount = tableNum + 1;
        // 使用UB缓存
        TPipe pipe;
        TBuf<TPosition::VECCALC> sharedMem;
        pipe.InitBuffer(sharedMem, calCount * sizeof(uint64_t));
        LocalTensor<uint64_t> sharedTensor = sharedMem.Get<uint64_t>();
        GlobalTensor<uint64_t> dUniqueOffsetsGT;
        dUniqueOffsetsGT.SetGlobalBuffer(dUniqueOffsetsGm, (calCount) * sizeof(uint64_t));

        CpGm2Local(sharedTensor, dUniqueOffsetsGT, calCount);

        __ubuf__ uint64_t* dUniqueOffsetsUB = reinterpret_cast<__ubuf__ uint64_t*>(sharedTensor.GetPhyAddr());

        // 使用UB缓存
        TBuf<TPosition::VECCALC> sharedMem2;
        pipe.InitBuffer(sharedMem2, calCount * sizeof(int64_t));
        LocalTensor<int64_t> sharedTensor2 = sharedMem2.Get<int64_t>();
        GlobalTensor<int64_t> dTableOffsetsInFeatureGT;
        dTableOffsetsInFeatureGT.SetGlobalBuffer(dTableOffsetsInFeatureGm, (calCount) * sizeof(int64_t));

        CpGm2Local(sharedTensor2, dTableOffsetsInFeatureGT, calCount);

        __ubuf__ int64_t* dTableOffsetsInFeatureUB = reinterpret_cast<__ubuf__ int64_t*>(sharedTensor2.GetPhyAddr());

        __ubuf__ uint64_t* tableBucketsMagicShifts = nullptr;

        // 主动触发一次get 保证后续访问sharedTensor和sharedTensor2时数据已经在UB中
        sharedTensor.GetValue(0);
        sharedTensor2.GetValue(0);

        bool useQuickDivide = tableNum < MAX_TABLE_NUM_USE_QUICK_DIVIDE;
        if (useQuickDivide) {
            // 使用UB缓存
            TBuf<TPosition::VECCALC> sharedMem3;
            pipe.InitBuffer(sharedMem3, (tableNum) * 2 * sizeof(uint64_t));
            LocalTensor<uint64_t> sharedTensor3 = sharedMem3.Get<uint64_t>();
            tableBucketsMagicShifts = reinterpret_cast<__ubuf__ uint64_t*>(sharedTensor3.GetPhyAddr());
        }
        // 计算快除法的magic number和shift，按table预计算
        for (int tableId = 0; useQuickDivide && tableId < tableNum; ++tableId) {
            int64_t table_feature_count = dTableOffsetsInFeatureUB[tableId + 1] - dTableOffsetsInFeatureUB[tableId];
            int64_t table_buckets = table_feature_count * localBatchSize;
            uint64_t unique_num = dUniqueOffsetsUB[tableId + 1] - dUniqueOffsetsUB[tableId];
            if (table_buckets == 1) {
                useQuickDivide = false;
                break;
            }
            uint64_t magic = 0;
            uint64_t shift = 0;
            GetUintDivMagicAndShift(magic, shift, static_cast<uint64_t>(table_buckets));
            tableBucketsMagicShifts[tableId * 2] = magic;
            tableBucketsMagicShifts[tableId * 2 + 1] = shift - 1;
        }

        if (useQuickDivide) {
            Simt::VF_CALL<GetNewLengthAndOffsetsKernel<DTYPE_X, true>>(
                Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, dUniqueOffsetsUB, dTableOffsetsInFeatureUB,
                tableBucketsMagicShifts, newOffsetsGm, newLengthsGm, tableNum, newLengthsSize, localBatchSize);
        } else {
            Simt::VF_CALL<GetNewLengthAndOffsetsKernel<DTYPE_X, false>>(
                Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, dUniqueOffsetsUB, dTableOffsetsInFeatureUB,
                tableBucketsMagicShifts, newOffsetsGm, newLengthsGm, tableNum, newLengthsSize, localBatchSize);
        }

        SyncAll();
    });
}