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

#ifndef JAGGED_TO_PADDED_DENSE__KERNEL_FUN_H
#define JAGGED_TO_PADDED_DENSE__KERNEL_FUN_H

#include <cstdint>

#include "kernel_operator.h"
#include "utils.h"

using namespace AscendC;

namespace JaggedToPaddedDense {

constexpr int USE_QUEUE_NUM = 2;

struct Args {
    GM_ADDR values;
    GM_ADDR offsets;
    GM_ADDR out;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

class JaggedToPaddedDenseKernel {
public:
    __aicore__ inline JaggedToPaddedDenseKernel(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        totalBatch = tilingData.totalBatch;
        baseBatchLen = tilingData.baseBatchLen;
        tailSplitIndex = tilingData.tailSplitIndex;
        valuesDim0 = tilingData.valuesDim0;
        valuesDim1 = tilingData.valuesDim1;
        offsetDim0 = tilingData.offsetDim0;
        outDim1 = tilingData.outDim1;
        ubCanUsed = tilingData.ubCanUsed;
        bytesOfDataType = tilingData.bytesOfDataType;
        offsetDataType = tilingData.offsetDataType;

        values = args.values;
        offsets = args.offsets;
        out = args.out;
        workspace = args.workspace;

        // caculate this offset
        if (GetBlockIdx() >= tailSplitIndex) {
            lenOfThisCore = baseBatchLen;
            offsetOfThisCore = tailSplitIndex * (baseBatchLen + 1) + (GetBlockIdx() - tailSplitIndex) * baseBatchLen;
        } else {
            lenOfThisCore = baseBatchLen + 1;
            offsetOfThisCore = GetBlockIdx() * (baseBatchLen + 1);
        }

        valuesGT.SetGlobalBuffer(values, valuesDim0 * valuesDim1 * bytesOfDataType);
        outGT.SetGlobalBuffer(out, offsetDim0 * outDim1 * valuesDim1 * bytesOfDataType);

        // Init pipe
        pipe.InitBuffer(inQueueX, USE_QUEUE_NUM, ubCanUsed / USE_QUEUE_NUM);
        blockLen = ubCanUsed / USE_QUEUE_NUM;
    }

    template <typename T>
    __aicore__ inline void CpGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, int64_t len)
    {
        uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(T) - alignLen;

        GlobalTensor<uint16_t> uint16Gt;
        uint16Gt.SetGlobalBuffer((__gm__ uint16_t*)gt.GetPhyAddr(), len * sizeof(T) / 2);
        LocalTensor<uint16_t> uint16Lt = lt.template ReinterpretCast<uint16_t>();

        if (alignLen != 0) {
            DataCopy(uint16Lt, uint16Gt, alignLen/2);
        }

        if (unAlignLen != 0) {
#ifdef SUPPORT_V200
            DataCopyPadGm2Local(uint16Lt[alignLen/2], uint16Gt[alignLen/2], unAlignLen/2);
#else
            const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            const DataCopyPadExtParams<uint16_t> dataCopyPadExtParams{false, 0, 0, 0};
            DataCopyPad(uint16Lt[alignLen/2], uint16Gt[alignLen/2], dataCopyExtParams, dataCopyPadExtParams);
#endif
        }
    }

    __aicore__ inline void DataCopyPadGm2Local(const LocalTensor<uint16_t>& lt,
                                               const GlobalTensor<uint16_t>& gt, int64_t len)
    {
        DataCopy<uint16_t>(lt, gt, DATA_COPY_ALIGN_BYTES);
        uint64_t mask0 = (1uL << 16) - (1uL << len);
        uint64_t mask[2] = {mask0, 0};
        Duplicate<uint16_t>(lt, 0, mask, 1, 1, 1);
    }

    template <typename T>
    __aicore__ inline void CpLocal2Gm(const GlobalTensor<T>& gt, const LocalTensor<T>& lt, int64_t len)
    {
        uint32_t alignLen = len * sizeof(T) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        uint32_t unAlignLen = len * sizeof(T) - alignLen;

        GlobalTensor<uint16_t> uint16Gt;
        uint16Gt.SetGlobalBuffer((__gm__ uint16_t*)gt.GetPhyAddr(), len * sizeof(T) / 2);
        LocalTensor<uint16_t> uint16Lt = lt.template ReinterpretCast<uint16_t>();

        if (alignLen != 0) {
            DataCopy(uint16Gt, uint16Lt, alignLen/2);
        }
        if (unAlignLen != 0) {
#ifdef SUPPORT_V200
            DataCopyPadLocal2Gm(uint16Gt[alignLen/2], uint16Lt[alignLen/2], unAlignLen/2);
#else
            const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
            const DataCopyPadExtParams<uint16_t> dataCopyPadExtParams{false, 0, 0, 0};
            DataCopyPad(uint16Gt[alignLen/2], uint16Lt[alignLen/2], dataCopyExtParams);
#endif
        }
    }

    __aicore__ inline void DataCopyPadLocal2Gm(const GlobalTensor<uint16_t>& gt, const LocalTensor<uint16_t>& lt,
        int64_t len)
    {
        SetAtomicAdd<uint16_t>();
        uint64_t mask0 = (1uL << 16) - (1uL << len);
        uint64_t mask[2] = {mask0, 0};
        Duplicate<uint16_t>(lt, 0, mask, 1, 1, 1);
        pipe_barrier(PIPE_ALL);
        DataCopy(gt, lt, DATA_COPY_ALIGN_BYTES);
        SetAtomicNone();
    }

    __aicore__ inline void Compute()
    {
        for (int64_t i = offsetOfThisCore; i < lenOfThisCore + offsetOfThisCore; i++) {
            int64_t offsetThisIndex;
            int64_t offsetNextIndex;
            if (offsetDataType == DATA_TYPE_INT64) {
                __gm__ int64_t* offsetsPtr = (__gm__ int64_t*)offsets;
                offsetThisIndex = *(offsetsPtr + i);
                offsetNextIndex = *(offsetsPtr + i + 1);
            } else {
                __gm__ int32_t* offsetsPtr = (__gm__ int32_t*)offsets;
                offsetThisIndex = *(offsetsPtr + i);
                offsetNextIndex = *(offsetsPtr + i + 1);
            }
            int64_t valuesStartIndex = offsetThisIndex * valuesDim1 * bytesOfDataType;
            int64_t valuesEndIndex = offsetNextIndex * valuesDim1 * bytesOfDataType;

            int64_t outStartIndex = i * valuesDim1 * outDim1 * bytesOfDataType;
            int64_t outEndIndex = (i + 1) * valuesDim1 * outDim1 * bytesOfDataType;

            if ((valuesEndIndex - valuesStartIndex) < 0) {
                continue;
            }

            if ((valuesEndIndex - valuesStartIndex) > (outEndIndex - outStartIndex)) {
                valuesEndIndex = valuesStartIndex + outEndIndex - outStartIndex;
            }

            int64_t totalLen = valuesEndIndex - valuesStartIndex;
            int64_t remainLen = totalLen;
            while (remainLen > 0) {
                int64_t thisLen = blockLen;
                if (remainLen < blockLen) {
                    thisLen = remainLen;
                }
                LocalTensor<uint8_t> localTensor = inQueueX.AllocTensor<uint8_t>();

                CpGm2Local(localTensor, valuesGT[valuesStartIndex], thisLen);
                inQueueX.EnQue(localTensor);
                LocalTensor<uint8_t> outPutTensor = inQueueX.DeQue<uint8_t>();

                CpLocal2Gm(outGT[outStartIndex], outPutTensor, thisLen);

                outStartIndex += thisLen;
                valuesStartIndex += thisLen;
                inQueueX.FreeTensor(outPutTensor);
                remainLen = remainLen - thisLen;
            }
        }
    }

private:
    // GM_ADDR
    GM_ADDR values;
    GM_ADDR offsets;
    GM_ADDR out;
    GM_ADDR workspace;

    // Shape
    int64_t totalBatch;
    int64_t valuesDim0;
    int64_t valuesDim1;
    int64_t offsetDim0;
    int64_t outDim1;

    // DataType
    int64_t bytesOfDataType;
    int64_t offsetDataType;

    // Tiling
    int64_t baseBatchLen;
    int64_t tailSplitIndex;

    // Ub
    int64_t ubCanUsed;
    int64_t blockLen;

    // ThisCoreLen
    int64_t lenOfThisCore;
    int64_t offsetOfThisCore;

    // Tpipe
    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, USE_QUEUE_NUM> inQueueX;

    // ThisCoreAddr
    GlobalTensor<uint8_t> valuesGT;
    GlobalTensor<uint8_t> outGT;
};
}  // namespace JaggedToPaddedDense
#endif