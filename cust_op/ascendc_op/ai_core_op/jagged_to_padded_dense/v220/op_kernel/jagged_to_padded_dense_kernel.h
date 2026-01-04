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

#ifndef JAGGED_TO_PADDED_DENSE__KERNEL_FUN_H
#define JAGGED_TO_PADDED_DENSE__KERNEL_FUN_H

#include <cstdint>

#include "kernel_operator.h"
#include "utils.h"

using namespace AscendC;

namespace JaggedToPaddedDense {

constexpr int USE_QUEUE_NUM = 2;

/*
 PAD_UB_SIZE:
    The copy interface caps repeatTimes at 255, so PAD_UB_SIZE must stay within 256 * 255 = 63.75 KB.
    Keep it aligned with VECTOR_REG_WIDTH
    Ensure that PAD_UB_SIZE < MIN_UB_USED_SIZE
*/
constexpr int PAD_UB_SIZE = 8 * 1024;
constexpr int VECTOR_REG_WIDTH = 256;
constexpr int UB_BLOCK_SIZE = 32;
constexpr int COPY_TILE_DST_REPEAT_SIZE = (VECTOR_REG_WIDTH / UB_BLOCK_SIZE);

struct Args {
    GM_ADDR values;
    GM_ADDR offsets;
    GM_ADDR out;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename VALUE_TYPE, typename OFFSET_TYPE>
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
        bytesOfDataType = sizeof(VALUE_TYPE);
        offsetDataType = sizeof(OFFSET_TYPE);
        paddingValueFp32 = tilingData.paddingValueFp32;
        paddingValueInt64 = tilingData.paddingValueInt64;

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

        int64_t rowBytes = outDim1 * valuesDim1 * bytesOfDataType;
        int64_t alignRowBytes = AlignUp<int64_t, int64_t>(rowBytes, VECTOR_REG_WIDTH);
        padChunkBytes = alignRowBytes > PAD_UB_SIZE ? PAD_UB_SIZE : alignRowBytes;

        // Use a single contiguous chunk of memory for the tile padding region.
        int64_t totalPadFullByte = AlignUp<int64_t, int64_t>(padChunkBytes + UB_BLOCK_SIZE, DATA_ALIGN_BYTES);

        int64_t avail_for_queue = ubCanUsed - totalPadFullByte;
        blockLen = (avail_for_queue / USE_QUEUE_NUM / DATA_ALIGN_BYTES) * DATA_ALIGN_BYTES;
        pipe.InitBuffer(inQueueX, USE_QUEUE_NUM, blockLen);

        pipe.InitBuffer(paddingTbuf, totalPadFullByte);
        LocalTensor<uint8_t> paddingFull = paddingTbuf.AllocTensor<uint8_t>();
        paddingBufLt = paddingFull[UB_BLOCK_SIZE];
        InitializePaddingBuffer(paddingFull);
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

            FillPaddingRegion(outStartIndex, outEndIndex);
        }
    }

    /**
     * Use the pre-initialized padding buffer to fill the remaining output region.
     */
    __aicore__ inline void FillPaddingRegion(int64_t& outStartIndex, int64_t outEndIndex)
    {
        int64_t outRemain = outEndIndex - outStartIndex;
        while (outRemain > 0) {
            int64_t thisPad = padChunkBytes;
            if (outRemain < padChunkBytes) {
                thisPad = outRemain;
            }
            CpLocal2Gm(outGT[outStartIndex], paddingBufLt, thisPad);
            outStartIndex += thisPad;
            outRemain -= thisPad;
        }
    }

    /**
     * Initialize the padding buffer and choose routines based on the data type.
     */
    __aicore__ inline void InitializePaddingBuffer(const LocalTensor<uint8_t>& paddingFull)
    {
        if constexpr (std::is_same<VALUE_TYPE, int64_t>::value) {
            InitializeInt64Padding(paddingFull);
        } else {
            InitializeFloatPadding();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void InitializeFloatPadding()
    {
        LocalTensor<VALUE_TYPE> padF = paddingBufLt.ReinterpretCast<VALUE_TYPE>();
        VALUE_TYPE padValue;
        if constexpr (std::is_same<VALUE_TYPE, bfloat16_t>::value) {
            padValue = ToBfloat16(paddingValueFp32);
        } else {
            padValue = static_cast<VALUE_TYPE>(paddingValueFp32);
        }
        Duplicate<VALUE_TYPE>(padF, padValue, padChunkBytes / sizeof(VALUE_TYPE));
    }

    __aicore__ inline void InitializeInt64Padding(const LocalTensor<uint8_t>& paddingFull)
    {
#ifdef INT64_TYPE_USED_COPY_PADDING_UB
        InitializeInt64PaddingWithCopy(paddingFull);
#else
        LocalTensor<int64_t> dstInt64 = paddingBufLt.ReinterpretCast<int64_t>();
        Duplicate<int64_t>(dstInt64, paddingValueInt64, padChunkBytes / sizeof(int64_t));
#endif
    }

    /**
     * Initialize the int64 padding buffer via the Copy path.
     * Used when Duplicate does not support int64: build a 32B tile and then broadcast with Copy.
     * paddingFull provides the UB buffer that holds the temporary tile.
     */
    __aicore__ inline void InitializeInt64PaddingWithCopy(const LocalTensor<uint8_t>& paddingFull)
    {
        LocalTensor<uint32_t> dst32 = paddingBufLt.ReinterpretCast<uint32_t>();

        if (paddingValueInt64 == 0) {
            Duplicate<uint32_t>(dst32, 0, padChunkBytes / sizeof(uint32_t));
            return;
        }

        LocalTensor<int64_t> tile64 = paddingFull.ReinterpretCast<int64_t>();
        for (int32_t i = 0; i < (UB_BLOCK_SIZE / sizeof(int64_t)); i++) {
            tile64.SetValue(i, paddingValueInt64);
        }

        LocalTensor<uint32_t> tile32 = paddingFull.ReinterpretCast<uint32_t>();
        uint8_t reps = padChunkBytes / VECTOR_REG_WIDTH;
        if (reps != 0) {
            Copy<uint32_t>(dst32, tile32, VECTOR_REG_WIDTH / sizeof(uint32_t),
                reps, {1, 0, COPY_TILE_DST_REPEAT_SIZE, 0});
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
    float paddingValueFp32;
    int64_t paddingValueInt64;
    int64_t padChunkBytes;

    // ThisCoreLen
    int64_t lenOfThisCore;
    int64_t offsetOfThisCore;

    // Tpipe
    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, USE_QUEUE_NUM> inQueueX;
    TBuf<TPosition::VECOUT> paddingTbuf;

    // ThisCoreAddr
    GlobalTensor<uint8_t> valuesGT;
    GlobalTensor<uint8_t> outGT;
    LocalTensor<uint8_t> paddingBufLt;
};
}  // namespace JaggedToPaddedDense
#endif