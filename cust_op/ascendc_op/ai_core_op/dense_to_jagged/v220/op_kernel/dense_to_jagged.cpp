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

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

namespace DenseToJagged_Kernel {
constexpr int32_t ALIGN_32 = 32;
constexpr int32_t ALIGN_16 = 16;
constexpr int32_t TYPE_FLOAT = 0;
constexpr int32_t TYPE_INT32 = 3;
constexpr int32_t TYPE_INT64 = 9;

struct DenseToJaggedArgs {
    GM_ADDR dense;
    GM_ADDR offset;
    GM_ADDR jagged_dense;

    int32_t denseDim1;
    int32_t denseDim2;
    int32_t left;
    int32_t singleCoreBatch;
    int32_t singleLoopSize;
    int64_t denseTotal;
    int64_t jaggedTotal;
};

template<typename dType, typename tType>
class DenseToJagged {
public:
    __aicore__ inline DenseToJagged() {};

    __aicore__ inline void init(DenseToJaggedArgs *args, TPipe *pipe)
    {
        this->args = args;
        this->pipe = pipe;

        thisId = GetBlockIdx();
        if (thisId < args->left) {
            args->singleCoreBatch += 1;
            offsetStartPos = thisId * args->singleCoreBatch;
        } else {
            offsetStartPos = (args->singleCoreBatch + 1) * args->left + (thisId - args->left) * args->singleCoreBatch;
        }

        align = sizeof(dType);
        denseGb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args->dense), args->denseTotal * align);
        jaggedDenseGb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args->jagged_dense), args->jaggedTotal * align);

        pipe->InitBuffer(inQueue, 1, args->singleLoopSize);
        pipe->InitBuffer(outQueue, 1, args->singleLoopSize);
    }

    __aicore__ inline void Compute()
    {
        ComputeEachBatch();
    }
private:

    __aicore__ inline void DataCopyPadLocal2Gm(const GlobalTensor<uint8_t>& gt, const LocalTensor<uint8_t>& lt,
                                               uint32_t unAlignLen)
    {
        GlobalTensor<uint16_t> uint16Gt;
        uint16Gt.SetGlobalBuffer((__gm__ uint16_t*)gt.GetPhyAddr(), unAlignLen/2);
        LocalTensor<uint16_t> uint16Lt = lt.template ReinterpretCast<uint16_t>();

        uint32_t len = unAlignLen / 2;
        SetAtomicAdd<uint16_t>();
        uint64_t mask0 = (1uL << 16) - (1uL << len);
        uint64_t mask[2] = {mask0, 0};
        Duplicate<uint16_t>(uint16Lt, 0, mask, 1, 1, 1);
        pipe_barrier(PIPE_ALL);

        DataCopy(uint16Gt, uint16Lt, ALIGN_16);
        SetAtomicNone();
    }

    __aicore__ inline void ComputeEachBatch()
    {
        tType jaggedPos;
        tType jaggedPosNext;

        __gm__ tType *oPtr = (__gm__ tType *)args->offset;

        for (int i = 0; i < args->singleCoreBatch; i++) {
            // Get information form offset tensor to jag dense
            jaggedPos = *(oPtr + offsetStartPos + i);
            jaggedPosNext = *(oPtr + offsetStartPos + i + 1);
            int copyRows = jaggedPosNext - jaggedPos;

            // Get jagged Global tensor with offset
            GlobalTensor<uint8_t> jaggedDenseCopyGb = jaggedDenseGb[jaggedPos * args->denseDim2 * align];
            int64_t densePos = static_cast<int64_t>(offsetStartPos + i) * args->denseDim2 * args->denseDim1 * align;
            GlobalTensor<uint8_t> denseCopyGb = denseGb[densePos];

            // When offset[n] - offset[n + 1] > dense dim1, only need to copy dense dim1 * dim2
            // otherwise, copy (offset[n] - offset[n + 1]) * dense dim2
            int64_t remainLen = copyRows > args->denseDim1 ? (args->denseDim1 * args->denseDim2 * align) :
                (copyRows * args->denseDim2 * align);
            while (remainLen > 0) {
                // args->singleLoopSize - ALIGN_32 to avoid overAlignLen exceed singleLoopSize
                int64_t thisLen = args->singleLoopSize - ALIGN_32;
                if (remainLen < (args->singleLoopSize - ALIGN_32)) {
                    thisLen = remainLen;
                }

                LocalTensor<uint8_t> localIn = inQueue.AllocTensor<uint8_t>();
                LocalTensor<uint8_t> localOut = outQueue.AllocTensor<uint8_t>();

                uint32_t overAlignLen = (thisLen + ALIGN_32 - 1) / ALIGN_32 * ALIGN_32;
                uint32_t alignLen = thisLen / ALIGN_32 * ALIGN_32;
                uint32_t unAlignLen = thisLen - alignLen;

                // Copy over aligned size to avoid dealing unaligned tail
                DataCopy(localIn, denseCopyGb, overAlignLen);
                inQueue.EnQue(localIn);

                LocalTensor<uint8_t> localInCopy = inQueue.DeQue<uint8_t>();

                // Copy from input to output queue
                DataCopy(localOut, localInCopy, overAlignLen);
                outQueue.EnQue(localOut);

                LocalTensor<uint8_t> localOutCopy = outQueue.DeQue<uint8_t>();

                // Copy aligned size, left unaligned tail for DataCopyPad to deal with
                if (alignLen != 0) {
                    DataCopy(jaggedDenseCopyGb, localOutCopy, alignLen);
                }

                if (unAlignLen != 0) {
#ifndef SUPPORT_V200
                    const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
                    DataCopyPad(jaggedDenseCopyGb[alignLen], localOutCopy[alignLen], dataCopyExtParams);
#else
                    DataCopyPadLocal2Gm(jaggedDenseCopyGb[alignLen], localOutCopy[alignLen], unAlignLen);
#endif
                }

                jaggedDenseCopyGb = jaggedDenseCopyGb[thisLen];
                denseCopyGb = denseCopyGb[thisLen];
                inQueue.FreeTensor(localInCopy);
                outQueue.FreeTensor(localOutCopy);
                remainLen = remainLen - thisLen;
            }
        }
    }

    TPipe *pipe;
    int32_t align;
    int32_t thisId;
    int32_t offsetStartPos;
    DenseToJaggedArgs *args;

    GlobalTensor<uint8_t> denseGb;
    GlobalTensor<uint8_t> jaggedDenseGb;

    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
};
} // DenseToJagged_Kernel

// call of kernel function
extern "C" __global__ __aicore__ void dense_to_jagged(GM_ADDR dense, GM_ADDR offset, GM_ADDR jagged_dense,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    DenseToJagged_Kernel::DenseToJaggedArgs args {
        dense, offset, jagged_dense, tiling_data.denseDim1, tiling_data.denseDim2, tiling_data.left,
        tiling_data.singleCoreBatch, tiling_data.singleLoopSize, tiling_data.denseTotal, tiling_data.jaggedTotal
    };

    TPipe pipe;
    int32_t floatType = DenseToJagged_Kernel::TYPE_FLOAT; // 0
    int32_t int32Type = DenseToJagged_Kernel::TYPE_INT32; // 3
    int32_t int64Type = DenseToJagged_Kernel::TYPE_INT64; // 9

    // Init DenseToJagged class with different data type according to dense and offset data type.
    if (tiling_data.denseType == floatType && tiling_data.offsetType == int32Type) {
        DenseToJagged_Kernel::DenseToJagged<float, int32_t> kernel;
        kernel.init(&args, &pipe);
        kernel.Compute();
    } else if (tiling_data.denseType == floatType && tiling_data.offsetType == int64Type) {
        DenseToJagged_Kernel::DenseToJagged<float, int64_t> kernel;
        kernel.init(&args, &pipe);
        kernel.Compute();
    } else if (tiling_data.denseType == int64Type && tiling_data.offsetType == int64Type) {
        DenseToJagged_Kernel::DenseToJagged<int64_t, int64_t> kernel;
        kernel.init(&args, &pipe);
        kernel.Compute();
    } else if (tiling_data.denseType == int64Type && tiling_data.offsetType == int32Type) {
        DenseToJagged_Kernel::DenseToJagged<int64_t, int32_t> kernel;
        kernel.init(&args, &pipe);
        kernel.Compute();
    }
}