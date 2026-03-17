/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#ifndef OFFSETS_RANGE_KERNEL_FUN_H
#define OFFSETS_RANGE_KERNEL_FUN_H

#include <cstdint>

#include "kernel_operator.h"
#include "kernel_common_utils.h"

using namespace AscendC;

namespace OffsetsRange {

constexpr int DATA_ALIGN_BYTES = 32;

struct Args {
    GM_ADDR offsets;
    GM_ADDR result;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename T>
class OffsetsRangeKernel {
public:
    __aicore__ inline OffsetsRangeKernel(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        offsetsLen = tilingData.offsetsLen;
        totalRows = tilingData.totalRows;
        rangeSize = tilingData.rangeSize;

        lutSize = tilingData.lutSize;
        blockDim = static_cast<int64_t>(GetBlockNum());

        offsets = args.offsets;
        result = args.result;
        workspace = args.workspace;

        offsetsGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args.offsets), offsetsLen);
        resultGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args.result), rangeSize);

        Tiling();
        pipe.InitBuffer(lutBuf, lutSize * sizeof(T));
        pipe.InitBuffer(queOut, 2, lutSize * sizeof(T));

        LUTInitUB();
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void Compute()
    {
        // 允许 0 元素，跳过空区间
        if (outEnd <= outStart) {
            return;
        }

        // pos 是全局输出位置
        int64_t pos = outStart;

        // 找初始 row：row = upper_bound(offsets, pos) - 1
        // upper_bound(pos) 等价于 lower_bound(pos+1)
        int64_t row = LowerBound(0, totalRows, pos + 1) - 1;
        if (row < 0) {
            row = 0;
        }

        // 初始化当前行的边界
        int64_t rowStart = static_cast<int64_t>(offsetsGT.GetValue(row));
        int64_t rowEnd = (row + 1 < totalRows) ? static_cast<int64_t>(offsetsGT.GetValue(row + 1)) : rangeSize;

        while (pos < outEnd) {
            // 推进 row：处理空行（rowEnd==rowStart）或 pos 跨过行尾的情况
            while (pos >= rowEnd && row + 1 < totalRows) {
                row++;
                rowStart = static_cast<int64_t>(offsetsGT.GetValue(row));
                rowEnd = (row + 1 < totalRows) ? static_cast<int64_t>(offsetsGT.GetValue(row + 1)) : rangeSize;
            }

            // 异常 offsets（非递增）避免死循环
            if (rowEnd < rowStart) {
                return;
            }

            // 行内偏移 written
            int64_t written = pos - rowStart;

            // 本行剩余、本核剩余
            int64_t remainInRow = rowEnd - pos;
            int64_t remainInCore = outEnd - pos;

            // chunk 受三者限制：lutSize、行剩余、核剩余
            int64_t chunk = (remainInRow < remainInCore) ? remainInRow : remainInCore;
            if (chunk > lutSize)
                chunk = lutSize;

            // 生成 [written, written+chunk) 并写回 result[pos:pos+chunk)
            if (written == 0) {
                CpLocal2Gm(resultGT[pos], lutLT, chunk);
                pipe_barrier(PIPE_MTE3);
            } else {
                auto rangeLt = queOut.AllocTensor<T>();
                AscendC::Adds<T>(rangeLt, lutLT, static_cast<T>(written), static_cast<int32_t>(chunk));
                pipe_barrier(PIPE_ALL);
                queOut.EnQue(rangeLt);

                auto outputLt = queOut.DeQue<T>();
                CpLocal2Gm(resultGT[pos], outputLt, chunk);
                pipe_barrier(PIPE_MTE3);
                queOut.FreeTensor(outputLt);
            }

            pos += chunk;
        }
    }

private:
    __aicore__ inline void LUTInitUB()
    {
        lutLT = lutBuf.Get<T>(lutSize);
        AscendC::CreateVecIndex(lutLT, static_cast<T>(0), static_cast<int32_t>(lutSize));
    }

    __aicore__ inline int64_t LowerBound(int64_t low, int64_t high, int64_t key)
    {
        while (low < high) {
            int64_t mid = (low + high) >> 1;
            int64_t value = static_cast<int64_t>(offsetsGT.GetValue(mid));
            if (value < key) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    }

    __aicore__ inline void Tiling()
    {
        // 输出区间：本核负责写 result[outStart, outEnd)
        outStart = 0;
        outEnd = 0;

        if (blockDim <= 0 || rangeSize <= 0) {
            return;
        }

        int32_t bid = static_cast<int32_t>(GetBlockIdx());
        if (bid < 0 || bid >= blockDim) {
            return;
        }

        outStart = (rangeSize * static_cast<int64_t>(bid)) / blockDim;
        outEnd = (rangeSize * static_cast<int64_t>(bid + 1)) / blockDim;

        // 最后一个核强制覆盖尾部，保证全集覆盖
        if (bid == static_cast<int32_t>(blockDim - 1)) {
            outEnd = rangeSize;
        }

        // 允许 0 元素
        if (outEnd <= outStart) {
            outStart = 0;
            outEnd = 0;
        }
    }

private:
    // GM_ADDR
    GM_ADDR offsets;
    GM_ADDR result;
    GM_ADDR workspace;

    // Shape
    int64_t offsetsLen;
    int64_t totalRows;
    int64_t rangeSize;

    // Tiling
    int64_t lutSize;
    int64_t blockDim;

    int64_t outStart;
    int64_t outEnd;

    // Pipe
    TPipe pipe;
    TQue<TPosition::VECOUT, 2> queOut;

    // Buffer
    TBuf<TPosition::VECCALC> lutBuf;

    // Global Tensor
    GlobalTensor<T> offsetsGT;
    GlobalTensor<T> resultGT;
    LocalTensor<T> lutLT;
};
}  // namespace OffsetsRange

#endif
