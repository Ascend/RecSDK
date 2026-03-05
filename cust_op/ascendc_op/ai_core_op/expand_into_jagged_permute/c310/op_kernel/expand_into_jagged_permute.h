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

#ifndef EXPAND_INTO_JAGGED_PERMUTE_H
#define EXPAND_INTO_JAGGED_PERMUTE_H

#include "common.h"

namespace ExpandIntoJaggedPermute {

template <typename DataType>
__simt_vf__ __launch_bounds__(SIMT_LAUNCH_BOUND)
    inline void ExpandIntoJaggedPermuteSimt(
        __gm__ DataType* output_permute,
        int64_t outputStart,
        int64_t segmentLength,
        int64_t inputStart)
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < static_cast<int64_t>(segmentLength);
         i += static_cast<int64_t>(blockDim.x)) {
        output_permute[outputStart + i] = static_cast<DataType>(inputStart + i);
    }
}

template <typename OutputType>
class ExpandIntoJaggedPermuteKernel {
public:
    __aicore__ inline ExpandIntoJaggedPermuteKernel(Args& args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        InitTilingParams(tilingData);
        InitGmParams(args);
        InitUbParams();
    }

    __aicore__ inline void Compute(Args& args)
    {
        ProcessTables(args);
    }

private:
    __aicore__ inline void InitTilingParams(const ExpandIntoJaggedPermuteTilingData& tilingData)
    {
        permuteLen = tilingData.permuteLen;
        offsetLen = permuteLen + 1;
        outputSize = tilingData.outputSize;
        splitBaseLen = tilingData.splitBaseLen;
        tailSplitIndex = tilingData.tailSplitIndex;
        ubCanUsed = tilingData.ubCanUsed;
    }

    __aicore__ inline void InitGmParams(const Args& args)
    {
        permuteGT.SetGlobalBuffer(reinterpret_cast<__gm__ OutputType*>(args.permute), permuteLen);
        inputOffsetsGT.SetGlobalBuffer(reinterpret_cast<__gm__ OutputType*>(args.input_offsets), offsetLen);
        outputOffsetsGT.SetGlobalBuffer(reinterpret_cast<__gm__ OutputType*>(args.output_offsets), offsetLen);
        outputPermuteGT.SetGlobalBuffer(reinterpret_cast<__gm__ OutputType*>(args.output_permute), outputSize);
    }

    __aicore__ inline void InitUbParams()
    {
        int64_t ubRemaining = static_cast<int64_t>(ubCanUsed);
        int64_t chunkBytes = AlignTo32(ubRemaining);

        pipe.InitBuffer(outChunkQ, 1, chunkBytes);

        chunkElements = chunkBytes / static_cast<int64_t>(sizeof(OutputType));
    }

    __aicore__ inline void ProcessTables(Args& args)
    {
        int64_t coreIdx = GetBlockIdx();
        int64_t tableCount;
        int64_t tableStart;
        if (coreIdx < tailSplitIndex) {
            tableCount = splitBaseLen + 1;
            tableStart = coreIdx * tableCount;
        } else {
            tableCount = splitBaseLen;
            tableStart = tailSplitIndex * (splitBaseLen + 1) + (coreIdx - tailSplitIndex) * splitBaseLen;
        }

        for (int64_t i = 0; i < tableCount; ++i) {
            int64_t tableIdx = tableStart + i;
            if (tableIdx >= permuteLen) {
                break;
            }
            ProcessSingleTable(args, tableIdx);
        }
    }

    __aicore__ inline void ProcessSingleTable(
        Args& args,
        int64_t tableIdx)
    {
        int64_t outputStart = static_cast<int64_t>(outputOffsetsGT.GetValue(tableIdx));
        int64_t outputEnd = static_cast<int64_t>(outputOffsetsGT.GetValue(tableIdx + 1));
        int64_t segmentLength = outputEnd - outputStart;
        if (segmentLength <= 0) {
            return;
        }

        int64_t originIdx = static_cast<int64_t>(permuteGT.GetValue(tableIdx));
        int64_t inputStart = static_cast<int64_t>(inputOffsetsGT.GetValue(originIdx));

        if (segmentLength <= THRESHOLD) {
            ProcessSimt(args, outputStart, segmentLength, inputStart);
        } else {
            ProcessSimd(outputStart, segmentLength, inputStart);
        }
    }

    __aicore__ inline void ProcessSimt(
        Args& args,
        int64_t outputStart,
        int64_t segmentLength,
        int64_t inputStart)
    {
        __gm__ OutputType* output_permute = reinterpret_cast<__gm__ OutputType*>(args.output_permute);

        asc_vf_call<ExpandIntoJaggedPermuteSimt<OutputType>>(
            dim3{static_cast<uint32_t>(DIM), 1, 1},
            output_permute,
            outputStart,
            segmentLength,
            inputStart);
    }

    __aicore__ inline void ProcessSimd(
        int64_t outputStart,
        int64_t segmentLength,
        int64_t inputStart)
    {
        LocalTensor<OutputType> chunkLt = outChunkQ.AllocTensor<OutputType>();
        int64_t processed = 0;
        while (processed < segmentLength) {
            int64_t remain = segmentLength - processed;
            int64_t cur = remain < chunkElements ? remain : chunkElements;

            CreateVecIndex(chunkLt, static_cast<OutputType>(inputStart + processed), cur);
            outChunkQ.EnQue(chunkLt);
            chunkLt = outChunkQ.DeQue<OutputType>();

            CpLocal2Gm(outputPermuteGT[outputStart + processed], chunkLt, cur);
            processed += cur;
        }

        pipe_barrier(PIPE_ALL);
        outChunkQ.FreeTensor(chunkLt);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECOUT, 1> outChunkQ;

    GlobalTensor<OutputType> permuteGT;
    GlobalTensor<OutputType> inputOffsetsGT;
    GlobalTensor<OutputType> outputOffsetsGT;
    GlobalTensor<OutputType> outputPermuteGT;

    int64_t permuteLen;
    int64_t offsetLen;
    int64_t outputSize;
    int64_t splitBaseLen;
    int64_t tailSplitIndex;
    uint64_t ubCanUsed;
    int64_t chunkElements;
};

}  // namespace ExpandIntoJaggedPermute

#endif  // EXPAND_INTO_JAGGED_PERMUTE_H