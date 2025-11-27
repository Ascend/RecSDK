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

namespace ReverseSequence_Kernel {
constexpr int32_t ALIGN_32 = 32;
constexpr int32_t ALIGN_16 = 16;

struct ReverseSequenceArgs {
    GM_ADDR input;
    GM_ADDR seqLengths;
    GM_ADDR output;

    int64_t batchSize;
    int64_t maxSeqLen;
    int64_t dataDim;

    int64_t handleNumOneTime;
    int64_t left;
    int64_t handleTotalCount;
};

template <typename dType, typename tType>
class ReverseSequence {
public:
    __aicore__ inline ReverseSequence(){};

    __aicore__ inline void init(ReverseSequenceArgs* args, TPipe* pipe)
    {
        this->args = args;
        this->pipe = pipe;

        thisId = GetBlockIdx();
        if (thisId < args->left) {
            args->handleTotalCount += 1;
            offsetStartPos = thisId * args->handleTotalCount;  // 当前ai core从哪一条data数据开始处理
        } else {
            offsetStartPos = (args->handleTotalCount + 1) * args->left + (thisId - args->left) * args->handleTotalCount;
        }

        intputGM.SetGlobalBuffer(reinterpret_cast<__gm__ dType*>(args->input),
                                 args->batchSize * args->maxSeqLen * args->dataDim);
        seqLengthsGM.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args->seqLengths), args->batchSize);
        outputGM.SetGlobalBuffer(reinterpret_cast<__gm__ dType*>(args->output),
                                 args->batchSize * args->maxSeqLen * args->dataDim);

        pipe->InitBuffer(inQueue, 1, args->dataDim * sizeof(dType));
        pipe->InitBuffer(outQueue, 1, args->dataDim * sizeof(dType));
    }

    __aicore__ inline void Compute()
    {
        for (int i = 0; i < args->handleTotalCount; ++i) {
            int64_t currentDataIndex = offsetStartPos + i;  // 当前是第多少条数据
            LocalTensor<dType> localIn = inQueue.AllocTensor<dType>();
            DataCopy(localIn, intputGM[currentDataIndex * args->dataDim], args->dataDim);
            inQueue.EnQue(localIn);

            LocalTensor<dType> localOut = outQueue.AllocTensor<dType>();
            LocalTensor<dType> localInCopy = inQueue.DeQue<dType>();
            DataCopy(localOut, localInCopy, args->dataDim);
            outQueue.EnQue(localOut);
            LocalTensor<dType> localOutCopy = outQueue.DeQue<dType>();

            int64_t bsIndex = currentDataIndex / args->maxSeqLen;  // 当前数据属于第几个sample
            int64_t seqLenValue = seqLengthsGM.GetValue(bsIndex);  // 当前sample对应的seqLen值
            int64_t maxSeqLenIndex = currentDataIndex % args->maxSeqLen;  // 当前数据在当前sample内的索引
            if (maxSeqLenIndex < seqLenValue) {
                // 执行逆序拷贝
                int64_t maxSeqLenReverseIndex = seqLenValue - maxSeqLenIndex - 1;
                int64_t globalTargetIndex = bsIndex * args->maxSeqLen + maxSeqLenReverseIndex;  // 拷贝到该条数据对应位置
                DataCopy(outputGM[globalTargetIndex * args->dataDim], localOutCopy, args->dataDim);
            }
            inQueue.FreeTensor(localInCopy);
            outQueue.FreeTensor(localOutCopy);
        }
    }

    TPipe* pipe;
    int64_t align;
    int64_t thisId;
    int64_t offsetStartPos;
    ReverseSequenceArgs* args;

    GlobalTensor<dType> intputGM;
    GlobalTensor<tType> seqLengthsGM;
    GlobalTensor<dType> outputGM;

    TQue<QuePosition::VECIN, 1> seqLenInQueue;

    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
};
}  // namespace ReverseSequence_Kernel

// call of kernel function
extern "C" __global__ __aicore__ void reverse_sequence(GM_ADDR input, GM_ADDR seq_lengths, GM_ADDR output,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    ReverseSequence_Kernel::ReverseSequenceArgs args{input,
                                                     seq_lengths,
                                                     output,
                                                     tiling_data.batchSize,
                                                     tiling_data.maxSeqLen,
                                                     tiling_data.dataDim,
                                                     tiling_data.handleNumOneTime,
                                                     tiling_data.left,
                                                     tiling_data.handleTotalCount};

    TPipe pipe;
    ReverseSequence_Kernel::ReverseSequence<DTYPE_INPUT, DTYPE_SEQ_LENGTHS> kernel;
    kernel.init(&args, &pipe);
    kernel.Compute();
}