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

#ifndef CONCAT_JAGGED_TENSOR_GRAD_H
#define CONCAT_JAGGED_TENSOR_GRAD_H
#include "kernel_operator.h"

using namespace AscendC;

#define HardEvent_SetWait_Flag(event_name, SRC, DST)                                                             \
    {                                                                                                            \
        event_t event_name = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::SRC##_##DST)); \
        set_flag(PIPE_##SRC, PIPE_##DST, event_name);                                                            \
        wait_flag(PIPE_##SRC, PIPE_##DST, event_name);                                                           \
    }

namespace AscendC {
    template <typename T>
    class ConcatJaggedTensorGradKernel {
    public:
    __aicore__ inline ConcatJaggedTensorGradKernel() = default;

    __aicore__ inline void Init(GM_ADDR values, GM_ADDR result, GM_ADDR workspace,
    ConcatJaggedTensorGradTilingData *tilingData)
    {
        blockIdx_ = GetBlockIdx();

        jtNum_ = tilingData->jtNum;
        inputColSize_ = tilingData->inputColSize;
        ubMaxLength_ = tilingData->ubMaxLength;

        formerCore_ = tilingData->formerCore;
        tailCore_ = tilingData->tailCore;
        batchNumInTail_ = tilingData->batchNumInTail;
        batchNumInFormer_ = tilingData->batchNumInFormer;

        batchNum_ = blockIdx_ < formerCore_ ? tilingData->batchNumInFormer : tilingData->batchNumInTail;
        if (blockIdx_ < formerCore_) {
            blockBatchOffsetBase_ = blockIdx_ * tilingData->batchNumInFormer;
        } else {
            blockBatchOffsetBase_ = formerCore_ * tilingData->batchNumInFormer +
                                    (blockIdx_ - formerCore_) * tilingData->batchNumInTail;
        }

        pipe_.InitBuffer(moveQue_, ubMaxLength_ * sizeof(T));
    }
    __aicore__ inline void Process(GM_ADDR values, GM_ADDR result, GM_ADDR workspace,
    ConcatJaggedTensorGradTilingData *tilingData)
    {
        for (uint32_t i = 0; i < batchNum_; i++) {
            int64_t index = blockBatchOffsetBase_ + i;
            int64_t input_offset_base = tilingData->inputOffsetBegin[index] * inputColSize_;
            int64_t slice_size = tilingData->sliceSize[index] * inputColSize_;
            int64_t input_offset = 0;

            inputGm_.SetGlobalBuffer((__gm__ T *)values + input_offset_base);
            outputGm_.SetGlobalBuffer(
                (__gm__ T *)GetTensorAddr(result, index % jtNum_) +
                tilingData->outputOffsetBegin[index] * inputColSize_
            );
            while (slice_size > 0) {
                int64_t copySize = slice_size > ubMaxLength_ ? ubMaxLength_ : slice_size;

                LocalTensor<T> moveTensor = moveQue_.AllocTensor<T>();
                DataCopyExtParams copyParams{(uint16_t)1, static_cast<uint32_t>(copySize * sizeof(T)), 0, 0, 0};
                DataCopyPad(moveTensor, inputGm_[input_offset], copyParams, padParams);
                HardEvent_SetWait_Flag(xLocal_copyin, MTE2, MTE3);

                DataCopyPad(outputGm_[input_offset], moveTensor, copyParams);
                slice_size -= copySize;
                input_offset += copySize;
                HardEvent_SetWait_Flag(xLocal_copyout, MTE3, MTE2);

                moveQue_.FreeTensor(moveTensor);
            }
        }
    }
    private:
    __aicore__ inline __gm__ T *GetTensorAddr(GM_ADDR tensorList, uint32_t index)
    {
        __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorList);
        uint64_t tensorPtrOffset = *dataAddr;

        __gm__ uint64_t *tensorPtr = dataAddr + (tensorPtrOffset >> 3);
        return reinterpret_cast<__gm__ T *>(*(tensorPtr + index));
    }

    private:
    TPipe pipe_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    TBuf<QuePosition::VECCALC> moveQue_;

    uint32_t blockIdx_;
    uint32_t jtNum_;
    uint32_t inputColSize_;
    uint32_t ubMaxLength_;
    uint32_t formerCore_;
    uint32_t tailCore_;
    uint32_t batchNumInTail_;
    uint32_t batchNumInFormer_;

    uint32_t batchNum_;
    uint32_t blockBatchOffsetBase_;

    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    };
}
#endif
