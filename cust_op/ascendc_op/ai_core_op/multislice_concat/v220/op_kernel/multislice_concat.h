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

#ifndef MULTISLICE_CONCAT_FUN_H
#define MULTISLICE_CONCAT_FUN_H

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLOCK_SIZE = 32;

namespace AscendC {
template <typename T>
class MultisliceConcatKernel {
public:
    __aicore__ inline MultisliceConcatKernel() = delete;
    __aicore__ inline MultisliceConcatKernel(GM_ADDR input, GM_ADDR workspace,
                                             const MultisliceConcatTilingData* __restrict tiling)
    {
        blockIdx_ = GetBlockIdx();
        inputCol_ = tiling->colSize;
        formerCore_ = tiling->formerCore;
        concatNum_ = tiling->concatNum;
        maxTransColumnSize_ = tiling->maxProColumnNum;
        batchNum_ = blockIdx_ < formerCore_ ? tiling->batchNumInFormer : tiling->batchNumInTail;
        if (blockIdx_ < formerCore_) {
            inputOffset_ = blockIdx_ * tiling->batchNumInFormer;
        } else {
            inputOffset_ = formerCore_ * tiling->batchNumInFormer + (blockIdx_ - formerCore_) * tiling->batchNumInTail;
        }
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input) + inputOffset_ * inputCol_);
        dataInBlock_ = BLOCK_SIZE / sizeof(T);
        int64_t bufferAlign = ((batchNum_ * maxTransColumnSize_ + dataInBlock_ - 1) / dataInBlock_) * dataInBlock_;
        pipe_.InitBuffer(moveQue_, BUFFER_NUM, bufferAlign * sizeof(T));
    }

    __aicore__ inline void Process(GM_ADDR outputs, const MultisliceConcatTilingData* __restrict tiling)
    {
        int64_t offset = 0;
        for (int64_t j = 0; j < concatNum_; j++) {
            ConcatMove(j, offset, outputs, tiling);
            offset += tiling->concatSize[j];
        }
    }

private:
    __aicore__ inline void ConcatMove(int64_t concatId, int64_t offset, GM_ADDR outputs,
                                      const MultisliceConcatTilingData* __restrict tiling)
    {
        int64_t concatOffset = 0;
        int64_t outputCol = 0;
        for (int32_t i = 0; i < tiling->concatSize[concatId]; i++) {
            int32_t sliceOffset = offset + i;
            outputCol += tiling->sliceLength[sliceOffset];
        }

        __gm__ T* outputBaseTensorAddr = reinterpret_cast<__gm__ T*>(GetOutputTensorAddr(outputs, concatId));
        GlobalTensor<T> outputGlobal;
        for (int32_t i = 0; i < tiling->concatSize[concatId]; i++) {
            int64_t sliceOffset = offset + i;
            int64_t inputOffset = tiling->sliceBegin[sliceOffset];
            int64_t sliceLength = tiling->sliceLength[sliceOffset];
            int64_t transSize = 0;
            int64_t remainSize = sliceLength;
            while (remainSize > 0) {
                transSize = (remainSize > maxTransColumnSize_) ? maxTransColumnSize_ : remainSize;
                DataCopyExtParams copyParams{batchNum_, static_cast<uint32_t>(transSize * sizeof(T)),
                                             static_cast<uint32_t>((inputCol_ - transSize) * sizeof(T)), 0, 0};
                DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
                LocalTensor<T> moveTensor = moveQue_.AllocTensor<T>();
#ifdef UNSUPPORT_PADDING_MODE
                DataCopyPad<T>(moveTensor, inputGm_[inputOffset], copyParams, padParams);
#else
                DataCopyPad<T, PaddingMode::Compact>(moveTensor, inputGm_[inputOffset], copyParams, padParams);
#endif  // UNSUPPORT_PADDING_MODE
                moveQue_.EnQue<T>(moveTensor);

                moveTensor = moveQue_.DeQue<T>();
                outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputBaseTensorAddr) +
                                             inputOffset_ * outputCol + concatOffset);
                DataCopyExtParams copyOutParams{batchNum_, static_cast<uint32_t>(transSize * sizeof(T)), 0,
                                                static_cast<uint32_t>((outputCol - transSize) * sizeof(T)), 0};
#ifdef UNSUPPORT_PADDING_MODE
                DataCopyPad<T>(outputGlobal, moveTensor, copyOutParams);
#else
                DataCopyPad<T, PaddingMode::Compact>(outputGlobal, moveTensor, copyOutParams);
#endif  // UNSUPPORT_PADDING_MODE
                moveQue_.FreeTensor<T>(moveTensor);
                remainSize = remainSize - transSize;
                inputOffset += transSize;
                concatOffset += transSize;
            }
        }
    }

    __aicore__ inline __gm__ T* GetOutputTensorAddr(GM_ADDR outputs, uint16_t index)
    {
        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(outputs);
        uint64_t tensorPtrOffset = *dataAddr;

        __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
        return reinterpret_cast<__gm__ T*>(*(tensorPtr + index));
    }

private:
    TPipe pipe_;
    GlobalTensor<T> inputGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> moveQue_;

    uint32_t blockIdx_;
    uint64_t inputCol_;
    uint64_t formerCore_;
    uint16_t batchNum_;
    uint64_t inputOffset_;
    uint64_t concatNum_;
    uint64_t dataInBlock_;
    uint64_t maxTransColumnSize_;
};
}  // namespace AscendC
#endif