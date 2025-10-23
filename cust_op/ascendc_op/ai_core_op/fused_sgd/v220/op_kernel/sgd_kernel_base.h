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

#ifndef SGD_KERNEL_BASE_H
#define SGD_KERNEL_BASE_H

#include "kernel_operator.h"

template <typename T1, typename T2, bool isEnableDecay>
class SgdKernelBase {
protected:
    static constexpr uint32_t T1_DATA_BLOCK = AscendC::ONE_BLK_SIZE / sizeof(T1);
    static constexpr uint32_t T2_DATA_BLOCK = AscendC::ONE_BLK_SIZE / sizeof(T2);
    static constexpr uint32_t BUFFER_NUM = 1;

    __aicore__ inline SgdKernelBase() = default;

    __aicore__ inline void InitBase(GM_ADDR gradient, GM_ADDR indices, GM_ADDR inputVar, GM_ADDR learningRate,
                                    GM_ADDR outputVar, const SgdTilingData &tilingData)
    {
        if (AscendC::GetBlockIdx() < tilingData.splitCoreIndex) {
            procBs = tilingData.splitPrevCoreProcBs;
            offsetBs = AscendC::GetBlockIdx() * tilingData.splitPrevCoreProcBs;
        } else if (AscendC::GetBlockIdx() < tilingData.actualCoreNum) {
            procBs = tilingData.splitNextCoreProcBs;
            offsetBs = tilingData.splitCoreIndex * tilingData.splitPrevCoreProcBs +
                       (AscendC::GetBlockIdx() - tilingData.splitCoreIndex) * tilingData.splitNextCoreProcBs;
        } else {
            procBs = 0;
            offsetBs = 0;
        }

        batchSize = tilingData.batchSize;
        tableSize = tilingData.tableSize;
        dimSize = tilingData.dimSize;
        alignDimSize = AscendC::AlignUp(dimSize, T1_DATA_BLOCK);
        isDimAlign = (dimSize == alignDimSize);

        lrGm.SetGlobalBuffer((__gm__ T1*)learningRate, sizeof(float));
        neLr = T1(-1) * lrGm.GetValue(0);

        gradGm.SetGlobalBuffer((__gm__ T1 *)gradient + offsetBs * dimSize, batchSize * dimSize);
        indicesGm.SetGlobalBuffer((__gm__ T2 *)indices + offsetBs, batchSize);
        outputGm.SetGlobalBuffer((__gm__ T1 *)outputVar, tableSize * dimSize);

        eventIdMTE2ToS = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <typename T>
    __aicore__ inline void DataCopyIn(const AscendC::LocalTensor<T> &lt, const AscendC::GlobalTensor<T> &gt,
                                      uint32_t copyCnt, uint32_t copyLens, bool isAlign)
    {
        if (isAlign) {
            AscendC::DataCopy(lt, gt, copyCnt * copyLens);
        } else {
            AscendC::DataCopyParams copyParams = {
                static_cast<uint16_t>(copyCnt),
                static_cast<uint16_t>(copyLens * sizeof(T)),
                0,
                0,
            };

            AscendC::DataCopyPadParams padParams = { false, 0, 0, 0 };
            AscendC::DataCopyPad(lt, gt, copyParams, padParams);
        }
    }

    template <typename T>
    __aicore__ inline void DataCopyOut(const AscendC::GlobalTensor<T> &gt, const AscendC::LocalTensor<T> &lt,
                                       uint32_t copyCnt, uint32_t copyLens, bool isAlign)
    {
        if (isAlign) {
            AscendC::DataCopy(gt, lt, copyCnt * copyLens);
        } else {
            AscendC::DataCopyParams copyParams = {
                static_cast<uint16_t>(copyCnt),
                static_cast<uint16_t>(copyLens * sizeof(T)),
                0,
                0,
            };
            AscendC::DataCopyPad(gt, lt, copyParams);
        }
    }

    // mte2
    __aicore__ inline void DataCopyInGrad(int64_t offset, int64_t cnt)
    {
        gradUb = gradQue.AllocTensor<T1>();
        DataCopyIn<T1>(gradUb, gradGm[offset * dimSize], cnt, dimSize, isDimAlign);
        gradQue.EnQue(gradUb);
    }

    // mte2
    __aicore__ inline void DataCopyInIndices(int64_t offset, int64_t cnt)
    {
        indicesUb = indicesQue.AllocTensor<T2>();

        DataCopyIn<T2>(indicesUb, indicesGm[offset], 1, cnt, ((cnt % T2_DATA_BLOCK) == 0));
        indicesQue.EnQue(indicesUb);
    }

    __aicore__ inline void ScatterVarByIndices(int64_t cnt)
    {
        outputUb = outputQue.template DeQue<T1>();

        for (int i = 0; i < cnt; i++) {
            int64_t indices = indicesUb.GetValue(i);
            if (likely(indices >= 0)) {
                this->template DataCopyOut<T1>(outputGm[indices * dimSize], outputUb[i * alignDimSize], 1, dimSize,
                                               isDimAlign);
            }
        }

        indicesQue.template FreeTensor<T2>(indicesUb);
        outputQue.template FreeTensor<T1>(outputUb);
    }

    uint32_t batchSize{ 0 };
    uint32_t tableSize{ 0 };
    uint32_t dimSize{ 0 };
    uint32_t alignDimSize{ 0 };
    uint32_t offsetBs{ 0 };
    uint32_t procBs{ 0 };
    uint32_t nLoopBs{ 0 };
    bool isDimAlign{ true };
    T1 neLr{ 0.0f };

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T1> gradGm, outputGm, lrGm;
    AscendC::GlobalTensor<T2> indicesGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> gradQue, indicesQue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outputQue;
    AscendC::LocalTensor<T1> gradUb, outputUb;
    AscendC::LocalTensor<T2> indicesUb;

    event_t eventIdMTE2ToS;
};

#endif