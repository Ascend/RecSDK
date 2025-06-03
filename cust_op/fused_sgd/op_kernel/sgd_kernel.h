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
#ifndef SGD_KERNEL_H
#define SGD_KERNEL_H

#include "sgd_kernel_base.h"

// 主模板声明（可无具体实现）
template <typename T1, typename T2, bool isEnableDecay>
class SgdKernel;

// 特化版本1 针对不带weight decay场景的SGD
template <typename T1, typename T2>
class SgdKernel<T1, T2, false> : public SgdKernelBase<T1, T2, false> {
public:
    using Base = SgdKernelBase<T1, T2, false>;
    using Base::Base;
    static constexpr uint32_t UB_BUFFER_SIZE = 2;

    __aicore__ inline SgdKernel() = default;

    __aicore__ inline void Init(GM_ADDR gradient, GM_ADDR indices, GM_ADDR inputVar, GM_ADDR learningRate,
                                GM_ADDR outputVar, const SgdTilingData &tilingData)
    {
        Base::InitBase(gradient, indices, inputVar, learningRate, outputVar, tilingData);
        this->InitUbBuffer(tilingData.ubFreeSize);
    }

    __aicore__ inline void Process()
    {
        auto nLoop = Base::procBs / Base::nLoopBs;
        for (auto i = 0; i < nLoop; i++) {
            this->DataCopyInIndices(i * Base::nLoopBs, Base::nLoopBs);
            this->DataCopyInGrad(i * Base::nLoopBs, Base::nLoopBs);
            this->Compute(Base::nLoopBs);
            this->ScatterVarByIndices(Base::nLoopBs);
        }

        auto nTail = Base::procBs % Base::nLoopBs;
        if (nTail != 0) {
            this->DataCopyInIndices(nLoop * Base::nLoopBs, nTail);
            this->DataCopyInGrad(nLoop * Base::nLoopBs, nTail);
            this->Compute(nTail);
            this->ScatterVarByIndices(nTail);
        }
    }

private:
    __aicore__ inline void InitUbBuffer(uint32_t ubFreeByteSize)
    {
        Base::nLoopBs = ubFreeByteSize / (Base::alignDimSize * (UB_BUFFER_SIZE * sizeof(T1)) + sizeof(T2));
        Base::nLoopBs = Base::nLoopBs / Base::T2_DATA_BLOCK * Base::T2_DATA_BLOCK;
        ASSERT(Base::nLoopBs != 0 && "nLoopBs cant be zeros!");

        Base::pipe.InitBuffer(Base::gradQue, Base::BUFFER_NUM, Base::nLoopBs * Base::alignDimSize * sizeof(T1));
        Base::pipe.InitBuffer(Base::indicesQue, Base::BUFFER_NUM, Base::nLoopBs * sizeof(T2));
        Base::pipe.InitBuffer(Base::outputQue, Base::BUFFER_NUM, Base::nLoopBs * Base::alignDimSize * sizeof(T1));
    }

    __aicore__ inline void ScatterVarByIndices(int64_t cnt)
    {
        Base::indicesQue.template DeQue<T2>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(Base::eventIdMTE2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(Base::eventIdMTE2ToS);

        AscendC::SetAtomicAdd<T1>();
        Base::ScatterVarByIndices(cnt);
        AscendC::SetAtomicNone();
    }

    __aicore__ inline void Compute(int64_t cnt)
    {
        auto ComputeLength = cnt * Base::alignDimSize;
        Base::outputUb = Base::outputQue.template AllocTensor<T1>();

        Base::gradUb = Base::gradQue.template DeQue<T1>();
        AscendC::Muls<T1>(Base::outputUb, Base::gradUb, Base::neLr, ComputeLength);
        Base::gradQue.template FreeTensor<T1>(Base::gradUb);

        Base::outputQue.EnQue(Base::outputUb);
    }
};

// 特化版本2 针对带weight decay场景的SGD
template <typename T1, typename T2>
class SgdKernel<T1, T2, true> : public SgdKernelBase<T1, T2, true> {
public:
    using Base = SgdKernelBase<T1, T2, true>;
    using Base::Base;
    static constexpr uint32_t UB_BUFFER_SIZE = 3;

    __aicore__ inline SgdKernel() = default;

    __aicore__ inline void Init(GM_ADDR gradient, GM_ADDR indices, GM_ADDR inputVar, GM_ADDR learningRate,
                                GM_ADDR outputVar, const SgdTilingData &tilingData)
    {
        Base::InitBase(gradient, indices, inputVar, learningRate, outputVar, tilingData);

        inputVarGm.SetGlobalBuffer((__gm__ T1 *)inputVar, Base::tableSize * Base::dimSize);

        this->InitUbBuffer(tilingData.ubFreeSize);
        weightDecay = tilingData.weightDecay;
    }

    __aicore__ inline void Process()
    {
        auto nLoop = Base::procBs / Base::nLoopBs;
        for (auto i = 0; i < nLoop; i++) {
            this->DataCopyInIndices(i * Base::nLoopBs, Base::nLoopBs);
            this->GatherVarByIndices(i * Base::nLoopBs, Base::nLoopBs);
            this->DataCopyInGrad(i * Base::nLoopBs, Base::nLoopBs);
            this->Compute(Base::nLoopBs);
            this->ScatterVarByIndices(Base::nLoopBs);
        }

        auto nTail = Base::procBs % Base::nLoopBs;
        if (nTail != 0) {
            this->DataCopyInIndices(nLoop * Base::nLoopBs, nTail);
            this->GatherVarByIndices(nLoop * Base::nLoopBs, nTail);
            this->DataCopyInGrad(nLoop * Base::nLoopBs, nTail);
            this->Compute(nTail);
            this->ScatterVarByIndices(nTail);
        }
    }

private:
    __aicore__ inline void InitUbBuffer(uint32_t ubFreeByteSize)
    {
        Base::nLoopBs = ubFreeByteSize / (Base::alignDimSize * (UB_BUFFER_SIZE * sizeof(T1)) + sizeof(T2));
        Base::nLoopBs = Base::nLoopBs / Base::T2_DATA_BLOCK * Base::T2_DATA_BLOCK;
        ASSERT(Base::nLoopBs != 0 && "nLoopBs cant be zeros!");

        Base::pipe.InitBuffer(varQue, Base::BUFFER_NUM, Base::nLoopBs * Base::alignDimSize * sizeof(T1));
        Base::pipe.InitBuffer(Base::gradQue, Base::BUFFER_NUM, Base::nLoopBs * Base::alignDimSize * sizeof(T1));
        Base::pipe.InitBuffer(Base::indicesQue, Base::BUFFER_NUM, Base::nLoopBs * sizeof(T2));
        Base::pipe.InitBuffer(Base::outputQue, Base::BUFFER_NUM, Base::nLoopBs * Base::alignDimSize * sizeof(T1));
    }

    __aicore__ inline void GatherVarByIndices(int64_t offset, int64_t cnt)
    {
        Base::indicesQue.template DeQue<T2>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(Base::eventIdMTE2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(Base::eventIdMTE2ToS);

        varUb = varQue.template AllocTensor<T1>();
        for (int i = 0; i < cnt; i++) {
            int64_t indices = Base::indicesUb.GetValue(i);
            if (likely(indices >= 0)) {
                this->template DataCopyIn<T1>(varUb[i * Base::alignDimSize], inputVarGm[indices * Base::dimSize], 1,
                                              Base::dimSize, Base::isDimAlign);
            }
        }

        varQue.EnQue(varUb);
    }

    __aicore__ inline void Compute(int64_t cnt)
    {
        auto ComputeLength = cnt * Base::alignDimSize;
        Base::outputUb = Base::outputQue.template AllocTensor<T1>();

        varUb = varQue.template DeQue<T1>();
        AscendC::Muls<T1>(Base::outputUb, varUb, this->weightDecay, ComputeLength);
        varQue.template FreeTensor<T1>(varUb);

        Base::gradUb = Base::gradQue.template DeQue<T1>();
        AscendC::Add<T1>(Base::outputUb, Base::gradUb, Base::outputUb, ComputeLength);
        Base::gradQue.template FreeTensor<T1>(Base::gradUb);

        AscendC::Muls<T1>(Base::outputUb, Base::outputUb, Base::neLr, ComputeLength);
        Base::outputQue.EnQue(Base::outputUb);
    }

    __aicore__ inline void ScatterVarByIndices(int64_t cnt)
    {
        AscendC::SetAtomicAdd<T1>();
        Base::ScatterVarByIndices(cnt);
        AscendC::SetAtomicNone();
    }

    AscendC::GlobalTensor<T1> inputVarGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, Base::BUFFER_NUM> varQue;
    AscendC::LocalTensor<T1> varUb;
    T1 weightDecay{ 0.0f };
};

#endif