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
#pragma once

#include <cstdint>
#include "kernel_operator.h"
#include "adamw_simd_tiling.h"
#include "adamw_simd_dtype.h"

using namespace AscendC;

namespace dyn_emb_adamw_fused_simd {

using dyn_emb_adamw_simd::CopyGmToUbAsFloat;
using dyn_emb_adamw_simd::CopyGmToUbAsFloatPad;
using dyn_emb_adamw_simd::CopyUbFloatToGm;
using dyn_emb_adamw_simd::CopyUbFloatToGmPad;
using dyn_emb_adamw_simd::NeedsGmCopyPad;

template <typename GradT, typename WeightT>
class AdamWFusedSimd {
public:
    static constexpr uint64_t kUbSlotCount = 5ULL;
    static constexpr uint32_t kFloatElemsPer32B = 8U;

    __aicore__ inline static uint32_t GradDimAlignedElemCount(uint32_t gradDim)
    {
        return ((gradDim + kFloatElemsPer32B - 1U) / kFloatElemsPer32B) * kFloatElemsPer32B;
    }

    __aicore__ inline static bool IsGradDim32BAligned(uint32_t gradDim)
    {
        return (gradDim % kFloatElemsPer32B) == 0U;
    }

    __aicore__ inline explicit AdamWFusedSimd(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR values, const __gm__ AdamWSimdTilingData* tiling)
    {
        tiling_ = tiling;
        const int64_t numRows = static_cast<int64_t>(tiling_->numRows);
        const int64_t gradDim = static_cast<int64_t>(tiling_->gradDim);
        const int64_t valDim = static_cast<int64_t>(tiling_->valDim);
        const int64_t rowsPerGroup = static_cast<int64_t>(tiling_->rowsPerGroup);
        const int64_t totalGradElems = numRows * gradDim;
        const int64_t totalValElems = numRows * valDim;
        gradsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GradT*>(grads), static_cast<uint64_t>(totalGradElems));
        valuesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ WeightT*>(values), static_cast<uint64_t>(totalValElems));
        const uint32_t gradDimUb = GradDimAlignedElemCount(tiling_->gradDim);
        const uint64_t ubBytes =
            static_cast<uint64_t>(rowsPerGroup) * static_cast<uint64_t>(gradDimUb) * sizeof(float) * kUbSlotCount;
        pipe_->InitBuffer(ubMem_, ubBytes);
    }

    __aicore__ inline void CopyInRow(int64_t rowValBase, int64_t gradBase, uint32_t gradDim, uint32_t ubOff,
                                     const LocalTensor<float>& u0, const LocalTensor<float>& u1,
                                     const LocalTensor<float>& u2, const LocalTensor<float>& u3,
                                     const LocalTensor<float>& scratch) const
    {
        const LocalTensor<float> gUb = u0[ubOff];
        const LocalTensor<float> wUb = u1[ubOff];
        const LocalTensor<float> mUb = u2[ubOff];
        const LocalTensor<float> vUb = u3[ubOff];
        const uint32_t alignedLen = GradDimAlignedElemCount(gradDim);
        const bool gradDirectCopy = IsGradDim32BAligned(gradDim) && !NeedsGmCopyPad<GradT>(gradDim);
        const bool weightDirectCopy = IsGradDim32BAligned(gradDim) && !NeedsGmCopyPad<WeightT>(gradDim);
        if (gradDirectCopy) {
            CopyGmToUbAsFloat<GradT>(gUb, gradsGm_, gradBase, gradDim, scratch);
        } else {
            CopyGmToUbAsFloatPad<GradT>(gUb, gradsGm_, gradBase, gradDim, alignedLen, scratch);
        }
        if (weightDirectCopy) {
            CopyGmToUbAsFloat<WeightT>(wUb, valuesGm_, rowValBase, gradDim, scratch);
            CopyGmToUbAsFloat<WeightT>(mUb, valuesGm_, rowValBase + static_cast<int64_t>(gradDim), gradDim, scratch);
            CopyGmToUbAsFloat<WeightT>(vUb, valuesGm_, rowValBase + static_cast<int64_t>(gradDim) * 2, gradDim,
                                       scratch);
        } else {
            CopyGmToUbAsFloatPad<WeightT>(wUb, valuesGm_, rowValBase, gradDim, alignedLen, scratch);
            CopyGmToUbAsFloatPad<WeightT>(mUb, valuesGm_, rowValBase + static_cast<int64_t>(gradDim), gradDim,
                                          alignedLen, scratch);
            CopyGmToUbAsFloatPad<WeightT>(vUb, valuesGm_, rowValBase + static_cast<int64_t>(gradDim) * 2, gradDim,
                                          alignedLen, scratch);
        }
    }

    __aicore__ inline void CopyOutRow(int64_t rowValBase, uint32_t gradDim, uint32_t ubOff,
                                      const LocalTensor<float>& u1, const LocalTensor<float>& u2,
                                      const LocalTensor<float>& u3, const LocalTensor<float>& scratch) const
    {
        const LocalTensor<float> wUb = u1[ubOff];
        const LocalTensor<float> mUb = u2[ubOff];
        const LocalTensor<float> vUb = u3[ubOff];
        const bool weightDirectCopy = IsGradDim32BAligned(gradDim) && !NeedsGmCopyPad<WeightT>(gradDim);
        if (weightDirectCopy) {
            CopyUbFloatToGm<WeightT>(valuesGm_, rowValBase, wUb, gradDim, scratch);
            CopyUbFloatToGm<WeightT>(valuesGm_, rowValBase + static_cast<int64_t>(gradDim), mUb, gradDim, scratch);
            CopyUbFloatToGm<WeightT>(valuesGm_, rowValBase + static_cast<int64_t>(gradDim) * 2, vUb, gradDim, scratch);
        } else {
            CopyUbFloatToGmPad<WeightT>(valuesGm_, rowValBase, wUb, gradDim, scratch);
            CopyUbFloatToGmPad<WeightT>(valuesGm_, rowValBase + static_cast<int64_t>(gradDim), mUb, gradDim, scratch);
            CopyUbFloatToGmPad<WeightT>(valuesGm_, rowValBase + static_cast<int64_t>(gradDim) * 2, vUb, gradDim,
                                        scratch);
        }
    }

    __aicore__ inline void CopyInGroup(uint32_t rowsInGroup, int64_t rowValBase, int64_t gradBase, uint32_t gradDim,
                                       uint32_t valDim, LocalTensor<float>& u0, LocalTensor<float>& u1,
                                       LocalTensor<float>& u2, LocalTensor<float>& u3, LocalTensor<float>& scratch)
    {
        for (uint32_t r = 0; r < rowsInGroup; ++r) {
            const int64_t rowOff = static_cast<int64_t>(r) * static_cast<int64_t>(valDim);
            const int64_t gradOff = static_cast<int64_t>(r) * static_cast<int64_t>(gradDim);
            const uint32_t ubOff = r * gradDim;
            CopyInRow(rowValBase + rowOff, gradBase + gradOff, gradDim, ubOff, u0, u1, u2, u3, scratch);
        }
    }

    __aicore__ inline void CopyOutGroup(uint32_t rowsInGroup, int64_t rowValBase, uint32_t gradDim, uint32_t valDim,
                                        LocalTensor<float>& u1, LocalTensor<float>& u2, LocalTensor<float>& u3,
                                        LocalTensor<float>& scratch)
    {
        for (uint32_t r = 0; r < rowsInGroup; ++r) {
            const int64_t rowOff = static_cast<int64_t>(r) * static_cast<int64_t>(valDim);
            const uint32_t ubOff = r * gradDim;
            CopyOutRow(rowValBase + rowOff, gradDim, ubOff, u1, u2, u3, scratch);
        }
    }

    __aicore__ inline void ComputeAdamW(uint32_t len, LocalTensor<float>& u0, LocalTensor<float>& u1,
                                        LocalTensor<float>& u2, LocalTensor<float>& u3, LocalTensor<float>& u4) const
    {
        const float beta1 = tiling_->beta1;
        const float beta2 = tiling_->beta2;
        const float oneMinusBeta1 = tiling_->oneMinusBeta1;
        const float oneMinusBeta2 = tiling_->oneMinusBeta2;
        const float stepSize = tiling_->stepSize;
        const float invVHatDenom = tiling_->invVHatDenom;
        const float decayFactor = tiling_->decayFactor;
        const float eps = tiling_->eps;

        Mul<float>(u4, u0, u0, len);
        Muls<float>(u2, u2, beta1, len);
        Muls<float>(u0, u0, oneMinusBeta1, len);
        Add<float>(u2, u2, u0, len);

        Muls<float>(u3, u3, beta2, len);
        Muls<float>(u4, u4, oneMinusBeta2, len);
        Add<float>(u3, u3, u4, len);

        Muls<float>(u4, u3, invVHatDenom, len);
        Sqrt<float>(u4, u4, len);
        Adds<float>(u4, u4, eps, len);
        Div<float>(u0, u2, u4, len);
        Muls<float>(u0, u0, stepSize, len);
        Muls<float>(u1, u1, decayFactor, len);
        Sub<float>(u1, u1, u0, len);
    }

    __aicore__ inline void ProcessGroupPerRow(int32_t rowIdx, uint32_t rowsInGroup, uint32_t gradDim, uint32_t valDim)
    {
        const uint32_t computeLen = GradDimAlignedElemCount(gradDim);
        for (uint32_t r = 0; r < rowsInGroup; ++r) {
            const int32_t absRow = rowIdx + static_cast<int32_t>(r);
            const int64_t rowValBase = static_cast<int64_t>(absRow) * static_cast<int64_t>(valDim);
            const int64_t gradBase = static_cast<int64_t>(absRow) * static_cast<int64_t>(gradDim);
            LocalTensor<float> u = ubMem_.Get<float>();
            LocalTensor<float> u0 = u[0];
            LocalTensor<float> u1 = u[computeLen];
            LocalTensor<float> u2 = u[computeLen * 2U];
            LocalTensor<float> u3 = u[computeLen * 3U];
            LocalTensor<float> u4 = u[computeLen * 4U];

            CopyInRow(rowValBase, gradBase, gradDim, 0U, u0, u1, u2, u3, u4);
            event_t eMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eMte2ToV);
            ComputeAdamW(computeLen, u0, u1, u2, u3, u4);
            event_t eVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eVToMte3);
            CopyOutRow(rowValBase, gradDim, 0U, u1, u2, u3, u4);
            event_t eMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eMte3ToMte2);
        }
    }

    __aicore__ inline void ProcessGroupVector(int32_t rowIdx, uint32_t rowsInGroup, uint32_t gradDim, uint32_t valDim)
    {
        const uint32_t len = rowsInGroup * gradDim;
        const int64_t rowValBase = static_cast<int64_t>(rowIdx) * static_cast<int64_t>(valDim);
        const int64_t gradBase = static_cast<int64_t>(rowIdx) * static_cast<int64_t>(gradDim);
        LocalTensor<float> u = ubMem_.Get<float>();
        LocalTensor<float> u0 = u[0];
        LocalTensor<float> u1 = u[len];
        LocalTensor<float> u2 = u[len * 2U];
        LocalTensor<float> u3 = u[len * 3U];
        LocalTensor<float> u4 = u[len * 4U];

        CopyInGroup(rowsInGroup, rowValBase, gradBase, gradDim, valDim, u0, u1, u2, u3, u4);
        event_t eMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eMte2ToV);
        ComputeAdamW(len, u0, u1, u2, u3, u4);
        event_t eVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eVToMte3);
        CopyOutGroup(rowsInGroup, rowValBase, gradDim, valDim, u1, u2, u3, u4);
        event_t eMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eMte3ToMte2);
    }

    __aicore__ inline void Process()
    {
        const int32_t coreId = static_cast<int32_t>(GetBlockIdx());
        if (coreId >= tiling_->needCoreNum) {
            return;
        }
        const uint32_t gradDim = tiling_->gradDim;
        const uint32_t valDim = tiling_->valDim;
        const int32_t numRows = tiling_->numRows;
        const uint32_t rowsPerGroup = tiling_->rowsPerGroup;
        const int32_t stride = tiling_->needCoreNum;
        const int32_t numGroups =
            (numRows + static_cast<int32_t>(rowsPerGroup) - 1) / static_cast<int32_t>(rowsPerGroup);
        const bool usePerRow = !IsGradDim32BAligned(gradDim) || rowsPerGroup == 1U;

        for (int32_t groupIdx = coreId; groupIdx < numGroups; groupIdx += stride) {
            const int32_t rowIdx = groupIdx * static_cast<int32_t>(rowsPerGroup);
            const uint32_t rowsInGroup = static_cast<uint32_t>(
                (rowIdx + static_cast<int32_t>(rowsPerGroup) <= numRows) ? rowsPerGroup : (numRows - rowIdx));
            if (usePerRow) {
                ProcessGroupPerRow(rowIdx, rowsInGroup, gradDim, valDim);
            } else {
                ProcessGroupVector(rowIdx, rowsInGroup, gradDim, valDim);
            }
        }
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> ubMem_;
    GlobalTensor<GradT> gradsGm_;
    GlobalTensor<WeightT> valuesGm_;
    const __gm__ AdamWSimdTilingData* tiling_;
};

}  // namespace dyn_emb_adamw_fused_simd
