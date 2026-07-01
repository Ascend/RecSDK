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

namespace dyn_emb_adamw_simd {

template <typename T>
__aicore__ inline void AdamWCompute(__local_mem__ T* dstW, __local_mem__ T* dstM1, __local_mem__ T* dstM2,
                                    __local_mem__ T* srcG, __local_mem__ T* srcW, __local_mem__ T* srcM1,
                                    __local_mem__ T* srcM2, uint32_t calCount, uint16_t repeatCount, uint32_t oneRepeat,
                                    float eps, float beta1, float oneMinusBeta1, float beta2, float oneMinusBeta2,
                                    float stepSize, float invVHatDenom, float decayFactor)
{
    AscendC::MicroAPI::RegTensor<T> dstVregG;
    AscendC::MicroAPI::RegTensor<T> dstVregW;
    AscendC::MicroAPI::RegTensor<T> dstVregM1;
    AscendC::MicroAPI::RegTensor<T> dstVregM2;
    AscendC::MicroAPI::RegTensor<T> srcVregG;
    AscendC::MicroAPI::RegTensor<T> srcVregW;
    AscendC::MicroAPI::RegTensor<T> srcVregM1;
    AscendC::MicroAPI::RegTensor<T> srcVregM2;
    AscendC::MicroAPI::MaskReg mask;

    for (uint16_t i = 0; i < repeatCount; ++i) {
        mask = AscendC::MicroAPI::UpdateMask<uint32_t>(calCount);
        AscendC::MicroAPI::DataCopy(srcVregG, srcG + i * oneRepeat);
        AscendC::MicroAPI::DataCopy(srcVregW, srcW + i * oneRepeat);
        AscendC::MicroAPI::DataCopy(srcVregM1, srcM1 + i * oneRepeat);
        AscendC::MicroAPI::DataCopy(srcVregM2, srcM2 + i * oneRepeat);

        AscendC::MicroAPI::Muls(dstVregM1, srcVregM1, beta1, mask);
        AscendC::MicroAPI::Muls(dstVregG, srcVregG, oneMinusBeta1, mask);
        AscendC::MicroAPI::Add(dstVregM1, dstVregM1, dstVregG, mask);

        AscendC::MicroAPI::Muls(dstVregM2, srcVregM2, beta2, mask);
        AscendC::MicroAPI::Mul(dstVregG, srcVregG, srcVregG, mask);
        AscendC::MicroAPI::Muls(dstVregG, dstVregG, oneMinusBeta2, mask);
        AscendC::MicroAPI::Add(dstVregM2, dstVregM2, dstVregG, mask);

        AscendC::MicroAPI::Muls(srcVregM2, dstVregM2, invVHatDenom, mask);
        AscendC::MicroAPI::Sqrt(srcVregM2, srcVregM2, mask);
        AscendC::MicroAPI::Adds(srcVregM2, srcVregM2, eps, mask);
        AscendC::MicroAPI::Div(dstVregG, dstVregM1, srcVregM2, mask);
        AscendC::MicroAPI::Muls(dstVregG, dstVregG, stepSize, mask);
        AscendC::MicroAPI::Muls(dstVregW, srcVregW, decayFactor, mask);
        AscendC::MicroAPI::Sub(dstVregW, dstVregW, dstVregG, mask);

        AscendC::MicroAPI::DataCopy(dstW + i * oneRepeat, dstVregW, mask);
        AscendC::MicroAPI::DataCopy(dstM1 + i * oneRepeat, dstVregM1, mask);
        AscendC::MicroAPI::DataCopy(dstM2 + i * oneRepeat, dstVregM2, mask);
    }
}

/// 扁平 grads(GradT) + 每行 WeightT*（w||m||v 连续 3*gradDim）；可选 founds 掩码
template <typename GradT, typename WeightT>
class AdamWSimd {
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

    __aicore__ inline explicit AdamWSimd(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR rowPtrs, GM_ADDR founds,
                                const __gm__ AdamWSimdTilingData* tiling)
    {
        tiling_ = tiling;
        const int64_t numRows = static_cast<int64_t>(tiling_->numRows);
        const int64_t gradDim = static_cast<int64_t>(tiling_->gradDim);
        const int64_t rowsPerGroup = static_cast<int64_t>(tiling_->rowsPerGroup);
        const int64_t totalGradElems = numRows * gradDim;
        gradsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GradT*>(grads), static_cast<uint64_t>(totalGradElems));
        rowPtrsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(rowPtrs), static_cast<uint64_t>(numRows));
        useFounds_ = (founds != nullptr);
        if (useFounds_) {
            foundsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(founds), static_cast<uint64_t>(numRows));
        }

        const uint32_t gradDimUb = GradDimAlignedElemCount(tiling_->gradDim);
        const uint64_t ubBytes =
            static_cast<uint64_t>(rowsPerGroup) * static_cast<uint64_t>(gradDimUb) * sizeof(float) * kUbSlotCount;
        pipe_->InitBuffer(ubMem_, ubBytes);
    }

    __aicore__ inline bool IsRowActive(int32_t rowIdx) const
    {
        if (!useFounds_) {
            return true;
        }
        return foundsGm_.GetValue(static_cast<uint32_t>(rowIdx)) != 0U;
    }

    __aicore__ inline __gm__ WeightT* GetRowPtr(int32_t rowIdx) const
    {
        const uint64_t ptrVal = rowPtrsGm_.GetValue(static_cast<uint32_t>(rowIdx));
        return reinterpret_cast<__gm__ WeightT*>(ptrVal);
    }

    __aicore__ inline void CopyInGradToUb(const LocalTensor<float>& dst, int64_t gradBase, uint32_t gradDim,
                                          uint32_t alignedLen, const LocalTensor<float>& scratch) const
    {
        if (IsGradDim32BAligned(gradDim) && !NeedsGmCopyPad<GradT>(gradDim)) {
            CopyGmToUbAsFloat<GradT>(dst, gradsGm_, gradBase, gradDim, scratch);
        } else {
            CopyGmToUbAsFloatPad<GradT>(dst, gradsGm_, gradBase, gradDim, alignedLen, scratch);
        }
    }

    __aicore__ inline void CopyInRow(uint32_t gradDim, int32_t rowIdx, int64_t gradBase, uint32_t ubOff,
                                     const LocalTensor<float>& u0, const LocalTensor<float>& u1,
                                     const LocalTensor<float>& u2, const LocalTensor<float>& u3,
                                     const LocalTensor<float>& scratch) const
    {
        const LocalTensor<float> gUb = u0[ubOff];
        const LocalTensor<float> wUb = u1[ubOff];
        const LocalTensor<float> mUb = u2[ubOff];
        const LocalTensor<float> vUb = u3[ubOff];
        __gm__ WeightT* rowPtr = GetRowPtr(rowIdx);
        const uint32_t alignedLen = GradDimAlignedElemCount(gradDim);
        CopyInGradToUb(gUb, gradBase, gradDim, alignedLen, scratch);
        const bool weightDirectCopy = IsGradDim32BAligned(gradDim) && !NeedsGmCopyPad<WeightT>(gradDim);
        if (weightDirectCopy) {
            CopyGmToUbAsFloat<WeightT>(wUb, rowPtr, gradDim, scratch);
            CopyGmToUbAsFloat<WeightT>(mUb, rowPtr + static_cast<int64_t>(gradDim), gradDim, scratch);
            CopyGmToUbAsFloat<WeightT>(vUb, rowPtr + static_cast<int64_t>(gradDim) * 2, gradDim, scratch);
        } else {
            CopyGmToUbAsFloatPad<WeightT>(wUb, rowPtr, gradDim, alignedLen, scratch);
            CopyGmToUbAsFloatPad<WeightT>(mUb, rowPtr + static_cast<int64_t>(gradDim), gradDim, alignedLen, scratch);
            CopyGmToUbAsFloatPad<WeightT>(vUb, rowPtr + static_cast<int64_t>(gradDim) * 2, gradDim, alignedLen,
                                          scratch);
        }
    }

    __aicore__ inline void CopyOutRow(uint32_t gradDim, int32_t rowIdx, uint32_t ubOff, const LocalTensor<float>& u1,
                                      const LocalTensor<float>& u2, const LocalTensor<float>& u3,
                                      const LocalTensor<float>& scratch) const
    {
        const LocalTensor<float> wUb = u1[ubOff];
        const LocalTensor<float> mUb = u2[ubOff];
        const LocalTensor<float> vUb = u3[ubOff];
        __gm__ WeightT* rowPtr = GetRowPtr(rowIdx);
        const bool weightDirectCopy = IsGradDim32BAligned(gradDim) && !NeedsGmCopyPad<WeightT>(gradDim);
        if (weightDirectCopy) {
            CopyUbFloatToGm<WeightT>(rowPtr, wUb, gradDim, scratch);
            CopyUbFloatToGm<WeightT>(rowPtr + static_cast<int64_t>(gradDim), mUb, gradDim, scratch);
            CopyUbFloatToGm<WeightT>(rowPtr + static_cast<int64_t>(gradDim) * 2, vUb, gradDim, scratch);
        } else {
            CopyUbFloatToGmPad<WeightT>(rowPtr, wUb, gradDim, scratch);
            CopyUbFloatToGmPad<WeightT>(rowPtr + static_cast<int64_t>(gradDim), mUb, gradDim, scratch);
            CopyUbFloatToGmPad<WeightT>(rowPtr + static_cast<int64_t>(gradDim) * 2, vUb, gradDim, scratch);
        }
    }

    __aicore__ inline void ComputeAdamW(uint32_t len, LocalTensor<float>& u0, LocalTensor<float>& u1,
                                        LocalTensor<float>& u2, LocalTensor<float>& u3, LocalTensor<float>& u4) const
    {
        (void)u4;
        const float beta1 = tiling_->beta1;
        const float beta2 = tiling_->beta2;
        const float oneMinusBeta1 = tiling_->oneMinusBeta1;
        const float oneMinusBeta2 = tiling_->oneMinusBeta2;
        const float stepSize = tiling_->stepSize;
        const float invVHatDenom = tiling_->invVHatDenom;
        const float decayFactor = tiling_->decayFactor;
        const float eps = tiling_->eps;

        __local_mem__ float* srcG = (__local_mem__ float*)u0.GetPhyAddr();
        __local_mem__ float* srcW = (__local_mem__ float*)u1.GetPhyAddr();
        __local_mem__ float* srcM1 = (__local_mem__ float*)u2.GetPhyAddr();
        __local_mem__ float* srcM2 = (__local_mem__ float*)u3.GetPhyAddr();
        __local_mem__ float* dstW = (__local_mem__ float*)u1.GetPhyAddr();
        __local_mem__ float* dstM1 = (__local_mem__ float*)u2.GetPhyAddr();
        __local_mem__ float* dstM2 = (__local_mem__ float*)u3.GetPhyAddr();

        constexpr uint32_t vecLen = AscendC::GetVecLen();
        constexpr uint32_t oneRepeat = vecLen / static_cast<uint32_t>(sizeof(float));
        const uint16_t repeatCount = static_cast<uint16_t>((len + oneRepeat - 1U) / oneRepeat);

        VF_CALL<AdamWCompute<float>>(dstW, dstM1, dstM2, srcG, srcW, srcM1, srcM2, len, repeatCount, oneRepeat, eps,
                                     beta1, oneMinusBeta1, beta2, oneMinusBeta2, stepSize, invVHatDenom, decayFactor);
    }

    __aicore__ inline void SyncMte2V() const
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(e);
        WaitFlag<HardEvent::MTE2_V>(e);
    }

    __aicore__ inline void SyncVMte3() const
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e);
        WaitFlag<HardEvent::V_MTE3>(e);
    }

    __aicore__ inline void SyncMte3Mte2() const
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(e);
        WaitFlag<HardEvent::MTE3_MTE2>(e);
    }

    __aicore__ inline void ProcessGroupVector(int32_t rowIdx, uint32_t rowsInGroup, uint32_t gradDim, int64_t gradBase)
    {
        const uint32_t len = rowsInGroup * gradDim;
        LocalTensor<float> u = ubMem_.Get<float>();
        LocalTensor<float> u0 = u[0];
        LocalTensor<float> u1 = u[len];
        LocalTensor<float> u2 = u[len * 2U];
        LocalTensor<float> u3 = u[len * 3U];
        LocalTensor<float> u4 = u[len * 4U];

        for (uint32_t r = 0; r < rowsInGroup; ++r) {
            const int32_t absRow = rowIdx + static_cast<int32_t>(r);
            const int64_t rowGradBase = gradBase + static_cast<int64_t>(r) * static_cast<int64_t>(gradDim);
            const uint32_t ubOff = r * gradDim;
            CopyInRow(gradDim, absRow, rowGradBase, ubOff, u0, u1, u2, u3, u4);
        }
        SyncMte2V();
        ComputeAdamW(len, u0, u1, u2, u3, u4);
        SyncVMte3();
        for (uint32_t r = 0; r < rowsInGroup; ++r) {
            const int32_t absRow = rowIdx + static_cast<int32_t>(r);
            const uint32_t ubOff = r * gradDim;
            CopyOutRow(gradDim, absRow, ubOff, u1, u2, u3, u4);
        }
        SyncMte3Mte2();
    }

    __aicore__ inline void ProcessGroupPerRow(int32_t rowIdx, uint32_t rowsInGroup, uint32_t gradDim, int64_t gradBase)
    {
        for (uint32_t r = 0; r < rowsInGroup; ++r) {
            const int32_t absRow = rowIdx + static_cast<int32_t>(r);
            if (!IsRowActive(absRow)) {
                continue;
            }
            const int64_t rowGradBase = gradBase + static_cast<int64_t>(r) * static_cast<int64_t>(gradDim);
            const uint32_t computeLen = GradDimAlignedElemCount(gradDim);
            LocalTensor<float> u = ubMem_.Get<float>();
            LocalTensor<float> u0 = u[0];
            LocalTensor<float> u1 = u[computeLen];
            LocalTensor<float> u2 = u[computeLen * 2U];
            LocalTensor<float> u3 = u[computeLen * 3U];
            LocalTensor<float> u4 = u[computeLen * 4U];

            CopyInRow(gradDim, absRow, rowGradBase, 0U, u0, u1, u2, u3, u4);
            SyncMte2V();
            ComputeAdamW(computeLen, u0, u1, u2, u3, u4);
            SyncVMte3();
            CopyOutRow(gradDim, absRow, 0U, u1, u2, u3, u4);
            SyncMte3Mte2();
        }
    }

    __aicore__ inline void Process()
    {
        const int32_t coreId = static_cast<int32_t>(GetBlockIdx());
        if (coreId >= tiling_->needCoreNum) {
            return;
        }

        const uint32_t gradDim = tiling_->gradDim;
        const int32_t numRows = tiling_->numRows;
        const uint32_t rowsPerGroup = tiling_->rowsPerGroup;
        const int32_t stride = tiling_->needCoreNum;
        const int32_t numGroups =
            (numRows + static_cast<int32_t>(rowsPerGroup) - 1) / static_cast<int32_t>(rowsPerGroup);

        for (int32_t groupIdx = coreId; groupIdx < numGroups; groupIdx += stride) {
            const int32_t rowIdx = groupIdx * static_cast<int32_t>(rowsPerGroup);
            const uint32_t rowsInGroup = static_cast<uint32_t>(
                (rowIdx + static_cast<int32_t>(rowsPerGroup) <= numRows) ? rowsPerGroup : (numRows - rowIdx));
            const int64_t gradBase = static_cast<int64_t>(rowIdx) * static_cast<int64_t>(gradDim);

            if (useFounds_) {
                ProcessGroupPerRow(rowIdx, rowsInGroup, gradDim, gradBase);
            } else {
                bool allActive = true;
                for (uint32_t r = 0; r < rowsInGroup; ++r) {
                    if (!IsRowActive(rowIdx + static_cast<int32_t>(r))) {
                        allActive = false;
                        break;
                    }
                }
                if (!allActive || !IsGradDim32BAligned(gradDim) || rowsPerGroup == 1U) {
                    ProcessGroupPerRow(rowIdx, rowsInGroup, gradDim, gradBase);
                } else {
                    ProcessGroupVector(rowIdx, rowsInGroup, gradDim, gradBase);
                }
            }
        }
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> ubMem_;
    GlobalTensor<GradT> gradsGm_;
    GlobalTensor<uint64_t> rowPtrsGm_;
    GlobalTensor<uint8_t> foundsGm_;
    bool useFounds_{false};
    const __gm__ AdamWSimdTilingData* tiling_;
};

}  // namespace dyn_emb_adamw_simd
