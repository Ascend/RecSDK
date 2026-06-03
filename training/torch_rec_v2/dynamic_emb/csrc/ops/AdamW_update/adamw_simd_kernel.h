/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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

using namespace AscendC;

namespace dyn_emb_adamw_simd {

/// 扁平 grads + 每行 float*（w||m||v 连续 3*gradDim）；可选 founds 掩码
class AdamWSimd {
public:
    // UB 五槽：u0=g, u1=w, u2=m, u3=v, u4=scratch
    static constexpr uint64_t kUbSlotCount = 5ULL;
    // float 矢量搬运/计算需 32B 对齐，见 CANN「非对齐场景」
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
        const int64_t totalGradFloats = numRows * gradDim;
        gradsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(grads), static_cast<uint64_t>(totalGradFloats));
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

    __aicore__ inline __gm__ float* GetRowPtr(int32_t rowIdx) const
    {
        const uint64_t ptrVal = rowPtrsGm_.GetValue(static_cast<uint32_t>(rowIdx));
        return reinterpret_cast<__gm__ float*>(ptrVal);
    }

    __aicore__ inline void CopyGmToUb(const LocalTensor<float>& dst, __gm__ float* srcGm, uint32_t len) const
    {
        GlobalTensor<float> srcTensor;
        srcTensor.SetGlobalBuffer(srcGm, static_cast<uint64_t>(len));
        DataCopy(dst, srcTensor, len);
    }

    __aicore__ inline void CopyUbToGm(__gm__ float* dstGm, const LocalTensor<float>& src, uint32_t len) const
    {
        GlobalTensor<float> dstTensor;
        dstTensor.SetGlobalBuffer(dstGm, static_cast<uint64_t>(len));
        DataCopy(dstTensor, src, len);
    }

    __aicore__ inline void CopyGmToUbPad(const LocalTensor<float>& dst, __gm__ float* srcGm, uint32_t gradDim) const
    {
        const uint32_t alignedLen = GradDimAlignedElemCount(gradDim);
        GlobalTensor<float> srcTensor;
        srcTensor.SetGlobalBuffer(srcGm, static_cast<uint64_t>(alignedLen));
        const DataCopyExtParams copyParams{1, static_cast<uint32_t>(gradDim * sizeof(float)), 0, 0, 0};
        const DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
        DataCopyPad(dst, srcTensor, copyParams, padParams);
    }

    __aicore__ inline void CopyUbToGmPad(__gm__ float* dstGm, const LocalTensor<float>& src, uint32_t gradDim) const
    {
        const uint32_t alignedLen = GradDimAlignedElemCount(gradDim);
        GlobalTensor<float> dstTensor;
        dstTensor.SetGlobalBuffer(dstGm, static_cast<uint64_t>(alignedLen));
        const DataCopyExtParams copyParams{1, static_cast<uint32_t>(gradDim * sizeof(float)), 0, 0, 0};
        DataCopyPad(dstTensor, src, copyParams);
    }

    __aicore__ inline void CopyInGradToUbPad(const LocalTensor<float>& dst, int64_t gradBase, uint32_t gradDim) const
    {
        const DataCopyExtParams copyParams{1, static_cast<uint32_t>(gradDim * sizeof(float)), 0, 0, 0};
        const DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
        DataCopyPad(dst, gradsGm_[gradBase], copyParams, padParams);
    }

    __aicore__ inline void CopyInRow(uint32_t gradDim, int32_t rowIdx, int64_t gradBase, uint32_t ubOff,
                                     const LocalTensor<float>& u0, const LocalTensor<float>& u1,
                                     const LocalTensor<float>& u2, const LocalTensor<float>& u3) const
    {
        const LocalTensor<float> gUb = u0[ubOff];
        const LocalTensor<float> wUb = u1[ubOff];
        const LocalTensor<float> mUb = u2[ubOff];
        const LocalTensor<float> vUb = u3[ubOff];
        __gm__ float* rowPtr = GetRowPtr(rowIdx);
        if (IsGradDim32BAligned(gradDim)) {
            DataCopy(gUb, gradsGm_[gradBase], gradDim);
            CopyGmToUb(wUb, rowPtr, gradDim);
            CopyGmToUb(mUb, rowPtr + static_cast<int64_t>(gradDim), gradDim);
            CopyGmToUb(vUb, rowPtr + static_cast<int64_t>(gradDim) * 2, gradDim);
        } else {
            CopyInGradToUbPad(gUb, gradBase, gradDim);
            CopyGmToUbPad(wUb, rowPtr, gradDim);
            CopyGmToUbPad(mUb, rowPtr + static_cast<int64_t>(gradDim), gradDim);
            CopyGmToUbPad(vUb, rowPtr + static_cast<int64_t>(gradDim) * 2, gradDim);
        }
    }

    __aicore__ inline void CopyOutRow(uint32_t gradDim, int32_t rowIdx, uint32_t ubOff, const LocalTensor<float>& u1,
                                      const LocalTensor<float>& u2, const LocalTensor<float>& u3) const
    {
        const LocalTensor<float> wUb = u1[ubOff];
        const LocalTensor<float> mUb = u2[ubOff];
        const LocalTensor<float> vUb = u3[ubOff];
        __gm__ float* rowPtr = GetRowPtr(rowIdx);
        if (IsGradDim32BAligned(gradDim)) {
            CopyUbToGm(rowPtr, wUb, gradDim);
            CopyUbToGm(rowPtr + static_cast<int64_t>(gradDim), mUb, gradDim);
            CopyUbToGm(rowPtr + static_cast<int64_t>(gradDim) * 2, vUb, gradDim);
        } else {
            CopyUbToGmPad(rowPtr, wUb, gradDim);
            CopyUbToGmPad(rowPtr + static_cast<int64_t>(gradDim), mUb, gradDim);
            CopyUbToGmPad(rowPtr + static_cast<int64_t>(gradDim) * 2, vUb, gradDim);
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
        Adds<float>(u4, u4, eps, len);
        Sqrt<float>(u4, u4, len);
        Div<float>(u0, u2, u4, len);
        Muls<float>(u0, u0, stepSize, len);
        Muls<float>(u1, u1, decayFactor, len);
        Sub<float>(u1, u1, u0, len);
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
        DataCopy(u0, gradsGm_[gradBase], len);
        for (uint32_t r = 0; r < rowsInGroup; ++r) {
            const int32_t absRow = rowIdx + static_cast<int32_t>(r);
            const uint32_t ubOff = r * gradDim;
            __gm__ float* rowPtr = GetRowPtr(absRow);
            CopyGmToUb(u1[ubOff], rowPtr, gradDim);
            CopyGmToUb(u2[ubOff], rowPtr + static_cast<int64_t>(gradDim), gradDim);
            CopyGmToUb(u3[ubOff], rowPtr + static_cast<int64_t>(gradDim) * 2, gradDim);
        }
        SyncMte2V();
        ComputeAdamW(len, u0, u1, u2, u3, u4);
        SyncVMte3();
        for (uint32_t r = 0; r < rowsInGroup; ++r) {
            const int32_t absRow = rowIdx + static_cast<int32_t>(r);
            const uint32_t ubOff = r * gradDim;
            CopyOutRow(gradDim, absRow, ubOff, u1, u2, u3);
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

            CopyInRow(gradDim, absRow, rowGradBase, 0U, u0, u1, u2, u3);
            SyncMte2V();
            ComputeAdamW(computeLen, u0, u1, u2, u3, u4);
            SyncVMte3();
            CopyOutRow(gradDim, absRow, 0U, u1, u2, u3);
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
                if (!allActive || !IsGradDim32BAligned(gradDim)) {
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
    GlobalTensor<float> gradsGm_;
    GlobalTensor<uint64_t> rowPtrsGm_;
    GlobalTensor<uint8_t> foundsGm_;
    bool useFounds_{false};
    const __gm__ AdamWSimdTilingData* tiling_;
};

}  // namespace dyn_emb_adamw_simd
