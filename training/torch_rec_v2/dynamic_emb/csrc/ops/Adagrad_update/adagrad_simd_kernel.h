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
#include "adagrad_simd_tiling.h"

using namespace AscendC;

namespace dyn_emb_adagrad_simd {

/// 扁平 grads + 每行 float*（w||acc 连续 2*gradDim）；可选 founds 掩码
class AdagradSimd {
public:
    // UB 四槽：u0=g, u1=w, u2=acc, u3=scratch
    static constexpr uint64_t kUbSlotCount = 4ULL;

    __aicore__ inline explicit AdagradSimd(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR rowPtrs, GM_ADDR founds,
        const __gm__ AdagradSimdTilingData* tiling)
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

        const uint64_t ubBytes = static_cast<uint64_t>(rowsPerGroup) * static_cast<uint64_t>(tiling_->gradDim) *
            sizeof(float) * kUbSlotCount;
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

    __aicore__ inline void CopyInRow(uint32_t gradDim, int32_t rowIdx, int64_t gradBase, uint32_t ubOff,
        const LocalTensor<float>& u0, const LocalTensor<float>& u1, const LocalTensor<float>& u2) const
    {
        const LocalTensor<float> gUb = u0[ubOff];
        const LocalTensor<float> wUb = u1[ubOff];
        const LocalTensor<float> aUb = u2[ubOff];
        DataCopy(gUb, gradsGm_[gradBase], gradDim);
        __gm__ float* rowPtr = GetRowPtr(rowIdx);
        CopyGmToUb(wUb, rowPtr, gradDim);
        CopyGmToUb(aUb, rowPtr + static_cast<int64_t>(gradDim), gradDim);
    }

    __aicore__ inline void CopyOutRow(uint32_t gradDim, int32_t rowIdx, uint32_t ubOff, const LocalTensor<float>& u1,
        const LocalTensor<float>& u2) const
    {
        const LocalTensor<float> wUb = u1[ubOff];
        const LocalTensor<float> aUb = u2[ubOff];
        __gm__ float* rowPtr = GetRowPtr(rowIdx);
        CopyUbToGm(rowPtr, wUb, gradDim);
        CopyUbToGm(rowPtr + static_cast<int64_t>(gradDim), aUb, gradDim);
    }

    __aicore__ inline void ComputeAdagrad(uint32_t len, LocalTensor<float>& u0, LocalTensor<float>& u1,
        LocalTensor<float>& u2, LocalTensor<float>& u3) const
    {
        const float lr = tiling_->lr;
        const float eps = tiling_->eps;
        Mul<float>(u3, u0, u0, len);
        Add<float>(u2, u2, u3, len);
        Sqrt<float>(u3, u2, len);
        Adds<float>(u3, u3, eps, len);
        Div<float>(u0, u0, u3, len);
        Muls<float>(u0, u0, lr, len);
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
        DataCopy(u0, gradsGm_[gradBase], len);
        for (uint32_t r = 0; r < rowsInGroup; ++r) {
            const int32_t absRow = rowIdx + static_cast<int32_t>(r);
            const uint32_t ubOff = r * gradDim;
            __gm__ float* rowPtr = GetRowPtr(absRow);
            CopyGmToUb(u1[ubOff], rowPtr, gradDim);
            CopyGmToUb(u2[ubOff], rowPtr + static_cast<int64_t>(gradDim), gradDim);
        }
        SyncMte2V();
        ComputeAdagrad(len, u0, u1, u2, u3);
        SyncVMte3();
        for (uint32_t r = 0; r < rowsInGroup; ++r) {
            const int32_t absRow = rowIdx + static_cast<int32_t>(r);
            const uint32_t ubOff = r * gradDim;
            CopyOutRow(gradDim, absRow, ubOff, u1, u2);
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
            const uint32_t len = gradDim;
            LocalTensor<float> u = ubMem_.Get<float>();
            LocalTensor<float> u0 = u[0];
            LocalTensor<float> u1 = u[len];
            LocalTensor<float> u2 = u[len * 2U];
            LocalTensor<float> u3 = u[len * 3U];

            CopyInRow(gradDim, absRow, rowGradBase, 0U, u0, u1, u2);
            SyncMte2V();
            ComputeAdagrad(len, u0, u1, u2, u3);
            SyncVMte3();
            CopyOutRow(gradDim, absRow, 0U, u1, u2);
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
        const int32_t numGroups = (numRows + static_cast<int32_t>(rowsPerGroup) - 1) / static_cast<int32_t>(rowsPerGroup);

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
                if (!allActive) {
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
    bool useFounds_ {false};
    const __gm__ AdagradSimdTilingData* tiling_;
};

} // namespace dyn_emb_adagrad_simd
