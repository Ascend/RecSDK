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
#include "sgd_simd_tiling.h"
#include "sgd_simd_dtype.h"
#include "../ops_utils.h"

using namespace AscendC;

namespace dyn_emb_sgd_simd {

template <typename GradT, typename WeightT>
class SgdSimd {
public:
    // UB槽位数，分别存放梯度/权重/中间变量
    static constexpr uint64_t kUbSlotCount = 3ULL;

    __aicore__ inline explicit SgdSimd(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR rowPtrs, GM_ADDR founds, const __gm__ SgdSimdTilingData* tiling)
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

    __aicore__ inline void CopyInRow(uint32_t gradDim, int32_t rowIdx, int64_t gradBase, uint32_t ubOff,
                                     const LocalTensor<float>& u0, const LocalTensor<float>& u1,
                                     const LocalTensor<float>& scratch) const
    {
        const LocalTensor<float> gUb = u0[ubOff];
        const LocalTensor<float> wUb = u1[ubOff];
        const uint32_t alignedLen = GradDimAlignedElemCount(gradDim);
        if (CanUseDirectGmCopy<GradT>(gradDim)) {
            CopyGmToUbAsFloat<GradT>(gUb, gradsGm_, gradBase, gradDim, scratch);
        } else {
            CopyGmToUbAsFloatPad<GradT>(gUb, gradsGm_, gradBase, gradDim, alignedLen, scratch);
        }
        if (CanUseDirectGmCopy<WeightT>(gradDim)) {
            CopyGmToUbAsFloat<WeightT>(wUb, GetRowPtr(rowIdx), gradDim, scratch);
        } else {
            CopyGmToUbAsFloatPad<WeightT>(wUb, GetRowPtr(rowIdx), gradDim, alignedLen, scratch);
        }
    }

    __aicore__ inline void CopyOutRow(uint32_t gradDim, int32_t rowIdx, uint32_t ubOff, const LocalTensor<float>& u1,
                                      const LocalTensor<float>& scratch) const
    {
        const LocalTensor<float> wUb = u1[ubOff];
        if (CanUseDirectGmCopy<WeightT>(gradDim)) {
            CopyUbFloatToGm<WeightT>(GetRowPtr(rowIdx), wUb, gradDim, scratch);
        } else {
            CopyUbFloatToGmPad<WeightT>(GetRowPtr(rowIdx), wUb, gradDim, scratch);
        }
    }

    __aicore__ inline void ProcessGroupVector(int32_t rowIdx, uint32_t rowsInGroup, uint32_t gradDim, int64_t gradBase)
    {
        const uint32_t len = rowsInGroup * gradDim;
        LocalTensor<float> u = ubMem_.Get<float>();
        LocalTensor<float> u0 = u[0];
        LocalTensor<float> u1 = u[len];
        LocalTensor<float> scratch = u[len * 2U];

        for (uint32_t r = 0; r < rowsInGroup; ++r) {
            const int32_t absRow = rowIdx + static_cast<int32_t>(r);
            const int64_t rowGradBase = gradBase + static_cast<int64_t>(r) * static_cast<int64_t>(gradDim);
            const uint32_t ubOff = r * gradDim;
            CopyInRow(gradDim, absRow, rowGradBase, ubOff, u0, u1, scratch);
        }
        ops_utils::SyncMte2V();
        ComputeSgdSimd(tiling_, len, u0, u1);
        ops_utils::SyncVMte3();
        for (uint32_t r = 0; r < rowsInGroup; ++r) {
            const int32_t absRow = rowIdx + static_cast<int32_t>(r);
            const uint32_t ubOff = r * gradDim;
            CopyOutRow(gradDim, absRow, ubOff, u1, scratch);
        }
        ops_utils::SyncMte3Mte2();
    }

    __aicore__ inline void ProcessGroupPerRow(int32_t rowIdx, uint32_t rowsInGroup, uint32_t gradDim, int64_t gradBase)
    {
        const uint32_t computeLen = GradDimAlignedElemCount(gradDim);
        for (uint32_t r = 0; r < rowsInGroup; ++r) {
            const int32_t absRow = rowIdx + static_cast<int32_t>(r);
            if (!IsRowActive(absRow)) {
                continue;
            }
            const int64_t rowGradBase = gradBase + static_cast<int64_t>(r) * static_cast<int64_t>(gradDim);
            LocalTensor<float> u = ubMem_.Get<float>();
            LocalTensor<float> u0 = u[0];
            LocalTensor<float> u1 = u[computeLen];
            LocalTensor<float> scratch = u[computeLen * 2U];

            CopyInRow(gradDim, absRow, rowGradBase, 0U, u0, u1, scratch);
            ops_utils::SyncMte2V();
            ComputeSgdSimd(tiling_, computeLen, u0, u1);
            ops_utils::SyncVMte3();
            CopyOutRow(gradDim, absRow, 0U, u1, scratch);
            ops_utils::SyncMte3Mte2();
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
    const __gm__ SgdSimdTilingData* tiling_;
};

}  // namespace dyn_emb_sgd_simd
