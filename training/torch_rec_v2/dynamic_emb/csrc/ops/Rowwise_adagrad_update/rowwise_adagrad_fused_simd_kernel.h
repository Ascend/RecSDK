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
#include <type_traits>

#include "Adagrad_update/adagrad_simd_dtype.h"
#include "ops_utils.h"
#include "kernel_operator.h"
#include "rowwise_adagrad_simd_tiling.h"

using namespace AscendC;

namespace dyn_emb_rowwise_adagrad_fused_simd {

constexpr int32_t kDataAlignBytes = 32;

template <typename GradT, typename WeightT>
struct RowwiseAdagradFusedSimdUsesCast {
    static constexpr bool value =
        dyn_emb_adagrad_simd::AdagradNeedsCast<GradT>::value || dyn_emb_adagrad_simd::AdagradNeedsCast<WeightT>::value;
};

template <typename T>
__aicore__ inline void CpGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, int64_t len)
{
    const uint32_t alignLen = static_cast<uint32_t>(len) * sizeof(T) / static_cast<uint32_t>(kDataAlignBytes) *
                              static_cast<uint32_t>(kDataAlignBytes);
    const uint32_t unAlignLen = static_cast<uint32_t>(len) * sizeof(T) - alignLen;
    DataCopy(lt, gt, alignLen / sizeof(T));
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
        const DataCopyPadExtParams<T> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyPad(lt[alignLen / sizeof(T)], gt[alignLen / sizeof(T)], dataCopyExtParams, dataCopyPadExtParams);
    }
}

template <typename T>
__aicore__ inline void CpLocal2Gm(const GlobalTensor<T>& gt, const LocalTensor<T>& lt, int64_t len)
{
    const uint32_t alignLen = static_cast<uint32_t>(len) * sizeof(T) / static_cast<uint32_t>(kDataAlignBytes) *
                              static_cast<uint32_t>(kDataAlignBytes);
    const uint32_t unAlignLen = static_cast<uint32_t>(len) * sizeof(T) - alignLen;
    DataCopy(gt, lt, alignLen / sizeof(T));
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
        DataCopyPad(gt[alignLen / sizeof(T)], lt[alignLen / sizeof(T)], dataCopyExtParams);
    }
}

/// 连续 values(WeightT) + 扁平 grads(GradT)；rowwise state 标量位于每行 gradDim 偏移
template <typename GradT, typename WeightT>
class RowwiseAdagradFusedSimd {
public:
    static constexpr uint64_t kUbSlotCount = RowwiseAdagradFusedSimdUsesCast<GradT, WeightT>::value ? 5ULL : 4ULL;
    static constexpr uint64_t kScalarFloats = 8ULL;
    static constexpr uint64_t kScalarBytes = kScalarFloats * sizeof(float);
    static constexpr uint32_t kFloatElemsPer32B = 8U;

    __aicore__ inline static uint32_t GradDimAlignedElemCount(uint32_t gradDim)
    {
        return ((gradDim + kFloatElemsPer32B - 1U) / kFloatElemsPer32B) * kFloatElemsPer32B;
    }

    __aicore__ inline static bool IsGradDim32BAligned(uint32_t gradDim)
    {
        return (gradDim % kFloatElemsPer32B) == 0U;
    }

    __aicore__ inline explicit RowwiseAdagradFusedSimd(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR values, const __gm__ RowwiseAdagradSimdTilingData* tiling)
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
        pipe_->InitBuffer(scalarMem_, kScalarBytes);
    }

    __aicore__ inline void CopyInGradToUb(const LocalTensor<float>& dst, int64_t gradBase, uint32_t gradDim) const
    {
        const DataCopyExtParams copyParams{1, static_cast<uint32_t>(gradDim * sizeof(GradT)), 0, 0, 0};
        const DataCopyPadExtParams<GradT> padParams{true, 0, 0, 0};
        DataCopyPad(dst, gradsGm_[gradBase], copyParams, padParams);
    }

    __aicore__ inline void CopyInGradFp32(int64_t gradBase, uint32_t gradDim, const LocalTensor<float>& u0) const
    {
        if (IsGradDim32BAligned(gradDim)) {
            DataCopy(u0, gradsGm_[gradBase], gradDim);
        } else {
            CopyInGradToUb(u0, gradBase, gradDim);
        }
    }

    __aicore__ inline void CopyInWeightFp32(int64_t rowValBase, uint32_t gradDim, const LocalTensor<float>& u1) const
    {
        if (IsGradDim32BAligned(gradDim)) {
            DataCopy(u1, valuesGm_[rowValBase], gradDim);
            return;
        }
        const DataCopyExtParams copyParams{1, static_cast<uint32_t>(gradDim * sizeof(WeightT)), 0, 0, 0};
        const DataCopyPadExtParams<WeightT> padParams{true, 0, 0, 0};
        DataCopyPad(u1, valuesGm_[rowValBase], copyParams, padParams);
    }

    __aicore__ inline void CopyOutWeightFp32(int64_t rowValBase, uint32_t gradDim, const LocalTensor<float>& u1) const
    {
        if (IsGradDim32BAligned(gradDim)) {
            DataCopy(valuesGm_[rowValBase], u1, gradDim);
            return;
        }
        const DataCopyExtParams copyParams{1, static_cast<uint32_t>(gradDim * sizeof(WeightT)), 0, 0, 0};
        DataCopyPad(valuesGm_[rowValBase], u1, copyParams);
    }

    __aicore__ inline void CopyInGradCast(int64_t gradBase, uint32_t gradDim, const LocalTensor<float>& u0,
                                          const LocalTensor<float>& scratch) const
    {
        const uint32_t alignedLen = GradDimAlignedElemCount(gradDim);
        const bool gradDirectCopy =
            IsGradDim32BAligned(gradDim) && !dyn_emb_adagrad_simd::NeedsGmCopyPad<GradT>(gradDim);
        if (gradDirectCopy) {
            dyn_emb_adagrad_simd::CopyGmToUbAsFloat<GradT>(u0, gradsGm_, gradBase, gradDim, scratch);
        } else {
            dyn_emb_adagrad_simd::CopyGmToUbAsFloatPad<GradT>(u0, gradsGm_, gradBase, gradDim, alignedLen, scratch);
        }
    }

    __aicore__ inline void CopyInWeightCast(int64_t rowValBase, uint32_t gradDim, const LocalTensor<float>& u1,
                                            const LocalTensor<float>& scratch) const
    {
        const uint32_t alignedLen = GradDimAlignedElemCount(gradDim);
        const bool weightDirectCopy =
            IsGradDim32BAligned(gradDim) && !dyn_emb_adagrad_simd::NeedsGmCopyPad<WeightT>(gradDim);
        if (weightDirectCopy) {
            dyn_emb_adagrad_simd::CopyGmToUbAsFloat<WeightT>(u1, valuesGm_, rowValBase, gradDim, scratch);
        } else {
            dyn_emb_adagrad_simd::CopyGmToUbAsFloatPad<WeightT>(u1, valuesGm_, rowValBase, gradDim, alignedLen,
                                                                scratch);
        }
    }

    __aicore__ inline void CopyOutWeightCast(int64_t rowValBase, uint32_t gradDim, const LocalTensor<float>& u1,
                                             const LocalTensor<float>& scratch) const
    {
        const bool weightDirectCopy =
            IsGradDim32BAligned(gradDim) && !dyn_emb_adagrad_simd::NeedsGmCopyPad<WeightT>(gradDim);
        if (weightDirectCopy) {
            dyn_emb_adagrad_simd::CopyUbFloatToGm<WeightT>(valuesGm_, rowValBase, u1, gradDim, scratch);
        } else {
            dyn_emb_adagrad_simd::CopyUbFloatToGmPad<WeightT>(valuesGm_, rowValBase, u1, gradDim, scratch);
        }
    }

    __aicore__ inline void LoadStateScalar(const LocalTensor<float>& sc, int64_t stateOff,
                                           const LocalTensor<float>& scratch) const
    {
        constexpr uint32_t kStateScalarCount = 1U;
        if constexpr (std::is_same_v<WeightT, float>) {
            CpGm2Local<float>(sc, valuesGm_[stateOff], kStateScalarCount);
        } else {
            const uint32_t alignedLen = GradDimAlignedElemCount(kStateScalarCount);
            dyn_emb_adagrad_simd::CopyGmToUbAsFloatPad<WeightT>(sc, valuesGm_, stateOff, kStateScalarCount, alignedLen,
                                                                scratch);
        }
    }

    __aicore__ inline void StoreStateScalar(const LocalTensor<float>& sc, int64_t stateOff,
                                            const LocalTensor<float>& scratch) const
    {
        constexpr uint32_t kStateScalarCount = 1U;
        if constexpr (std::is_same_v<WeightT, float>) {
            CpLocal2Gm<float>(valuesGm_[stateOff], sc, kStateScalarCount);
        } else {
            dyn_emb_adagrad_simd::CopyUbFloatToGmPad<WeightT>(valuesGm_, stateOff, sc, kStateScalarCount, scratch);
        }
    }

    __aicore__ inline void ComputeRow(uint32_t gradDim, uint32_t computeLen, int64_t stateOff, LocalTensor<float>& u0,
                                      LocalTensor<float>& u1, LocalTensor<float>& u2, LocalTensor<float>& u3,
                                      LocalTensor<float>& sc, const LocalTensor<float>& scratch)
    {
        const float lr = tiling_->lr;
        const float eps = tiling_->eps;
        const float invGradDim = 1.0f / static_cast<float>(gradDim);

        Mul<float>(u3, u0, u0, computeLen);
        uint32_t srcShape[2] = {1, computeLen};
        ReduceSum<float, Pattern::Reduce::AR, false>(u2, u3, srcShape, false);
        ops_utils::SyncMte2V();
        Muls<float>(u2, u2, invGradDim, 1);

        LoadStateScalar(sc, stateOff, scratch);
        ops_utils::SyncMte2V();
        Add<float>(u2, sc, u2, 1);

        ops_utils::SyncVMte3();
        StoreStateScalar(u2, stateOff, scratch);
        ops_utils::SyncMte3Mte2();

        Sqrt<float>(sc, u2, 1);
        Adds<float>(sc, sc, eps, 1);
        Reciprocal<float>(sc, sc, 1);
        Muls<float>(sc, sc, lr, 1);
        Duplicate<float>(u3, sc, computeLen);
        Mul<float>(u3, u0, u3, computeLen);
        Sub<float>(u1, u1, u3, computeLen);
    }

    __aicore__ inline void ProcessGroupPerRow(int32_t rowIdx, uint32_t rowsInGroup, uint32_t gradDim, uint32_t valDim,
                                              int64_t gradBase)
    {
        const uint32_t computeLen = GradDimAlignedElemCount(gradDim);
        for (uint32_t r = 0; r < rowsInGroup; ++r) {
            const int32_t absRow = rowIdx + static_cast<int32_t>(r);
            const int64_t rowGradBase = gradBase + static_cast<int64_t>(r) * static_cast<int64_t>(gradDim);
            const int64_t rowValBase = static_cast<int64_t>(absRow) * static_cast<int64_t>(valDim);
            const int64_t stateOff = rowValBase + static_cast<int64_t>(gradDim);
            LocalTensor<float> u = ubMem_.Get<float>();
            LocalTensor<float> u0 = u[0];
            LocalTensor<float> u1 = u[computeLen];
            LocalTensor<float> u2 = u[computeLen * 2U];
            LocalTensor<float> u3 = u[computeLen * 3U];
            LocalTensor<float> sc = scalarMem_.Get<float>();

            if constexpr (RowwiseAdagradFusedSimdUsesCast<GradT, WeightT>::value) {
                LocalTensor<float> scratch = u[computeLen * 4U];
                CopyInGradCast(rowGradBase, gradDim, u0, scratch);
                CopyInWeightCast(rowValBase, gradDim, u1, scratch);
                ops_utils::SyncMte2V();
                ComputeRow(gradDim, computeLen, stateOff, u0, u1, u2, u3, sc, scratch);
                ops_utils::SyncVMte3();
                CopyOutWeightCast(rowValBase, gradDim, u1, scratch);
            } else {
                CopyInGradFp32(rowGradBase, gradDim, u0);
                CopyInWeightFp32(rowValBase, gradDim, u1);
                ops_utils::SyncMte2V();
                LocalTensor<float> dummyScratch = u3;
                ComputeRow(gradDim, computeLen, stateOff, u0, u1, u2, u3, sc, dummyScratch);
                ops_utils::SyncVMte3();
                CopyOutWeightFp32(rowValBase, gradDim, u1);
            }
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
        const uint32_t valDim = tiling_->valDim;
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
            ProcessGroupPerRow(rowIdx, rowsInGroup, gradDim, valDim, gradBase);
        }
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> ubMem_;
    TBuf<TPosition::VECCALC> scalarMem_;
    GlobalTensor<GradT> gradsGm_;
    GlobalTensor<WeightT> valuesGm_;
    const __gm__ RowwiseAdagradSimdTilingData* tiling_;
};

}  // namespace dyn_emb_rowwise_adagrad_fused_simd
