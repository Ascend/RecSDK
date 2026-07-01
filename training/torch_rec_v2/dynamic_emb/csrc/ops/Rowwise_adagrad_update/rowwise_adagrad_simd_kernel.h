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

namespace dyn_emb_rowwise_adagrad_simd {

constexpr int32_t kDataAlignBytes = 32;

template <typename T>
__aicore__ inline void RowwiseAdagradCompute(__local_mem__ T* dstW, __local_mem__ T* dstState, __local_mem__ T* srcG,
                                             __local_mem__ T* srcW, __local_mem__ T* srcState, uint32_t calCount,
                                             uint16_t repeatCount, uint32_t oneRepeat, float eps, float lr,
                                             float invGradDim)
{
    uint32_t scalarLen = 1;
    auto maskScalar = AscendC::MicroAPI::UpdateMask<uint32_t>(scalarLen);

    AscendC::MicroAPI::RegTensor<float> vSumTotal;
    AscendC::MicroAPI::Duplicate(vSumTotal, 0.0f, maskScalar);

    uint32_t offset;
    uint32_t remaining;
    uint32_t blockLen;
    AscendC::MicroAPI::RegTensor<float> vGrad;
    AscendC::MicroAPI::RegTensor<float> vGradSq;
    AscendC::MicroAPI::RegTensor<float> vSumSq;

    for (uint16_t i = 0; i < repeatCount; ++i) {
        offset = i * oneRepeat;
        remaining = calCount - offset;
        blockLen = (remaining > oneRepeat) ? oneRepeat : remaining;
        auto maskVec = AscendC::MicroAPI::UpdateMask<uint32_t>(blockLen);

        AscendC::MicroAPI::DataCopy(vGrad, srcG + offset);
        AscendC::MicroAPI::Mul(vGradSq, vGrad, vGrad, maskVec);
        AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(vSumSq, vGradSq, maskVec);
        AscendC::MicroAPI::Add(vSumTotal, vSumTotal, vSumSq, maskScalar);
    }

    AscendC::MicroAPI::Muls(vSumTotal, vSumTotal, invGradDim, maskScalar);

    AscendC::MicroAPI::RegTensor<float> vMomentOld;
    AscendC::MicroAPI::RegTensor<float> vNewMoment;
    AscendC::MicroAPI::DataCopy(vMomentOld, srcState);
    AscendC::MicroAPI::Add(vNewMoment, vMomentOld, vSumTotal, maskScalar);

    AscendC::MicroAPI::RegTensor<float> vDenom;
    AscendC::MicroAPI::RegTensor<float> vAdaptiveLr;
    AscendC::MicroAPI::Sqrt(vDenom, vNewMoment, maskScalar);
    AscendC::MicroAPI::Adds(vDenom, vDenom, eps, maskScalar);
    AscendC::MicroAPI::Duplicate(vAdaptiveLr, lr, maskScalar);
    AscendC::MicroAPI::Div(vAdaptiveLr, vAdaptiveLr, vDenom, maskScalar);

    uint32_t fullVecLen = oneRepeat;
    auto fullMask = AscendC::MicroAPI::UpdateMask<uint32_t>(fullVecLen);
    AscendC::MicroAPI::Duplicate(vAdaptiveLr, vAdaptiveLr, fullMask);

    AscendC::MicroAPI::RegTensor<float> vWeight;
    AscendC::MicroAPI::RegTensor<float> vUpdate;
    for (uint16_t i = 0; i < repeatCount; ++i) {
        offset = i * oneRepeat;
        remaining = calCount - offset;
        blockLen = (remaining > oneRepeat) ? oneRepeat : remaining;
        auto maskVec = AscendC::MicroAPI::UpdateMask<uint32_t>(blockLen);

        AscendC::MicroAPI::DataCopy(vGrad, srcG + offset);
        AscendC::MicroAPI::DataCopy(vWeight, srcW + offset);
        AscendC::MicroAPI::Mul(vUpdate, vAdaptiveLr, vGrad, maskVec);
        AscendC::MicroAPI::Sub(vWeight, vWeight, vUpdate, maskVec);
        AscendC::MicroAPI::DataCopy(dstW + offset, vWeight, maskVec);
    }

    AscendC::MicroAPI::DataCopy(dstState, vNewMoment, maskScalar);
}

template <typename GradT, typename WeightT>
struct RowwiseAdagradSimdUsesCast {
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

/// 扁平 grads(GradT) + 每行 WeightT*（w||rowwise state 位于 gradDim）；可选 founds 掩码
template <typename GradT, typename WeightT>
class RowwiseAdagradSimd {
public:
    static constexpr uint64_t kUbSlotCount = RowwiseAdagradSimdUsesCast<GradT, WeightT>::value ? 5ULL : 4ULL;
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

    __aicore__ inline explicit RowwiseAdagradSimd(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR rowPtrs, GM_ADDR founds,
                                const __gm__ RowwiseAdagradSimdTilingData* tiling)
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
        pipe_->InitBuffer(scalarMem_, kScalarBytes);
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

    __aicore__ inline void CopyInGradToUbPad(const LocalTensor<float>& dst, int64_t gradBase, uint32_t gradDim) const
    {
        const DataCopyExtParams copyParams{1, static_cast<uint32_t>(gradDim * sizeof(GradT)), 0, 0, 0};
        const DataCopyPadExtParams<GradT> padParams{true, 0, 0, 0};
        DataCopyPad(dst, gradsGm_[gradBase], copyParams, padParams);
    }

    __aicore__ inline void CopyInRowFp32(uint32_t gradDim, int64_t gradBase, __gm__ WeightT* rowPtr,
                                         const LocalTensor<float>& u0, const LocalTensor<float>& u1) const
    {
        if (IsGradDim32BAligned(gradDim)) {
            DataCopy(u0, gradsGm_[gradBase], gradDim);
            GlobalTensor<WeightT> wGm;
            wGm.SetGlobalBuffer(rowPtr, gradDim);
            DataCopy(u1, wGm, gradDim);
        } else {
            CopyInGradToUbPad(u0, gradBase, gradDim);
            const uint32_t alignedLen = GradDimAlignedElemCount(gradDim);
            GlobalTensor<WeightT> wGm;
            wGm.SetGlobalBuffer(rowPtr, alignedLen);
            const DataCopyExtParams copyParams{1, static_cast<uint32_t>(gradDim * sizeof(WeightT)), 0, 0, 0};
            const DataCopyPadExtParams<WeightT> padParams{true, 0, 0, 0};
            DataCopyPad(u1, wGm, copyParams, padParams);
        }
    }

    __aicore__ inline void CopyOutRowFp32(uint32_t gradDim, __gm__ WeightT* rowPtr, const LocalTensor<float>& u1) const
    {
        if (IsGradDim32BAligned(gradDim)) {
            GlobalTensor<WeightT> wGm;
            wGm.SetGlobalBuffer(rowPtr, gradDim);
            DataCopy(wGm, u1, gradDim);
            return;
        }
        const DataCopyExtParams copyParams{1, static_cast<uint32_t>(gradDim * sizeof(WeightT)), 0, 0, 0};
        GlobalTensor<WeightT> wGm;
        wGm.SetGlobalBuffer(rowPtr, gradDim);
        DataCopyPad(wGm, u1, copyParams);
    }

    __aicore__ inline void CopyInRowCast(uint32_t gradDim, int64_t gradBase, __gm__ WeightT* rowPtr,
                                         const LocalTensor<float>& u0, const LocalTensor<float>& u1,
                                         const LocalTensor<float>& scratch) const
    {
        const uint32_t alignedLen = GradDimAlignedElemCount(gradDim);
        const bool gradDirectCopy =
            IsGradDim32BAligned(gradDim) && !dyn_emb_adagrad_simd::NeedsGmCopyPad<GradT>(gradDim);
        const bool weightDirectCopy =
            IsGradDim32BAligned(gradDim) && !dyn_emb_adagrad_simd::NeedsGmCopyPad<WeightT>(gradDim);
        if (gradDirectCopy) {
            dyn_emb_adagrad_simd::CopyGmToUbAsFloat<GradT>(u0, gradsGm_, gradBase, gradDim, scratch);
        } else {
            dyn_emb_adagrad_simd::CopyGmToUbAsFloatPad<GradT>(u0, gradsGm_, gradBase, gradDim, alignedLen, scratch);
        }
        if (weightDirectCopy) {
            dyn_emb_adagrad_simd::CopyGmToUbAsFloat<WeightT>(u1, rowPtr, gradDim, scratch);
        } else {
            dyn_emb_adagrad_simd::CopyGmToUbAsFloatPad<WeightT>(u1, rowPtr, gradDim, alignedLen, scratch);
        }
    }

    __aicore__ inline void CopyOutRowCast(uint32_t gradDim, __gm__ WeightT* rowPtr, const LocalTensor<float>& u1,
                                          const LocalTensor<float>& scratch) const
    {
        const bool weightDirectCopy =
            IsGradDim32BAligned(gradDim) && !dyn_emb_adagrad_simd::NeedsGmCopyPad<WeightT>(gradDim);
        if (weightDirectCopy) {
            dyn_emb_adagrad_simd::CopyUbFloatToGm<WeightT>(rowPtr, u1, gradDim, scratch);
        } else {
            dyn_emb_adagrad_simd::CopyUbFloatToGmPad<WeightT>(rowPtr, u1, gradDim, scratch);
        }
    }

    __aicore__ inline void LoadStateScalar(const LocalTensor<float>& sc, __gm__ WeightT* rowPtr,
                                           const LocalTensor<float>& scratch) const
    {
        constexpr uint32_t kStateScalarCount = 1U;
        if constexpr (std::is_same_v<WeightT, float>) {
            GlobalTensor<float> stateGm;
            stateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rowPtr + tiling_->gradDim), kStateScalarCount);
            CpGm2Local<float>(sc, stateGm, kStateScalarCount);
        } else {
            const uint32_t alignedLen = GradDimAlignedElemCount(kStateScalarCount);
            dyn_emb_adagrad_simd::CopyGmToUbAsFloatPad<WeightT>(sc, rowPtr + tiling_->gradDim, kStateScalarCount,
                                                                alignedLen, scratch);
        }
    }

    __aicore__ inline void StoreStateScalar(const LocalTensor<float>& sc, __gm__ WeightT* rowPtr,
                                            const LocalTensor<float>& scratch) const
    {
        constexpr uint32_t kStateScalarCount = 1U;
        if constexpr (std::is_same_v<WeightT, float>) {
            GlobalTensor<float> stateGm;
            stateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rowPtr + tiling_->gradDim), kStateScalarCount);
            CpLocal2Gm<float>(stateGm, sc, kStateScalarCount);
        } else {
            dyn_emb_adagrad_simd::CopyUbFloatToGmPad<WeightT>(rowPtr + tiling_->gradDim, sc, kStateScalarCount,
                                                              scratch);
        }
    }

    __aicore__ inline void ComputeRow(uint32_t gradDim, uint32_t computeLen, LocalTensor<float>& u0,
                                      LocalTensor<float>& u1, LocalTensor<float>& u2, LocalTensor<float>& u3,
                                      LocalTensor<float>& sc, __gm__ WeightT* rowPtr, const LocalTensor<float>& scratch)
    {
        (void)u2;
        (void)u3;
        const float lr = tiling_->lr;
        const float eps = tiling_->eps;
        const float invGradDim = 1.0f / static_cast<float>(gradDim);

        LoadStateScalar(sc, rowPtr, scratch);
        ops_utils::SyncMte2V();

        __local_mem__ float* srcG = (__local_mem__ float*)u0.GetPhyAddr();
        __local_mem__ float* srcW = (__local_mem__ float*)u1.GetPhyAddr();
        __local_mem__ float* dstW = (__local_mem__ float*)u1.GetPhyAddr();
        __local_mem__ float* srcState = (__local_mem__ float*)sc.GetPhyAddr();
        __local_mem__ float* dstState = (__local_mem__ float*)sc.GetPhyAddr();

        constexpr uint32_t vecLen = AscendC::GetVecLen();
        constexpr uint32_t oneRepeat = vecLen / static_cast<uint32_t>(sizeof(float));
        const uint16_t repeatCount = static_cast<uint16_t>((computeLen + oneRepeat - 1U) / oneRepeat);

        VF_CALL<RowwiseAdagradCompute<float>>(dstW, dstState, srcG, srcW, srcState, computeLen, repeatCount, oneRepeat,
                                              eps, lr, invGradDim);

        ops_utils::SyncVMte3();
        StoreStateScalar(sc, rowPtr, scratch);
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
            LocalTensor<float> u2 = u[computeLen * 2U];
            LocalTensor<float> u3 = u[computeLen * 3U];
            LocalTensor<float> sc = scalarMem_.Get<float>();
            __gm__ WeightT* rowPtr = GetRowPtr(absRow);

            if constexpr (RowwiseAdagradSimdUsesCast<GradT, WeightT>::value) {
                LocalTensor<float> scratch = u[computeLen * 4U];
                CopyInRowCast(gradDim, rowGradBase, rowPtr, u0, u1, scratch);
            } else {
                CopyInRowFp32(gradDim, rowGradBase, rowPtr, u0, u1);
            }
            ops_utils::SyncMte2V();
            if constexpr (RowwiseAdagradSimdUsesCast<GradT, WeightT>::value) {
                LocalTensor<float> scratch = u[computeLen * 4U];
                ComputeRow(gradDim, computeLen, u0, u1, u2, u3, sc, rowPtr, scratch);
            } else {
                LocalTensor<float> dummyScratch = u3;
                ComputeRow(gradDim, computeLen, u0, u1, u2, u3, sc, rowPtr, dummyScratch);
            }
            ops_utils::SyncVMte3();
            if constexpr (RowwiseAdagradSimdUsesCast<GradT, WeightT>::value) {
                LocalTensor<float> scratch = u[computeLen * 4U];
                CopyOutRowCast(gradDim, rowPtr, u1, scratch);
            } else {
                CopyOutRowFp32(gradDim, rowPtr, u1);
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
            ProcessGroupPerRow(rowIdx, rowsInGroup, gradDim, gradBase);
        }
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> ubMem_;
    TBuf<TPosition::VECCALC> scalarMem_;
    GlobalTensor<GradT> gradsGm_;
    GlobalTensor<uint64_t> rowPtrsGm_;
    GlobalTensor<uint8_t> foundsGm_;
    bool useFounds_{false};
    const __gm__ RowwiseAdagradSimdTilingData* tiling_;
};

}  // namespace dyn_emb_rowwise_adagrad_simd
