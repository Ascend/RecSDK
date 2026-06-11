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
#include <type_traits>
#include "ops_utils.h"
#include "kernel_operator.h"

namespace dyn_emb_adagrad_simd {

using namespace AscendC;

template <typename T>
struct AdagradNeedsCast {
    static constexpr bool value = !std::is_same_v<T, float>;
};

template <typename GradT, typename WeightT>
struct AdagradSimdUsesCast {
    static constexpr bool value = AdagradNeedsCast<GradT>::value || AdagradNeedsCast<WeightT>::value;
};

template <typename T>
__aicore__ inline bool NeedsGmCopyPad(uint32_t elemCount)
{
    return (elemCount * static_cast<uint32_t>(sizeof(T))) % 32U != 0U;
}

template <typename SrcT>
__aicore__ inline void CopyGmToUbAsFloat(const LocalTensor<float>& dst, const GlobalTensor<SrcT>& srcGm, int64_t offset,
                                         uint32_t elemCount, const LocalTensor<float>& scratch)
{
    if constexpr (std::is_same_v<SrcT, float>) {
        DataCopy(dst, srcGm[offset], elemCount);
    } else {
        LocalTensor<SrcT> tmp = scratch.template ReinterpretCast<SrcT>();
        DataCopy(tmp, srcGm[offset], elemCount);
        ops_utils::SyncMte2V();
        Cast(dst, tmp, RoundMode::CAST_NONE, elemCount);
        ops_utils::SyncVMte2();
    }
}

template <typename SrcT>
__aicore__ inline void CopyGmToUbAsFloat(const LocalTensor<float>& dst, __gm__ SrcT* srcGm, uint32_t elemCount,
                                         const LocalTensor<float>& scratch)
{
    GlobalTensor<SrcT> srcTensor;
    srcTensor.SetGlobalBuffer(srcGm, static_cast<uint64_t>(elemCount));
    CopyGmToUbAsFloat(dst, srcTensor, 0, elemCount, scratch);
}

template <typename DstT>
__aicore__ inline void CopyUbFloatToGm(const GlobalTensor<DstT>& dstGm, int64_t offset, const LocalTensor<float>& src,
                                       uint32_t elemCount, const LocalTensor<float>& scratch)
{
    if constexpr (std::is_same_v<DstT, float>) {
        DataCopy(dstGm[offset], src, elemCount);
    } else {
        LocalTensor<DstT> tmp = scratch.template ReinterpretCast<DstT>();
        Cast(tmp, src, RoundMode::CAST_RINT, elemCount);
        ops_utils::SyncVMte3();
        DataCopy(dstGm[offset], tmp, elemCount);
        ops_utils::SyncMte3V();
    }
}

template <typename DstT>
__aicore__ inline void CopyUbFloatToGm(__gm__ DstT* dstGm, const LocalTensor<float>& src, uint32_t elemCount,
                                       const LocalTensor<float>& scratch)
{
    GlobalTensor<DstT> dstTensor;
    dstTensor.SetGlobalBuffer(dstGm, static_cast<uint64_t>(elemCount));
    CopyUbFloatToGm(dstTensor, 0, src, elemCount, scratch);
}

template <typename SrcT>
__aicore__ inline void CopyGmToUbAsFloatPad(const LocalTensor<float>& dst, const GlobalTensor<SrcT>& srcGm,
                                            int64_t offset, uint32_t gradDim, uint32_t alignedLen,
                                            const LocalTensor<float>& scratch)
{
    const DataCopyExtParams copyParams{1, static_cast<uint32_t>(gradDim * sizeof(SrcT)), 0, 0, 0};
    if constexpr (std::is_same_v<SrcT, float>) {
        const DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
        DataCopyPad(dst, srcGm[offset], copyParams, padParams);
    } else {
        LocalTensor<SrcT> tmp = scratch.template ReinterpretCast<SrcT>();
        const DataCopyPadExtParams<SrcT> padParams{true, 0, 0, 0};
        DataCopyPad(tmp, srcGm[offset], copyParams, padParams);
        ops_utils::SyncMte2V();
        Cast(dst, tmp, RoundMode::CAST_NONE, alignedLen);
        ops_utils::SyncVMte2();
    }
}

template <typename SrcT>
__aicore__ inline void CopyGmToUbAsFloatPad(const LocalTensor<float>& dst, __gm__ SrcT* srcGm, uint32_t gradDim,
                                            uint32_t alignedLen, const LocalTensor<float>& scratch)
{
    GlobalTensor<SrcT> srcTensor;
    srcTensor.SetGlobalBuffer(srcGm, static_cast<uint64_t>(alignedLen));
    CopyGmToUbAsFloatPad(dst, srcTensor, 0, gradDim, alignedLen, scratch);
}

template <typename DstT>
__aicore__ inline void CopyUbFloatToGmPad(const GlobalTensor<DstT>& dstGm, int64_t offset,
                                          const LocalTensor<float>& src, uint32_t gradDim,
                                          const LocalTensor<float>& scratch)
{
    const DataCopyExtParams copyParams{1, static_cast<uint32_t>(gradDim * sizeof(DstT)), 0, 0, 0};
    if constexpr (std::is_same_v<DstT, float>) {
        DataCopyPad(dstGm[offset], src, copyParams);
    } else {
        LocalTensor<DstT> tmp = scratch.template ReinterpretCast<DstT>();
        Cast(tmp, src, RoundMode::CAST_RINT, gradDim);
        ops_utils::SyncVMte3();
        DataCopyPad(dstGm[offset], tmp, copyParams);
        ops_utils::SyncMte3V();
    }
}

template <typename DstT>
__aicore__ inline void CopyUbFloatToGmPad(__gm__ DstT* dstGm, const LocalTensor<float>& src, uint32_t gradDim,
                                          const LocalTensor<float>& scratch)
{
    GlobalTensor<DstT> dstTensor;
    dstTensor.SetGlobalBuffer(dstGm, static_cast<uint64_t>(gradDim));
    CopyUbFloatToGmPad(dstTensor, 0, src, gradDim, scratch);
}

}  // namespace dyn_emb_adagrad_simd
