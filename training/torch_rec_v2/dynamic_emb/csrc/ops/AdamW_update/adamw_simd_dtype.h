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
#include "kernel_operator.h"

namespace dyn_emb_adamw_simd {

using namespace AscendC;

// UB 内统一用 float 做矢量计算；GM 侧 grad/weight 可为 fp32/fp16/bf16
template <typename T>
struct AdamWNeedsCast {
    static constexpr bool value = !std::is_same_v<T, float>;
};

// MTE 单次搬运至少 32B；fp16/bf16 在 gradDim=8 时仅 16B，必须走 DataCopyPad
template <typename T>
__aicore__ inline bool NeedsGmCopyPad(uint32_t elemCount)
{
    return (elemCount * static_cast<uint32_t>(sizeof(T))) % 32U != 0U;
}

__aicore__ inline void SyncMte2ToV()
{
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e);
    WaitFlag<HardEvent::MTE2_V>(e);
}

__aicore__ inline void SyncVToMte2()
{
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(e);
    WaitFlag<HardEvent::V_MTE2>(e);
}

__aicore__ inline void SyncVToMte3()
{
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e);
    WaitFlag<HardEvent::V_MTE3>(e);
}

__aicore__ inline void SyncMte3ToV()
{
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(e);
    WaitFlag<HardEvent::MTE3_V>(e);
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
        SyncMte2ToV();
        Cast(dst, tmp, RoundMode::CAST_NONE, elemCount);
        // scratch 下次被 MTE2 写入前，需等 V 完成 Cast
        SyncVToMte2();
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
        SyncVToMte3();
        DataCopy(dstGm[offset], tmp, elemCount);
        // scratch 下次被 V Cast 写入前，需等 MTE3 完成读 tmp
        SyncMte3ToV();
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
        SyncMte2ToV();
        Cast(dst, tmp, RoundMode::CAST_NONE, alignedLen);
        SyncVToMte2();
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
        SyncVToMte3();
        DataCopyPad(dstGm[offset], tmp, copyParams);
        SyncMte3ToV();
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

}  // namespace dyn_emb_adamw_simd
