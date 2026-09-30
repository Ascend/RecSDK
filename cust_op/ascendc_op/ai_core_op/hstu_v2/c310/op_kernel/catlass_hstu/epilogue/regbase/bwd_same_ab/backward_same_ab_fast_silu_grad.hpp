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
============================================================================== */

#pragma once

#include "../fast_silu_grad.hpp"

namespace catlass::Epilogue::RegBase::SameAB {

template <typename Type, typename AccType, bool HAS_RAB, bool NEED_MASK, typename RegType>
__simd_callee__ inline void SiluScoreComputeNoStore(__ubuf__ AccType* scorePtr, __ubuf__ Type* rabPtr,
                                                    __ubuf__ Type* maskPtr, RegType& a, RegType& z, RegType& silu,
                                                    RegType& ones, RegType& zeros, AccType alpha,
                                                    AscendC::MicroAPI::MaskReg& mask)
{
    AscendC::MicroAPI::LoadAlign(a, scorePtr);
    // SameAB Fixpipe is NO_QUANT. Apply QK alpha in-register so Score does
    // not need a separate full FP32 UB read/modify/write pass.
    AscendC::MicroAPI::Muls(a, a, alpha, mask);
    AddFusedRab<Type, AccType, HAS_RAB, NEED_MASK>(a, z, rabPtr, maskPtr, alpha, mask);
    Sigmoid(z, a, ones, zeros, mask);
    AscendC::MicroAPI::Mul(silu, z, a, mask);
}

template <typename Type, typename AccType, typename GrabType, bool HAS_RAB, bool NEED_MASK>
__simd_callee__ inline void FastSiluGradVfImpl(__ubuf__ AccType* scorePtr, __ubuf__ Type* rabPtr,
                                               __ubuf__ Type* maskPtr, __ubuf__ Type* probPtr,
                                               __ubuf__ GrabType* grabPartPtr, AccType alpha, AccType probScale,
                                               uint32_t count, uint32_t repeatTimes)
{
    constexpr uint32_t ONE_REPEAT = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(AccType));

    AscendC::MicroAPI::RegTensor<AccType> a;
    AscendC::MicroAPI::RegTensor<AccType> z;
    AscendC::MicroAPI::RegTensor<AccType> t;
    AscendC::MicroAPI::RegTensor<AccType> silu;
    AscendC::MicroAPI::RegTensor<AccType> ones;
    AscendC::MicroAPI::RegTensor<AccType> zeros;
    AscendC::MicroAPI::MaskReg mask;

    AscendC::MicroAPI::Duplicate(ones, 1.0f);
    AscendC::MicroAPI::Duplicate(zeros, 0.0f);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        uint32_t activeCount = count < ONE_REPEAT ? count : ONE_REPEAT;
        mask = AscendC::MicroAPI::UpdateMask<AccType>(count);
        auto offset = i * ONE_REPEAT;

        // Keep the unscaled SiLU value only in a register. The legacy
        // SiluScore API stores that intermediate to UB, but Same AB must
        // publish only the scaled Prob consumed by dV.
        SiluScoreComputeNoStore<Type, AccType, HAS_RAB, NEED_MASK>(scorePtr + offset, rabPtr + offset, maskPtr + offset,
                                                                   a, z, silu, ones, zeros, alpha, mask);

        // Sigmoid uses the predicate internally. Rebuild the lane mask before
        // the derivative and the only Prob store instead of relying on the
        // post-Sigmoid MaskReg state.
        uint32_t postComputeCount = activeCount;
        mask = AscendC::MicroAPI::UpdateMask<AccType>(postComputeCount);

        AscendC::MicroAPI::Sub(t, ones, z, mask);
        AscendC::MicroAPI::MulAddDst(z, silu, t, mask);

        AscendC::MicroAPI::Muls(silu, silu, probScale, mask);
        if constexpr (!std::is_same<Type, AccType>::value) {
            CastDownStore<Type, AccType>(probPtr + offset, silu, mask);
        } else {
            AscendC::MicroAPI::StoreAlign(probPtr + offset, silu, mask);
        }

        if constexpr (!std::is_same<GrabType, AccType>::value) {
            CastDownStore<GrabType, AccType>(grabPartPtr + offset, z, mask);
        } else {
            AscendC::MicroAPI::StoreAlign(grabPartPtr + offset, z, mask);
        }
    }
}

template <typename Type, typename AccType, typename GrabType, bool HAS_RAB>
__simd_vf__ inline void FastSiluGradVf(__ubuf__ AccType* scorePtr, __ubuf__ Type* rabPtr, __ubuf__ Type* maskPtr,
                                       __ubuf__ Type* probPtr, __ubuf__ GrabType* grabPartPtr, AccType alpha,
                                       AccType probScale, uint32_t count, uint32_t repeatTimes, bool needMask)
{
    if (needMask) {
        FastSiluGradVfImpl<Type, AccType, GrabType, HAS_RAB, true>(scorePtr, rabPtr, maskPtr, probPtr, grabPartPtr,
                                                                   alpha, probScale, count, repeatTimes);
    } else {
        FastSiluGradVfImpl<Type, AccType, GrabType, HAS_RAB, false>(scorePtr, rabPtr, maskPtr, probPtr, grabPartPtr,
                                                                    alpha, probScale, count, repeatTimes);
    }
}

}  // namespace catlass::Epilogue::RegBase::SameAB
