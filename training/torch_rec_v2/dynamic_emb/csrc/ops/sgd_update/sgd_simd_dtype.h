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
#include "../AdamW_update/adamw_simd_dtype.h"
#include "sgd_simd_tiling.h"

namespace dyn_emb_sgd_simd {

using namespace AscendC;

using dyn_emb_adamw_simd::CopyGmToUbAsFloat;
using dyn_emb_adamw_simd::CopyGmToUbAsFloatPad;
using dyn_emb_adamw_simd::CopyUbFloatToGm;
using dyn_emb_adamw_simd::CopyUbFloatToGmPad;
using dyn_emb_adamw_simd::NeedsGmCopyPad;

static constexpr uint32_t kFloatElemsPer32B = 8U;

__aicore__ inline uint32_t GradDimAlignedElemCount(uint32_t gradDim)
{
    return ((gradDim + kFloatElemsPer32B - 1U) / kFloatElemsPer32B) * kFloatElemsPer32B;
}

__aicore__ inline bool IsGradDim32BAligned(uint32_t gradDim)
{
    return (gradDim % kFloatElemsPer32B) == 0U;
}

template <typename T>
__aicore__ inline bool CanUseDirectGmCopy(uint32_t gradDim)
{
    return IsGradDim32BAligned(gradDim) && !NeedsGmCopyPad<T>(gradDim);
}

template <typename T>
__aicore__ inline void SgdCompute(__local_mem__ T* dstW, __local_mem__ T* srcG, __local_mem__ T* srcW,
                                  uint32_t calCount, uint16_t repeatCount, uint32_t oneRepeat, float lr)
{
    AscendC::MicroAPI::RegTensor<T> dstVregG;
    AscendC::MicroAPI::RegTensor<T> dstVregW;
    AscendC::MicroAPI::RegTensor<T> srcVregG;
    AscendC::MicroAPI::RegTensor<T> srcVregW;
    AscendC::MicroAPI::MaskReg mask;

    for (uint16_t i = 0; i < repeatCount; ++i) {
        mask = AscendC::MicroAPI::UpdateMask<uint32_t>(calCount);
        AscendC::MicroAPI::DataCopy(srcVregG, srcG + i * oneRepeat);
        AscendC::MicroAPI::DataCopy(srcVregW, srcW + i * oneRepeat);

        AscendC::MicroAPI::Muls(dstVregG, srcVregG, lr, mask);
        AscendC::MicroAPI::Sub(dstVregW, srcVregW, dstVregG, mask);

        AscendC::MicroAPI::DataCopy(dstW + i * oneRepeat, dstVregW, mask);
    }
}

__aicore__ inline void ComputeSgdSimd(const __gm__ SgdSimdTilingData* tiling, uint32_t len, LocalTensor<float>& u0,
                                      LocalTensor<float>& u1)
{
    const float lr = tiling->lr;

    __local_mem__ float* srcG = (__local_mem__ float*)u0.GetPhyAddr();
    __local_mem__ float* srcW = (__local_mem__ float*)u1.GetPhyAddr();
    __local_mem__ float* dstW = (__local_mem__ float*)u1.GetPhyAddr();

    constexpr uint32_t vecLen = AscendC::GetVecLen();
    constexpr uint32_t oneRepeat = vecLen / static_cast<uint32_t>(sizeof(float));
    const uint16_t repeatCount = static_cast<uint16_t>((len + oneRepeat - 1U) / oneRepeat);

    VF_CALL<SgdCompute<float>>(dstW, srcG, srcW, len, repeatCount, oneRepeat, lr);
}

}  // namespace dyn_emb_sgd_simd
