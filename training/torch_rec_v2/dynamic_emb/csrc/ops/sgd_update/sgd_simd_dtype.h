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

__aicore__ inline void ComputeSgdSimd(const __gm__ SgdSimdTilingData* tiling, uint32_t len, LocalTensor<float>& u0,
                                      LocalTensor<float>& u1)
{
    const float lr = tiling->lr;
    Muls<float>(u0, u0, lr, len);
    Sub<float>(u1, u1, u0, len);
}

}  // namespace dyn_emb_sgd_simd
