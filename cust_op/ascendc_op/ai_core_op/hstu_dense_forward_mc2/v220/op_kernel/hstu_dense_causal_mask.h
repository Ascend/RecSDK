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

#ifndef HSTU_DENSE_CAUSAL_MASK_H
#define HSTU_DENSE_CAUSAL_MASK_H

#include <unistd.h>

#include <cstdint>
#include <type_traits>

#include "kernel_log.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace HstuDenseForward {

/**
 * @brief Causal mask types supported by the HSTU Dense Forward operator.
 *
 * Currently supported mask types:
 *   - MASK_TRIL  (0): Lower triangular mask (tril). Fills the lower triangle
 *                     with zeros, upper triangle with the given value.
 *   - MASK_NONE  (2): No mask. All elements are set to zero; only offsets
 *                     within the range are filled with the given value.
 *   - MASK_CUSTOM (3): User-defined custom mask provided via a GM buffer.
 *                     Loaded and applied separately in DoMaskInitOptional.
 *
 * NOT supported (will assert at runtime):
 *   - MASK_TRIU  (1): Upper triangular mask. DoCausalMask asserts unimplemented.
 */
enum class CausalMaskT {
    MASK_TRIL = 0,  // Lower triangular mask  (supported)
    MASK_TRIU,      // Upper triangular mask  (NOT supported)
    MASK_NONE,      // No mask                (supported)
    MASK_CUSTOM,    // User-defined custom mask (supported)
};

template <typename qType, CausalMaskT maskType>
__aicore__ inline void DoCausalMask(LocalTensor<qType>& inMaskLt, int64_t maskOffset, int64_t maskLens,
                                    int64_t maskStride, int64_t repeatTimes, qType value)
{
    if constexpr (maskType == CausalMaskT::MASK_TRIL) {
        Duplicate<qType>(inMaskLt, 0, maskLens);
        for (int i = 0; i < repeatTimes; i++) {
            int64_t thisIndexMask = maskOffset + i + 1;
            Duplicate<qType>(inMaskLt[i * maskStride], value, thisIndexMask);
        }
    } else if constexpr (maskType == CausalMaskT::MASK_TRIU) {
        ASCENDC_ASSERT((false), "DoCausalMask triu is not implemented");
    } else if constexpr (maskType == CausalMaskT::MASK_NONE) {
        Duplicate<qType>(inMaskLt, 0, maskLens);
        for (int i = 0; i < repeatTimes; i++) {
            Duplicate<qType>(inMaskLt[i * maskStride], value, maskOffset);
        }
    } else {
        ASCENDC_ASSERT((false), "DoCausalMask custom is not implemented");
    }
}
}  // namespace HstuDenseForward
#endif
