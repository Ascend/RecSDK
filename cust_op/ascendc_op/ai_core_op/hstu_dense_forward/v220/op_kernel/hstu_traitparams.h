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

#ifndef HSTU_TRAIPARAMS_H
#define HSTU_TRAIPARAMS_H

#include <unistd.h>
#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "matmul_constexpr.h"
#include "hstu_dense_causal_mask.h"

namespace HstuForward {

template <typename qTypeTemplate, typename oTypeTemplate, bool bias,
    bool determin, CausalMaskT maskedType, int tilingM, int tilingN, int tilingK>
struct TraitParams {
    using qType = qTypeTemplate;
    using oType = oTypeTemplate;
    static constexpr bool enableBias = bias;
    static constexpr bool deterministic = determin;
    static constexpr CausalMaskT maskType = maskedType;
    static constexpr int blockM = tilingM;
    static constexpr int blockN = tilingN;
    static constexpr int blockK = tilingK;
};


}  // namespace HstuForward

#endif  // HSTU_TRAIPARAMS_H

