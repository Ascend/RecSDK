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

enum class CubeToVecT {
    PING_PONG_FULL_UB = 0,
    PING_PONG_FULL_GM = 1,
};

enum class VecToCubeT {
    PING_PONG_FULL_TSCM = 0,
    PING_PONG_FULL_GM = 1,
};

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
    static constexpr int blockArea = tilingM * tilingN;
    static constexpr auto Config = FindShape<sizeof(qType), blockM, blockN, blockK>();

    __aicore__ static constexpr bool UseL1Cache()
    {
        return !std::is_same<qType, float>::value && (blockK < MAX_BLOCK_DIM);
    }

    __aicore__ static constexpr size_t GetL1CacheSize()
    {
        if constexpr (UseL1Cache()) {
            return MATMUL_L1_SIZE - 2 * sizeof(qType) * blockM * blockN;
        } else {
            return MATMUL_L1_SIZE;
        }
    }
    
    __aicore__ static constexpr bool EnableMask()
    {
        return maskedType != CausalMaskT::MASK_NONE;
    }

    __aicore__ static constexpr bool EnableBias()
    {
        return enableBias;
    }

    __aicore__ static constexpr CubeToVecT GetCubeToVecType()
    {
        if constexpr (std::is_same<qType, float>::value || std::is_same<qType, fp8_e4m3fn_t>::value) {
            return CubeToVecT::PING_PONG_FULL_GM;
        }

        return blockArea <= BASIC_K_64 * BASIC_K_64 ? CubeToVecT::PING_PONG_FULL_UB : CubeToVecT::PING_PONG_FULL_GM;
    }

    __aicore__ static constexpr VecToCubeT GetVecToCubeType()
    {
        return VecToCubeT::PING_PONG_FULL_GM;
    }

    __aicore__ static constexpr uint32_t GetInTQueNumber()
    {
        return GetCubeToVecType() == CubeToVecT::PING_PONG_FULL_GM ? 1 : 2;
    }

    __aicore__ static constexpr uint32_t GetOutTQueNumber()
    {
        return GetVecToCubeType() == VecToCubeT::PING_PONG_FULL_GM ? 1 : 2;
    }

    // only used when enable CV passthrough
    __aicore__ static constexpr uint32_t GetUbBlockElem()
    {
        int32_t returnElementNumer = 1; // init fot cast
        returnElementNumer += EnableMask();
        returnElementNumer += EnableBias();
        returnElementNumer += GetInTQueNumber();
        returnElementNumer += GetOutTQueNumber();
        uint32_t blockElem = UB_SIZE / sizeof(float) / returnElementNumer;
        return AlignDown(blockElem, ALIGN_1024);
    }

    __aicore__ static constexpr auto GetQKMatmulConfig()
    {
        return Config.getQKMatmulConfig();
    }

    __aicore__ static constexpr auto GetSVMatmulConfig()
    {
        return Config.getSVMatmulConfig();
    }
};
}
#endif