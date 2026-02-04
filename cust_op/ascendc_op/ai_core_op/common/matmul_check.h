/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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
#ifndef MATMUL_CHECK_H
#define MATMUL_CHECK_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "ops_log.h"

namespace MatmulTilingCheck {
    inline bool SafeCheckMultiply(int a, int b)
    {
        constexpr int INT_MAX = std::numeric_limits<int>::max();
        if (a <= 0 || b <= 0) {
            return false;
        }
        return a <= INT_MAX / b;
    }

    inline bool CheckBaseMNK(optiling::TCubeTiling& tiling, int inputDataType, int outputDataType)
    {
        int baseM = tiling.get_baseM();
        int baseN = tiling.get_baseN();
        int baseK = tiling.get_baseK();
        int M = tiling.get_M();
        int N = tiling.get_N();
        int Ka = tiling.get_Ka();
        int Kb = tiling.get_Kb();
        OPS_CHECK(!SafeCheckMultiply(baseM, baseK), OPS_LOG_E("", "baseM * baseK out_of_range"), return false);
        OPS_CHECK(!SafeCheckMultiply(M, Ka), OPS_LOG_E("", "M * Ka out_of_range"), return false);
        OPS_CHECK(!SafeCheckMultiply(baseN, baseK), OPS_LOG_E("", "baseN * baseK out_of_range"), return false);
        OPS_CHECK(!SafeCheckMultiply(N, Kb), OPS_LOG_E("", "N * Kb out_of_range"), return false);
        OPS_CHECK(!SafeCheckMultiply(baseM, baseN), OPS_LOG_E("", "baseM * baseN out_of_range"), return false);
        OPS_CHECK(!SafeCheckMultiply(M, N), OPS_LOG_E("", "M * N out_of_range"), return false);
        return true;
    }
};

#endif // COMMON_HOST_H