/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef MATMUL_CONSTEXPR_H
#define MATMUL_CONSTEXPR_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

constexpr int BLOCK_HEIGHT_256 = 256;
constexpr int BLOCK_HEIGHT_128 = 128;
constexpr int BASIC_K_32 = 32;
constexpr int BASIC_K_64 = 64;
constexpr int MAX_BLOCK_DIM = 512;
constexpr int MATMUL_L1_SIZE = 524288;  // 512KB

constexpr MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;

#ifdef SUPPORT_910_95
    // FP16/BFP16
    constexpr MatmulShapeParams qkShapeFp16Params = {
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, MAX_BLOCK_DIM,
        BLOCK_HEIGHT_128, BLOCK_HEIGHT_256, BASIC_K_64
    };

    constexpr MatmulShapeParams svShapeFp16Params = {
        BLOCK_HEIGHT_256, MAX_BLOCK_DIM, BLOCK_HEIGHT_256,
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, BASIC_K_64
    };

    // FP32
    constexpr MatmulShapeParams qkShapeFp32Params = {
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, MAX_BLOCK_DIM,
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, BASIC_K_32
    };

    constexpr MatmulShapeParams svShapeFp32Params = {
        BLOCK_HEIGHT_256, MAX_BLOCK_DIM, BLOCK_HEIGHT_256,
        BLOCK_HEIGHT_128, MAX_BLOCK_DIM, BASIC_K_32
    };
#else
    // FP16/BFP16
    constexpr MatmulShapeParams qkShapeFp16Params = {
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, MAX_BLOCK_DIM,
        BLOCK_HEIGHT_128, BLOCK_HEIGHT_256, BASIC_K_64
    };

    constexpr MatmulShapeParams svShapeFp16Params = {
        BLOCK_HEIGHT_256, MAX_BLOCK_DIM, BLOCK_HEIGHT_256,
        BLOCK_HEIGHT_128, BLOCK_HEIGHT_256, BASIC_K_64
    };

    // FP32
    constexpr MatmulShapeParams qkShapeFp32Params = {
        BLOCK_HEIGHT_256, BLOCK_HEIGHT_256, MAX_BLOCK_DIM,
        BLOCK_HEIGHT_128, BLOCK_HEIGHT_256, BASIC_K_32
    };

    constexpr MatmulShapeParams svShapeFp32Params = {
        BLOCK_HEIGHT_256, MAX_BLOCK_DIM, BLOCK_HEIGHT_256,
        BLOCK_HEIGHT_128, BLOCK_HEIGHT_256, BASIC_K_32
    };
#endif


constexpr MatmulQuantParams quantParams = {false, false};
constexpr MatmulBatchParams batchParams = {false, BatchMode::NONE};
constexpr MatmulFuncParams qkFuncParams = {false, false};
constexpr MatmulFuncParams svFuncParams = {false, false};

constexpr MatmulConfig mmStaticConfigQKFp16 =
    GetMMConfig<configMode>(qkShapeFp16Params, quantParams, batchParams, qkFuncParams);
constexpr MatmulConfig mmStaticConfigSVFp16 =
    GetMMConfig<configMode>(svShapeFp16Params, quantParams, batchParams, svFuncParams);

constexpr MatmulConfig mmStaticConfigQKFp32 =
    GetMMConfig<configMode>(qkShapeFp32Params, quantParams, batchParams, qkFuncParams);
constexpr MatmulConfig mmStaticConfigSVFp32 =
    GetMMConfig<configMode>(svShapeFp32Params, quantParams, batchParams, svFuncParams);

#endif