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

#ifndef HSTU_DENSE_FORWARD_TILING_H
#define HSTU_DENSE_FORWARD_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "tiling_policy_define.h"

/**
 * @brief Tiling data structure for HSTU Dense Forward MC2 operator.
 *
 * This struct holds all parameters computed on the host and passed to the device-side kernel.
 * It includes MC2 communication init/config, Matmul tiling shapes, sequence/shape info,
 * and per-core block partitioning.
 */
struct HstuDenseForwardTilingData {
    Mc2InitTiling mc2InitTiling;  // MC2 communication init tiling
    Mc2CcTiling mc2CcTiling;      // MC2 communication tiling config

    uint32_t size;         // Total size of tiling data
    TCubeTiling qkMatmul;  // Q*K Matmul tiling parameters
    TCubeTiling svMatmul;  // Score*V Matmul tiling parameters

    int64_t batchSize;    // Batch size (xDim0)
    int64_t seqLen;       // Sequence length per card (xDim1)
    int64_t headNum;      // Number of attention heads (xDim2)
    int64_t dim;          // Head dimension (xDim3)
    int64_t blockHeight;  // Block height for tiled computation (e.g., 256)

    int32_t qkBaseM;  // QK Matmul base M (tiling base)
    int32_t qkBaseN;  // QK Matmul base N (tiling base)
    int32_t svBaseM;  // SV Matmul base M (tiling base)
    int32_t svBaseN;  // SV Matmul base N (tiling base)

    uint32_t seqOffset[HSTU_MAX_BATCH_SIZE + 1];      // Variable-length sequence offsets
    uint32_t eachCoreStartBlockId[HSTU_MAX_AIV_NUM];  // Starting block ID per AIV core
    uint32_t eachCoreEndBlockId[HSTU_MAX_AIV_NUM];    // Ending block ID per AIV core

    uint32_t enableBias;  // Whether attn_bias is provided (1/0)
    uint32_t rankId;      // Current rank ID in the communication group
    uint32_t rankSize;    // Total number of ranks in the communication group
    uint32_t maskType;    // Causal mask type: 0=TRIL, 1=TRIU, 2=NONE, 3=CUSTOM
    int64_t maxSeqLen;    // Maximum sequence length across all batches
    float siluScale;      // Scale factor for SiLU activation
    uint64_t contextGM;   // HCCL context global memory address
};

#endif  // HSTU_DENSE_FORWARD_TILING_H
