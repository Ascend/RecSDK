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

struct HstuDenseForwardTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;

    uint32_t size;
    TCubeTiling qkMatmul;
    TCubeTiling svMatmul;

    int64_t batchSize;
    int64_t seqLen;
    int64_t headNum;
    int64_t dim;
    int64_t blockHeight;

    int32_t qkBaseM;
    int32_t qkBaseN;
    int32_t svBaseM;
    int32_t svBaseN;

    uint32_t seqOffset[HSTU_MAX_BATCH_SIZE + 1];
    uint32_t eachCoreStartBlockId[HSTU_MAX_AIV_NUM];
    uint32_t eachCoreEndBlockId[HSTU_MAX_AIV_NUM];

    uint32_t enableBias;
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t maskType;
    int64_t maxSeqLen;
    float siluScale;
    uint64_t contextGM;
};

#endif  // HSTU_DENSE_FORWARD_TILING_H
