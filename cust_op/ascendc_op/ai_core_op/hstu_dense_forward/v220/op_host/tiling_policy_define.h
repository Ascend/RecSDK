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


#ifndef TILING_POLICY_DEFINE_H
#define TILING_POLICY_DEFINE_H

#include "ops_log.h"

namespace HstuDenseForward {

#include <cstdio>
#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace INDEX_T {
    constexpr int INDEX_0 = 0;
    constexpr int INDEX_1 = 1;
    constexpr int INDEX_2 = 2;
    constexpr int INDEX_3 = 3;
    constexpr int INDEX_4 = 4;
    constexpr int INDEX_5 = 5;
}

namespace INPUT_INDEX_T {
    constexpr int Q_INDEX = 0;
    constexpr int K_INDEX = 1;
    constexpr int V_INDEX = 2;
    constexpr int MASK_INDEX = 3;
    constexpr int ATTN_BIAS_INDEX = 4;
    constexpr int SEQ_OFFSET_Q_INDEX = 5;
    constexpr int SEQ_OFFSET_K_INDEX = 6;
    constexpr int SEQ_OFFSET_T_INDEX = 7;
    constexpr int KV_CACHE_INDEX = 8;
    constexpr int PAGE_OFFSETS_INDEX = 9;
    constexpr int PAGE_IDS_INDEX = 10;
    constexpr int LAST_PAGE_LEN_INDEX = 11;
    constexpr int NUM_CONTEXT_INDEX = 12;
    constexpr int NUM_TARGET_INDEX = 13;
}

namespace ATTR_INDEX_T {
    constexpr int MASKTYPE_INDEX = 0;
    constexpr int MAX_SEQ_Q_INDEX = 1;
    constexpr int MAX_SEQ_K_INDEX = 2;
    constexpr int SILU_SCALE_INDEX = 3;
    constexpr int LAYOUT_INDEX = 4;
    constexpr int TARGET_GROUP_SIZE_INDEX = 5;
    constexpr int IS_DELTA_QK_INDEX = 6;
    constexpr int ALPHA_INDEX = 7;
}

constexpr int NORMAL_TILING_KEY = 0;
constexpr int JAGGED_TILING_KEY = 1;
constexpr int PAGED_TILING_KEY = 2;

constexpr int NORMAL_DIM_NUM = 4;
constexpr int JAGGED_DIM_NUM = 3;
constexpr int CONTEXT_DIM_NUM = 1;

constexpr int MAX_AIV_NUM = 48;
constexpr int MAX_BATCH_SIZE = 2048;
#ifdef SUPPORT_V200
    constexpr int BLOCK_HEIGHT = 128;
#else
    constexpr int BLOCK_HEIGHT = 256;
#endif
constexpr int VCORE_NUM_IN_ONE_AIC = 2;
constexpr int COMPUTE_PIPE_NUM = 3;
constexpr int TRANS_PIPE_NUM = 4;
constexpr int TRANS_TASK_NUM = 3;

constexpr int MAX_NUM_CONTEXT = BLOCK_HEIGHT;
constexpr int MAX_NUM_TARGET = 512;

constexpr int MIN_PAGE_SIZE = 32;

enum class MASK_TYPE {
    MASK_TRIL = 0,
    MATRIX_TRIU = 1,
    MATRIX_NONE = 2,
    MATRIX_CUSTOM = 3,
};
}

#endif