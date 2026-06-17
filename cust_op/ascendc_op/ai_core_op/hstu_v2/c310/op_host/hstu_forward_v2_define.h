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

#ifndef HSTU_FORWARD_V2_DEFINE_H
#define HSTU_FORWARD_V2_DEFINE_H

enum class DIM_INDEX : uint32_t {
    ZERO = 0,
    ONE,
    TWO,
    THREE
};

enum class IN_INDEX : uint32_t {
    Q = 0,
    K,
    V,
    MASK,
    RAB,
    SEQ_OFFSET_Q,
    SEQ_OFFSET_K,
    NUM_CONTEXT,
    NUM_TARGET
};

enum class OUT_INDEX : uint32_t {
    ATTN_OUTPUT = 0
};

enum class ATTR_INDEX : uint32_t {
    MAX_SEQLEN_Q = 0,
    MAX_SEQLEN_K = 1,
    SCALE,
    TARGET_GROUP_SIZE,
    ALPHA
};

#endif
