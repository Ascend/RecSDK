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


#ifndef MXREC_HSTU_COMMON_CONST_H
#define MXREC_HSTU_COMMON_CONST_H

namespace HstuDenseForward {

constexpr uint32_t MAX_BATCH_SIZE = 2048;
constexpr uint32_t MAX_HEAD_NUM = 8;
constexpr int USE_QUEUE_NUM = 1;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int MAX_INDICS_ONE_BLOCK = 100;

constexpr int INVALID_TASK_ID = -1;
constexpr int BLOCK_M = 256;
constexpr int BLOCK_MN = 256 * 256;

template <typename T>
__aicore__ inline T CeilDiv(T dividend, T divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (dividend + divisor - 1) / divisor;
}

// (maskedType << 4) | (enableBias << 3) | (isQkUseUb << 2) | typeTilingKey
// maskedType: { 0:mask_tril, 2:mask_none, 3:mask_custom }
// enableBias: { 0:no_bias, 1:bias }
// isQkUseUb: { 0:no_ub, 1:ub }
// typeTilingKey: { 0:normal, 1:jagged, 2:paged }
#define NORMAL_QK_WS_NO_BIAS_CAUSAL_MASK 0UL  // (0 << 4 | 0 << 3 | 0 << 2 | 0)
#define JAGGED_QK_WS_NO_BIAS_CAUSAL_MASK 1UL  // (0 << 4 | 0 << 3 | 0 << 2 | 1)
#define PAGED_QK_WS_NO_BIAS_CAUSAL_MASK 2UL  // (0 << 4 | 0 << 3 | 0 << 2 | 2)
#define NORMAL_QK_UB_NO_BIAS_CAUSAL_MASK 4UL  // (0 << 4 | 0 << 3 | 1 << 2 | 0)
#define JAGGED_QK_UB_NO_BIAS_CAUSAL_MASK 5UL  // (0 << 4 | 0 << 3 | 1 << 2 | 1)
#define PAGED_QK_UB_NO_BIAS_CAUSAL_MASK 6UL  // (0 << 4 | 0 << 3 | 1 << 2 | 2)
#define NORMAL_QK_WS_BIAS_CAUSAL_MASK 8UL  // (0 << 4 | 1 << 3 | 0 << 2 | 0)
#define JAGGED_QK_WS_BIAS_CAUSAL_MASK 9UL  // (0 << 4 | 1 << 3 | 0 << 2 | 1)
#define PAGED_QK_WS_BIAS_CAUSAL_MASK 10UL  // (0 << 4 | 1 << 3 | 0 << 2 | 2)
#define NORMAL_QK_UB_BIAS_CAUSAL_MASK 12UL  // (0 << 4 | 1 << 3 | 1 << 2 | 0)
#define JAGGED_QK_UB_BIAS_CAUSAL_MASK 13UL  // (0 << 4 | 1 << 3 | 1 << 2 | 1)
#define PAGED_QK_UB_BIAS_CAUSAL_MASK 14UL  // (0 << 4 | 1 << 3 | 1 << 2 | 2)

#define NORMAL_QK_WS_NO_BIAS_NO_MASK 32UL  // (2 << 4 | 0 << 3 | 0 << 2 | 0)
#define JAGGED_QK_WS_NO_BIAS_NO_MASK 33UL  // (2 << 4 | 0 << 3 | 0 << 2 | 1)
#define PAGED_QK_WS_NO_BIAS_NO_MASK 34UL  // (2 << 4 | 0 << 3 | 0 << 2 | 2)
#define NORMAL_QK_UB_NO_BIAS_NO_MASK 36UL  // (2 << 4 | 0 << 3 | 1 << 2 | 0)
#define JAGGED_QK_UB_NO_BIAS_NO_MASK 37UL  // (2 << 4 | 0 << 3 | 1 << 2 | 1)
#define PAGED_QK_UB_NO_BIAS_NO_MASK 38UL  // (2 << 4 | 0 << 3 | 1 << 2 | 2)
#define NORMAL_QK_WS_BIAS_NO_MASK 40UL  // (2 << 4 | 1 << 3 | 0 << 2 | 0)
#define JAGGED_QK_WS_BIAS_NO_MASK 41UL  // (2 << 4 | 1 << 3 | 0 << 2 | 1)
#define PAGED_QK_WS_BIAS_NO_MASK 42UL  // (2 << 4 | 1 << 3 | 0 << 2 | 2)
#define NORMAL_QK_UB_BIAS_NO_MASK 44UL  // (2 << 4 | 1 << 3 | 1 << 2 | 0)
#define JAGGED_QK_UB_BIAS_NO_MASK 45UL  // (2 << 4 | 1 << 3 | 1 << 2 | 1)
#define PAGED_QK_UB_BIAS_NO_MASK 46UL  // (2 << 4 | 1 << 3 | 1 << 2 | 2)

#define NORMAL_QK_WS_NO_BIAS_CUSTOM_MASK 48UL  // (3 << 4 | 0 << 3 | 0 << 2 | 0)
#define JAGGED_QK_WS_NO_BIAS_CUSTOM_MASK 49UL  // (3 << 4 | 0 << 3 | 0 << 2 | 1)
#define PAGED_QK_WS_NO_BIAS_CUSTOM_MASK 50UL  // (3 << 4 | 0 << 3 | 0 << 2 | 2)
#define NORMAL_QK_UB_NO_BIAS_CUSTOM_MASK 52UL  // (3 << 4 | 0 << 3 | 1 << 2 | 0)
#define JAGGED_QK_UB_NO_BIAS_CUSTOM_MASK 53UL  // (3 << 4 | 0 << 3 | 1 << 2 | 1)
#define PAGED_QK_UB_NO_BIAS_CUSTOM_MASK 54UL  // (3 << 4 | 0 << 3 | 1 << 2 | 2)
#define NORMAL_QK_WS_BIAS_CUSTOM_MASK 56UL  // (3 << 4 | 1 << 3 | 0 << 2 | 0)
#define JAGGED_QK_WS_BIAS_CUSTOM_MASK 57UL  // (3 << 4 | 1 << 3 | 0 << 2 | 1)
#define PAGED_QK_WS_BIAS_CUSTOM_MASK 58UL  // (3 << 4 | 1 << 3 | 0 << 2 | 2)
#define NORMAL_QK_UB_BIAS_CUSTOM_MASK 60UL  // (3 << 4 | 1 << 3 | 1 << 2 | 0)
#define JAGGED_QK_UB_BIAS_CUSTOM_MASK 61UL  // (3 << 4 | 1 << 3 | 1 << 2 | 1)
#define PAGED_QK_UB_BIAS_CUSTOM_MASK 62UL  // (3 << 4 | 1 << 3 | 1 << 2 | 2)

}  // namespace HstuDenseForward

#endif  // MXREC_HSTU_COMMON_CONST_H

