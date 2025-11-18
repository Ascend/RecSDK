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

}  // namespace HstuDenseForward

#endif  // MXREC_HSTU_COMMON_CONST_H

