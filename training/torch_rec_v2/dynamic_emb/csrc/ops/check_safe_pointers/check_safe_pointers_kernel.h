/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ASCENDC_CHECK_SAFE_POINTERS_KERNEL_H_
#define ASCENDC_CHECK_SAFE_POINTERS_KERNEL_H_

#include <simt_api/common_functions.h>
#include "kernel_operator.h"

using namespace AscendC;
constexpr uint32_t BLOCK_THREAD_NUM = 2048;

template <typename T>
__simt_vf__ __aicore__
LAUNCH_BOUND(BLOCK_THREAD_NUM) inline void check_safe_pointers_kernel_vf(const uint64_t n,
    const __gm__ T* __gm__* ptrs, __gm__ uint64_t* counter, const uint64_t thread_all) {
    for (uint64_t id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; id < n; id += thread_all) {
        const __gm__ T *ptr = ptrs[id];
        if (ptr == nullptr) {
            asc_atomic_add(counter, 1);
        }
    }
}

template <typename T>
__global__ __vector__ void check_safe_pointers_kernel(const uint64_t n, const __gm__ T* __gm__* ptrs, __gm__ uint64_t* counter) {
    const uint64_t thread_all = BLOCK_THREAD_NUM * GetBlockNum();
    asc_vf_call<check_safe_pointers_kernel_vf<T>>(dim3{BLOCK_THREAD_NUM}, n, ptrs, counter, thread_all);
}

#endif  // ASCENDC_CHECK_SAFE_POINTERS_KERNEL_H_
