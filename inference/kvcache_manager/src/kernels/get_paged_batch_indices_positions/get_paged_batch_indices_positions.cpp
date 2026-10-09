/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#include "kernel_operator.h"
#include "get_paged_batch_indices_positions_kernel.h"

extern "C" __global__ __aicore__ void get_paged_batch_indices_positions(int32_t batch_size, GM_ADDR append_indptr,
                                                                        GM_ADDR seq_lens_ptr, GM_ADDR batch_indices_ptr,
                                                                        GM_ADDR positions_ptr)
{
    const __gm__ int32_t* aip = reinterpret_cast<const __gm__ int32_t*>(append_indptr);
    const __gm__ int32_t* slp = reinterpret_cast<const __gm__ int32_t*>(seq_lens_ptr);
    __gm__ int32_t* bip = reinterpret_cast<__gm__ int32_t*>(batch_indices_ptr);
    __gm__ int32_t* pp = reinterpret_cast<__gm__ int32_t*>(positions_ptr);

    AscendC::Simt::VF_CALL<GetPagedBatchIndicesPositions>(AscendC::Simt::Dim3{BATCH_INDICES_BLOCK_SIZE, 1, 1},
                                                          batch_size, aip, slp, bip, pp);
}
