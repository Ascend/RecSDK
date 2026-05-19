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

#ifndef ASCENDC_LOAD_OR_INITIALIZE_EMBEDDINGS_KERNEL_H_
#define ASCENDC_LOAD_OR_INITIALIZE_EMBEDDINGS_KERNEL_H_

#include <simt_api/common_functions.h>
#include "kernel_operator.h"
#include "device_utils.h"

using namespace AscendC;
constexpr uint32_t BLOCK_THREAD_NUM_1024 = 1024;

template <typename T, typename EmbeddingGenerator>
__simt_vf__ __aicore__
LAUNCH_BOUND(BLOCK_THREAD_NUM_1024) inline void load_or_initialize_embeddings_kernel_vf(
    const uint64_t n,
    const int emb_dim,
    __gm__ T* outputs,
    __gm__ T* __gm__* inputs_ptr,
    __gm__ bool* masks,
    typename EmbeddingGenerator::Args generator_args)
{
    EmbeddingGenerator emb_gen(generator_args);

    for (uint64_t emb_id = AscendC::Simt::GetBlockIdx(); emb_id < n; emb_id += AscendC::Simt::GetBlockNum()) {
        __gm__ T* input_ptr = inputs_ptr[emb_id];
        bool mask = masks[emb_id];

        if (mask) {
            for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
                outputs[emb_id * emb_dim + i] = input_ptr[i];
            }
        } else {
            for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
                auto tmp = emb_gen.generate(emb_id);
                outputs[emb_id * emb_dim + i] = dyn_emb::TypeConvertFunc<T, float>::convert(tmp);
            }
        }
    }

    emb_gen.destroy();
}

template <typename T, typename EmbeddingGenerator>
__global__ __vector__ void load_or_initialize_embeddings_kernel(
    const uint64_t n,
    const int emb_dim,
    __gm__ T* outputs,
    __gm__ T* __gm__* inputs_ptr,
    __gm__ bool* masks,
    typename EmbeddingGenerator::Args generator_args)
{
    asc_vf_call<load_or_initialize_embeddings_kernel_vf<T, EmbeddingGenerator>>(
        dim3{BLOCK_THREAD_NUM_1024}, n, emb_dim, outputs, inputs_ptr, masks, generator_args);
}

#endif  // ASCENDC_LOAD_OR_INITIALIZE_EMBEDDINGS_KERNEL_H_
