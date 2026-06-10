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

#ifndef ASCENDC_INITIALIZE_WITH_INDEX_KERNEL_H_
#define ASCENDC_INITIALIZE_WITH_INDEX_KERNEL_H_

#include <simt_api/common_functions.h>
#include "kernel_operator.h"
#include "device_utils.h"

using namespace AscendC;
constexpr uint32_t INITIALIZE_WITH_INDEX_BLOCK_THREAD_NUM = 1024;

template <typename ValueT, typename IndexT, typename GeneratorT>
__simt_vf__ __aicore__ LAUNCH_BOUND(INITIALIZE_WITH_INDEX_BLOCK_THREAD_NUM) inline void initialize_with_index_kernel_vf(
    int64_t num, int64_t dim, int64_t stride, __gm__ ValueT* buffer, __gm__ IndexT const* indices,
    typename GeneratorT::Args generator_args)
{
    GeneratorT gen(generator_args);

    for (uint64_t emb_id = AscendC::Simt::GetBlockIdx(); emb_id < static_cast<uint64_t>(num);
         emb_id += AscendC::Simt::GetBlockNum()) {
        int64_t index = indices[emb_id];
        __gm__ ValueT* dst = buffer + index * stride;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            auto tmp = gen.generate(index);
            dst[i] = dyn_emb::TypeConvertFunc<ValueT, float>::convert(tmp);
        }
    }

    gen.destroy();
}

template <typename ValueT, typename IndexT, typename GeneratorT>
__global__ __vector__ void initialize_with_index_kernel(int64_t num, int64_t dim, int64_t stride, __gm__ ValueT* buffer,
                                                        __gm__ IndexT const* indices,
                                                        typename GeneratorT::Args generator_args)
{
    asc_vf_call<initialize_with_index_kernel_vf<ValueT, IndexT, GeneratorT>>(
        dim3{INITIALIZE_WITH_INDEX_BLOCK_THREAD_NUM}, num, dim, stride, buffer, indices, generator_args);
}

#endif  // ASCENDC_INITIALIZE_WITH_INDEX_KERNEL_H_
