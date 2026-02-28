/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include <type_traits>
#include "pooling_embeddings_kernel.h"
#include "kernel_operator.h"
#include "../ops_utils.h"

constexpr int32_t BLOCK_THREADS = PoolingEmbeddingsSimt::MAX_THREADS_PER_BLOCK;
constexpr int32_t ELEMS_PER_THREAD = PoolingEmbeddingsSimt::MAX_ELEMENTS_PER_THREAD;

extern "C" __global__ __aicore__ void pooling_embeddings(GM_ADDR src_data, GM_ADDR dst_data, GM_ADDR offset_data,
                                                    GM_ADDR inverse_data, int32_t combiner, int32_t total_dims,
                                                    int32_t accum_dims, int32_t ev_size, int32_t num_vec,
                                                    int32_t batch_size, int32_t totalBlocks, int32_t blocksPerCore,
                                                    int32_t remainderBlocks, bool isSmall, uint32_t src_type_num,
                                                    uint32_t dst_type_num, uint32_t offset_type_num)
{
    int32_t coreId = AscendC::GetBlockIdx();

    dyn_emb::DataType src_type = static_cast<dyn_emb::DataType>(src_type_num);
    dyn_emb::DataType dst_type = static_cast<dyn_emb::DataType>(dst_type_num);
    dyn_emb::DataType offset_type = static_cast<dyn_emb::DataType>(offset_type_num);
    INT_TYPE_DISPATCH(offset_type, offset_t, {
        FLOAT_TYPE_DISPATCH(src_type, src_t, {
            FLOAT_TYPE_DISPATCH(dst_type, dst_t, {
                __gm__ src_t* src = reinterpret_cast<__gm__ src_t*>(src_data);
                __gm__ dst_t* dst = reinterpret_cast<__gm__ dst_t*>(dst_data);
                __gm__ offset_t* offset = reinterpret_cast<__gm__ offset_t*>(offset_data);
                __gm__ offset_t* inverse = reinterpret_cast<__gm__ offset_t*>(inverse_data);

                if (isSmall) {
                    AscendC::Simt::VF_CALL<PoolingEmbeddingsSimt::SimtSmallDataCompute<src_t, dst_t, offset_t>>(
                        AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, src, dst, offset, inverse,
                        combiner, total_dims, accum_dims, ev_size, num_vec, batch_size);

                } else {
                    int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
                    int32_t blockStartIdx = coreId * blocksPerCore +
                        ((coreId < remainderBlocks) ? coreId : remainderBlocks);

                    AscendC::Simt::VF_CALL<PoolingEmbeddingsSimt::SimtLargeDataCompute<src_t, dst_t, offset_t>>(
                        AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, src, dst, offset, inverse, combiner,
                        total_dims, accum_dims, ev_size, num_vec, batch_size, totalBlocks,
                        blockStartIdx, curBlocksCount);
                }
            });
        });
    });
}