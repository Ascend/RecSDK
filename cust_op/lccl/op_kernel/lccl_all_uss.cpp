/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#include "kernel_operator.h"
#include "all2all_uss.h"
#include "all_uss_deterministic.h"

extern "C" __global__ __aicore__ void lccl_all_uss(GM_ADDR send_data, GM_ADDR send_count_matrix,
                                                   GM_ADDR shape_vec, GM_ADDR peer_mem, GM_ADDR restore,
                                                   GM_ADDR rev_data, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    if (tiling_data.deterministic) {
        AllUssDeterministic<float> opKernel(tiling_data.rank, tiling_data.rankSize, (1 << 2));
        opKernel.Init(send_data, send_count_matrix, shape_vec, peer_mem, restore, rev_data,
                      tiling_data.rank, tiling_data.rankSize, tiling_data.magic, tiling_data.dim, tiling_data.outShape);
        opKernel.Process();
    } else {
        AllUss<float> opKernel(tiling_data.rank, tiling_data.rankSize, (1 << 2));
        opKernel.Init(send_data, send_count_matrix, shape_vec, peer_mem, restore, rev_data,
                      tiling_data.rank, tiling_data.rankSize, tiling_data.magic, tiling_data.dim, tiling_data.outShape);
        opKernel.Process();
    }
}