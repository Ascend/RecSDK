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

#include "permute_pooled_embs_kernel.h"
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void permute_pooled_embs(
    GM_ADDR pooled_embs,
    GM_ADDR offset_dim_list,
    GM_ADDR permute_list,
    GM_ADDR inv_offset_dim_list,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    TPipe pipe;
    PermutePooledEmbs::Args args{pooled_embs, offset_dim_list, permute_list,
                                 inv_offset_dim_list, output, tiling};
    PermutePooledEmbs::PermutePooledEmbsKernel<DTYPE_POOLED_EMBS> kernel(args, &pipe);
    kernel.PermuteColumns();
}
