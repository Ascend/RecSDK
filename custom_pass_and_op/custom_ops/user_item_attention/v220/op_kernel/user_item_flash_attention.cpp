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
#include "user_item_flash_attention.h"

#include "kernel_operator.h"

extern "C" __global__ __aicore__ void user_item_flash_attention(GM_ADDR query, GM_ADDR key_user, GM_ADDR value_user,
                                                                GM_ADDR mask_len, GM_ADDR key_item, GM_ADDR value_item,
                                                                GM_ADDR attn_out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data_in, tiling);
    AscendC::TPipe pipe;
    using namespace matmul;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    kernels::UserItemFlashAttentionKernel<DTYPE_QUERY, float, DTYPE_QUERY> user_item_fa(&pipe);
    REGIST_MATMUL_OBJ(user_item_fa.pipe_, GetSysWorkSpacePtr(), user_item_fa.gemm_qk_, &tiling_data_in.gemm_qk_tiling,
                      user_item_fa.gemm_pv_, &tiling_data_in.gemm_pv_tiling);
    user_item_fa.init(query, key_user, value_user, mask_len, key_item, value_item, workspace, &tiling_data_in,
                      attn_out);
    user_item_fa.process();
}
