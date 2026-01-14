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
#pragma once

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/matrix/matmul_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(UserItemFlashAttentionShapeInfo)
TILING_DATA_FIELD_DEF(uint32_t, item_batch);
TILING_DATA_FIELD_DEF(uint32_t, q_head_num);
TILING_DATA_FIELD_DEF(uint32_t, kv_head_num);
TILING_DATA_FIELD_DEF(uint32_t, head_dim);
TILING_DATA_FIELD_DEF(uint32_t, item_seq_len);
TILING_DATA_FIELD_DEF(uint32_t, user_seq_len);
TILING_DATA_FIELD_DEF(uint32_t, item_batch_stride);
TILING_DATA_FIELD_DEF(uint32_t, item_head_num_stride);
TILING_DATA_FIELD_DEF(uint32_t, user_seq_stride);
TILING_DATA_FIELD_DEF(uint32_t, user_head_num_stride);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(UserItemFlashAttentionShapeInfoOp, UserItemFlashAttentionShapeInfo);

BEGIN_TILING_DATA_DEF(UserItemFlashAttentionTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmax_tiling);
TILING_DATA_FIELD_DEF_STRUCT(UserItemFlashAttentionShapeInfo, shape_info);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, gemm_qk_tiling);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, gemm_pv_tiling);
TILING_DATA_FIELD_DEF(uint32_t, total_task_num);
TILING_DATA_FIELD_DEF(uint32_t, task_num_per_core);
TILING_DATA_FIELD_DEF(uint32_t, item_batch_block_num);
TILING_DATA_FIELD_DEF(uint32_t, used_aiv_num);
TILING_DATA_FIELD_DEF(uint32_t, item_batch_block);
TILING_DATA_FIELD_DEF(uint32_t, user_kv_seq_block);
TILING_DATA_FIELD_DEF(uint32_t, vec_softmax_batch_block);
TILING_DATA_FIELD_DEF(uint32_t, vec_aggr_batch_block);
TILING_DATA_FIELD_DEF(uint32_t, softmax_buf_size);
TILING_DATA_FIELD_DEF(uint32_t, gemm_qk_res_size);
TILING_DATA_FIELD_DEF(uint32_t, gemm_pv_res_size);
TILING_DATA_FIELD_DEF(uint32_t, gemm_aggr_res_size);
TILING_DATA_FIELD_DEF(uint32_t, has_item_kv);
TILING_DATA_FIELD_DEF(float, score_scale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UserItemFlashAttention, UserItemFlashAttentionTilingData);
}  // namespace optiling
