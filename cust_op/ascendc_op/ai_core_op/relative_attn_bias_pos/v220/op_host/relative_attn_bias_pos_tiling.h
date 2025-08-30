/**
 * @file relative_attn_bias_pos_tiling.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_POS_TILING_H
#define MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_POS_TILING_H
#include "register/tilingdata_base.h"
constexpr int MAX_BATCH_SIZE = 512;

namespace optiling {
BEGIN_TILING_DATA_DEF(RelativeAttnBiasPosTilingData)
TILING_DATA_FIELD_DEF(int64_t, s);
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, stride);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_BATCH_SIZE, pastValidLens);

TILING_DATA_FIELD_DEF(int, dataType);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RelativeAttnBiasPos, RelativeAttnBiasPosTilingData)
}  // namespace optiling
#endif  // MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_POS_TILING_H
