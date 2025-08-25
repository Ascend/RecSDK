/**
 * @file relative_attn_bias_time_tiling.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_TIME_TILING_H
#define MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_TIME_TILING_H
#include "register/tilingdata_base.h"
constexpr int MAX_BATCH_SIZE = 512;

namespace optiling {
BEGIN_TILING_DATA_DEF(RelativeAttnBiasTimeTilingData)
TILING_DATA_FIELD_DEF(int64_t, s);
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, stride);

TILING_DATA_FIELD_DEF(float, bucketDivisor);
TILING_DATA_FIELD_DEF(int64_t, numBuckets);
TILING_DATA_FIELD_DEF(int64_t, numLayer);
TILING_DATA_FIELD_DEF(float, clampMax);

TILING_DATA_FIELD_DEF(int, tswType);
TILING_DATA_FIELD_DEF(int, tsType);
TILING_DATA_FIELD_DEF(int, buffSize);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RelativeAttnBiasTime, RelativeAttnBiasTimeTilingData)
}  // namespace optiling
#endif  // MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_TIME_TILING_H
