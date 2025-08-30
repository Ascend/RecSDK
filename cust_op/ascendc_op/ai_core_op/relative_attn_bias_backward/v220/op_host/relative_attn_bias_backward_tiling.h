/**
 * @file relative_attn_bias_tiling.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_TILING_H
#define MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RelativeAttnBiasBackwardTilingData)
TILING_DATA_FIELD_DEF(int64_t, s);
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, timeStride);

TILING_DATA_FIELD_DEF(float, bucketDivisor);
TILING_DATA_FIELD_DEF(int64_t, numBuckets);
TILING_DATA_FIELD_DEF(int64_t, numLayer);

TILING_DATA_FIELD_DEF(int, gradDataType);
TILING_DATA_FIELD_DEF(int, indexDataType);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RelativeAttnBiasBackward, RelativeAttnBiasBackwardTilingData)
}  // namespace optiling
#endif  // MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_TILING_H
