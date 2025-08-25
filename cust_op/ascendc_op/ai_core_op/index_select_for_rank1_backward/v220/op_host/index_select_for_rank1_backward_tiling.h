/**
 * @file index_select_for_rank1_backward_tiling.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef INDEX_SELECT_RANK1_BACKWARD_TILING
#define INDEX_SELECT_RANK1_BACKWARD_TILING
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexSelectForRank1BackwardTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalLen);
TILING_DATA_FIELD_DEF(int64_t, xDim0);
TILING_DATA_FIELD_DEF(int64_t, baseLen);
TILING_DATA_FIELD_DEF(int64_t, tailSplitIndex);
TILING_DATA_FIELD_DEF(int64_t, stride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexSelectForRank1Backward, IndexSelectForRank1BackwardTilingData)
}  // namespace optiling
#endif