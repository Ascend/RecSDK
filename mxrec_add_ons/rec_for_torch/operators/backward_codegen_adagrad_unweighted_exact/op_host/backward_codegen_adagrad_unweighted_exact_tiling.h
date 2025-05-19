/**
 * @file backward_codegen_adagrad_unweighted_exact_tiling.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef BACKWARD_CODEGEN_ADAGRAD_UNWEIGHTED_EXACT_TILING
#define BACKWARD_CODEGEN_ADAGRAD_UNWEIGHTED_EXACT_TILING
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BackwardCodegenAdagradUnweightedExactTilingData)
TILING_DATA_FIELD_DEF(int64_t, gradOutputDim0);
TILING_DATA_FIELD_DEF(int64_t, gradOutputDim1);
TILING_DATA_FIELD_DEF(int64_t, devWeightsDim0);
TILING_DATA_FIELD_DEF(int64_t, weightsOffsetsDim0);
TILING_DATA_FIELD_DEF(int64_t, dOffsetsDim0);
TILING_DATA_FIELD_DEF(int64_t, indicesDim0);
TILING_DATA_FIELD_DEF(int64_t, offsetsDim0);
TILING_DATA_FIELD_DEF(int64_t, outDim0);
TILING_DATA_FIELD_DEF(int64_t, bytesOfDataType);
TILING_DATA_FIELD_DEF(int64_t, offsetDataType);
TILING_DATA_FIELD_DEF(int64_t, splitBaseLen);
TILING_DATA_FIELD_DEF(int64_t, tailSplitIndex);
TILING_DATA_FIELD_DEF(int64_t, ubCanUsed);
TILING_DATA_FIELD_DEF(int64_t, poolMode);
TILING_DATA_FIELD_DEF(int64_t, maxD);
TILING_DATA_FIELD_DEF(int64_t, uniqueIdDim0);
TILING_DATA_FIELD_DEF(int64_t, uniqueHashDim0);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, learningRate);
TILING_DATA_FIELD_DEF(bool, enableHash);
TILING_DATA_FIELD_DEF(float, beta1);
TILING_DATA_FIELD_DEF(float, beta2);
TILING_DATA_FIELD_DEF(float, beta1pow);
TILING_DATA_FIELD_DEF(float, beta2pow);
TILING_DATA_FIELD_DEF(int64_t, iter);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BackwardCodegenAdagradUnweightedExact, BackwardCodegenAdagradUnweightedExactTilingData)
}  // namespace optiling
#endif