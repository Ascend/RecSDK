/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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
TILING_DATA_FIELD_DEF(float, beta2sqrt);
TILING_DATA_FIELD_DEF(int64_t, iter);
TILING_DATA_FIELD_DEF(bool, useOptimize); // where to use optimizer update
TILING_DATA_FIELD_DEF(bool, useRegBase);
TILING_DATA_FIELD_DEF(int64_t, momentumDim0);
TILING_DATA_FIELD_DEF(int64_t, totalHashSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BackwardCodegenAdagradUnweightedExact, BackwardCodegenAdagradUnweightedExactTilingData)
}  // namespace optiling
#endif
