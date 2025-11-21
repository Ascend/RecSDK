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

#ifndef DENSE_EMBEDDING_CODEGEN_LOOKUP_FUNCTION_GRAD_TILING_H
#define DENSE_EMBEDDING_CODEGEN_LOOKUP_FUNCTION_GRAD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DenseEmbeddingCodegenLookupFunctionGradTilingData)
TILING_DATA_FIELD_DEF(int32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(int32_t, formerCoreLength);
TILING_DATA_FIELD_DEF(int32_t, formerTileNum);
TILING_DATA_FIELD_DEF(int32_t, formerTileLength);
TILING_DATA_FIELD_DEF(int32_t, formerLastTileLength);
TILING_DATA_FIELD_DEF(int32_t, tailCoreNum);
TILING_DATA_FIELD_DEF(int32_t, tailCoreLength);
TILING_DATA_FIELD_DEF(int32_t, tailTileNum);
TILING_DATA_FIELD_DEF(int32_t, tailTileLength);
TILING_DATA_FIELD_DEF(int32_t, tailLastTileLength);
TILING_DATA_FIELD_DEF(int32_t, weightsOffsetsLength);
TILING_DATA_FIELD_DEF(int32_t, batchSize);
TILING_DATA_FIELD_DEF(int32_t, embedDimLength);
TILING_DATA_FIELD_DEF(int32_t, indicesAllLength);
TILING_DATA_FIELD_DEF(int32_t, devWeightsLength);
TILING_DATA_FIELD_DEF(int32_t, alignedEmbedDimLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DenseEmbeddingCodegenLookupFunctionGrad, DenseEmbeddingCodegenLookupFunctionGradTilingData)
} // namespace optiling

#endif // DENSE_EMBEDDING_CODEGEN_LOOKUP_FUNCTION_GRAD_TILING_H
