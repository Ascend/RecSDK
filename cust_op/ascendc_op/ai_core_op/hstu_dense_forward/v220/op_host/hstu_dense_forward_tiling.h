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


#ifndef HSTU_DENSE_FORWARD_TILING_H
#define HSTU_DENSE_FORWARD_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_policy_define.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HstuDenseForwardTilingData)
TILING_DATA_FIELD_DEF(uint32_t, size);

TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, seqLen);
TILING_DATA_FIELD_DEF(int64_t, headNum);
TILING_DATA_FIELD_DEF(int64_t, headNumK);
TILING_DATA_FIELD_DEF(int64_t, headRatio);
TILING_DATA_FIELD_DEF(int64_t, dim);
TILING_DATA_FIELD_DEF(int64_t, blockHeight);

#ifdef SUPPORT_V200
    TILING_DATA_FIELD_DEF(int32_t, tmpUbSize);
#else
    TILING_DATA_FIELD_DEF(int64_t, pageSize);
#endif

TILING_DATA_FIELD_DEF(uint32_t, enableBias);
TILING_DATA_FIELD_DEF(uint32_t, maskType);
TILING_DATA_FIELD_DEF(int64_t, maxSeqLen);
TILING_DATA_FIELD_DEF(float, siluScale);
TILING_DATA_FIELD_DEF(int64_t, maxSeqLenq);
TILING_DATA_FIELD_DEF(int64_t, maxSeqLenk);
TILING_DATA_FIELD_DEF(bool, enableNumContext);
TILING_DATA_FIELD_DEF(bool, enableNumTarget);
TILING_DATA_FIELD_DEF(int64_t, targetGroupSize);
TILING_DATA_FIELD_DEF(float, alpha);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HstuDenseForward, HstuDenseForwardTilingData)
}  // namespace optiling
#endif