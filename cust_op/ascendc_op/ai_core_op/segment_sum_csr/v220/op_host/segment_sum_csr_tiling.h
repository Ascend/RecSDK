/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#ifndef MXREC_SEGMENT_SUM_CSR_TILING_H
#define MXREC_SEGMENT_SUM_CSR_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SegmentSumCsrTilingData)
TILING_DATA_FIELD_DEF(size_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, segmentNums);
TILING_DATA_FIELD_DEF(int64_t, totalLength);
TILING_DATA_FIELD_DEF(int64_t, csrSegLength);
TILING_DATA_FIELD_DEF(int64_t, baseCoreSegments);
TILING_DATA_FIELD_DEF(int64_t, remainedSegments);
TILING_DATA_FIELD_DEF(int64_t, formerCoreSegments);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SegmentSumCsr, SegmentSumCsrTilingData)
}  // namespace optiling

#endif  // MXREC_SEGMENT_SUM_CSR_TILING_H
