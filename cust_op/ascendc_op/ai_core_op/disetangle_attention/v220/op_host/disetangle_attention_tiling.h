/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
*/

#ifndef DISETANGLE_ATTENTION_TILING_H
#define DISETANGLE_ATTENTION_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DisetangleAttentionTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, headNum);
TILING_DATA_FIELD_DEF(uint32_t, seqLen);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, useCoreNum);  // 实际使用的aiv corenum
TILING_DATA_FIELD_DEF(uint32_t, aivCoreNum);  // 最大可使用的aiv corenum
TILING_DATA_FIELD_DEF(uint32_t, splitNextCoreAccSN);
TILING_DATA_FIELD_DEF(uint32_t, splitPrevCoreAccSN);
TILING_DATA_FIELD_DEF(uint32_t, splitCoreIdx);
TILING_DATA_FIELD_DEF(uint32_t, accS);  // 基本加速块的大小
TILING_DATA_FIELD_DEF(float, scoreScale);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, SFT);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, QK_MM);  // 矩阵乘 [acc_s, d] * [acc_s ,d](transpose)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, MM);     // 矩阵乘 [acc_s, d] * [d, 2 * acc_s]
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, SV_MM);  // 矩阵乘 [acc_s, acc_s] * [acc_s, d]
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DisetangleAttention, DisetangleAttentionTilingData)
}  // namespace optiling

#endif