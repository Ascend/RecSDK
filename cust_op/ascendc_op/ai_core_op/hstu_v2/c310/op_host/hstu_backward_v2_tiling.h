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

/**
 * @file hstu_backward_v2_tiling.h
 * @brief HSTU Backward V2 算子的 Tiling 数据结构定义
 * @description 定义算子 tiling 所需的各种参数，包括批次、头数、维度、序列长度等
 */

#ifndef HSTU_BACKWARD_V2_TILING_H
#define HSTU_BACKWARD_V2_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

/**
 * @brief HSTU Backward V2 算子的 Tiling 数据结构
 * @description 用于在算子执行时传递 tiling 参数，包括：
 * - batch: 批次大小
 * - heads: 注意力头数
 * - dimQK: Query 和 Key 的特征维度
 * - dimGV: Value 和 Gradient 的特征维度
 * - totalSeqLenQ: Query 的总序列长度
 * - totalSeqLenK: Key/Value 的总序列长度
 * - maxSeqLenQ: Query 的最大序列长度
 * - maxSeqLenK: Key/Value 的最大序列长度
 * - targetGroupSize: 目标分组大小
 * - alpha: 注意力分数的缩放系数
 * - scale: 缩放因子
 */
BEGIN_TILING_DATA_DEF(HstuBackwardV2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);           ///< 批次大小
TILING_DATA_FIELD_DEF(uint32_t, heads);           ///< 注意力头数
TILING_DATA_FIELD_DEF(uint32_t, dimQK);           ///< Query 和 Key 的特征维度
TILING_DATA_FIELD_DEF(uint32_t, dimGV);           ///< Value 和 Gradient 的特征维度
TILING_DATA_FIELD_DEF(uint32_t, totalSeqLenQ);    ///< Query 的总序列长度
TILING_DATA_FIELD_DEF(uint32_t, totalSeqLenK);    ///< Key/Value 的总序列长度
TILING_DATA_FIELD_DEF(uint32_t, maxSeqLenQ);      ///< Query 的最大序列长度
TILING_DATA_FIELD_DEF(uint32_t, maxSeqLenK);      ///< Key/Value 的最大序列长度
TILING_DATA_FIELD_DEF(int32_t, targetGroupSize);  ///< 目标分组大小
TILING_DATA_FIELD_DEF(float, alpha);              ///< 注意力分数的缩放系数
TILING_DATA_FIELD_DEF(float, scale);              ///< 缩放因子
TILING_DATA_FIELD_DEF(int32_t, windowSizeLeft);   ///< 注意力窗口左侧宽度（语义：-1 无限）
TILING_DATA_FIELD_DEF(int32_t, windowSizeRight);  ///< 注意力窗口右侧宽度（语义：-1 无限，0 因果）
END_TILING_DATA_DEF;

/**
 * @brief 注册 HstuBackwardV2 算子的 Tiling 数据类
 * @description 将 HstuBackwardV2TilingData 注册到 tiling 系统，使其可在算子中使用
 */
REGISTER_TILING_DATA_CLASS(HstuBackwardV2, HstuBackwardV2TilingData)

}  // namespace optiling

#endif
