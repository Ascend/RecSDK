/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================== */

/*!
 * \file mask/no_mask_predictor.h
 * \brief 无 mask 权重策略（已实现）。
 *
 * 权重 = 该 batch 的 Q 轴块数 qBlkCnt（内层 Q 块全部要算），与行块位置无关：
 *   q=[256,128]、k=[256,256]、BLOCK_M=BLOCK_N=128 时，
 *   序列 0 的每个 K 行块权重 2，序列 1 的每个 K 行块权重 4。
 *
 * 优化：get_head_weight 用闭式 kBlkCnt × qBlkCnt，避免逐行块循环；
 * 与基类默认实现的逐行块裸值累加恒等。
 */

#ifndef HSTU_ATTN_METADATA_BACKWARD_AICPU_MASK_NO_MASK_PREDICTOR_H
#define HSTU_ATTN_METADATA_BACKWARD_AICPU_MASK_NO_MASK_PREDICTOR_H

#include <cstddef>
#include <cstdint>

#include "mask_predictor.h"

namespace aicpu {
namespace backward {

class NoMaskPredictor : public MaskPredictor {
public:
    /*! \brief 无 mask：行块权重 = 该 batch 的 Q 轴块数，与 kBlkId 无关。 */
    uint32_t get_blk_weight(const RowWeightCtx& c) const override
    {
        return (c.batchId < c.in.qBlkCnt.size()) ? c.in.qBlkCnt[c.batchId] : 0U;
    }

    /*! \brief 闭式：kBlkCnt × qBlkCnt，与基类逐行块裸值累加恒等。 */
    uint64_t get_head_weight(const BackwardInput& in, size_t batchId) const override
    {
        const uint32_t blkCnt = (batchId < in.kBlkCnt.size()) ? in.kBlkCnt[batchId] : 0U;
        const uint32_t qBlkCnt = (batchId < in.qBlkCnt.size()) ? in.qBlkCnt[batchId] : 0U;
        return static_cast<uint64_t>(blkCnt) * qBlkCnt;
    }
};

}  // namespace backward
}  // namespace aicpu

#endif  // HSTU_ATTN_METADATA_BACKWARD_AICPU_MASK_NO_MASK_PREDICTOR_H
