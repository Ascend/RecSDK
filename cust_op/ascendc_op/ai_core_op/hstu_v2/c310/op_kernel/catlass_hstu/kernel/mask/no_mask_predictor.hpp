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
============================================================================== */

/**
 * @file no_mask_predictor.hpp
 * @brief NoMaskPredictor — 无 mask 场景 predictor 框架
 *
 *   四个部分全部内聚在 struct 内:
 *     1. BlockPredParams         — per-block 参数 (嵌套 struct, 纯数据)
 *     2. MakeBlockPredParams()   — static 方法, 从 coord + kernel 构造参数
 *     3. classifier() / isSkip() — 分类 & 跳过判断
 *     4. ApplyMask()             — 写入 RAB mask (本题: no-op)
 */

#pragma once

namespace Catlass::Kernel::Mask {

template <uint32_t BLOCK_M, uint32_t BLOCK_N>
struct NoMaskPredictor {
    // =========================================================================
    // 1. BlockPredParams — per-block 预测参数 (纯数据)
    // =========================================================================
    struct BlockPredParams {
        uint32_t qSeqId;
        uint32_t blockHeight;
        uint32_t seqlenQ;
        uint32_t swizzleDir;
    };

    // =========================================================================
    // 1.5. Construct — mainloop 侧 predictor 初始化 (nomask 无需初始化)
    //   与 Arbitrary/Causal predictor 同构签名, mainloop 可统一调用
    // =========================================================================
    template <typename Kernel>
    CATLASS_DEVICE void Construct(Kernel* kernel, const typename Kernel::Params& params)
    {
    }

    // =========================================================================
    // 2. MakeBlockPredParams — static, 从 coord + kernel 构造参数
    // =========================================================================
    template <typename Kernel, typename Coord>
    CATLASS_DEVICE static BlockPredParams MakeBlockPredParams(Coord blockCoord, Kernel* kernel, uint32_t seqlenQ,
                                                              uint32_t seqlenK, uint32_t swizzleDir)
    {
        BlockPredParams bp;
        uint32_t sq = tla::get<2>(blockCoord);

        bp.qSeqId = sq;
        bp.blockHeight = BLOCK_M;
        bp.seqlenQ = seqlenQ;
        bp.swizzleDir = swizzleDir;
        return bp;
    }

    // =========================================================================
    // 3a. classifier — 判定当前 block 是否需要 mask
    // =========================================================================
    CATLASS_DEVICE void Classifier(BlockPredParams& bp) {}

    // =========================================================================
    // 3b. isSkip — nomask始终不跳过
    // =========================================================================
    CATLASS_DEVICE bool IsSkip()
    {
        return false;
    }

    // =========================================================================
    // 4. ApplyMask — 写入 RAB mask
    // =========================================================================
    template <typename Elem, class Coord, class Shape>
    CATLASS_DEVICE void ApplyMask(AscendC::LocalTensor<Elem>& mask, Coord const& coord, Shape const& shape,
                                  uint32_t alignRows, uint32_t alignCols) const
    {
    }

    CATLASS_DEVICE bool IsInnerLoopFirstQBlock(BlockPredParams& bp)
    {
        return (bp.swizzleDir == 1) ? bp.qSeqId == 0 : bp.qSeqId == (CeilDiv(bp.seqlenQ, bp.blockHeight) - 1);
    }

    CATLASS_DEVICE bool IsInnerLoopLastQBlock(BlockPredParams& bp)
    {
        return (bp.swizzleDir == 1) ? bp.qSeqId == (CeilDiv(bp.seqlenQ, bp.blockHeight) - 1) : bp.qSeqId == 0;
    }

    bool needMask = false;
};
}  // namespace Catlass::Kernel::Mask
