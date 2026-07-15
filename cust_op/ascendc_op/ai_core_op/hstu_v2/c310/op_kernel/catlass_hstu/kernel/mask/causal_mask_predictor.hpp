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
 * @file causal_mask_predictor.hpp
 * @brief CausalMaskPredictor — causal / context / target mask 场景 predictor
 *
 *   四个部分全部内聚在 struct 内:
 *     1. BlockPredParams         — per-block 参数 (嵌套 struct, 纯数据)
 *     2. MakeBlockPredParams()   — static 方法, 从 coord + kernel 构造参数
 *     3. Classifier() / IsSkip() — 分类 & 跳过判断
 *     4. ApplyMask()             — 写入 mask buffer
 */

#pragma once

namespace Catlass::Kernel::Mask {

template <uint32_t BLOCK_M, uint32_t BLOCK_N>
struct CausalMaskPredictor {
    static constexpr uint8_t HAS_CONTEXT = 0x1;
    static constexpr uint8_t HAS_HISTORY = 0x2;
    static constexpr uint8_t HAS_TARGET = 0x4;
    static constexpr uint32_t DATA_ALIGN_BYTES = 32;

    // =========================================================================
    // 1. BlockPredParams — per-block 预测参数 (纯数据)
    // =========================================================================
    struct BlockPredParams {
        // 块几何信息 — 当前 block 在矩阵中的位置
        struct Geo {
            uint32_t qSeqId;
            uint32_t kSeqId;
            uint32_t seqlenQ;
            uint32_t seqlenK;
            uint32_t swizzleDir;
        } geo;

        // Mask 配置 — 分类结果 + runtime 参数
        struct Mask {
            uint32_t numContext;
            uint32_t numTarget;
            uint32_t targetGroupSize;
            uint8_t fusedMaskType;  // HAS_CONTEXT | HAS_HISTORY | HAS_TARGET
        } mask;
    };

    // =========================================================================
    // 2. MakeBlockPredParams — static, 从 coord + kernel 构造参数
    // =========================================================================
    template <typename Kernel, typename Coord>
    CATLASS_DEVICE static BlockPredParams MakeBlockPredParams(Coord blockCoord, Kernel* kernel, uint32_t seqlenQ,
                                                              uint32_t seqlenK, uint32_t swizzleDir)
    {
        BlockPredParams bp;

        uint32_t b = tla::get<0>(blockCoord);
        uint32_t sq = tla::get<2>(blockCoord);
        uint32_t sk = tla::get<3>(blockCoord);

        bp.geo.qSeqId = sq;
        bp.geo.kSeqId = sk;
        bp.geo.seqlenQ = seqlenQ;
        bp.geo.seqlenK = seqlenK;
        if constexpr (Kernel::IS_CONTEXT_V) {
            bp.mask.numContext = kernel->gNumContext.GetValue(b);
        } else {
            bp.mask.numContext = 0;
        }
        if constexpr (Kernel::IS_TARGET_V) {
            bp.mask.numTarget = kernel->gNumTarget.GetValue(b);
        } else {
            bp.mask.numTarget = 0;
        }
        bp.mask.targetGroupSize = kernel->targetGroupSize;
        bp.mask.fusedMaskType = 0;
        bp.geo.swizzleDir = swizzleDir;
        return bp;
    }

    // =========================================================================
    // MaskRegion — 块级几何判定
    // =========================================================================
    struct MaskRegion {
        const BlockPredParams& p;

        // 构造函数中一次性提取，后续方法直接用，避免重复 p.geo.xxx / p.mask.xxx
        const uint32_t& qSeqId;
        const uint32_t& kSeqId;
        const uint32_t& seqlenK;
        const uint32_t& seqlenQ;
        const uint32_t& numContext;
        const uint32_t& numTarget;
        const uint32_t& targetGroupSize;
        const int32_t deltaQK;

        CATLASS_DEVICE MaskRegion(const BlockPredParams& params)
            : p(params),
              qSeqId(params.geo.qSeqId),
              kSeqId(params.geo.kSeqId),
              seqlenK(params.geo.seqlenK),
              seqlenQ(params.geo.seqlenQ),
              numContext(params.mask.numContext),
              numTarget(params.mask.numTarget),
              targetGroupSize(params.mask.targetGroupSize),
              deltaQK(seqlenK - seqlenQ)
        {
        }

        CATLASS_DEVICE bool NeedContextMask() const
        {
            if (numContext <= 0)
                return false;
            const uint32_t numBlkQ = CeilDiv(numContext, BLOCK_M);
            const uint32_t numBlkK = CeilDiv(seqlenK - numTarget, BLOCK_N);
            return (qSeqId < numBlkQ) && (kSeqId < numBlkK);
        }

        CATLASS_DEVICE void NeedHistoryMask(bool& belowDig, bool& diagonal) const
        {
            const int qBase = qSeqId * BLOCK_M + deltaQK;
            const int rcol = qBase + BLOCK_M - 1;
            const int rblk = rcol / BLOCK_N;
            belowDig = (kSeqId <= rblk);

            if (!belowDig) {
                diagonal = false;
                return;
            }
            const int lblk = qBase / BLOCK_N;
            diagonal = (kSeqId >= lblk);
        }

        CATLASS_DEVICE bool NeedTargetMask(bool trilMask) const
        {
            if (numTarget <= 0 || targetGroupSize <= 0)
                return false;
            auto tbase = (seqlenK - numTarget) / BLOCK_N;
            return (tbase <= kSeqId) && trilMask;
        }
    };

    // =========================================================================
    // 3a. Classifier — 判定当前 block 的 mask 类型, 返回 needMask
    // =========================================================================
    CATLASS_DEVICE void Classifier(BlockPredParams& bp)
    {
        MaskRegion region(bp);
        this->contextMask = region.NeedContextMask();

        bool belowDig, diagonal;
        region.NeedHistoryMask(belowDig, diagonal);
        this->trilMask = belowDig;
        this->historyMask = diagonal;

        this->targetMask = region.NeedTargetMask(this->trilMask);
        this->needMask = this->trilMask ? (this->historyMask || this->targetMask) : this->contextMask;
        if (this->contextMask) {
            bp.mask.fusedMaskType |= HAS_CONTEXT;
        }
        if (this->historyMask) {
            bp.mask.fusedMaskType |= HAS_HISTORY;
        }
        if (this->targetMask) {
            bp.mask.fusedMaskType |= HAS_TARGET;
        }
        this->params = bp;
    }

    // =========================================================================
    // 3b. IsSkip — 上三角且无 context 的块跳过
    // =========================================================================
    CATLASS_DEVICE bool IsSkip() const
    {
        return !this->trilMask && !this->contextMask;
    }

    // =========================================================================
    // 4. ApplyMask — 写入 mask buffer
    // =========================================================================
    template <typename Elem, class Coord, class Shape>
    CATLASS_DEVICE void ApplyMask(AscendC::LocalTensor<Elem>& mask, Coord const& coord, Shape const& shape,
                                  uint32_t alignRows, uint32_t alignCols) const
    {
        if (!this->needMask) {
            return;
        }
        int32_t rowCoord = static_cast<int32_t>(tla::get<2>(coord));
        int32_t colCoord = static_cast<int32_t>(tla::get<3>(coord));
        auto mSize = tla::get<0>(shape);
        auto nSize = tla::get<1>(shape);
        auto count = alignRows * alignCols;
        AscendC::NumericLimits<Elem>::NegativeInfinity(mask, count);

        if (this->params.mask.fusedMaskType & HAS_HISTORY) {
            ApplyHistoryMask(mask, rowCoord, colCoord, mSize, nSize, alignRows, alignCols);
        }
        if (this->params.mask.fusedMaskType & HAS_CONTEXT) {
            ApplyContextMask(mask, rowCoord, colCoord, mSize, nSize, alignRows, alignCols);
        }
        if (this->params.mask.fusedMaskType & HAS_TARGET) {
            if (!this->historyMask) {
                AscendC::Duplicate<Elem>(mask, 0, count);
            }
            ApplyTargetMask(mask, rowCoord, colCoord, mSize, nSize, alignRows, alignCols);
        }
    }

    CATLASS_DEVICE bool IsInnerLoopFirstQBlock(BlockPredParams& bp)
    {
        this->isFirstQBlock = IsFirst(bp);
        this->isLastQBlock = IsLast(bp);
        return (bp.geo.swizzleDir == 1) ? this->isFirstQBlock : this->isLastQBlock;
    }

    CATLASS_DEVICE bool IsInnerLoopLastQBlock(BlockPredParams& bp)
    {
        return (bp.geo.swizzleDir == 1) ? this->isLastQBlock : this->isFirstQBlock;
    }

private:
    // =========================================================================
    // 4a. ApplyContextMask — context Q rows × history K cols → 补 0
    // =========================================================================
    template <typename Elem>
    CATLASS_DEVICE void ApplyContextMask(AscendC::LocalTensor<Elem>& mask, int32_t rowCoord, int32_t colCoord,
                                         int32_t mSize, int32_t nSize, uint32_t alignRows, uint32_t alignCols) const
    {
        auto numCtx = this->params.mask.numContext;
        auto histKEnd = this->params.geo.seqlenK - this->params.mask.numTarget;
        int validWidth = histKEnd - colCoord;
        if (validWidth > nSize) {
            validWidth = nSize;
        }
        int validHeight = mSize;
        if (rowCoord + mSize > numCtx) {
            validHeight = numCtx - rowCoord;
        }
        for (int i = 0; i < validHeight; i++) {
            AscendC::Duplicate<Elem>(mask[i * alignCols], 0, validWidth);
        }
    }

    // =========================================================================
    // 4b. ApplyHistoryMask — 下三角可视区域补 0 (对角线以上为 -inf)
    // =========================================================================
    template <typename Elem>
    CATLASS_DEVICE void ApplyHistoryMask(AscendC::LocalTensor<Elem>& mask, int32_t rowCoord, int32_t colCoord,
                                         int32_t mSize, int32_t nSize, uint32_t alignRows, uint32_t alignCols) const
    {
        int deltaQK = this->params.geo.seqlenK - this->params.geo.seqlenQ;
        int startMaskWidth = deltaQK + rowCoord - colCoord;

        int startMaskRow = 0;
        if (startMaskWidth < 0) {
            startMaskRow = -startMaskWidth;
            if (startMaskRow >= mSize) {
                return;
            }
            startMaskWidth = 0;
        }
        int thisIndexMask = startMaskWidth + 1;
        int endMaskRow = mSize;
        for (int i = startMaskRow; i < endMaskRow; i++) {
            thisIndexMask = thisIndexMask > nSize ? nSize : thisIndexMask;
            AscendC::Duplicate<Elem>(mask[i * alignCols], 0, thisIndexMask);
            thisIndexMask++;
        }
    }

    // =========================================================================
    // 4c. ApplyTargetMask — target region mask
    // =========================================================================
    template <typename Elem>
    CATLASS_DEVICE void ApplyTargetMask(AscendC::LocalTensor<Elem>& mask, int32_t rowCoord, int32_t colCoord,
                                        int32_t mSize, int32_t nSize, uint32_t alignRows, uint32_t alignCols) const
    {
        int tbaseQ = this->params.geo.seqlenQ - this->params.mask.numTarget;
        int tbaseK = this->params.geo.seqlenK - this->params.mask.numTarget;
        int64_t tbaseQInBlk = tbaseQ - rowCoord;
        int64_t tbaseKInBlk = tbaseK - colCoord;
        for (int i = 0; i < mSize; i++) {
            int64_t triNum = (i - tbaseQInBlk) / this->params.mask.targetGroupSize;
            int64_t triRight = tbaseKInBlk + triNum * this->params.mask.targetGroupSize;
            if (triNum < 1 || triRight <= 0) {
                continue;
            }
            int64_t validRbound = (triRight >= nSize) ? nSize : triRight;
            int64_t validLbound = (tbaseKInBlk >= 0) ? tbaseKInBlk : 0;
            int alignStart = validLbound * sizeof(Elem) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES / sizeof(Elem);
            int64_t validWidth = validRbound - alignStart;
            AscendC::NumericLimits<Elem>::NegativeInfinity(mask[i * alignCols + alignStart], validWidth);
            if (alignStart != validLbound) {
                int unalignlen = validLbound - alignStart;
                AscendC::Duplicate<Elem>(mask[i * alignCols + alignStart], 0, unalignlen);
            }
        }
    }

    CATLASS_DEVICE bool IsFirst(BlockPredParams& bp) const
    {
        const uint32_t numBlockForContextMaskK = CeilDiv(bp.geo.seqlenK - bp.mask.numTarget, BLOCK_N);
        bool kIdNotInContext = (bp.mask.numContext <= 0) || (bp.geo.kSeqId >= numBlockForContextMaskK);
        const uint32_t deltaBlock = CeilDiv(bp.geo.seqlenK - bp.geo.seqlenQ, BLOCK_N);
        bool isDiagonal =
            (bp.geo.kSeqId == bp.geo.qSeqId * 2 + deltaBlock) || (bp.geo.kSeqId == bp.geo.qSeqId * 2 + deltaBlock + 1);
        return bp.geo.qSeqId == 0 || (isDiagonal && kIdNotInContext);
    }

    CATLASS_DEVICE bool IsLast(BlockPredParams& bp) const
    {
        return bp.geo.qSeqId == CeilDiv(bp.geo.seqlenQ, BLOCK_M) - 1;
    }

public:
    BlockPredParams params{};

    bool needMask = false;
    bool contextMask = false;
    bool historyMask = false;
    bool targetMask = false;
    bool trilMask = false;
    bool isFirstQBlock = false;
    bool isLastQBlock = false;
};

}  // namespace Catlass::Kernel::Mask
