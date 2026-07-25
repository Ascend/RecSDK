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
        uint32_t qSeqId;
        uint32_t kSeqId;
        uint32_t seqlenQ;
        uint32_t seqlenK;
        uint32_t swizzleDir;

        // Mask 配置 — 分类结果 + runtime 参数
        uint32_t numContext;
        uint32_t numTarget;
        uint32_t targetGroupSize;
        uint8_t fusedMaskType;  // HAS_CONTEXT | HAS_HISTORY | HAS_TARGET
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

        bp.qSeqId = sq;
        bp.kSeqId = sk;
        bp.seqlenQ = seqlenQ;
        bp.seqlenK = seqlenK;
        if constexpr (Kernel::IS_CONTEXT_V) {
            bp.numContext = kernel->gNumContext.GetValue(b);
        } else {
            bp.numContext = 0;
        }
        if constexpr (Kernel::IS_TARGET_V) {
            bp.numTarget = kernel->gNumTarget.GetValue(b);
        } else {
            bp.numTarget = 0;
        }
        bp.targetGroupSize = kernel->targetGroupSize;
        bp.fusedMaskType = 0;
        bp.swizzleDir = swizzleDir;
        return bp;
    }

    // =========================================================================
    // MaskRegion — 块级几何判定
    // =========================================================================
    struct MaskRegion {
        const BlockPredParams& p;

        CATLASS_DEVICE MaskRegion(const BlockPredParams& params) : p(params) {}

        CATLASS_DEVICE static bool NeedContextMask(const BlockPredParams& p)
        {
            if (p.numContext <= 0)
                return false;
            const uint32_t numBlkQ = CeilDiv(p.numContext, BLOCK_M);
            const uint32_t numBlkK = CeilDiv(p.seqlenK - p.numTarget, BLOCK_N);
            return (p.qSeqId < numBlkQ) && (p.kSeqId < numBlkK);
        }

        CATLASS_DEVICE static void NeedHistoryMask(const BlockPredParams& p, bool& belowDig, bool& diagonal)
        {
            const int deltaQK = p.seqlenK - p.seqlenQ;
            const int qBase = p.qSeqId * BLOCK_M + deltaQK;
            const int rcol = qBase + BLOCK_M - 1;
            const int rblk = rcol / BLOCK_N;
            belowDig = (p.kSeqId <= rblk);

            if (!belowDig) {
                diagonal = false;
                return;
            }
            const int lblk = qBase / BLOCK_N;
            diagonal = (p.kSeqId >= lblk);
        }

        CATLASS_DEVICE static bool NeedTargetMask(const BlockPredParams& p, bool trilMask)
        {
            if (p.numTarget <= 0 || p.targetGroupSize <= 0)
                return false;
            auto tbase = (p.seqlenK - p.numTarget) / BLOCK_N;
            return (tbase <= p.kSeqId) && trilMask;
        }
    };

    // =========================================================================
    // 3a. Classifier — 判定当前 block 的 mask 类型, 返回 needMask
    // =========================================================================
    CATLASS_DEVICE void Classifier(BlockPredParams& bp)
    {
        MaskRegion region(bp);
        const bool contextMask = MaskRegion::NeedContextMask(bp);

        bool belowDig, diagonal;
        MaskRegion::NeedHistoryMask(bp, belowDig, diagonal);
        const bool trilMask = belowDig;
        const bool historyMask = diagonal;

        const bool targetMask = MaskRegion::NeedTargetMask(bp, trilMask);
        const bool needMask = trilMask ? (historyMask || targetMask) : contextMask;
        if (contextMask) {
            bp.fusedMaskType |= HAS_CONTEXT;
        }
        if (historyMask) {
            bp.fusedMaskType |= HAS_HISTORY;
        }
        if (targetMask) {
            bp.fusedMaskType |= HAS_TARGET;
        }

        this->contextMask = contextMask;
        this->trilMask = trilMask;
        this->historyMask = historyMask;
        this->targetMask = targetMask;
        this->needMask = needMask;
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

        if (this->params.fusedMaskType & HAS_HISTORY) {
            ApplyHistoryMask(mask, rowCoord, colCoord, mSize, nSize, alignRows, alignCols);
        }
        if (this->params.fusedMaskType & HAS_CONTEXT) {
            ApplyContextMask(mask, rowCoord, colCoord, mSize, nSize, alignRows, alignCols);
        }
        if (this->params.fusedMaskType & HAS_TARGET) {
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
        return (bp.swizzleDir == 1) ? this->isFirstQBlock : this->isLastQBlock;
    }

    CATLASS_DEVICE bool IsInnerLoopLastQBlock(BlockPredParams& bp)
    {
        return (bp.swizzleDir == 1) ? this->isLastQBlock : this->isFirstQBlock;
    }

private:
    // =========================================================================
    // 4a. ApplyContextMask — context Q rows × history K cols → 补 0
    // =========================================================================
    template <typename Elem>
    CATLASS_DEVICE void ApplyContextMask(AscendC::LocalTensor<Elem>& mask, int32_t rowCoord, int32_t colCoord,
                                         int32_t mSize, int32_t nSize, uint32_t alignRows, uint32_t alignCols) const
    {
        auto numCtx = this->params.numContext;
        auto histKEnd = this->params.seqlenK - this->params.numTarget;
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
        int deltaQK = this->params.seqlenK - this->params.seqlenQ;
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
        int tbaseQ = this->params.seqlenQ - this->params.numTarget;
        int tbaseK = this->params.seqlenK - this->params.numTarget;
        int64_t tbaseQInBlk = tbaseQ - rowCoord;
        int64_t tbaseKInBlk = tbaseK - colCoord;
        for (int i = 0; i < mSize; i++) {
            int64_t triNum = (i - tbaseQInBlk) / this->params.targetGroupSize;
            int64_t triRight = tbaseKInBlk + triNum * this->params.targetGroupSize;
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
        const uint32_t numBlockForContextMaskK = CeilDiv(bp.seqlenK - bp.numTarget, BLOCK_N);
        bool kIdNotInContext = (bp.numContext <= 0) || (bp.kSeqId >= numBlockForContextMaskK);
        const uint32_t deltaBlock = CeilDiv(bp.seqlenK - bp.seqlenQ, BLOCK_N);
        bool isDiagonal = (bp.kSeqId == bp.qSeqId * 2 + deltaBlock) || (bp.kSeqId == bp.qSeqId * 2 + deltaBlock + 1);
        return bp.qSeqId == 0 || (isDiagonal && kIdNotInContext);
    }

    CATLASS_DEVICE bool IsLast(BlockPredParams& bp) const
    {
        return bp.qSeqId == CeilDiv(bp.seqlenQ, BLOCK_M) - 1;
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
