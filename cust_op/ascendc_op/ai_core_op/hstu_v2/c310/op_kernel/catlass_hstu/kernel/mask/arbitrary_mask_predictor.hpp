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
 * @file arbitrary_mask_predictor.hpp
 * @brief ArbitraryMaskPredictor — arbitrary mask 场景 predictor 框架
 *
 *   由 sparse_info (6 个 int32 tensor: mask_cnt/mask_offset/mask_idx/
 *   full_cnt/full_offset/full_idx) 驱动,对每个 (Q_block, K_block) 判定:
 *     - mask  类: 需写 mask → needMask=true,  IsSkip=false
 *     - full  类: 无需 mask → needMask=false, IsSkip=false
 *     - empty 类: 无计算量 → IsSkip=true (不计入 sparse_info, 直接跳过)
 *
 *   四个部分全部内聚在 struct 内 (与 NoMask/Causal predictor 同构):
 *     1. BlockPredParams         — per-block 参数 (嵌套 struct, 纯数据)
 *     2. MakeBlockPredParams()   — 从 coord + kernel 构造参数
 *     3. Classifier() / IsSkip() — 判断block类型 (full / mask) & 跳过判断
 *     4. ApplyMask()             — 写入 mask buffer
 */

#pragma once

namespace Catlass::Kernel::Mask {
constexpr uint32_t INVALID_U32 = 0xFFFFFFFFU;
constexpr size_t SPARSE_INFO_TENSOR_NUM = 6;

enum SparseInfoIndex : size_t {
    MASK_CNT = 0,     // mask 类 block 计数
    MASK_OFFSET = 1,  // mask 类 block 偏移 (mask_cnt 前缀和)
    MASK_IDX = 2,     // mask 类 block 索引
    FULL_CNT = 3,     // full 类 block 计数
    FULL_OFFSET = 4,  // full 类 block 偏移 (full_cnt 前缀和)
    FULL_IDX = 5,     // full 类 block 索引
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, bool IS_FWD = true>
struct ArbitraryMaskPredictor {
    // =========================================================================
    // 1. BlockPredParams — per-block 预测参数 (纯数据)
    // =========================================================================
    // 本行 (blkIdM 对应行) sparse_info 描述, blkIdM 未变化时跨调用复用
    // 稀疏索引区段: [start, start+cnt) 描述 mask_idx/full_idx 中本行对应的连续区间
    struct Span {
        int32_t start{-1};
        int32_t cnt{-1};
    };
    struct BlockSparseParams {
        Span mask;  // mask_idx 区段
        Span full;  // full_idx 区段
        // 本行 sparse info 中的非空首末 blk 编号
        int32_t firstBlk{-1};
        int32_t lastBlk{-1};
    };

    struct BlockPredParams {
        // 块几何信息 — 当前 block 在矩阵中的位置
        int32_t batchId{-1};
        int32_t qSeqId{-1};   // Q_block 编号 (列方向, swizzleDir 决定遍历顺序)
        int32_t kSeqId{-1};   // K_block 编号 (行方向)
        int32_t qSeqLen{-1};  // Q_seq 长度
        int32_t kSeqLen{-1};  // K_seq 长度

        uint32_t swizzleDir{INVALID_U32};

        // 本行 sparse info 描述 (mask/full 段范围 + 非空首末 blk 编号)
        BlockSparseParams sparse{};

        // 当前(qSeqId, kSeqId)是否为mask / full block
        bool isMaskBlock{false};
        bool isFullBlock{false};
    };

    // =========================================================================
    // 1.5. Construct — mainloop 侧 predictor 初始化
    //   从 params 直接初始化 arbitrary_func 与 sparse_info GlobalTensor。
    //   与 NoMask/Causal predictor 同构签名 (template <typename Kernel>),
    //   sparse_info 6 个 tensor 顺序固定: mask_cnt, mask_offset, mask_idx,
    //   full_cnt, full_offset, full_idx (对应 SparseInfoIndex 0..5)。
    // =========================================================================
    template <typename Kernel>
    CATLASS_DEVICE void Construct(Kernel* kernel, const typename Kernel::Params& params)
    {
        GET_TILING_DATA_MEMBER(HstuBackwardV2TilingData, groups, groupsVal, kernel->ptrTiling);
        this->groups = groupsVal;
        gArbitraryFunc.SetGlobalBuffer((__gm__ int32_t*)params.ptrArbitraryFunc);

        AscendC::ListTensorDesc sparseListDesc;
        sparseListDesc.Init(params.ptrSparseInfo);
        GM_ADDR spTensor;
        spTensor = (__gm__ uint8_t*)sparseListDesc.GetDataPtr<__gm__ uint8_t>(SparseInfoIndex::MASK_CNT);
        gMaskCnt.SetGlobalBuffer((__gm__ int32_t*)spTensor);
        spTensor = (__gm__ uint8_t*)sparseListDesc.GetDataPtr<__gm__ uint8_t>(SparseInfoIndex::MASK_OFFSET);
        gMaskOffset.SetGlobalBuffer((__gm__ int32_t*)spTensor);
        spTensor = (__gm__ uint8_t*)sparseListDesc.GetDataPtr<__gm__ uint8_t>(SparseInfoIndex::MASK_IDX);
        gMaskIdx.SetGlobalBuffer((__gm__ int32_t*)spTensor);
        spTensor = (__gm__ uint8_t*)sparseListDesc.GetDataPtr<__gm__ uint8_t>(SparseInfoIndex::FULL_CNT);
        gFullCnt.SetGlobalBuffer((__gm__ int32_t*)spTensor);
        spTensor = (__gm__ uint8_t*)sparseListDesc.GetDataPtr<__gm__ uint8_t>(SparseInfoIndex::FULL_OFFSET);
        gFullOffset.SetGlobalBuffer((__gm__ int32_t*)spTensor);
        spTensor = (__gm__ uint8_t*)sparseListDesc.GetDataPtr<__gm__ uint8_t>(SparseInfoIndex::FULL_IDX);
        gFullIdx.SetGlobalBuffer((__gm__ int32_t*)spTensor);

        maxSeqLenQ = kernel->maxSeqLenQ;
        maxSeqLenK = kernel->maxSeqLenK;
        maxBlkCntQ = CeilDiv<BLOCK_M>(maxSeqLenQ);
        maxBlkCntK = CeilDiv<BLOCK_N>(maxSeqLenK);
    }

    // =========================================================================
    // 2. MakeBlockPredParams — 从 coord + kernel 构造参数
    //    coord = (batchId, headId, qBlockId, kBlockId)
    //    rowBlockId 即 K_block 编号 (kSeqId)
    //    实例方法: 维护 mPtr/fPtr 跨调用复用, blkIdM 变化时按 swizzleDir 重置
    // =========================================================================
    template <typename Kernel, typename Coord>
    CATLASS_DEVICE BlockPredParams MakeBlockPredParams(Coord blockCoord, Kernel* kernel, uint32_t seqlenQ,
                                                       uint32_t seqlenK, uint32_t swizzleDir)
    {
        BlockPredParams bp;

        // 块几何信息 — 当前 block 在矩阵中的位置
        bp.batchId = tla::get<0>(blockCoord);
        bp.qSeqId = tla::get<2>(blockCoord);
        bp.kSeqId = tla::get<3>(blockCoord);
        bp.qSeqLen = seqlenQ;
        bp.kSeqLen = seqlenK;
        bp.swizzleDir = swizzleDir;

        if (IsSparseCached(bp)) {
            bp.sparse = this->params.sparse;
        } else {
            ParseBlkSparseToBP(bp);
        }

        // 推进 mPtr/fPtr 查找 blkIdN, 返回命中类型 (mask 优先)
        auto hit = LookupBlockType(bp);
        if (hit == BlockType::MASK) {
            bp.isMaskBlock = true;
        } else if (hit == BlockType::FULL) {
            bp.isFullBlock = true;
        }
        return bp;
    }

    // =========================================================================
    // 3a. Classifier — 判定当前 (Q_block, K_block) 的稀疏类型
    //   mask : kSeqId ∈ mask_idx[offset[qSeqId], offset[qSeqId+1]) → needMask=true
    //   full : kSeqId ∈ full_idx[offset[qSeqId], offset[qSeqId+1]) → needMask=false
    //   empty: 两者都不在 → IsSkip 返回 true
    // =========================================================================
    CATLASS_DEVICE void Classifier(BlockPredParams& bp)
    {
        this->params = bp;
        this->isMaskBlock = bp.isMaskBlock;
        this->isFullBlock = bp.isFullBlock;
        this->isEmptyBlock = !bp.isMaskBlock && !bp.isFullBlock;
        this->needMask = bp.isMaskBlock;
    }

    // =========================================================================
    // 3b. IsSkip — 遍历mask和full block，无信息的为empty会自动跳过
    // =========================================================================
    CATLASS_DEVICE bool IsSkip() const
    {
        return this->isEmptyBlock;
    }

    // =========================================================================
    // 4b. ApplyMaskSimd — SIMD 版 ApplyMask
    //   与 ApplyMask (SIMT 版) 语义等价, 但用单核向量指令 (Duplicate) 逐行写连续段,
    //   不派发 SIMT 线程块, 适合 mask 段较规整、无需细粒度线程并行的场景。
    //
    //   语义: 对块内每行查 arbitrary_func, 将 [start, end) 与本 block 列范围交集
    //         内的位置补 0 (允许关注), 其余保持 -inf。
    //
    //   注: mask buffer 为 block-local, 形状 [alignRows, alignCols], 故写位置使用
    //       block 内列偏移 (col - sk), 而非绝对列号。
    //       沿用 SIMT 版的 (batchId * maxSeqLenQ + sq) * (groups*2) 索引方式,
    //       headId 与 N 维 TODO 在集成时统一修正。
    // =========================================================================
    template <typename Elem, class Coord, class Shape>
    CATLASS_DEVICE void ApplyMask(AscendC::LocalTensor<Elem>& mask, Coord const& coord, Shape const& shape,
                                  uint32_t alignRows, uint32_t alignCols) const
    {
        if (!this->needMask) {
            return;
        }
        uint32_t batchId = tla::get<0>(coord);
        uint32_t sq = tla::get<2>(coord);
        uint32_t sk = tla::get<3>(coord);
        auto mSize = tla::get<0>(shape);
        auto nSize = tla::get<1>(shape);
        auto count = alignRows * alignCols;
        // 1. 全量初始化为 -inf, 后续对 [start, end) 段补 0
        AscendC::NumericLimits<Elem>::NegativeInfinity(mask, count);

        // TODO: arbitrary_func shape [B, N, MAX_S, 2*groups], 当前索引缺少 headId 与 N 维,
        //       maxSeqLenQ 仅作 batch 内 stride 占位, 集成时需按真实布局修正 (与 SIMT 版一致)
        const uint32_t rowStride = this->groups * 2;
        __gm__ int32_t* afBase =
            (__gm__ int32_t*)this->gArbitraryFunc[(batchId * this->maxSeqLenQ + sq) * rowStride].GetPhyAddr();

        // 2. 逐行处理: 每行 groups 个 [start, end) 段, 与本 block 列范围求交后用 Duplicate 补 0
        //    block 列范围 [blkColL, blkColR) 对应绝对列 [sk, sk + nSize)
        //    Duplicate 要求首地址 32 字节对齐, 参考 CausalMaskPredictor::ApplyTargetMask 的对齐处理:
        //    向下取整到对齐位置 alignOffset, 先填 0 覆盖 [alignOffset, colR),
        //    再将 padding 区 [alignOffset, colL) 恢复为 -inf
        constexpr uint32_t DATA_ALIGN_BYTES = 32;
        const int32_t blkColL = static_cast<int32_t>(sk);
        const int32_t blkColR = static_cast<int32_t>(sk + nSize);
        for (int32_t row = 0; row < mSize; row++) {
            __gm__ int32_t* rowAf = afBase + row * rowStride;
            for (int32_t g = this->groups - 1; g >= 0; g--) {
                int32_t start = rowAf[2 * g];
                int32_t end = rowAf[2 * g + 1];
                if (start >= end) {
                    continue;  // 占位段或空段, 跳过
                }
                // 求 [start, end) 与 [blkColL, blkColR) 的交集
                int32_t colL = (start > blkColL) ? start : blkColL;
                int32_t colR = (end < blkColR) ? end : blkColR;
                if (colL >= colR) {
                    continue;
                }
                uint32_t offset = static_cast<uint32_t>(colL - blkColL);  // block 内列偏移
                uint32_t length = static_cast<uint32_t>(colR - colL);
                // 向下取整到 32 字节对齐位置 (element 单位)
                uint32_t alignOffset = offset * sizeof(Elem) / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES / sizeof(Elem);
                uint32_t paddingLen = offset - alignOffset;
                // 1. 在对齐位置填 0, 长度 = length + padding_len (覆盖 [alignOffset, colR))
                AscendC::Duplicate<Elem>(mask[row * alignCols + alignOffset], 0, length + paddingLen);
                // 2. padding 区 [alignOffset, offset) 恢复为 -inf
                if (paddingLen > 0) {
                    AscendC::NumericLimits<Elem>::NegativeInfinity(mask[row * alignCols + alignOffset], paddingLen);
                }
            }
        }
    }

    // =========================================================================
    // 5. IsInnerLoopFirstQBlock / IsInnerLoopLastQBlock — 判断当前 block 是否为(非空)首尾块
    // =========================================================================
    CATLASS_DEVICE bool IsInnerLoopFirstQBlock(BlockPredParams& bp)
    {
        if constexpr (IS_FWD) {
            return (bp.swizzleDir == 1) ? bp.qSeqId == 0 : bp.qSeqId == (CeilDiv(bp.qSeqLen, BLOCK_M) - 1);
        } else {
            return (bp.swizzleDir == 1) ? bp.qSeqId == bp.sparse.firstBlk : bp.qSeqId == bp.sparse.lastBlk;
        }
    }

    CATLASS_DEVICE bool IsInnerLoopLastQBlock(BlockPredParams& bp)
    {
        if constexpr (IS_FWD) {
            return (bp.swizzleDir == 1) ? bp.qSeqId == (CeilDiv(bp.qSeqLen, BLOCK_M) - 1) : bp.qSeqId == 0;
        } else {
            return (bp.swizzleDir == 1) ? bp.qSeqId == bp.sparse.lastBlk : bp.qSeqId == bp.sparse.firstBlk;
        }
    }

public:
    BlockPredParams params{};

    bool needMask = false;
    bool isMaskBlock = false;
    bool isFullBlock = false;
    bool isEmptyBlock = false;

private:
    uint32_t maxSeqLenQ{0};
    uint32_t maxSeqLenK{0};
    uint32_t maxBlkCntQ{0};
    uint32_t maxBlkCntK{0};

    CATLASS_DEVICE bool IsSparseCached(BlockPredParams& bp)
    {
        if constexpr (IS_FWD) {
            return bp.batchId == this->params.batchId && bp.qSeqId == this->params.qSeqId;
        } else {
            return bp.batchId == this->params.batchId && bp.kSeqId == this->params.kSeqId;
        }
    }

    // 推进 mPtr/fPtr 在 mask_idx / full_idx 中查找 blkIdN, 返回命中类型 (mask 优先):
    //   swizzleDir==1 (正向): val > blkIdN 时提前终止 (后续更大不可能命中)
    //   swizzleDir==0 (逆向): val < blkIdN 时提前终止 (前面更小不可能命中)
    enum class BlockType : uint8_t {
        EMPTY,
        MASK,
        FULL
    };
    CATLASS_DEVICE BlockType LookupBlockType(BlockPredParams& bp)
    {
        // 列方向目标 block 编号: fwd=K, bwd=Q (IS_FWD 为编译期常量, 编译器折叠)
        const uint32_t blkIdN = IS_FWD ? bp.kSeqId : bp.qSeqId;
        const bool ascending = (bp.swizzleDir == 1);
        const int32_t step = ascending ? 1 : -1;
        // mask 优先: 扫描 mask_idx
        while (this->mPtr >= 0 && this->mPtr < bp.sparse.mask.cnt) {
            uint32_t v = this->gMaskIdx.GetValue(bp.sparse.mask.start + this->mPtr);
            if (v == blkIdN) {
                return BlockType::MASK;
            }
            if (ascending ? (v > blkIdN) : (v < blkIdN)) {
                break;  // 单调性破坏, 后续不可能命中
            }
            this->mPtr += step;
        }
        // mask 未命中: 扫描 full_idx
        while (this->fPtr >= 0 && this->fPtr < bp.sparse.full.cnt) {
            uint32_t v = this->gFullIdx.GetValue(bp.sparse.full.start + this->fPtr);
            if (v == blkIdN) {
                return BlockType::FULL;
            }
            if (ascending ? (v > blkIdN) : (v < blkIdN)) {
                break;
            }
            this->fPtr += step;
        }
        return BlockType::EMPTY;
    }

    CATLASS_DEVICE void ParseBlkSparseToBP(BlockPredParams& bp)
    {
        int32_t pos = IS_FWD ? bp.batchId * maxBlkCntQ + bp.qSeqId : bp.batchId * maxBlkCntK + bp.kSeqId;
        int32_t maskStart = this->gMaskOffset.GetValue(pos);
        int32_t maskEnd = this->gMaskOffset.GetValue(pos + 1);
        int32_t maskCnt = maskEnd - maskStart;
        int32_t fullStart = this->gFullOffset.GetValue(pos);
        int32_t fullEnd = this->gFullOffset.GetValue(pos + 1);
        int32_t fullCnt = fullEnd - fullStart;

        bp.sparse.mask.cnt = maskCnt;
        bp.sparse.full.cnt = fullCnt;
        bp.sparse.mask.start = maskStart;
        bp.sparse.full.start = fullStart;

        if (maskCnt > 0 && fullCnt <= 0) {
            bp.sparse.firstBlk = this->gMaskIdx.GetValue(maskStart);
            bp.sparse.lastBlk = this->gMaskIdx.GetValue(maskEnd - 1);
        } else if (fullCnt > 0 && maskCnt <= 0) {
            bp.sparse.firstBlk = this->gFullIdx.GetValue(fullStart);
            bp.sparse.lastBlk = this->gFullIdx.GetValue(fullEnd - 1);
        } else {
            int32_t mskBlk = this->gMaskIdx.GetValue(maskStart);
            int32_t fullBlk = this->gFullIdx.GetValue(fullStart);
            bp.sparse.firstBlk = (mskBlk < fullBlk) ? mskBlk : fullBlk;
            mskBlk = this->gMaskIdx.GetValue(maskEnd - 1);
            fullBlk = this->gFullIdx.GetValue(fullEnd - 1);
            bp.sparse.lastBlk = (mskBlk > fullBlk) ? mskBlk : fullBlk;
        }

        // 初始化 mPtr/fPtr: 用于 LookupBlockType中扫描mask_idx和full_idx
        //   swizzleDir==1 (正向遍历): mPtr/fPtr 初始化为 0, 沿升序推进
        //   swizzleDir==0 (逆向遍历): mPtr/fPtr 初始化为 cnt-1 (cnt==0 时置 -1 跳过空数组)
        const bool ascending = (bp.swizzleDir == 1);
        if (ascending) {
            this->mPtr = 0;
            this->fPtr = 0;
        } else {
            this->mPtr = (maskCnt > 0) ? maskCnt - 1 : -1;
            this->fPtr = (fullCnt > 0) ? fullCnt - 1 : -1;
        }
    }

    // 双指针状态: 跨调用复用 mask_idx / full_idx 的扫描位置
    //   mPtr 指向 mask_idx 中下一个待检查的偏移 (相对 maskStart)
    //   fPtr 指向 full_idx 中下一个待检查的偏移 (相对 fullStart)
    //   blkIdM 变化时按 swizzleDir 重置 (见 ResetPtrs)
    int32_t mPtr{-1};
    int32_t fPtr{-1};
    // arbitrary_func tensor 引用 (由 Construct 从 params 初始化)
    // shape [B, N, MAX_S, 2*groups]: 每行 [start, end) 表示Mask掩码段
    AscendC::GlobalTensor<int32_t> gArbitraryFunc;
    uint32_t groups{0};

    // sparse_info tensor 引用 (由 Construct 从 params 初始化)
    AscendC::GlobalTensor<int32_t> gMaskCnt;
    AscendC::GlobalTensor<int32_t> gMaskOffset;
    AscendC::GlobalTensor<int32_t> gMaskIdx;
    AscendC::GlobalTensor<int32_t> gFullCnt;
    AscendC::GlobalTensor<int32_t> gFullOffset;
    AscendC::GlobalTensor<int32_t> gFullIdx;
};

}  // namespace Catlass::Kernel::Mask
