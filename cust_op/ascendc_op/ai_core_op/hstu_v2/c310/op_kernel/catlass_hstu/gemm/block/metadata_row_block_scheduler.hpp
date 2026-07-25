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

/**
 * @file metadata_row_block_scheduler.hpp
 *
 * @brief HSTU 行方向「metadata 驱动」块调度器(独立类,与 RowBlockScheduler 接口同构)。
 *
 * @description 不改动 RowBlockScheduler。本类消费由 flash_attn_metadata(HSTU 模式)产出的
 *              metadata(HEAD + FA + FD 布局),按 FA 记录把每核负责的行块区间(可跨多个 section)
 *              展平后依次遍历。对外表现为一条扁平的行块序列 —— 主循环 `while(IsValid()){++}` 与
 *              ColumnBlockScheduler(模板化于本类型)均无需改动。
 *
 *              分核语义、正确性已由 metadata_sched/mock_validation 的纯 CPU 用例验证:
 *                - 当 metadata 编码设备分核时,本类逐核逐块与 RowBlockScheduler 完全一致;
 *                - 任意合法 metadata(含多 section)全体核并起来恰好覆盖整张行块网格。
 */

#pragma once

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Block {

// =============================================================================
// flash_attn_metadata 布局常量(source of truth:
//   flash_attn_metadata/op_kernel_aicpu/flash_attn_metadata.h 的 optiling 命名空间)
// 这里在设备侧本地镜像一份,避免把 host(AICPU)头引入 kernel 编译单元。若上游布局变更,需同步。
// =============================================================================
namespace flash_meta_layout {
constexpr uint32_t AIC_CORE_NUM = 36U;
constexpr uint32_t HEAD_METADATA_STRIDE = 16U;
constexpr uint32_t FA_METADATA_STRIDE = 16U;

constexpr uint32_t HEAD_SECTION_NUM_INDEX = 0U;
constexpr uint32_t HEAD_IS_FD_INDEX = 1U;
constexpr uint32_t HEAD_M_BASE_SIZE_INDEX = 2U;
constexpr uint32_t HEAD_S2_BASE_SIZE_INDEX = 3U;

constexpr uint32_t FA_BN2_START_INDEX = 0U;
constexpr uint32_t FA_M_START_INDEX = 1U;
constexpr uint32_t FA_BN2_END_INDEX = 3U;
constexpr uint32_t FA_M_END_INDEX = 4U;
}  // namespace flash_meta_layout

/**
 * @brief 行方向 metadata 驱动块调度器
 *
 * @tparam ElementOffset_ 序列偏移量数据类型
 * @tparam BLOCK_M 行方向块大小(HSTU 反向即行轴 seqK 的块 Rk;由 BlockSchedulerBuilder 传入 = get<1>(L1TileShape))
 * @tparam BLOCK_N 列方向块大小(仅为与 RowBlockScheduler 接口对齐,本类不用)
 *
 * @note 接口与 RowBlockScheduler 完全对齐: Init/IsValid/IsLast/operator++/GetMeta/
 *       GetHeadBlockId/GetCurrentSeqLen/GetTile,可无缝替换到 ColumnBlockScheduler 与主循环。
 */
template <class ElementOffset_, uint32_t BLOCK_M, uint32_t BLOCK_N>
class MetadataRowBlockScheduler {
public:
    using ElementOffset = ElementOffset_;

    /**
     * @param batchSize 批次大小
     * @param headNum 注意力头数(== kvHeadNum, G=1)
     * @param seqOffsetM 行轴序列偏移量(HSTU 反向 = seqOffsetK)
     * @param seqOffsetN 列轴序列偏移量(接口对齐用)
     * @param metadata flash_attn_metadata 输出(int32,HEAD+FA+FD 布局)
     */
    CATLASS_DEVICE
    MetadataRowBlockScheduler(uint32_t batchSize, uint32_t headNum, GM_ADDR seqOffsetM, GM_ADDR seqOffsetN,
                              GM_ADDR metadata)
    {
        this->batchSize = batchSize;
        this->headNum = headNum;
        this->seqOffsetM.SetGlobalBuffer((__gm__ ElementOffset*)seqOffsetM);
        this->seqOffsetN.SetGlobalBuffer((__gm__ ElementOffset*)seqOffsetN);
        this->meta.SetGlobalBuffer((__gm__ int32_t*)metadata);
    }

    /**
     * @brief 初始化: 读 HEAD、定位本核首个非空 section,加载首段行块区间。
     */
    CATLASS_DEVICE
    void Init()
    {
        using namespace flash_meta_layout;
        this->sectionNum = static_cast<uint32_t>(meta.GetValue(HEAD_SECTION_NUM_INDEX));
        // 约定: HSTU 模式必须满足 HEAD[mBaseSize] == BLOCK_M(=Rk),否则展平网格与 kernel 编译期块不一致。
        // 设备侧无 assert,采用失败安全: 不一致则本核不出任务(blockCnt=0),由上层校验/回退。
        uint32_t mBase = static_cast<uint32_t>(meta.GetValue(HEAD_M_BASE_SIZE_INDEX));
        if (mBase != BLOCK_M) {
            this->blockCnt = 0;
            this->curSection = this->sectionNum;  // 直接置为“全部走完”
            return;
        }

        if ASCEND_IS_AIV {
            this->coreId = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        } else {
            this->coreId = AscendC::GetBlockIdx();
        }

        this->curSection = 0;
        AdvanceToNextNonEmptySection(/*fromCurrent=*/true);
    }

    CATLASS_DEVICE
    bool IsValid()
    {
        return this->blockCnt != 0;
    }

    // 最后一块: 当前段仅剩 1 块,且后续 section 本核均为空
    CATLASS_DEVICE
    bool IsLast()
    {
        return this->blockCnt == 1 && !HasMoreAfterCurrent();
    }

    CATLASS_DEVICE uint32_t GetHeadBlockId() const
    {
        return headBlockId;
    }
    CATLASS_DEVICE uint32_t GetCurrentSeqLen() const
    {
        return currentSeqLen;
    }

    /**
     * @brief 前缀递增: 段内推进一块;段内走完则自动跳到本核下一个非空 section。
     */
    CATLASS_DEVICE
    MetadataRowBlockScheduler& operator++()
    {
        this->blockCnt--;
        if (this->blockCnt == 0) {
            AdvanceToNextNonEmptySection(/*fromCurrent=*/false);
            return *this;
        }
        // 段内推进(与 RowBlockScheduler::operator++ 同款 batch→head→m 递进)
        headBlockId++;
        if (headBlockId >= headBlockCnt) {
            headBlockId = 0;
            headId++;
            if (headId >= headNum) {
                headId = 0;
                batchBaseOffset += currentSeqLen;
                batchId++;
                if (batchId < batchSize) {
                    currentSeqLen = seqOffsetM.GetValue(batchId + 1) - seqOffsetM.GetValue(batchId);
                    headBlockCnt = CeilDiv<BLOCK_M>(currentSeqLen);
                }
            }
        }
        Update();
        return *this;
    }

    CATLASS_DEVICE
    auto GetMeta() const
    {
        return tla::MakeCoord(batchId, headId, headBlockId);
    }

    template <class Tensor>
    CATLASS_DEVICE auto GetTile(Tensor const& tensor)
    {
        auto dim = tla::get<1>(tensor.shape());
        auto coord = tla::MakeCoord(this->blockOffset, 0);
        auto shape = tla::MakeShape(this->blockSize, dim);
        return tla::GetTile(tensor, coord, shape);
    }

private:
    // flatten(bn2, m) = Σ_{x<bn2} CeilDiv<BLOCK_M>(seqLen(x/headNum)) + m
    CATLASS_DEVICE
    uint32_t FlattenRowBlock(uint32_t bn2, uint32_t m)
    {
        uint32_t acc = 0;
        for (uint32_t x = 0; x < bn2; ++x) {
            uint32_t b = x / headNum;
            uint32_t seqLen = seqOffsetM.GetValue(b + 1) - seqOffsetM.GetValue(b);
            acc += CeilDiv<BLOCK_M>(seqLen);
        }
        return acc + m;
    }

    // 读 FA[sec][coreId] 的 (bn2Start,mStart,bn2End,mEnd),展平成 [rowBlockStart, rowBlockEnd)。
    // 返回该段本核块数(0 表示无任务)。rowBlockStart 通过出参返回。
    CATLASS_DEVICE
    uint32_t LoadSection(uint32_t sec, uint32_t& rowBlockStart)
    {
        using namespace flash_meta_layout;
        uint32_t base = HEAD_METADATA_STRIDE + (sec * AIC_CORE_NUM + coreId) * FA_METADATA_STRIDE;
        uint32_t bn2S = static_cast<uint32_t>(meta.GetValue(base + FA_BN2_START_INDEX));
        uint32_t mS = static_cast<uint32_t>(meta.GetValue(base + FA_M_START_INDEX));
        uint32_t bn2E = static_cast<uint32_t>(meta.GetValue(base + FA_BN2_END_INDEX));
        uint32_t mE = static_cast<uint32_t>(meta.GetValue(base + FA_M_END_INDEX));
        rowBlockStart = FlattenRowBlock(bn2S, mS);
        uint32_t rowBlockEnd = FlattenRowBlock(bn2E, mE);
        return (rowBlockEnd > rowBlockStart) ? (rowBlockEnd - rowBlockStart) : 0U;
    }

    // 定位下一个(含/不含当前)在本核非空的 section,并 InitBlock 到其起始行块。
    CATLASS_DEVICE
    void AdvanceToNextNonEmptySection(bool fromCurrent)
    {
        uint32_t sec = fromCurrent ? curSection : curSection + 1;
        for (; sec < sectionNum; ++sec) {
            uint32_t rowStart = 0;
            uint32_t cnt = LoadSection(sec, rowStart);
            if (cnt != 0U) {
                curSection = sec;
                this->blockCnt = cnt;
                InitBlock(rowStart);
                Update();
                return;
            }
        }
        curSection = sectionNum;
        this->blockCnt = 0;  // → IsValid()=false
    }

    // 是否还存在“当前段之后、本核非空”的 section(供 IsLast 判断)
    CATLASS_DEVICE
    bool HasMoreAfterCurrent()
    {
        for (uint32_t sec = curSection + 1; sec < sectionNum; ++sec) {
            uint32_t rowStart = 0;
            if (LoadSection(sec, rowStart) != 0U) {
                return true;
            }
        }
        return false;
    }

    // 展平起点 → (batchId, headId, headBlockId, currentSeqLen, headBlockCnt, batchBaseOffset)
    // 与 RowBlockScheduler::InitBlock 完全一致。
    CATLASS_DEVICE
    void InitBlock(uint32_t coreStartBlockId)
    {
        uint32_t blockOffsetAcc = 0;
        for (uint32_t b = 0; b < this->batchSize; ++b) {
            uint32_t seqLens = seqOffsetM.GetValue(b + 1) - seqOffsetM.GetValue(b);
            uint32_t mBlockCnt = CeilDiv<BLOCK_M>(seqLens);
            blockOffsetAcc += mBlockCnt * this->headNum;
            if (coreStartBlockId < blockOffsetAcc) {
                this->batchId = b;
                coreStartBlockId -= (blockOffsetAcc - mBlockCnt * this->headNum);
                this->headId = coreStartBlockId / mBlockCnt;
                this->headBlockId = coreStartBlockId - mBlockCnt * this->headId;
                this->currentSeqLen = seqLens;
                this->headBlockCnt = mBlockCnt;
                this->batchBaseOffset = seqOffsetM.GetValue(b);
                return;
            }
        }
    }

    // 与 RowBlockScheduler::Update 完全一致
    CATLASS_DEVICE
    void Update()
    {
        if (batchId >= batchSize) {
            blockSize = 0;
            blockOffset = 0;
            return;
        }
        auto seqOffset = headBlockId * BLOCK_M;
        blockOffset = (batchBaseOffset + seqOffset) * headNum + headId;
        auto remainingSeq = currentSeqLen - seqOffset;
        blockSize = (remainingSeq < BLOCK_M) ? remainingSeq : BLOCK_M;
    }

    // —— 行块状态(与 RowBlockScheduler 同名)——
    uint32_t blockSize{0};
    uint32_t blockOffset{0};
    uint32_t batchId{0};
    uint32_t batchSize{0};
    uint32_t batchBaseOffset{0};
    uint32_t headId{0};
    uint32_t headBlockId{0};
    uint32_t headNum{0};
    uint32_t headBlockCnt{0};
    uint32_t currentSeqLen{0};
    uint32_t blockCnt{0};
    AscendC::GlobalTensor<ElementOffset> seqOffsetM;
    AscendC::GlobalTensor<ElementOffset> seqOffsetN;

    // —— metadata / section 专属状态 ——
    AscendC::GlobalTensor<int32_t> meta;
    uint32_t sectionNum{0};
    uint32_t curSection{0};
    uint32_t coreId{0};
};

// =============================================================================
// 行调度器工厂: 屏蔽“旧 RowBlockScheduler(4 参构造) vs MetadataRowBlockScheduler(5 参含 metadata)”的差异。
// 使主循环对两种调度器类型用同一行构造代码;旧 RowBlockScheduler 的构造签名保持不变(不改旧类)。
// C++17 保证复制消除 → 返回的 prvalue 直接就地构造到目标对象,不需要拷贝/移动构造(设备侧安全)。
// =============================================================================
template <class Sched>
struct RowSchedulerMaker {
    CATLASS_DEVICE static Sched Make(uint32_t b, uint32_t h, GM_ADDR seqM, GM_ADDR seqN, GM_ADDR /*metadata*/)
    {
        return Sched(b, h, seqM, seqN);
    }
};

template <class E, uint32_t M, uint32_t N>
struct RowSchedulerMaker<MetadataRowBlockScheduler<E, M, N>> {
    using S = MetadataRowBlockScheduler<E, M, N>;
    CATLASS_DEVICE static S Make(uint32_t b, uint32_t h, GM_ADDR seqM, GM_ADDR seqN, GM_ADDR metadata)
    {
        return S(b, h, seqM, seqN, metadata);
    }
};

template <class Sched>
CATLASS_DEVICE Sched MakeRowScheduler(uint32_t b, uint32_t h, GM_ADDR seqM, GM_ADDR seqN, GM_ADDR metadata)
{
    return RowSchedulerMaker<Sched>::Make(b, h, seqM, seqN, metadata);
}

}  // namespace Catlass::Gemm::Block
