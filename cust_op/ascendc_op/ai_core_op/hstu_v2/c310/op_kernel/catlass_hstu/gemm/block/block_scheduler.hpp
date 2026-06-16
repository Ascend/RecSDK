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
 • @file block_scheduler.hpp

 • @brief HSTU GEMM 块调度器实现

 • @description 提供行方向和列方向的块调度器，用于将计算任务分配到多个 AI Core 上

 •              支持静态负载均衡和持久化分块策略

 */

#pragma once

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Block {

/**
 • @brief 行方向块调度器

 • @tparam ElementOffset_ 序列偏移量的数据类型

 • @tparam BLOCK_M 行方向的块大小

 • @tparam BLOCK_N 列方向的块大小

 • @description 负责在行方向（head 维度）上进行任务分片，支持两种分片策略:

 •              - FastSplitCore: 批次大小为 1 时的快速分片

 •              - PersistentSplitCore: 多批次时的持久化负载均衡分片

 */
template <class ElementOffset_, uint32_t BLOCK_M, uint32_t BLOCK_N>
class RowBlockScheduler {
public:
    using ElementOffset = ElementOffset_;

    /**
     ◦ @brief 构造函数

     ◦ @param batchSize 批次大小

     ◦ @param headNum 注意力头数

     ◦ @param seqOffsetM M 方向的序列偏移量指针

     ◦ @param seqOffsetN N 方向的序列偏移量指针

     ◦ @description 初始化调度器，设置批次大小、头数，并加载序列偏移量

     */
    CATLASS_DEVICE
    RowBlockScheduler(uint32_t batchSize, uint32_t headNum, GM_ADDR seqOffsetM, GM_ADDR seqOffsetN)
    {
        this->batchSize = batchSize;
        this->headNum = headNum;

        this->seqOffsetM.SetGlobalBuffer((__gm__ ElementOffset*)seqOffsetM);
        this->seqOffsetN.SetGlobalBuffer((__gm__ ElementOffset*)seqOffsetN);
    }

    /**
     ◦ @brief 快速分片策略 (批次大小为 1 时使用)

     ◦ @param coreId 当前核心 ID

     ◦ @param coreNum 总核心数

     ◦ @description 当批次大小为 1 时，使用简单的静态分片策略，

     ◦              将 head * headBlockCnt 个块均匀分配到各个核心

     */
    CATLASS_DEVICE
    void FastSplitCore(uint32_t coreId, uint32_t coreNum)
    {
        this->currentSeqLen = (seqOffsetM.GetValue(1) - seqOffsetM.GetValue(0));
        if (this->currentSeqLen == 0) {
            return;
        }

        this->headBlockCnt = CeilDiv<BLOCK_M>(currentSeqLen);
        auto totalBlockCnt = this->headBlockCnt * this->headNum;
        auto splitNextCore = totalBlockCnt / coreNum;
        auto splitPrevCore = splitNextCore + 1;
        auto splitCoreIdx = totalBlockCnt % coreNum;

        uint32_t blockEnd = 0;
        uint32_t blockId = 0;
        if (coreId < splitCoreIdx) {
            blockId = coreId * splitPrevCore;
            blockEnd = blockId + splitPrevCore;
        } else if (coreId < coreNum) {
            blockId = splitCoreIdx * splitPrevCore + (coreId - splitCoreIdx) * splitNextCore;
            blockEnd = blockId + splitNextCore;
        }

        this->batchId = 0;
        this->batchBaseOffset = seqOffsetM.GetValue(0);
        this->headId = blockId / this->headBlockCnt;
        this->headBlockId = blockId - this->headBlockCnt * headId;
        this->blockCnt = blockEnd - blockId;
    }

    /**
     ◦ @brief 初始化块信息

     ◦ @param coreStartBlockId 核心起始块 ID

     ◦ @description 根据起始块 ID 计算对应的 batchId、headId、headBlockId 等信息

     */
    CATLASS_DEVICE
    void InitBlock(uint32_t coreStartBlockId)
    {
        uint32_t blockOffset = 0;
        for (auto b = 0; b < this->batchSize; ++b) {
            auto seqLens = (seqOffsetM.GetValue(b + 1) - seqOffsetM.GetValue(b));
            auto mBlockCnt = CeilDiv<BLOCK_M>(seqLens);
            blockOffset += (mBlockCnt * this->headNum);
            if (coreStartBlockId < blockOffset) {
                this->batchId = b;
                coreStartBlockId -= (blockOffset - mBlockCnt * this->headNum);
                this->headId = coreStartBlockId / mBlockCnt;
                this->headBlockId = coreStartBlockId - mBlockCnt * this->headId;
                this->currentSeqLen = seqLens;
                this->headBlockCnt = mBlockCnt;
                this->batchBaseOffset = seqOffsetM.GetValue(b);
                break;
            }
        }
    }

    /**
     ◦ @brief 持久化分片策略 (多批次时使用)

     ◦ @param coreId 当前核心 ID

     ◦ @param coreNum 总核心数

     ◦ @description 当批次大于 1 时，计算总工作量并均匀分配到各个核心，

     ◦              确保每个核心处理的块数量尽量均衡

     */
    CATLASS_DEVICE
    void PersistentSplitCore(uint32_t coreId, uint32_t coreNum)
    {
        uint32_t totalWorkLoad = 0;
        for (auto b = 0; b < batchSize; ++b) {
            uint32_t mBlockCnt = CeilDiv<BLOCK_M>((seqOffsetM.GetValue(b + 1) - seqOffsetM.GetValue(b))) * headNum;
            uint32_t nBlockCnt = CeilDiv<BLOCK_N>(seqOffsetN.GetValue(b + 1) - seqOffsetN.GetValue(b));
            totalWorkLoad += mBlockCnt * nBlockCnt;
        }

        uint32_t workLoadPerCore = CeilDiv(totalWorkLoad, coreNum);
        uint32_t workLoadThisCore = 0;
        uint32_t currentCore = 0;
        uint32_t coreStartBlockId = 0;
        uint32_t blockIdx = 0;

        for (auto b = 0; b < batchSize; ++b) {
            uint32_t mBlockCnt = CeilDiv<BLOCK_M>((seqOffsetM.GetValue(b + 1) - seqOffsetM.GetValue(b)));
            uint32_t nBlockCnt = CeilDiv<BLOCK_N>(seqOffsetN.GetValue(b + 1) - seqOffsetN.GetValue(b));

            for (auto h = 0; h < headNum; ++h) {
                for (auto m = 0; m < mBlockCnt; ++m) {
                    blockIdx++;
                    workLoadThisCore += nBlockCnt;

                    if (workLoadThisCore >= workLoadPerCore && currentCore < coreNum - 1) {
                        if (coreId == currentCore) {
                            this->blockCnt = blockIdx - coreStartBlockId;
                            InitBlock(coreStartBlockId);
                            return;
                        }

                        workLoadThisCore = 0;
                        coreStartBlockId = blockIdx;
                        currentCore++;
                    }

                    totalWorkLoad -= nBlockCnt;
                    if (totalWorkLoad == 0 && (coreId == currentCore)) {
                        this->blockCnt = blockIdx - coreStartBlockId;
                        InitBlock(coreStartBlockId);
                        return;
                    }
                }
            }
        }
    }

    /**
     ◦ @brief 初始化调度器

     ◦ @description 获取当前核心 ID 和总核心数，根据批次大小选择合适的分片策略

     */
    CATLASS_DEVICE
    void Init()
    {
        uint32_t coreId = 0;
        if ASCEND_IS_AIV {
            coreId = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        } else {
            coreId = AscendC::GetBlockIdx();
        }
        uint32_t coreNum = AscendC::GetBlockNum();

        if (this->batchSize == 1) {
            FastSplitCore(coreId, coreNum);
        } else {
            PersistentSplitCore(coreId, coreNum);
        }

        Update();
    }

    /**
     ◦ @brief 判断是否是最后一个块 (RowBlockScheduler)

     ◦ @return bool 如果剩余块数为 1 则返回 true

     */
    CATLASS_DEVICE
    bool IsLast()
    {
        return (this->blockCnt == 1);
    }

    /**
     ◦ @brief 判断块调度是否有效

     ◦ @return bool 如果块数不为 0 则返回 true

     */
    CATLASS_DEVICE
    bool IsValid()
    {
        return !(this->blockCnt == 0);
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
     ◦ @brief 前缀递增运算符，移动到下一个块

     ◦ @return RowBlockScheduler& 调度器引用

     ◦ @description 更新 batchId、headId、headBlockId 等信息，移动到下一个计算块

     */
    CATLASS_DEVICE
    RowBlockScheduler& operator++()
    {
        this->blockCnt--;
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
};

/**
 • @brief 列方向块调度器

 • @tparam RowBlockScheduler_ 行方向调度器类型

 • @tparam BLOCK_M 行方向的块大小

 • @tparam BLOCK_N 列方向的块大小

 • @tparam USE_SWIZZLE 是否启用 Swizzle 优化

 • @description 负责在列方向（seq 维度）上进行任务分片，支持双向遍历 (正序/逆序)，

 •              可选支持 Swizzle 优化以提高缓存命中率

 */
template <class RowBlockScheduler_, uint32_t BLOCK_M, uint32_t BLOCK_N, bool USE_SWIZZLE = false>
class ColumnBlockScheduler {
public:
    using RowBlockScheduler = RowBlockScheduler_;
    using ElementOffset = typename RowBlockScheduler::ElementOffset;

    /**
     ◦ @brief 构造函数

     ◦ @param batchSize 批次大小

     ◦ @param headNum 注意力头数

     ◦ @param seqOffset 序列偏移量指针

     ◦ @description 初始化列方向调度器

     */
    CATLASS_DEVICE
    ColumnBlockScheduler(uint32_t batchSize, uint32_t headNum, GM_ADDR seqOffset)
    {
        this->batchSize = batchSize;
        this->headNum = headNum;
        gSeqOffset.SetGlobalBuffer((__gm__ ElementOffset*)seqOffset);
    }

    /**
     ◦ @brief 初始化列调度器

     ◦ @param rowBlockScheduler 行方向调度器引用

     ◦ @description 根据行调度器的元数据初始化列调度器，包括 batchId、headId、序列长度等

     ◦              如果启用 Swizzle，还可以切换遍历方向

     */
    CATLASS_DEVICE
    void Init(RowBlockScheduler const& rowBlockScheduler)
    {
        auto meta = rowBlockScheduler.GetMeta();
        auto batchId = tla::get<0>(meta);  // 0 means batchId
        auto headId = tla::get<1>(meta);   // 1 means headId
        rowBlockId = tla::get<2>(meta);    // 2 means rowBlockId

        if constexpr (USE_SWIZZLE) {
            if (isInitialized && this->batchId == batchId && this->headId == headId) {
                swizzleDir = 1 - swizzleDir;
                triggerSwizzle = true;
            } else {
                swizzleDir = 1;
                isInitialized = true;
            }
        } else {
            swizzleDir = 1;
        }

        this->batchId = batchId;
        this->headId = headId;
        this->blockBaseOffset = gSeqOffset.GetValue(batchId);
        this->seqLens = gSeqOffset.GetValue(batchId + 1) - gSeqOffset.GetValue(batchId);
        this->blockCnt = CeilDiv<BLOCK_N>(seqLens);
        this->blockId = (swizzleDir == 1) ? 0 : blockCnt - 1;
        Update();
    }

    /**
     ◦ @brief 获取是否触发 Swizzle

     ◦ @return bool 是否触发 Swizzle

     ◦ @description 返回 triggerSwizzle 标志并将其复位，用于通知上层是否需要切换数据布局

     */
    CATLASS_DEVICE
    bool GetTriggerSwizzle()
    {
        auto result = triggerSwizzle;
        triggerSwizzle = false;
        return result;
    }

    /**
     ◦ @brief 判断是否是第一个块

     ◦ @return bool 根据遍历方向判断是否是起始块

     */
    CATLASS_DEVICE
    bool IsFirst()
    {
        return (swizzleDir == 1) ? blockId == 0 : blockId == (blockCnt - 1);
    }

    /**
     ◦ @brief 判断是否是最后一个块

     ◦ @return bool 根据遍历方向判断是否是结束块

     */
    CATLASS_DEVICE
    bool IsLast()
    {
        return (swizzleDir == 1) ? blockId == (blockCnt - 1) : blockId == 0;
    }

    /**
     ◦ @brief 判断块调度是否有效

     ◦ @return bool 根据遍历方向判断块 ID 是否在有效范围内

     */
    CATLASS_DEVICE
    bool IsValid()
    {
        return !((swizzleDir == 1) ? blockId >= blockCnt : blockId < 0);
    }

    CATLASS_DEVICE int32_t GetBlockId() const
    {
        return blockId;
    }
    CATLASS_DEVICE uint32_t GetRowBlockId() const
    {
        return rowBlockId;
    }
    CATLASS_DEVICE uint32_t GetSeqLens() const
    {
        return seqLens;
    }
    CATLASS_DEVICE uint32_t GetSwizzleDir() const
    {
        return swizzleDir;
    }

    /**
     ◦ @brief 前缀递增运算符，移动到下一个块

     ◦ @return ColumnBlockScheduler& 调度器引用

     ◦ @description 根据遍历方向 (swizzleDir) 更新 blockId，移动到下一个列块

     */
    CATLASS_DEVICE
    ColumnBlockScheduler& operator++()
    {
        blockId = (swizzleDir == 1) ? blockId + 1 : blockId - 1;
        Update();
        return *this;
    }

    /**
     ◦ @brief 获取当前块的张量切片

     ◦ @tparam Tensor 张量类型

     ◦ @param tensor 源张量

     ◦ @return TensorTile 从源张量中切取的块

     ◦ @description 根据当前 blockId、blockOffset、blockSize 等信息，从张量中切取对应的数据块

     */
    template <class Tensor>
    CATLASS_DEVICE auto GetTile(Tensor const& tensor)
    {
        auto dim = tla::get<1>(tensor.shape());
        auto coord = tla::MakeCoord(blockOffset + headId, 0);
        auto shape = tla::MakeShape(blockSize, dim);

        return tla::GetTile(tensor, coord, shape);
    }

    /**
     ◦ @brief 获取共享内存块的张量切片

     ◦ @tparam Tensor 张量类型

     ◦ @param tensor 源张量

     ◦ @param totalSeqLens 总序列长度

     ◦ @return TensorTile 共享内存块的张量切片

     ◦ @description 用于获取共享内存中的数据块，坐标计算考虑了 headId 和总序列长度

     */
    template <class Tensor>
    CATLASS_DEVICE auto GetShareTile(Tensor const& tensor, int64_t totalSeqLens)
    {
        auto dim = tla::get<1>(tensor.shape());
        auto coord = tla::MakeCoord(headId * totalSeqLens + blockOffset / headNum, 0);
        auto shape = tla::MakeShape(blockSize, dim);
        return tla::GetTile(tensor, coord, shape);
    }

    /**
     ◦ @brief 获取张量切片的映射信息

     ◦ @tparam Coord 坐标类型

     ◦ @tparam Shape 形状类型

     ◦ @param kCoord 关键坐标

     ◦ @param kShape 关键形状

     ◦ @return Tuple<Coord, Shape> 坐标和形状的元组

     ◦ @description 生成用于定位张量数据的坐标和形状信息，用于多核数据分发

     */
    template <class Coord, class Shape>
    CATLASS_DEVICE auto GetTileMapping(Coord const& kCoord, Shape const& kShape)
    {
        auto coord = tla::MakeCoord(batchId, headId, blockId * BLOCK_N, rowBlockId * BLOCK_M);
        auto shape = tla::MakeShape(blockSize, tla::get<0>(kShape));
        return tla::MakeTuple(coord, shape);
    }

private:
    /**
     ◦ @brief 更新块信息

     ◦ @description 根据当前 blockId 计算 blockSize 和 blockOffset

     */
    CATLASS_DEVICE
    void Update()
    {
        blockSize = (blockId == (blockCnt - 1)) ? (seqLens - blockId * BLOCK_N) : BLOCK_N;
        blockOffset = (blockBaseOffset + blockId * BLOCK_N) * headNum;
    }

    uint32_t blockSize{0};
    uint32_t blockOffset{0};
    uint32_t blockBaseOffset{0};
    uint32_t batchSize{0};
    uint32_t headNum{0};
    uint32_t batchId{0};
    uint32_t headId{0};
    uint32_t seqLens{0};
    int32_t blockId{0};
    uint32_t rowBlockId{0};
    uint32_t blockCnt{0};
    bool isInitialized{false};
    bool triggerSwizzle{false};
    uint32_t swizzleDir{1};
    AscendC::GlobalTensor<ElementOffset> gSeqOffset;
};

}  // namespace Catlass::Gemm::Block
