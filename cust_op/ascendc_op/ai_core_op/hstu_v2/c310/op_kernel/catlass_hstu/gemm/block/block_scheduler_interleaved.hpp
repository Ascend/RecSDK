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
 * @file block_scheduler_interleaved.hpp
 * @brief HSTU 交替分块调度器 — 提高 L2 Cache 跨核共享率
 *
 * @description
 * 与 RowBlockScheduler (连续分块) 不同，本调度器将 Q 行块 (rowBlock) 在各核
 * 之间以 round-robin 方式交替分配:
 *
 *   Core 0: rowBlock [0, N, 2N, ...]
 *   Core 1: rowBlock [1, N+1, 2N+1, ...]
 *   ...
 *
 */

#pragma once

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Block {

template <class ElementOffset_, uint32_t BLOCK_M, uint32_t BLOCK_N>
class InterleavedRowBlockScheduler {
public:
    using ElementOffset = ElementOffset_;

    CATLASS_DEVICE
    InterleavedRowBlockScheduler(uint32_t batchSize, uint32_t headNum, GM_ADDR seqOffsetM, GM_ADDR seqOffsetN)
    {
        this->batchSize = batchSize;
        this->headNum = headNum;

        this->seqOffsetM.SetGlobalBuffer((__gm__ ElementOffset*)seqOffsetM);
        this->seqOffsetN.SetGlobalBuffer((__gm__ ElementOffset*)seqOffsetN);
    }

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
            SingleBatchRoundRobin(coreId, coreNum);
        } else {
            MultiBatchRoundRobin(coreId, coreNum);
        }

        if (this->blockCnt > 0) {
            LocateBlock(currentFlatBlockId);
            Update();
        }
    }

    CATLASS_DEVICE bool IsLast()
    {
        return (this->blockCnt == 1);
    }
    CATLASS_DEVICE bool IsValid()
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

    CATLASS_DEVICE
    InterleavedRowBlockScheduler& operator++()
    {
        this->blockCnt--;
        if (this->blockCnt == 0) {
            return *this;
        }

        currentFlatBlockId += coreStride;
        AdvanceByStride(coreStride);
        Update();
        return *this;
    }

private:
    CATLASS_DEVICE
    void SingleBatchRoundRobin(uint32_t coreId, uint32_t coreNum)
    {
        this->currentSeqLen = seqOffsetM.GetValue(1) - seqOffsetM.GetValue(0);
        if (this->currentSeqLen == 0) {
            this->blockCnt = 0;
            return;
        }

        this->headBlockCnt = CeilDiv<BLOCK_M>(currentSeqLen);
        totalFlatBlocks = this->headBlockCnt * this->headNum;
        coreStride = coreNum;

        if (coreId >= totalFlatBlocks) {
            this->blockCnt = 0;
            return;
        }

        currentFlatBlockId = coreId;
        this->blockCnt = (totalFlatBlocks - coreId + coreNum - 1) / coreNum;

        this->batchId = 0;
        this->batchBaseOffset = seqOffsetM.GetValue(0);
    }

    CATLASS_DEVICE
    void MultiBatchRoundRobin(uint32_t coreId, uint32_t coreNum)
    {
        totalFlatBlocks = 0;
        for (uint32_t b = 0; b < batchSize; ++b) {
            uint32_t seqLen = seqOffsetM.GetValue(b + 1) - seqOffsetM.GetValue(b);
            uint32_t hbc = CeilDiv<BLOCK_M>(seqLen);
            totalFlatBlocks += hbc * headNum;
        }

        if (totalFlatBlocks == 0) {
            this->blockCnt = 0;
            return;
        }

        coreStride = coreNum;
        currentFlatBlockId = coreId;

        if (currentFlatBlockId >= totalFlatBlocks) {
            this->blockCnt = 0;
            return;
        }

        this->blockCnt = (totalFlatBlocks - coreId + coreNum - 1) / coreNum;
    }

    CATLASS_DEVICE
    void LocateBlock(uint32_t flatId)
    {
        uint32_t remaining = flatId;
        for (uint32_t b = 0; b < batchSize; ++b) {
            this->batchBaseOffset = seqOffsetM.GetValue(b);
            this->currentSeqLen = seqOffsetM.GetValue(b + 1) - seqOffsetM.GetValue(b);
            uint32_t hbc = CeilDiv<BLOCK_M>(this->currentSeqLen);
            uint32_t batchBlocks = hbc * headNum;

            if (remaining < batchBlocks) {
                this->batchId = b;
                this->headBlockCnt = hbc;
                this->headId = remaining / hbc;
                this->headBlockId = remaining - headId * hbc;
                return;
            }
            remaining -= batchBlocks;
        }

        this->blockCnt = 0;
    }

    CATLASS_DEVICE
    void AdvanceByStride(uint32_t stride)
    {
        uint32_t consumedInBatch = headId * headBlockCnt + headBlockId + 1;
        uint32_t blocksInCurrentBatch = headBlockCnt * headNum;
        auto hasRemaining = (blocksInCurrentBatch > consumedInBatch);
        uint32_t remainingInBatch = hasRemaining * (blocksInCurrentBatch - consumedInBatch);

        if (stride <= remainingInBatch) {
            uint32_t localIdx = headId * headBlockCnt + headBlockId + stride;
            headId = localIdx / headBlockCnt;
            headBlockId = localIdx % headBlockCnt;
            return;
        }

        stride -= remainingInBatch + 1;
        ;
        batchId++;

        while (batchId < batchSize) {
            batchBaseOffset = seqOffsetM.GetValue(batchId);
            currentSeqLen = seqOffsetM.GetValue(batchId + 1) - seqOffsetM.GetValue(batchId);
            headBlockCnt = CeilDiv<BLOCK_M>(currentSeqLen);

            if (headBlockCnt == 0) {
                batchId++;
                continue;
            }

            uint32_t blocksInBatch = headBlockCnt * headNum;
            if (stride < blocksInBatch) {
                headId = stride / headBlockCnt;
                headBlockId = stride % headBlockCnt;
                return;
            }

            stride -= blocksInBatch;
            batchId++;
        }

        this->blockCnt = 0;
    }

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
        auto isTail = (remainingSeq < BLOCK_M);
        blockSize = BLOCK_M + isTail * (remainingSeq - BLOCK_M);
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

    uint32_t currentFlatBlockId{0};
    uint32_t totalFlatBlocks{0};
    uint32_t coreStride{0};

    AscendC::GlobalTensor<ElementOffset> seqOffsetM;
    AscendC::GlobalTensor<ElementOffset> seqOffsetN;
};

}  // namespace Catlass::Gemm::Block
