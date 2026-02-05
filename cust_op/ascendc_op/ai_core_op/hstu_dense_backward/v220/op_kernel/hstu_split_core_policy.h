/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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

#ifndef MXREC_HSTU_SPLIT_CORE_POLICY_H
#define MXREC_HSTU_SPLIT_CORE_POLICY_H
#include "hstu_common_const.h"

constexpr int MAX_BATCH_SIZE = 2048;
constexpr int MAX_HEAD_NUM = 8;
constexpr int MAX_BXN = MAX_BATCH_SIZE * MAX_HEAD_NUM;

namespace HstuDenseBackward {
template <typename oType>
class BlockTaskAssign {
public:
    __aicore__ inline BlockTaskAssign(GlobalTensor<oType>& seqOffsetsQ,
                                      GlobalTensor<oType>& seqOffsetsK,
                                      uint32_t coreNum,
                                      uint32_t blockLen,
                                      uint32_t batchSize,
                                      uint32_t headNum)
    {
        this->seqOffsetsQ = seqOffsetsQ;
        this->seqOffsetsK = seqOffsetsK;
        this->coreNum = coreNum;
        this->blockLen = blockLen;
        this->batchSize = batchSize;
        this->headNum = headNum;

        this->bxn = batchSize * headNum;
    }

    __aicore__ inline void Compute(int* result, int coreId, bool isCol)
    {
        uint8_t blockNumber[MAX_BXN];
        uint32_t totalTaskNumber = 0;
        for (auto batchId = 0; batchId < batchSize; batchId++) {
            uint32_t batchBlockSize = this->seqOffsets.GetValue(batchId + 1) - this->seqOffsets.GetValue(batchId);
            uint32_t blk = CeilDiv(batchBlockSize, blockLen);
            uint32_t batchOffset = batchId * headNum;
            for (auto headId = 0; headId < headNum; headId++) {
                blockNumber[batchOffset + headId] = blk;
            }
            totalTaskNumber += headNum * blk * blk;
        }
        uint32_t eachCoreTaskNumLimit = CeilDiv(totalTaskNumber, this->coreNum);

        uint32_t batchId = 0;
        uint32_t batchTaskNum = blockNumber[batchId];
        uint32_t processBlockNum = 0;

        for (int i = 0; i < this->coreNum && batchId < bxn; i++) {
            uint32_t workLoads = 0;
            auto start = processBlockNum;

            while (workLoads < eachCoreTaskNumLimit) {
                workLoads += batchTaskNum;
                processBlockNum++;
                blockNumber[batchId]--;
                if (blockNumber[batchId] == 0 && batchId + 1 >= bxn) {
                    batchId++;
                    break;
                }
                if (blockNumber[batchId] == 0) {
                    batchId++;
                    batchTaskNum = blockNumber[batchId];
                }
            }

            if (i == coreId) {
                result[0] = start;
                result[1] = processBlockNum;
                break;
            }
        }
    }

    __aicore__ inline void ComputeCausal(int* result, int coreId, bool isCol)
    {
        uint32_t totalTaskNumber = 0;
        uint8_t blockNumber[MAX_BXN];
        uint32_t offsetQ = 0;
        uint32_t offsetK = 0;
        for (auto batchId = 0; batchId < batchSize; batchId++) {
            uint32_t nextOffsetQ = this->seqOffsetsQ.GetValue(batchId + 1);
            uint32_t nextOffsetK = this->seqOffsetsK.GetValue(batchId + 1);
            uint32_t seqlenQ = nextOffsetQ - offsetQ;
            uint32_t seqlenK = nextOffsetK - offsetK;
            offsetQ = nextOffsetQ;
            offsetK = nextOffsetK;
            uint32_t blkQ = CeilDiv(seqlenQ, blockLen);
            uint32_t blkK = CeilDiv(seqlenK, blockLen);
            uint32_t batchOffset = batchId * headNum;
            for (auto headId = 0; headId < headNum; headId++) {
                blockNumber[batchOffset + headId] = blkK;
            }
            uint32_t blkTop = GetBlockTop(seqlenQ, seqlenK, blockLen);
            uint32_t flag = blkTop + blkQ - 1 - blkK;
            uint32_t taskNum = (blkQ * (blkK + flag + blkTop) / 2 - flag);
            totalTaskNumber += headNum * taskNum;
        }
        uint32_t eachCoreTaskNumLimit = CeilDiv(totalTaskNumber, this->coreNum);
        uint32_t batchId = 0;
        uint32_t processBlockNum = 0;
        uint32_t taskNum = CeilDiv(this->seqOffsetsQ.GetValue(1), blockLen);
        uint32_t blkTop = GetBlockTop(this->seqOffsetsQ.GetValue(1),
                                      this->seqOffsetsK.GetValue(1),
                                      blockLen);
        uint32_t blkId = 0;
        for (int i = 0; i < this->coreNum && batchId < bxn; i++) {
            uint32_t workLoads = 0;
            auto start = processBlockNum;

            while (workLoads < eachCoreTaskNumLimit && batchId < bxn) {
                workLoads += taskNum;
                blkId++;
                taskNum = taskNum - static_cast<int>(blkId >= blkTop);
                processBlockNum++;
                blockNumber[batchId]--;
                if (blockNumber[batchId] == 0) {
                    batchId++;
                    if (batchId >= bxn) {
                        break;
                    }
                    uint32_t bid = batchId / headNum;
                    uint32_t seqlenQ = this->seqOffsetsQ.GetValue(bid + 1) - this->seqOffsetsQ.GetValue(bid);
                    uint32_t seqlenK = this->seqOffsetsK.GetValue(bid + 1) - this->seqOffsetsK.GetValue(bid);
                    blkTop = GetBlockTop(seqlenQ, seqlenK, blockLen);
                    taskNum = CeilDiv(seqlenQ, blockLen);
                    blkId = 0;
                }
            }
            if (i == coreId) {
                result[0] = start;
                result[1] = processBlockNum;
                return;
            }
        }
    }

private:
    GlobalTensor<oType> seqOffsetsQ;
    GlobalTensor<oType> seqOffsetsK;
    uint32_t coreNum = 0;
    uint32_t blockLen = 0;
    uint32_t batchSize = 0;
    uint32_t headNum = 0;

    uint32_t bxn;
    static __aicore__ inline uint32_t GetBlockTop(uint32_t seqlenQ, uint32_t seqlenK, uint32_t blockLen)
    {
        return (seqlenK - seqlenQ + 2 * blockLen - 1) / blockLen;
    }
};
} // namespace
#endif  // MXREC_HSTU_SPLIT_CORE_POLICY_H
