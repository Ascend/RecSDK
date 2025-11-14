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

constexpr int MAX_BATCH_SIZE = 2048;
constexpr int MAX_HEAD_NUM = 8;
constexpr int MAX_BXN = MAX_BATCH_SIZE * MAX_HEAD_NUM;

namespace HstuDenseBackward {
template <typename T>
__aicore__ inline T CeilDiv(T dividend, T divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename oType>
class BlockTaskAssign {
public:
    __aicore__ inline BlockTaskAssign(GlobalTensor<oType>& seqOffsets, uint32_t coreNum, uint32_t blockLen,
                                      uint32_t batchSize, uint32_t headNum)
    {
        this->seqOffsets = seqOffsets;
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
        for (auto batchId = 0; batchId < batchSize; batchId++) {
            uint32_t batchBlockSize = this->seqOffsets.GetValue(batchId + 1) - this->seqOffsets.GetValue(batchId);
            uint32_t blk = CeilDiv(batchBlockSize, blockLen);
            uint32_t batchOffset = batchId * headNum;
            for (auto headId = 0; headId < headNum; headId++) {
                blockNumber[batchOffset + headId] = blk;
            }
            totalTaskNumber += headNum * blk * (blk + 1) / 2;
        }
        uint32_t eachCoreTaskNumLimit = CeilDiv(totalTaskNumber, this->coreNum);

        uint32_t batchId = 0;
        uint32_t processBlockNum = 0;
        uint32_t taskNum = isCol ? blockNumber[0] : 1;

        for (int i = 0; i < this->coreNum && batchId < bxn; i++) {
            uint32_t workLoads = 0;
            auto start = processBlockNum;

            while (workLoads < eachCoreTaskNumLimit) {
                workLoads += taskNum;
                taskNum = isCol ? taskNum - 1 : taskNum + 1;
                processBlockNum++;
                blockNumber[batchId]--;
                if (blockNumber[batchId] == 0 && batchId + 1 >= bxn) {
                    batchId++;
                    break;
                }
                if (blockNumber[batchId] == 0) {
                    batchId++;
                    taskNum = isCol ? blockNumber[batchId] : 1;
                }
            }
            if (i == coreId) {
                result[0] = start;
                result[1] = processBlockNum;
                break;
            }
        }
    }

private:
    GlobalTensor<oType> seqOffsets;
    uint32_t coreNum = 0;
    uint32_t blockLen = 0;
    uint32_t batchSize = 0;
    uint32_t headNum = 0;

    uint32_t bxn;
};
} // namespace
#endif  // MXREC_HSTU_SPLIT_CORE_POLICY_H
