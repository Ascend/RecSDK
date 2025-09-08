/*
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef HSTU_SPLIT_CORE_POLICY_H
#define HSTU_SPLIT_CORE_POLICY_H

#include <unistd.h>

#include <cstdint>
#include <type_traits>

#include "kernel_log.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;


namespace HstuDenseForward {
    constexpr int CONST_2 = 2;
    class BlockTaskAssign {
    public:
        __aicore__ inline BlockTaskAssign(uint32_t *seqOffsets, uint32_t coreNum, uint32_t blockLen,
                                          uint32_t batchSize, uint32_t headNum, GlobalTensor<int64_t> &blockNumberGt)
        {
            this->seqOffsets = seqOffsets;
            this->coreNum = coreNum;
            this->blockLen = blockLen;
            this->batchSize = batchSize;
            this->headNum = headNum;
            this->blockNumberGt = blockNumberGt;
        }

        __aicore__ inline void PreInit()
        {
            // 得到每个batch 和 head的block个数
            for (auto batchId = 0; batchId < batchSize; batchId++) {
                auto batchBlockSize = this->seqOffsets[batchId + 1] - this->seqOffsets[batchId];

                for (auto headId = 0; headId < headNum; headId++) {
                    blockNumberGt.SetValue(batchId * headNum + headId,
                        (batchBlockSize + blockLen - 1) / blockLen);
                }
            }
        }

        __aicore__ inline bool BatchSwitch(
            uint32_t &batchId,
            uint32_t totalBatchSize,
            uint32_t &batchTaskNum)
        {
            if (blockNumberGt.GetValue(batchId) == 0) {
                batchId++;
                if (batchId >= totalBatchSize) {
                    return false;
                }
                batchTaskNum = blockNumberGt.GetValue(batchId);
            }
            return true;
        }

        __aicore__ inline void Compute()
        {
            uint32_t totalBatchSize = 0;
            int64_t eachCoreTaskNumLimit = 0;
            InitAndComputeLimit(false, totalBatchSize, eachCoreTaskNumLimit);

            // 遍历workers 计算得到每一个works的任务量
            uint32_t batchId = 0;
            uint32_t batchTaskNum = blockNumberGt.GetValue(batchId);
            uint32_t processBlockNum = 0;
            uint32_t processTaskNum = 0;
            uint32_t workLoads = 0;
            for (int i = 0; i < this->coreNum && batchId < totalBatchSize; i++) {
                blockNumberGt.SetValue(totalBatchSize + i, processBlockNum);
                workLoads = 0;
                while (workLoads < eachCoreTaskNumLimit) {
                    workLoads += batchTaskNum;
                    processTaskNum += batchTaskNum;
                    processBlockNum++;
                    blockNumberGt.SetValue(batchId, blockNumberGt.GetValue(batchId) - 1);
                    if (!BatchSwitch(batchId, totalBatchSize, batchTaskNum)) {
                        break;
                    }
                }
                blockNumberGt.SetValue(totalBatchSize + i + this->coreNum, processBlockNum);
            }
            DataCacheCleanAndInvalid<int64_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(blockNumberGt);
        }

        __aicore__ inline bool BatchSwitchCausal(
            uint32_t &batchId,
            uint32_t &taskNum,
            uint32_t totalBatchSize
        )
        {
            if (blockNumberGt.GetValue(batchId) == 0) {
                batchId++;
                taskNum = 1;
                if (batchId >= totalBatchSize) {
                    return false;
                }
            }
            return true;
        }

        __aicore__ inline  void ComputeCausal()
        {
            uint32_t totalBatchSize = 0;
            int64_t eachCoreTaskNumLimit = 0;
            InitAndComputeLimit(true, totalBatchSize, eachCoreTaskNumLimit);

            // 遍历workers 计算得到每一个works的任务量（因果场景）
            uint32_t batchId = 0;
            uint32_t taskNum = 1;
            uint32_t processBlockNum = 0;
            uint32_t processTaskNum = 0;
            uint32_t workLoads = 0;
            for (int i = 0; i < this->coreNum && batchId < totalBatchSize; i++) {
                blockNumberGt.SetValue(totalBatchSize + i, processBlockNum);
                workLoads = 0;
                while (workLoads < eachCoreTaskNumLimit) {
                    workLoads += taskNum;
                    processTaskNum += taskNum;
                    taskNum++;
                    processBlockNum++;
                    blockNumberGt.SetValue(batchId, blockNumberGt.GetValue(batchId) - 1);
                    if (!BatchSwitchCausal(batchId, taskNum, totalBatchSize)) {
                        break;
                    }
                }
                blockNumberGt.SetValue(totalBatchSize + i + this->coreNum, processBlockNum);
            }
            DataCacheCleanAndInvalid<int64_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(blockNumberGt);
        }

    private:
        __aicore__ inline void InitAndComputeLimit(
            bool isCausal,
            uint32_t &totalBatchSize,
            int64_t &eachCoreTaskNumLimit)
        {
            // 得到每个batch 和 head的block个数
            totalBatchSize = batchSize * headNum;
            PreInit();

            // 计算所有的task_num得到每个core 计算的task均值
            int64_t totalTaskNumber = 0;
            for (uint32_t i = 0; i < totalBatchSize; i++) {
                auto n = blockNumberGt.GetValue(i);
                if (isCausal) {
                    totalTaskNumber += n * (n + 1) / CONST_2;
                } else {
                    totalTaskNumber += n * n;
                }
            }
            eachCoreTaskNumLimit = (totalTaskNumber + this->coreNum - 1) / this->coreNum;
        }

        uint32_t *seqOffsets = nullptr;
        uint32_t coreNum = 0;
        uint32_t blockLen = 0;
        uint32_t batchSize = 0;
        uint32_t headNum = 0;
        GlobalTensor<int64_t>blockNumberGt;
    };
}
#endif