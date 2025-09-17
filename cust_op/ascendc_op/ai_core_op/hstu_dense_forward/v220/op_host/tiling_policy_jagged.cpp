/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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


#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cassert>

#include "common_host.h"
#include "register/op_def_registry.h"
#include "tiling_policy_factory.h"
#include "tiling_policy_jagged.h"

constexpr bool JAGGED_TASK_ASSIGN_DEBUG = false;

#if JAGGED_TASK_ASSIGN_DEBUG
#include <chrono>
#endif

constexpr uint32_t CONST_2 = 2;
constexpr int QKV_DIM = 3;

namespace {
    struct BlockTaskInfo {
        uint32_t startBlockId = 0;
        uint32_t endBlockId = 0;

        friend std::ostream& operator<<(std::ostream& os, const BlockTaskInfo& blockTask)
        {
            return os << "startBlockId:" << blockTask.startBlockId << " " <<
                         "endBlockId:" << blockTask.endBlockId << " ";
        }
    };

    class BlockTaskAssign {
    public:
        BlockTaskAssign(uint32_t *seqOffsets,
                        uint32_t coreNum, uint32_t blockLen, uint32_t batchSize, uint32_t headNum)
        {
#if JAGGED_TASK_ASSIGN_DEBUG
            OPS_LOG_D("BlockTaskAssign coreNum:%d blockLen:%d batchSize:%d headNum:%d\n",
                coreNum, blockLen, batchSize, headNum);
            OPS_LOG_D("BlockTaskAssign seqOffsets:");
            for (auto i = 0; i <= batchSize; i++) {
                OPS_LOG_D("%d ", seqOffsets[i]);
            }
            OPS_LOG_D("\n");
#endif
            this->seqOffsets = seqOffsets;
            this->coreNum = coreNum;
            this->blockLen = blockLen;
            this->batchSize = batchSize;
            this->headNum = headNum;
        }

        void PreInit(
            std::vector<BlockTaskInfo> &workTasks, std::vector<int> &workLoads, std::vector<int64_t> &blockNumber)
        {
            workTasks.resize(this->coreNum);
            workLoads.resize(this->coreNum, 0);

            // 得到每个batch 和 head的block个数
            for (auto batchId = 0; batchId < batchSize; batchId++) {
                auto batchBlockSize = this->seqOffsets[batchId + 1] - this->seqOffsets[batchId];

                for (auto headId = 0; headId < headNum; headId++) {
                    blockNumber[batchId * headNum + headId] =
                        (batchBlockSize + blockLen - 1) / blockLen;
                }
            }
        }

        bool BatchSwitch(
            std::vector<int64_t> &blockNumber,
            uint32_t &batchId,
            uint32_t totalBatchSize,
            uint32_t &batchTaskNum)
        {
            if (blockNumber[batchId] == 0) {
                batchId++;
                if (batchId >= totalBatchSize) {
                    return false;
                }
                batchTaskNum = blockNumber[batchId];
            }
            return true;
        }

        void Compute(std::vector<BlockTaskInfo> &workTasks, std::vector<int> &workLoads)
        {
            // 得到每个batch 和 head的block个数
            uint32_t totalBatchSize = batchSize * headNum;
            std::vector<int64_t> blockNumber(totalBatchSize, 0);
            PreInit(workTasks, workLoads, blockNumber);

            // 计算所有的task_num得到每个core 计算的task均值
            int64_t totalTaskNumber = 0;
            totalTaskNumber = std::accumulate(blockNumber.begin(),
                                              blockNumber.end(),
                                              totalTaskNumber,
                                              [](int64_t val, int64_t x) {
                                                  return val + x * x;
                                              });

            int64_t eachCoreTaskNumLimit = (totalTaskNumber + this->coreNum - 1) / this->coreNum;

#if JAGGED_TASK_ASSIGN_DEBUG
            int64_t total_block_number = 0;
            total_block_number = std::accumulate(blockNumber.begin(),
                                                 blockNumber.end(), total_block_number, [](int64_t val, int64_t x) {
                        return val + x;
                    });
            OPS_LOG_D("eachCoreTaskNumLimit :%d totalTaskNumber:%d total_block_number:%d\n",
                eachCoreTaskNumLimit, totalTaskNumber, total_block_number);
#endif

            // 遍历workers 计算得到每一个works的任务量
            uint32_t batchId = 0;
            uint32_t batchTaskNum = blockNumber[batchId];
            uint32_t processBlockNum = 0;
            uint32_t processTaskNum = 0;
            for (int i = 0; i < this->coreNum && batchId < totalBatchSize; i++) {
                BlockTaskInfo blockTask;
                blockTask.startBlockId = processBlockNum;

                while (workLoads[i] < eachCoreTaskNumLimit) {
                    workLoads[i] += batchTaskNum;
                    processTaskNum += batchTaskNum;
                    processBlockNum++;
                    blockNumber[batchId]--;
                    if (!BatchSwitch(blockNumber, batchId, totalBatchSize, batchTaskNum)) {
                        break;
                    }
                }

                blockTask.endBlockId = processBlockNum;
                workTasks[i] = blockTask;
            }

#if JAGGED_TASK_ASSIGN_DEBUG
            OPS_LOG_D("processTaskNum :%d processBlockNum:%d\n", processTaskNum, processBlockNum);
            assert(processTaskNum == totalTaskNumber);
            assert(processBlockNum == total_block_number);
#endif
        }

        bool BatchSwitchCausal(
            std::vector<int64_t> &blockNumber,
            uint32_t &batchId,
            uint32_t &taskNum,
            uint32_t totalBatchSize
        )
        {
            if (blockNumber[batchId] == 0) {
                batchId++;
                taskNum = 1;
                if (batchId >= totalBatchSize) {
                    return false;
                }
            }
            return true;
        }

        void ComputeCausal(std::vector<BlockTaskInfo> &workTasks, std::vector<int> &workLoads)
        {
            // 得到每个batch 和 head的block个数
            uint32_t totalBatchSize = batchSize * headNum;
            std::vector<int64_t> blockNumber(totalBatchSize, 0);
            PreInit(workTasks, workLoads, blockNumber);

            // 计算所有的task_num得到每个core 计算的task均值
            int64_t totalTaskNumber = 0;
            totalTaskNumber = std::accumulate(blockNumber.begin(),
                                              blockNumber.end(),
                                              totalTaskNumber,
                                              [](int64_t val, int64_t x) {
                                                  return val + x * (x + 1) / CONST_2;
                                              });

            int64_t eachCoreTaskNumLimit = (totalTaskNumber + this->coreNum - 1) / this->coreNum;

#if JAGGED_TASK_ASSIGN_DEBUG
            int64_t total_block_number = 0;
            total_block_number = std::accumulate(blockNumber.begin(),
                                                 blockNumber.end(), total_block_number, [](int64_t val, int64_t x) {
                        return val + x;
                    });
            OPS_LOG_D("eachCoreTaskNumLimit :%d totalTaskNumber:%d total_block_number:%d\n",
                eachCoreTaskNumLimit, totalTaskNumber, total_block_number);
#endif

            // 遍历workers 计算得到每一个works的任务量
            uint32_t batchId = 0;
            uint32_t taskNum = 1;
            uint32_t processBlockNum = 0;
            uint32_t processTaskNum = 0;
            for (int i = 0; i < this->coreNum && batchId < totalBatchSize; i++) {
                BlockTaskInfo blockTask;
                blockTask.startBlockId = processBlockNum;

                while (workLoads[i] < eachCoreTaskNumLimit) {
                    workLoads[i] += taskNum;
                    processTaskNum += taskNum;

                    taskNum++;
                    processBlockNum++;
                    blockNumber[batchId]--;
                    if (!BatchSwitchCausal(blockNumber, batchId, taskNum, totalBatchSize)) {
                        break;
                    }
                }

                blockTask.endBlockId = processBlockNum;
                workTasks[i] = blockTask;
            }

#if JAGGED_TASK_ASSIGN_DEBUG
            OPS_LOG_D("processTaskNum :%d processBlockNum:%d\n", processTaskNum, processBlockNum);
            assert(processTaskNum == totalTaskNumber);
            assert(processBlockNum == total_block_number);
#endif
        }

    private:
        uint32_t *seqOffsets = nullptr;
        uint32_t coreNum = 0;
        uint32_t blockLen = 0;
        uint32_t batchSize = 0;
        uint32_t headNum = 0;
    };
}

namespace HstuDenseForward {

REGISTER_POLICY(LAYOUT_TYPE::JAGGED, std::make_shared<TilingPolicyJagged>());

bool TilingPolicyJagged::TilingShape(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling)
{
    int64_t batchSize;
    int64_t headNum;
    int64_t headDIM;
    int64_t seqLens;

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, return false);

    const auto seqOffset = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_T::INDEX_4);
    OPS_CHECK_PTR_NULL(seqOffset, return false);

    auto *seqOffsetData = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(seqOffset->GetData()));
    OPS_CHECK_PTR_NULL(seqOffsetData, return false);

    int64_t seqOffsetLens = seqOffset->GetSize();
    batchSize = GetBatchSizeFromJaggedOffset(seqOffsetData, seqOffsetLens);
    OPS_CHECK((batchSize == 0 || batchSize > MAX_BATCH_SIZE),
        OPS_LOG_E("", "batchSize limit (0, %d], but get %lld\n", MAX_BATCH_SIZE, batchSize), return false);

    if (!QKVShapeCheck(context, QKV_DIM)) {
        return false;
    }
    auto queryShape = context->GetInputShape(INDEX_T::INDEX_0)->GetStorageShape();
    headNum = queryShape.GetDim(INDEX_T::INDEX_1);
    headDIM = queryShape.GetDim(INDEX_T::INDEX_2);
    seqLens = tiling.get_maxSeqLen();

    tiling.set_batchSize(batchSize);
    tiling.set_headNum(headNum);
    tiling.set_dim(headDIM);
    tiling.set_seqLen(seqLens);

    OPS_CHECK(!GeneralShapeCheck(batchSize, seqLens, headNum, headDIM),
        OPS_LOG_E("", "Jagged Shape Check failed"), return false);
    return true;
}

static void CallBlockAssign(
    uint32_t *seqOffsets,
    uint32_t coreNum,
    std::vector<BlockTaskInfo> &workTasks,
    std::vector<int> &workLoads,
    optiling::HstuDenseForwardTilingData &tiling)
{
    uint32_t batchSize = tiling.get_batchSize();
    uint32_t headNum = tiling.get_headNum();
    uint32_t maskType = tiling.get_maskType();

    auto taskAssigner = BlockTaskAssign(seqOffsets, coreNum, BLOCK_HEIGHT, batchSize, headNum);
    if (maskType == 0) {
        taskAssigner.ComputeCausal(workTasks, workLoads);
    } else {
        taskAssigner.Compute(workTasks, workLoads);
    }
}

bool TilingPolicyJagged::TilingCore(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, return false);

    const auto seqOffset = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_T::INDEX_4);
    OPS_CHECK_PTR_NULL(seqOffset, return false);

    auto *seqOffsetData = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(seqOffset->GetData()));
    int seqOffsetLens = seqOffset->GetSize();
    int64_t batchSize = GetBatchSizeFromJaggedOffset(seqOffsetData, seqOffsetLens);
    OPS_CHECK((batchSize == 0 || batchSize > MAX_BATCH_SIZE),
        OPS_LOG_E("", "batchSize limit (0, %d], but get %lld\n", MAX_BATCH_SIZE, batchSize), return false);

#if JAGGED_TASK_ASSIGN_DEBUG
    auto start = std::chrono::high_resolution_clock::now();
#endif

    std::vector<BlockTaskInfo> workTasks;
    std::vector<int> workLoads;

    uint32_t seqOffsets[MAX_BATCH_SIZE + 1] = {0};
    for (auto i = 0; i < batchSize + 1; i++) {
        seqOffsets[i] = seqOffsetData[i];
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum > MAX_AIV_NUM, OPS_LOG_E("", "vecCoreNum %d should be < %d\n", coreNum, MAX_AIV_NUM),
              return ge::GRAPH_FAILED);
    CallBlockAssign(seqOffsets, coreNum, workTasks, workLoads, tiling);

#if JAGGED_TASK_ASSIGN_DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;
    std::cout << "BlockTaskAssign Elapsed time: " << elapsed.count() << " us\n";

    for (auto i = 0; i < coreNum; i++) {
        OPS_LOG_E("", "aicore :%d startBlockId:%d endBlockId:%d totalTaskNumber:%d\n",
            i, workTasks[i].startBlockId, workTasks[i].endBlockId, workLoads[i]);
    }
#endif

    uint32_t startBlockId[MAX_AIV_NUM] = {0};
    uint32_t endBlockId[MAX_AIV_NUM] = {0};

    for (auto i = 0; i < coreNum; i++) {
        startBlockId[i] = workTasks[i].startBlockId;
        endBlockId[i] = workTasks[i].endBlockId;
    }

    tiling.set_seqOffset(seqOffsets);
    tiling.set_eachCoreStartBlockId(startBlockId);
    tiling.set_eachCoreEndBlockId(endBlockId);

    size_t aicCoreNum = ascendPlatform.GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);

    return true;
}

bool TilingPolicyJagged::TilingKeySet(gert::TilingContext* context, optiling::HstuDenseForwardTilingData &tiling)
{
    OPS_LOG_E_IF_NULL("query", context->GetInputTensor(0), return ge::GRAPH_FAILED);
    ge::DataType qTypeGe = context->GetInputTensor(0)->GetDataType();
    if (qTypeGe == ge::DataType::DT_FLOAT) {
        context->SetTilingKey(JAGGED_FLOAT_TILING_KEY);
    } else if (qTypeGe == ge::DataType::DT_FLOAT16) {
        context->SetTilingKey(JAGGED_FLOAT16_TILING_KEY);
    } else if (qTypeGe == ge::DataType::DT_BF16) {
        context->SetTilingKey(JAGGED_BF16_TILING_KEY);
    } else {
        OPS_LOG_E("", "invalid datatype, only support fp32, fp16, bf16");
        return false;
    }

    return true;
}

void TilingPolicyJagged::DumpTiling(optiling::HstuDenseForwardTilingData &tiling)
{
    this->TilingPolicy::DumpTiling(tiling);

    uint32_t *seqOffset = tiling.get_seqOffset();
    uint32_t *startBlockId = tiling.get_eachCoreStartBlockId();
    uint32_t *endBlockId = tiling.get_eachCoreEndBlockId();

    OPS_LOG_D("seq offset:");
    for (auto i = 0; i < (tiling.get_batchSize() + 1); i++) {
        OPS_LOG_D("%d ", seqOffset[i]);
    }
    OPS_LOG_D("\n");

    OPS_LOG_D("core block range:\n");
    for (auto i = 0; i < MAX_AIV_NUM; i++) {
        OPS_LOG_E("", "core_id:%d startBlockId:%d endBlockId:%d\n", i, startBlockId[i], endBlockId[i]);
    }
}

}