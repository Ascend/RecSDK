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

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cassert>

#include "register/op_def_registry.h"
#include "tiling_policy_factory.h"
#include "tiling_policy_jagged.h"

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
        }

    private:
        uint32_t *seqOffsets = nullptr;
        uint32_t coreNum = 0;
        uint32_t blockLen = 0;
        uint32_t batchSize = 0;
        uint32_t headNum = 0;
    };
}
    
namespace HstuDenseForwardFuxi {

REGISTER_POLICY(LAYOUT_TYPE::JAGGED, std::make_shared<TilingPolicyJagged>());

ge::graphStatus TilingPolicyJagged::InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* qShape = context->GetInputShape(INDEX_T::INDEX_0);
    OPS_LOG_E_IF_NULL("qShape", qShape, return ge::GRAPH_FAILED);

    gert::Shape* outputShape = context->GetOutputShape(INDEX_T::INDEX_0);
    OPS_LOG_E_IF_NULL("outputShape", outputShape, return ge::GRAPH_FAILED);

    outputShape->SetDimNum(OUTPUT_DIM_NUM);

    outputShape->SetDim(INDEX_T::INDEX_0, qShape->GetDim(INDEX_T::INDEX_0));

    // 获取算子可选参数：timestampBias和positionBias
    auto timestampBias = context->GetOptionalInputTensor(INDEX_T::INDEX_3);
    auto positionBias = context->GetOptionalInputTensor(INDEX_T::INDEX_4);
    if ((timestampBias == nullptr) || (positionBias == nullptr)) {
        outputShape->SetDim(INDEX_T::INDEX_1, qShape->GetDim(INDEX_T::INDEX_1) * qShape->GetDim(INDEX_T::INDEX_2));
    } else {
        // 3:attnOut + tsOut + posOut
        outputShape->SetDim(INDEX_T::INDEX_1,
            OUTPUT_DIM2_TIMES_3 * qShape->GetDim(INDEX_T::INDEX_1) * qShape->GetDim(INDEX_T::INDEX_2));
    }

    return ge::GRAPH_SUCCESS;
}

bool TilingPolicyJagged::TilingShape(gert::TilingContext* context, optiling::HstuDenseForwardFuxiTilingData &tiling)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return false);

    const auto seqOffset = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_T::INDEX_4);
    OPS_LOG_E_IF_NULL("seqOffset", seqOffset, return false);

    auto *seqOffsetData = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(seqOffset->GetData()));
    OPS_LOG_E_IF_NULL("seqOffsetData", seqOffsetData, return false);

    size_t seqOffsetLens = seqOffset->GetSize();
    int64_t batchSize = static_cast<int64_t>(seqOffsetLens - 1);
    OPS_LOG_E_IF(batchSize > MAX_BATCH_SIZE, context, return false,
        "batch size is over limit %d", MAX_BATCH_SIZE);

    if (!QKVShapeCheck(context, QKV_DIM)) {
        return false;
    }
    auto queryShape = context->GetInputShape(INDEX_T::INDEX_0)->GetStorageShape();
    int64_t headNum = queryShape.GetDim(INDEX_T::INDEX_1);
    int64_t headDim = queryShape.GetDim(INDEX_T::INDEX_2);
    int64_t seqLens = tiling.get_maxSeqLen();

    tiling.set_batchSize(batchSize);
    tiling.set_headNum(headNum);
    tiling.set_dim(headDim);
    tiling.set_seqLen(seqLens);

    OPS_LOG_E_IF(!GeneralShapeCheck(batchSize, seqLens, headNum, headDim),
                 context, return false, "Jagged Shape Check failed");
    return true;
}

static void CallBlockAssign(
    uint32_t *seqOffsets,
    uint32_t coreNum,
    std::vector<BlockTaskInfo> &workTasks,
    std::vector<int> &workLoads,
    optiling::HstuDenseForwardFuxiTilingData &tiling)
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

bool TilingPolicyJagged::TilingCore(gert::TilingContext* context, optiling::HstuDenseForwardFuxiTilingData &tiling)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return false);

    const auto seqOffset = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_T::INDEX_4);
    OPS_LOG_E_IF_NULL("seqOffset", seqOffset, return false);

    auto *seqOffsetData = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(seqOffset->GetData()));
    size_t seq_offset_lens = seqOffset->GetSize();
    if (seq_offset_lens > (MAX_BATCH_SIZE + 1)) {
        OPS_LOG_E(context, "seq_offset_lens exceed limit %d\n", MAX_BATCH_SIZE + 1);
        return false;
    }

    std::vector<BlockTaskInfo> workTasks;
    std::vector<int> workLoads;

    uint32_t seqOffsets[MAX_BATCH_SIZE + 1] = {0};
    for (auto i = 0; i < seq_offset_lens; i++) {
        seqOffsets[i] = seqOffsetData[i];
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum > MAX_AIV_NUM, OPS_LOG_E("", "vecCoreNum %d should be < %d\n", coreNum, MAX_AIV_NUM),
              return ge::GRAPH_FAILED);
    CallBlockAssign(seqOffsets, coreNum, workTasks, workLoads, tiling);

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

bool TilingPolicyJagged::TilingMatmul(gert::TilingContext* context,
    optiling::HstuDenseForwardFuxiTilingData &tiling, matmul_tiling::DataType dataType)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int64_t dim = tiling.get_dim();
    tiling.set_blockHeight(BLOCK_HEIGHT);

    // apply qk
    matmul_tiling::MatmulApiTiling qkMatmul(ascendPlatform);
    qkMatmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    qkMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    qkMatmul.SetOrgShape(BLOCK_HEIGHT, BLOCK_HEIGHT, dim);
    qkMatmul.SetShape(BLOCK_HEIGHT, BLOCK_HEIGHT, dim);
    qkMatmul.SetBias(false);
    qkMatmul.SetBufferSpace(-1, -1, -1);

    // sv
    matmul_tiling::MatmulApiTiling svMatmul(ascendPlatform);
    svMatmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    svMatmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    svMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    svMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    svMatmul.SetOrgShape(BLOCK_HEIGHT, dim, BLOCK_HEIGHT);
    svMatmul.SetShape(BLOCK_HEIGHT, dim, BLOCK_HEIGHT);
    svMatmul.SetBias(false);
    svMatmul.SetBufferSpace(-1, -1, -1);

    // tv
    matmul_tiling::MatmulApiTiling tvMatmul(ascendPlatform);
    tvMatmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    tvMatmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    tvMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tvMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    tvMatmul.SetOrgShape(BLOCK_HEIGHT, dim, BLOCK_HEIGHT);
    tvMatmul.SetShape(BLOCK_HEIGHT, dim, BLOCK_HEIGHT);
    tvMatmul.SetBias(false);
    tvMatmul.SetBufferSpace(-1, -1, -1);

    matmul_tiling::MatmulApiTiling pvMatmul(ascendPlatform);
    pvMatmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    pvMatmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    pvMatmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    pvMatmul.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);

    pvMatmul.SetOrgShape(BLOCK_HEIGHT, dim, BLOCK_HEIGHT);
    pvMatmul.SetShape(BLOCK_HEIGHT, dim, BLOCK_HEIGHT);
    pvMatmul.SetBias(false);
    pvMatmul.SetBufferSpace(-1, -1, -1);

    if ((qkMatmul.GetTiling(tiling.qkMatmul) == -1) || (svMatmul.GetTiling(tiling.svMatmul) == -1) ||
        (tvMatmul.GetTiling(tiling.tvMatmul) == -1) || (pvMatmul.GetTiling(tiling.pvMatmul) == -1)) {
        OPS_LOG_E(context, "GetTiling failed.");
        return false;
    }

    auto findResult = matmul_tiling::DTYPE_BYTE_TAB.find(dataType);
    if (findResult == matmul_tiling::DTYPE_BYTE_TAB.end()) {
        OPS_LOG_E("", "dataType not in DTYPE_BYTE_TAB");
        return ge::GRAPH_FAILED;
    }
    int dataTypeLength = findResult->second;
    if (!CheckBaseMNK(tiling.qkMatmul, dataTypeLength, sizeof(float)) ||
        !CheckBaseMNK(tiling.svMatmul, dataTypeLength, sizeof(float)) ||
        !CheckBaseMNK(tiling.pvMatmul, dataTypeLength, sizeof(float)) ||
        !CheckBaseMNK(tiling.tvMatmul, dataTypeLength, sizeof(float))) {
        return ge::GRAPH_FAILED;
    }
    return true;
}

bool TilingPolicyJagged::TilingHeighLevelApi(gert::TilingContext* context,
    optiling::HstuDenseForwardFuxiTilingData &tiling)
{
    int64_t dim = tiling.get_dim();

    matmul_tiling::DataType dataType;
    OPS_LOG_E_IF_NULL("query", context->GetInputTensor(0), return ge::GRAPH_FAILED);
    ge::DataType qTypeGe = context->GetInputTensor(0)->GetDataType();
    if (qTypeGe == ge::DataType::DT_FLOAT) {
        dataType = matmul_tiling::DataType::DT_FLOAT;
    } else if (qTypeGe == ge::DataType::DT_FLOAT16) {
        dataType = matmul_tiling::DataType::DT_FLOAT16;
    } else {
        dataType = matmul_tiling::DataType::DT_BFLOAT16;
    }

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);

    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    size_t coreNum = ascendPlatform.GetCoreNumAic();

    int64_t oneBlockMidElem = BLOCK_HEIGHT * BLOCK_HEIGHT * COMPUTE_PIPE_NUM * OUTPUT_DIM2_TIMES_3;
    int64_t oneCoreMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidElem;

    int64_t oneBlockMidTransElem = BLOCK_HEIGHT * dim * TRANS_PIPE_NUM * OUTPUT_DIM2_TIMES_3;
    int64_t oneCoreTransMidElem = coreNum * VCORE_NUM_IN_ONE_AIC * oneBlockMidTransElem;

    int64_t workspaceSize = (oneCoreMidElem + oneCoreTransMidElem) * sizeof(float);
    currentWorkspace[0] = workspaceSize + systemWorkspacesSize;

    OPS_LOG_E_IF(!TilingMatmul(context, tiling, dataType), context, return false, "TilingMatmul failed");

    tiling.set_qkBaseM(tiling.qkMatmul.get_baseM());
    tiling.set_qkBaseN(tiling.qkMatmul.get_baseN());

    tiling.set_svBaseM(tiling.svMatmul.get_baseM());
    tiling.set_svBaseN(tiling.svMatmul.get_baseN());

    tiling.set_tvBaseM(tiling.tvMatmul.get_baseM());
    tiling.set_tvBaseN(tiling.tvMatmul.get_baseN());

    tiling.set_pvBaseM(tiling.pvMatmul.get_baseM());
    tiling.set_pvBaseN(tiling.pvMatmul.get_baseN());

    return true;
}

}