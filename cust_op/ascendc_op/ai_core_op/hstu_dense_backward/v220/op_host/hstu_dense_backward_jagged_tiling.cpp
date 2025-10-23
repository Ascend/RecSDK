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


#include <vector>
#include <numeric>

#include "common_host.h"
#include "register/op_def_registry.h"
#include "hstu_dense_backward_jagged_tiling.h"

namespace {
struct BlockTaskInfo {
    uint32_t startBlockId = 0;
    uint32_t endBlockId = 0;

    friend std::ostream &operator<<(std::ostream &os, const BlockTaskInfo &blockTask)
    {
        return os << "startBlockId:" << blockTask.startBlockId << " " << "endBlockId:" << blockTask.endBlockId << " ";
    }
};

class BlockTaskAssign {
public:
    BlockTaskAssign(uint32_t *seqOffsets, uint32_t coreNum, uint32_t blockLen, uint32_t batchSize, uint32_t headNum)
    {
        this->seqOffsets = seqOffsets;
        this->coreNum = coreNum;
        this->blockLen = blockLen;
        this->batchSize = batchSize;
        this->headNum = headNum;
    }

    void PreInit(std::vector<BlockTaskInfo> &workTasks, std::vector<int> &workLoads, std::vector<int64_t> &blockNumber)
    {
        workTasks.resize(this->coreNum);
        workLoads.resize(this->coreNum, 0);

        for (auto batchId = 0; batchId < batchSize; batchId++) {
            auto batchBlockSize = this->seqOffsets[batchId + 1] - this->seqOffsets[batchId];
            
            for (auto headId = 0; headId < headNum; headId++) {
                blockNumber[batchId * headNum + headId] = (batchBlockSize + blockLen - 1) / blockLen;
            }
        }
    }

    void Compute(std::vector<BlockTaskInfo> &workTasks, std::vector<int> &workLoads)
    {
        uint32_t totalBatchSize = batchSize * headNum;
        std::vector<int64_t> blockNumber(totalBatchSize, 0);
        PreInit(workTasks, workLoads, blockNumber);

        int64_t totalTaskNumber = 0;
        totalTaskNumber = std::accumulate(blockNumber.begin(), blockNumber.end(), totalTaskNumber,
                                          [](int64_t val, int64_t x) { return val + x * x; });

        int64_t eachCoreTaskNumLimit = (totalTaskNumber + this->coreNum - 1) / this->coreNum;

        uint32_t batchId = 0;
        uint32_t batchTaskNum = blockNumber[batchId];
        uint32_t processBlockNum = 0;
        for (int i = 0; i < this->coreNum && batchId < totalBatchSize; i++) {
            BlockTaskInfo blockTask;
            blockTask.startBlockId = processBlockNum;

            while (workLoads[i] < eachCoreTaskNumLimit) {
                workLoads[i] += batchTaskNum;
                processBlockNum++;
                blockNumber[batchId]--;
                if (blockNumber[batchId] == 0 && batchId + 1 >= totalBatchSize) {
                    batchId++;
                    break;
                }
                if (blockNumber[batchId] == 0) {
                    batchId++;
                    batchTaskNum = blockNumber[batchId];
                }
            }

            blockTask.endBlockId = processBlockNum;
            workTasks[i] = blockTask;
        }
    }

    void ComputeCausal(std::vector<BlockTaskInfo> &workTasks, std::vector<int> &workLoads, bool isCol)
    {
        uint32_t totalBatchSize = batchSize * headNum;
        std::vector<int64_t> blockNumber(totalBatchSize, 0);
        PreInit(workTasks, workLoads, blockNumber);

        int64_t totalTaskNumber = 0;
        constexpr int two = 2;
        totalTaskNumber = std::accumulate(blockNumber.begin(), blockNumber.end(), totalTaskNumber,
                                          [](int64_t val, int64_t x) { return val + x * (x + 1) / two; });

        int64_t eachCoreTaskNumLimit = (totalTaskNumber + this->coreNum - 1) / this->coreNum;

        uint32_t batchId = 0;
        uint32_t processBlockNum = 0;
        uint32_t taskNum = isCol ? blockNumber[0] : 1;
        for (int i = 0; i < this->coreNum && batchId < totalBatchSize; i++) {
            BlockTaskInfo blockTask;
            blockTask.startBlockId = processBlockNum;

            while (workLoads[i] < eachCoreTaskNumLimit) {
                workLoads[i] += taskNum;
                taskNum = isCol ? taskNum - 1 : taskNum + 1;
                processBlockNum++;
                blockNumber[batchId]--;
                if (blockNumber[batchId] == 0 && batchId + 1 >= totalBatchSize) {
                    batchId++;
                    break;
                }
                if (blockNumber[batchId] == 0) {
                    batchId++;
                    taskNum = isCol ? blockNumber[batchId] : 1;
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
} // namespace

namespace optiling {
ge::graphStatus GetJaggedAttrsInfo(const gert::RuntimeAttrs *attrs, HstuDenseBackwardTilingData &tiling)
{
    const int32_t *maskType = attrs->GetAttrPointer<int32_t>(INDEX_T::INDEX_1);
    OPS_CHECK_PTR_NULL(maskType, return ge::GRAPH_FAILED);

    const int32_t *maxSeqLen = attrs->GetAttrPointer<int32_t>(INDEX_T::INDEX_2);
    OPS_CHECK_PTR_NULL(maxSeqLen, return ge::GRAPH_FAILED);

    const float *siluScale = attrs->GetAttrPointer<float>(INDEX_T::INDEX_3);
    OPS_CHECK_PTR_NULL(siluScale, return ge::GRAPH_FAILED);

    const auto seqOffset = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_T::INDEX_4);
    OPS_CHECK_PTR_NULL(seqOffset, return ge::GRAPH_FAILED);

    auto *seqOffsetData = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(seqOffset->GetData()));
    int seqOffsetLens = seqOffset->GetSize();
    int64_t batchSize = GetBatchSizeFromJaggedOffset(seqOffsetData, seqOffsetLens);
    OPS_CHECK((batchSize == 0 || batchSize > MAX_BATCH_SIZE),
        OPS_LOG_E("", "batchSize limit (0, %d], but get %lld\n", MAX_BATCH_SIZE, batchSize), return ge::GRAPH_FAILED);

    uint32_t seqOffsets[MAX_BATCH_SIZE + 1] = {0};
    for (auto i = 0; i < batchSize + 1; i++) {
        seqOffsets[i] = seqOffsetData[i];
    }
    
    tiling.set_maskType(*maskType);
    tiling.set_maxSeqLen(*maxSeqLen);
    tiling.set_siluScale(*siluScale);
    tiling.set_seqOffset(seqOffsets);
    tiling.set_batchSize(batchSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetJaggedBasicShapeInfo(gert::TilingContext *context, HstuDenseBackwardTilingData &tiling)
{
    int64_t maxSeqLen = tiling.get_maxSeqLen();

    auto gradShape = context->GetInputShape(INDEX_T::INDEX_0)->GetStorageShape();

    OPS_CHECK(gradShape.GetDimNum() != JAGGED_GRAD_DIM_NUM,
                OPS_LOG_E("", "hstu jagged backward only support input with dim %d\n", JAGGED_GRAD_DIM_NUM),
                return ge::GRAPH_FAILED);

    int64_t seqLen = gradShape.GetDim(INDEX_T::INDEX_0);
    int64_t headNum = gradShape.GetDim(INDEX_T::INDEX_1);
    int64_t headDim = gradShape.GetDim(INDEX_T::INDEX_2);
    int64_t biasGradSeqLen = 0;
    auto attnBiasGradShape = context->GetOutputShape(INDEX_T::INDEX_3);
    if (attnBiasGradShape != nullptr) {
        biasGradSeqLen = attnBiasGradShape->GetStorageShape().GetDim(INDEX_T::INDEX_2);
        OPS_CHECK(biasGradSeqLen < maxSeqLen,
            OPS_LOG_E("", "attnBiasGrad get seqLen less than maxSeqLen\n"),
            return ge::GRAPH_FAILED);
    } else {
        biasGradSeqLen = AlignUp(maxSeqLen, static_cast<int64_t>(BLOCK_256));
        OPS_CHECK((biasGradSeqLen == 0), OPS_LOG_E("", "attnBiasGrad get seqLen error\n"), return ge::GRAPH_FAILED);
    }

    tiling.set_seqLen(seqLen);
    tiling.set_headNum(headNum);
    tiling.set_headDim(headDim);
    tiling.set_biasGradSeqLen(biasGradSeqLen);

    tiling.set_isNormal(0);

    int64_t batchSize = tiling.get_batchSize();
    OPS_CHECK(!BasicShapeCheck(batchSize, maxSeqLen, headNum, headDim),
        OPS_LOG_E("", "jagged shape check failed\n"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InitJaggedTilingKey(gert::TilingContext *context, HstuDenseBackwardTilingData &tiling)
{
    int64_t dataTypeLength = 0;
    OPS_LOG_E_IF_NULL("grad", context->GetInputTensor(INDEX_T::INDEX_0), return ge::GRAPH_FAILED);
    ge::DataType gradType = context->GetInputTensor(INDEX_T::INDEX_0)->GetDataType();
    if (gradType == ge::DataType::DT_FLOAT) {
        dataTypeLength = DATA_TYPE_LENGTH_FLOAT;
        context->SetTilingKey(JAGGED_FLOAT_TILING_KEY);
        tiling.set_blockHeight(BLOCK_128);
    } else if (gradType == ge::DataType::DT_FLOAT16) {
        dataTypeLength = DATA_TYPE_LENGTH_FLOAT16;
        context->SetTilingKey(JAGGED_FLOAT16_TILING_KEY);
        tiling.set_blockHeight(BLOCK_256);
    } else if (gradType == ge::DataType::DT_BF16) {
        dataTypeLength = DATA_TYPE_LENGTH_FLOAT16;
        context->SetTilingKey(JAGGED_BF16_TILING_KEY);
        tiling.set_blockHeight(BLOCK_256);
    } else {
        OPS_LOG_E("", "invalid datatype, only support float/fp16/bf16");
        return ge::GRAPH_FAILED;
    }
    tiling.set_dataTypeLength(dataTypeLength);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingCore(gert::TilingContext *context,
                           HstuDenseBackwardTilingData &tiling)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t vecCoreNum = ascendPlatform.GetCoreNumAiv();
    OPS_CHECK(vecCoreNum > MAX_AIV_NUM, OPS_LOG_E("", "vecCoreNum %d should be < %d\n", vecCoreNum, MAX_AIV_NUM),
              return ge::GRAPH_FAILED);
    uint32_t batchSize = tiling.get_batchSize();
    uint32_t headNum = tiling.get_headNum();
    uint32_t blockHeight = tiling.get_blockHeight();
    auto maskType = tiling.get_maskType();
    auto seqOffsets = tiling.get_seqOffset();

    std::vector<BlockTaskInfo> colWorkTasks;
    std::vector<int> colWorkLoads;
    std::vector<BlockTaskInfo> rowWorkTasks;
    std::vector<int> rowWorkLoads;

    auto taskAssigner = BlockTaskAssign(seqOffsets, vecCoreNum, blockHeight, batchSize, headNum);
    if (IfMask(maskType, MaskType::MASK_TRIL)) {
        taskAssigner.ComputeCausal(colWorkTasks, colWorkLoads, true);
        taskAssigner.ComputeCausal(rowWorkTasks, rowWorkLoads, false);
    } else {
        taskAssigner.Compute(colWorkTasks, colWorkLoads);
        rowWorkTasks = colWorkTasks;
        rowWorkLoads = colWorkLoads;
    }
    uint32_t startColBlockId[MAX_AIV_NUM] = {0};
    uint32_t endColBlockId[MAX_AIV_NUM] = {0};
    uint32_t startRowBlockId[MAX_AIV_NUM] = {0};
    uint32_t endRowBlockId[MAX_AIV_NUM] = {0};
    for (auto i = 0; i < vecCoreNum; i++) {
        startColBlockId[i] = colWorkTasks[i].startBlockId;
        endColBlockId[i] = colWorkTasks[i].endBlockId;
        startRowBlockId[i] = rowWorkTasks[i].startBlockId;
        endRowBlockId[i] = rowWorkTasks[i].endBlockId;
    }
    tiling.set_eachCoreStartColBlockId(startColBlockId);
    tiling.set_eachCoreEndColBlockId(endColBlockId);
    tiling.set_eachCoreStartRowBlockId(startRowBlockId);
    tiling.set_eachCoreEndRowBlockId(endRowBlockId);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingJaggedFunc(gert::TilingContext *context,
                                 const gert::RuntimeAttrs *attrs,
                                 HstuDenseBackwardTilingData &tiling)
{
    OPS_CHECK(GetJaggedAttrsInfo(attrs, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "JaggedTiling GetJaggedAttrsInfo failed\n"), return ge::GRAPH_FAILED);
    
    OPS_CHECK(GetJaggedBasicShapeInfo(context, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "JaggedTiling GetJaggedBasicShapeInfo failed\n"), return ge::GRAPH_FAILED);
    
    OPS_CHECK(CheckMaskTypeAndBias(context, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "JaggedTiling CheckMaskTypeAndBias failed\n"), return ge::GRAPH_FAILED);

    OPS_CHECK(InitJaggedTilingKey(context, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "JaggedTiling InitJaggedTilingKey failed\n"), return ge::GRAPH_FAILED);

    OPS_CHECK(TilingCore(context, tiling) == ge::GRAPH_FAILED,
                OPS_LOG_E("", "JaggedTiling TilingCore failed\n"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus JaggedInferShape(gert::InferShapeContext *context)
{
    const gert::Shape *qShape = context->GetInputShape(INDEX_T::INDEX_1);
    OPS_CHECK_PTR_NULL(qShape, return ge::GRAPH_FAILED);

    // q_grad、k_grad、v_grad的shape与q一致
    gert::Shape *qGradShape = context->GetOutputShape(INDEX_T::INDEX_0);
    OPS_CHECK_PTR_NULL(qGradShape, return ge::GRAPH_FAILED);
    qGradShape->SetDimNum(qShape->GetDimNum());

    gert::Shape *kGradShape = context->GetOutputShape(INDEX_T::INDEX_1);
    OPS_CHECK_PTR_NULL(kGradShape, return ge::GRAPH_FAILED);
    kGradShape->SetDimNum(qShape->GetDimNum());

    gert::Shape *vGradShape = context->GetOutputShape(INDEX_T::INDEX_2);
    OPS_CHECK_PTR_NULL(vGradShape, return ge::GRAPH_FAILED);
    vGradShape->SetDimNum(qShape->GetDimNum());

    for (size_t i = 0; i < qShape->GetDimNum(); i++) {
        qGradShape->SetDim(i, qShape->GetDim(i));
        kGradShape->SetDim(i, qShape->GetDim(i));
        vGradShape->SetDim(i, qShape->GetDim(i));
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, return ge::GRAPH_FAILED);
    const int32_t *maxSeqLen = attrs->GetAttrPointer<int32_t>(INDEX_T::INDEX_2);
    OPS_CHECK_PTR_NULL(maxSeqLen, return ge::GRAPH_FAILED);

    const auto seqOffset = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_T::INDEX_4);
    OPS_CHECK_PTR_NULL(seqOffset, return ge::GRAPH_FAILED);

    auto *seqOffsetData = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(seqOffset->GetData()));
    int seqOffsetLens = seqOffset->GetSize();
    int64_t batchSize = GetBatchSizeFromJaggedOffset(seqOffsetData, seqOffsetLens);
    OPS_CHECK((batchSize == 0 || batchSize > MAX_BATCH_SIZE),
        OPS_LOG_E("", "batchSize limit (0, %d], but get %lld\n", MAX_BATCH_SIZE, batchSize), return ge::GRAPH_FAILED);

    gert::Shape *attnBiasGradShape = context->GetOutputShape(INDEX_T::INDEX_3);
    if (attnBiasGradShape != nullptr) {
        attnBiasGradShape->SetDimNum(BIAS_DIM_NUM);
        attnBiasGradShape->SetDim(INDEX_T::INDEX_0, batchSize);
        attnBiasGradShape->SetDim(INDEX_T::INDEX_1, qShape->GetDim(1));
        attnBiasGradShape->SetDim(INDEX_T::INDEX_2, *maxSeqLen);
        attnBiasGradShape->SetDim(INDEX_T::INDEX_3, *maxSeqLen);
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
