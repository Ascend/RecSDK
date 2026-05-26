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

#include "concat_jagged_tensor_tiling.h"
#include "register/op_def_registry.h"
#include "register/register.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"
#include <algorithm>

constexpr uint32_t RESERVED_UB_SIZE = 1024;
constexpr uint32_t WORKSPACE_SIZE = 4 * 1024 * 1024;
constexpr uint32_t ALIGN = 32;
constexpr uint32_t MAX_TENSOR_SIZE = 24 * 1024;  // 当Tensor长度大于此值时调整分核策略性能更优

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取硬件信息
    uint64_t ubSize = 0;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t maxUbSize = ubSize - RESERVED_UB_SIZE;
    uint64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        OPS_LOG_E(context, "[ERROR] need more than 0 ai core");
        return ge::GRAPH_FAILED;
    }
    // 计算偏移地址和大小
    auto offsets = context->GetAttrs()->GetListInt(0);
    int32_t offsetLen = *context->GetAttrs()->GetInt(1);
    int32_t jtNum = *context->GetAttrs()->GetInt(2);
    int32_t nPrefixFromRight = *context->GetAttrs()->GetInt(3);

    uint32_t sliceSize[MAX_SLICE_SIZE + 1] = {0};  // 当nPrefixFromRight > 0时,切片个数+1。
    uint32_t inputOffsetBegin[MAX_SLICE_SIZE + 1] = {0};
    uint32_t outputOffsetBegin[MAX_SLICE_SIZE + 1] = {0};

    uint32_t indexSliceForRight = offsetLen - 1;
    uint32_t outputOffset = 0;
    const int64_t* offsetsData = offsets->GetData();

    uint32_t maxTensorLen = std::max<int32_t>(offsetsData[indexSliceForRight], offsetsData[2 * offsetLen - 1]);

    // 判断是否需要优化切片策略：Tensor长度较长且分片较小时
    const bool needAdjustSliceStrategy = (offsetLen < coreNum) && (maxTensorLen > MAX_TENSOR_SIZE);

    if (nPrefixFromRight == 0) {
        // [A] + [B]
        for (int i = 0; i < offsetLen - 1; i++) {
            // A[:indexSliceForRight-1]
            inputOffsetBegin[i] = offsetsData[i];
            sliceSize[i] = offsetsData[i + 1] - offsetsData[i];
            // B[indexSliceForRight:]
            inputOffsetBegin[i + indexSliceForRight] = offsetsData[offsetLen + i];
            sliceSize[i + indexSliceForRight] = offsetsData[offsetLen + i + 1] - offsetsData[offsetLen + i];
        }
    } else {
        // [nPrefixFromB] + [A] + [remainB]
        // first offset [nPrefixFromB]
        int32_t bLength = offsetsData[offsetLen + 1] - offsetsData[offsetLen];
        // nPrefixFromRight
        sliceSize[indexSliceForRight] = std::min<int32_t>(nPrefixFromRight, bLength);
        inputOffsetBegin[indexSliceForRight] = 0;
        // [A] + ([remainB] + [nPrefixFromB])
        for (int i = 0; i < offsetLen - 2; i++) {
            // [A]
            inputOffsetBegin[i] = offsetsData[i];
            sliceSize[i] = offsetsData[i + 1] - offsetsData[i];
            // [remainB]
            int32_t remainB = std::max<int32_t>((bLength - nPrefixFromRight), 0);
            // [next nPrefixFromB]
            bLength = offsetsData[offsetLen + i + 2] - offsetsData[offsetLen + i + 1];
            int32_t prefixNextB = std::min<int32_t>(nPrefixFromRight, bLength);

            inputOffsetBegin[i + offsetLen] = inputOffsetBegin[offsetLen + i - 1] + sliceSize[i + offsetLen - 1];
            sliceSize[i + offsetLen] = remainB + prefixNextB;
        }
        // last offset [A] + [remainB]
        inputOffsetBegin[indexSliceForRight - 1] = offsetsData[indexSliceForRight - 1];
        sliceSize[indexSliceForRight - 1] = offsetsData[indexSliceForRight] - offsetsData[indexSliceForRight - 1];
        // B
        int32_t LastBLength = offsetsData[2 * offsetLen - 1] - offsetsData[2 * offsetLen - 2];
        int32_t lastRemainB = std::max<int32_t>((LastBLength - nPrefixFromRight), 0);

        inputOffsetBegin[2 * offsetLen - 2] = inputOffsetBegin[2 * offsetLen - 3] + sliceSize[2 * offsetLen - 3];
        sliceSize[2 * offsetLen - 2] = lastRemainB;
    }

    uint32_t batchSize = (nPrefixFromRight == 0) ? (jtNum * (offsetLen - 1)) : (jtNum * (offsetLen - 1) + 1);

    if (nPrefixFromRight == 0) {
        for (int i = 0; i < offsetLen - 1; i++) {
            outputOffsetBegin[i] = outputOffset;
            outputOffset += sliceSize[i];
            outputOffsetBegin[i + offsetLen - 1] = outputOffset;
            outputOffset += sliceSize[i + offsetLen - 1];
        }
    } else {
        outputOffsetBegin[offsetLen - 1] = outputOffset;
        outputOffset += sliceSize[offsetLen - 1];
        for (int i = 0; i < offsetLen - 1; i++) {
            outputOffsetBegin[i] = outputOffset;
            outputOffset += sliceSize[i];
            outputOffsetBegin[i + offsetLen] = outputOffset;
            outputOffset += sliceSize[i + offsetLen];
        }
    }

    uint32_t oneSliceSize = maxTensorLen / coreNum;
    uint32_t resetIndexSliceForRight = indexSliceForRight;

    std::vector<uint32_t> readjustSliceSize;
    std::vector<uint32_t> readjustInputOffsetBegin;
    std::vector<uint32_t> readjustOutputOffsetBegin;

    // Tensor长度较长且分片较小时，优化切片策略。
    if (needAdjustSliceStrategy) {
        for (int i = 0; i < batchSize; ++i) {
            if (i == indexSliceForRight) {
                resetIndexSliceForRight = readjustSliceSize.size();
            }

            uint32_t inputSize = sliceSize[i];
            uint32_t inputOffset = inputOffsetBegin[i];
            uint32_t outputOffset = outputOffsetBegin[i];

            if (inputSize <= oneSliceSize) {
                // 不需要切分，直接添加
                readjustSliceSize.push_back(inputSize);
                readjustInputOffsetBegin.push_back(inputOffset);
                readjustOutputOffsetBegin.push_back(outputOffset);
            } else {
                // 需要切分成多个分片
                uint32_t remaining = inputSize;
                uint32_t currentInputOffset = inputOffset;
                uint32_t currentOutputOffset = outputOffset;

                while (remaining > 0) {
                    uint32_t chunkSize = std::min(remaining, oneSliceSize);
                    readjustSliceSize.push_back(chunkSize);
                    readjustInputOffsetBegin.push_back(currentInputOffset);
                    readjustOutputOffsetBegin.push_back(currentOutputOffset);
                    currentInputOffset += chunkSize;
                    currentOutputOffset += chunkSize;
                    remaining -= chunkSize;
                }
            }
        }
    }

    uint64_t resetBatchSize = batchSize;
    if (needAdjustSliceStrategy) {
        resetBatchSize = readjustSliceSize.size();
    }

    // 计算切分逻辑
    uint64_t formerCore = 0;
    uint64_t tailCore = 0;
    uint64_t batchNumInTail = 0;
    uint64_t batchNumInFormer = 0;
    if (resetBatchSize < coreNum) {
        formerCore = resetBatchSize;
        batchNumInFormer = 1;
    } else {
        formerCore = resetBatchSize % coreNum;
        tailCore = coreNum - formerCore;
        batchNumInTail = resetBatchSize / coreNum;
        batchNumInFormer = batchNumInTail + 1;
    }

    // 计算copy最大行数
    OPS_CHECK_PTR_NULL(context->GetRequiredInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputXDesc", context->GetInputDesc(0), return ge::GRAPH_FAILED);
    uint32_t inputColSize = context->GetRequiredInputShape(0)->GetOriginShape().GetDim(1);
    auto inputXDesc = context->GetInputDesc(0);
    ge::DataType xDtype{ge::DT_FLOAT};
    xDtype = inputXDesc->GetDataType();
    uint32_t ubMaxLength = maxUbSize / ge::GetSizeByDataType(xDtype);
    ubMaxLength = ubMaxLength & ~(ALIGN - 1);

    // 设置tiling
    ConcatJaggedTensorTilingData tiling;
    tiling.set_jtNum(jtNum);
    tiling.set_inputColSize(inputColSize);
    tiling.set_ubMaxLength(ubMaxLength);
    tiling.set_formerCore(formerCore);
    tiling.set_tailCore(tailCore);
    tiling.set_batchNumInTail(batchNumInTail);
    tiling.set_batchNumInFormer(batchNumInFormer);

    if (needAdjustSliceStrategy) {
        // 填充剩余元素为 0
        while (readjustSliceSize.size() < MAX_SLICE_SIZE + 1) {
            readjustSliceSize.push_back(0);
            readjustInputOffsetBegin.push_back(0);
            readjustOutputOffsetBegin.push_back(0);
        }
        tiling.set_indexSliceForRight(resetIndexSliceForRight);
        tiling.set_inputOffsetBegin(readjustInputOffsetBegin.data());
        tiling.set_sliceSize(readjustSliceSize.data());
        tiling.set_outputOffsetBegin(readjustOutputOffsetBegin.data());

    } else {
        tiling.set_indexSliceForRight(indexSliceForRight);
        tiling.set_inputOffsetBegin(inputOffsetBegin);
        tiling.set_sliceSize(sliceSize);
        tiling.set_outputOffsetBegin(outputOffsetBegin);
    }

    context->SetBlockDim(coreNum);
    OPS_CHECK_PTR_NULL(context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
    currentWorkspace[0] = WORKSPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto offsets = context->GetAttrs()->GetListInt(0);
    int32_t offsetLen = *context->GetAttrs()->GetInt(1);
    int32_t jtNum = *context->GetAttrs()->GetInt(2);
    int64_t outputRow = 0;
    for (int i = 0; i < jtNum; i++) {
        outputRow += offsets->GetData()[i * offsetLen];
    }
    const gert::Shape* x_shape = context->GetInputShape(0);
    if (x_shape == nullptr) {
        OPS_LOG_E("[ERROR]", "InputShape should not be nullptr.");
        return ge::GRAPH_FAILED;
    }
    uint32_t dimNum = x_shape->GetDimNum();
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (y_shape == nullptr) {
        OPS_LOG_E("[ERROR]", "OutputShape should not be nullptr.");
        return ge::GRAPH_FAILED;
    }
    y_shape->SetDimNum(dimNum);
    y_shape->SetDim(0, outputRow);
    y_shape->SetDim(1, x_shape->GetDim(1));
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ConcatJaggedTensor : public OpDef {
public:
    explicit ConcatJaggedTensor(const char* name) : OpDef(name)
    {
        this->Input("values")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("offsets").ListInt();
        this->Attr("offsetLen").Int(0);
        this->Attr("jtNum").Int(0);
        this->Attr("nPrefixFromRight").Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend310p");
#ifdef SUPPORT_950
        this->AICore().AddConfig("ascend950");
#endif
    }
};
OP_ADD(ConcatJaggedTensor);
}  // namespace ops
