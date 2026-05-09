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
==============================================================================*/

#include <cstdint>
#include "gen_position_ids_reverse_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace {
    constexpr int32_t SEQLEN_INDEX = 0;
    constexpr int32_t SEQLEN_OFFSETS_INDEX = 1;
    constexpr int32_t RSPOS_INDEX = 2;
    constexpr int32_t OUTPUT_INDEX = 0;
    constexpr int32_t ATTR_BATCH_SIZE = 0;
    constexpr int32_t DIM0 = 0;
}  // namespace

namespace optiling {

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("seqlenShape", context->GetInputShape(SEQLEN_INDEX), return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("seqlenTensor", context->GetInputTensor(SEQLEN_INDEX), return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("seqlenOffsetsShape", context->GetInputShape(SEQLEN_OFFSETS_INDEX), return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("seqlenOffsetsTensor", context->GetInputTensor(SEQLEN_OFFSETS_INDEX), return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("rsposShape", context->GetInputShape(RSPOS_INDEX), return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("rsposTensor", context->GetInputTensor(RSPOS_INDEX), return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("attrs", context->GetAttrs(), return ge::GRAPH_FAILED);
        if (context->GetRawTilingData() == nullptr) {
            return ge::GRAPH_FAILED;
        }

        const gert::Shape& seqlenShape = context->GetInputShape(SEQLEN_INDEX)->GetOriginShape();
        const gert::Shape& seqlenOffsetsShape = context->GetInputShape(SEQLEN_OFFSETS_INDEX)->GetOriginShape();
        const gert::Shape& rsposShape = context->GetInputShape(RSPOS_INDEX)->GetOriginShape();

        const ge::DataType seqlenDtype = context->GetInputTensor(SEQLEN_INDEX)->GetDataType();
        const ge::DataType seqlenOffDtype = context->GetInputTensor(SEQLEN_OFFSETS_INDEX)->GetDataType();
        const ge::DataType rsposDtype = context->GetInputTensor(RSPOS_INDEX)->GetDataType();
        if (seqlenDtype != ge::DT_INT32 || seqlenOffDtype != ge::DT_INT32 || rsposDtype != ge::DT_INT32) {
            OPS_LOG_E("Tiling", "seqlen, seqlen_offsets, rspos must be int32.");
            return ge::GRAPH_FAILED;
        }

        OPS_LOG_E_IF(seqlenShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED, "[ERROR] seqlen must be 1D.");
        OPS_LOG_E_IF(seqlenOffsetsShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED,
                     "[ERROR] seqlen_offsets must be 1D.");
        OPS_LOG_E_IF(rsposShape.GetDimNum() != 1, context, return ge::GRAPH_FAILED, "[ERROR] rspos must be 1D.");

        const auto* attrs = context->GetAttrs();
        int64_t seqlenDim0 = seqlenShape.GetDim(DIM0);
        int64_t batchSize = 0;
        const int64_t* pBatchI64 = attrs->GetAttrPointer<int64_t>(ATTR_BATCH_SIZE);
        const int32_t* pBatchI32 = attrs->GetAttrPointer<int32_t>(ATTR_BATCH_SIZE);
        if (pBatchI64 != nullptr) {
            batchSize = *pBatchI64;
        } else if (pBatchI32 != nullptr) {
            batchSize = static_cast<int64_t>(*pBatchI32);
        } else {
            OPS_LOG_E("Tiling", "batchSize attr must be int32 or int64.");
            return ge::GRAPH_FAILED;
        }

        OPS_LOG_E_IF(
                seqlenDim0 != batchSize, context, return ge::GRAPH_FAILED,
                "[ERROR] seqlen dim0 must equal attr batchSize, got seqlen %lld vs batchSize %lld.", (long long)seqlenDim0,
                (long long)batchSize);
        int64_t offsetsLen = seqlenOffsetsShape.GetDim(DIM0);
        int64_t rsposLen = rsposShape.GetDim(DIM0);

        OPS_LOG_E_IF(batchSize < 0, context, return ge::GRAPH_FAILED, "[ERROR] batchSize must be >= 0, got %lld.",
                     (long long)batchSize);
        OPS_LOG_E_IF(offsetsLen != batchSize + 1, context, return ge::GRAPH_FAILED,
                     "[ERROR] seqlen_offsets length must be batchSize+1, expected %lld, got %lld.",
                     (long long)(batchSize + 1), (long long)offsetsLen);
        OPS_LOG_E_IF(rsposLen != batchSize, context, return ge::GRAPH_FAILED,
                     "[ERROR] rspos length must be batchSize, expected %lld, got %lld.",
                     (long long)batchSize, (long long)rsposLen);

        auto ascendCPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const size_t maxCores = ascendCPlatform.GetCoreNumAiv();
        OPS_LOG_E_IF(maxCores == 0, context, return ge::GRAPH_FAILED, "[ERROR] aiv core num is zero.");

        const int64_t totalBlocks = batchSize;
        size_t actualCoreNum = totalBlocks > static_cast<int64_t>(maxCores) ? maxCores : static_cast<size_t>(totalBlocks);
        int64_t blocksPerCore = 0;
        int32_t remainderBlocks = 0;
        if (actualCoreNum > 0) {
            blocksPerCore = totalBlocks / static_cast<int64_t>(actualCoreNum);
            remainderBlocks = static_cast<int32_t>(totalBlocks % static_cast<int64_t>(actualCoreNum));
        }

        size_t* currentWorkspace = context->GetWorkspaceSizes(1U);
        OPS_LOG_E_IF_NULL("currentWorkspace", currentWorkspace, return ge::GRAPH_FAILED);
        const size_t systemWorkspacesSize = ascendCPlatform.GetLibApiWorkSpaceSize();
        currentWorkspace[0] = systemWorkspacesSize;

        GenPositionIdsReverseV2TilingData tiling;
        tiling.set_batchSize(batchSize);
        tiling.set_blocksPerCore(blocksPerCore);
        tiling.set_remainderBlocks(remainderBlocks);

        context->SetBlockDim(actualCoreNum > 0 ? static_cast<uint32_t>(actualCoreNum) : 1U);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
// 输出一维长度 = sum(seqlen) = seqlen_offsets[batchSize]；Infer 阶段在常量折叠可用时从 seqlen Host 求和
//（无 total_seq_len 属性，与 PyTorch 侧由调用方先分配同长度 position_ids 一致）。
// 若 GetInputTensor/GetData 不可用，请按所用 CANN 版本替换为可获取常量 seqlen 的接口。
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

        const gert::Shape* seqlenShape = context->GetInputShape(SEQLEN_INDEX);
        const gert::Shape* offsetsShape = context->GetInputShape(SEQLEN_OFFSETS_INDEX);
        OPS_LOG_E_IF_NULL("seqlenShape", seqlenShape, return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("offsetsShape", offsetsShape, return ge::GRAPH_FAILED);

        int64_t batchSize = seqlenShape->GetDim(DIM0);
        if (offsetsShape->GetDim(DIM0) != batchSize + 1) {
            return ge::GRAPH_FAILED;
        }

        const gert::Tensor* seqlenTensor = context->GetInputTensor(SEQLEN_INDEX);
        OPS_LOG_E_IF_NULL("seqlen tensor for infer", seqlenTensor, return ge::GRAPH_FAILED);
        const int32_t* seqlenData = seqlenTensor->GetData<int32_t>();
        OPS_LOG_E_IF_NULL("seqlen data ptr", seqlenData, return ge::GRAPH_FAILED);
        int64_t totalOutputLen = 0;
        for (int64_t b = 0; b < batchSize; ++b) {
            totalOutputLen += static_cast<int64_t>(seqlenData[static_cast<size_t>(b)]);
        }

        gert::Shape* positionIdsShape = context->GetOutputShape(OUTPUT_INDEX);
        OPS_LOG_E_IF_NULL("positionIdsShape", positionIdsShape, return ge::GRAPH_FAILED);
        positionIdsShape->SetDimNum(1);
        positionIdsShape->SetDim(0, totalOutputLen);
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
    {
        OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
        if (ge::GRAPH_SUCCESS != context->SetOutputDataType(OUTPUT_INDEX, ge::DT_INT32)) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class GenPositionIdsReverseV2 : public OpDef {
    public:
        explicit GenPositionIdsReverseV2(const char* name) : OpDef(name)
        {
            this->Input("seqlen")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("seqlen_offsets")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("rspos")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("position_ids")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Attr("batchSize").Int();

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore().SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend950");
        }
    };

    OP_ADD(GenPositionIdsReverseV2);
}  // namespace ops