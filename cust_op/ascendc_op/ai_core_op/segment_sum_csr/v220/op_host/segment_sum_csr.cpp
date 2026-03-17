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

#include "segment_sum_csr_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ops_log.h"

namespace optiling {
    constexpr int RESERVER_UB_SIZE = 40 * 1024;

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("csrSeg", context->GetInputTensor(0), return ge::GRAPH_FAILED);
        OPS_LOG_E_IF_NULL("values", context->GetInputTensor(1), return ge::GRAPH_FAILED);

        // platform info
        auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
        currentWorkspace[0] = systemWorkspacesSize;
        size_t coreNum = ascendPlatform.GetCoreNumAiv();
        if (coreNum == 0) {
            OPS_LOG_E("[ERROR]", "ai core num is zero.");
            return ge::GRAPH_FAILED;
        }

        // ub
        uint64_t ubCanUsed;
        ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCanUsed);
        ubCanUsed = ubCanUsed - RESERVER_UB_SIZE;

        // 输入shape
        const gert::StorageShape* x1_shape = context->GetInputShape(0);
        const gert::Shape shape1 = x1_shape->GetStorageShape();
        const auto segmentNums = shape1.GetDim(0) - 1;
        const gert::StorageShape* x2_shape = context->GetInputShape(1);
        const gert::Shape shape2 = x2_shape->GetStorageShape();
        const auto totalLength = shape2.GetDim(0);

        // 输入类型
        ge::DataType csrType = context->GetInputTensor(0)->GetDataType();

        int64_t baseCoreSegments = segmentNums / coreNum;    // 均分到每个核的数据段数
        int64_t remainedSegments = segmentNums % coreNum;    // 剩余的数据段数，并将剩余的段数均分到前面的核
        int64_t formerCoreSegments = baseCoreSegments + 1;

        // 设置TilingKey
        int64_t tilingKey = 0;
        tilingKey = csrType == ge::DataType::DT_INT64 ? 1 : 0;

        // 设置tiling参数
        SegmentSumCsrTilingData tiling;
        tiling.set_coreNum(coreNum);
        tiling.set_segmentNums(segmentNums);
        tiling.set_csrSegLength(shape1.GetDim(0));
        tiling.set_totalLength(totalLength);
        tiling.set_baseCoreSegments(baseCoreSegments);
        tiling.set_remainedSegments(remainedSegments);
        tiling.set_formerCoreSegments(formerCoreSegments);
        if (csrType == ge::DataType::DT_INT32) {
            const int32_t* batchSize = context->GetAttrs()->GetAttrPointer<int32_t>(0);
            tiling.set_batchSize(static_cast<int64_t>(*batchSize));
        } else {
            const int64_t* batchSize = context->GetAttrs()->GetAttrPointer<int64_t>(0);
            tiling.set_batchSize(*batchSize);
        }
        context->SetBlockDim(coreNum);
        context->SetTilingKey(tilingKey);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* csrSegshape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    int64_t input_dim0 = csrSegshape->GetDim(0);
    yShape->SetDimNum(1);
    yShape->SetDim(0, input_dim0 - 1);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SegmentSumCsr : public OpDef {
public:
    explicit SegmentSumCsr(const char* name) : OpDef(name)
    {
        this->Input("csrSeg")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("values")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("batchSize").Int();
        this->SetInferShape(ge::InferShape);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend310p", aicore_config);
#ifdef SUPPORT_950
        this->AICore().AddConfig("ascend950", aicore_config);
#endif
    }
};

OP_ADD(SegmentSumCsr);
}  // namespace ops