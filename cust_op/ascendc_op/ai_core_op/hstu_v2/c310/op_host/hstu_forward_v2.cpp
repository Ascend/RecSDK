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

#include <cstdint>
#include <algorithm>
#include <array>

#include "register/op_def_registry.h"

#include "ops_log.h"
#include "hstu_forward_v2_define.h"
#include "hstu_forward_v2_tiling.h"
#include "../op_kernel/catlass_hstu/kernel/fwd/forward_kernel_tiling.hpp"

namespace optiling {
static bool ParseShape(gert::TilingContext* context, HstuForwardV2TilingData& tilingData)
{
    static constexpr std::array<int64_t, 2> supportHeadRange = {1, 16};

    const auto qShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::Q))->GetStorageShape();
    const auto vShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::V))->GetStorageShape();
    const auto seqOffsetShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::SEQ_OFFSET_Q))->GetStorageShape();

    auto totalSeqLenQ = qShape.GetDim(static_cast<size_t>(DIM_INDEX::ZERO));
    auto totalSeqLenK = vShape.GetDim(static_cast<size_t>(DIM_INDEX::ZERO));
    auto batch = seqOffsetShape.GetDim(static_cast<size_t>(DIM_INDEX::ZERO)) - 1;
    auto headQ = qShape.GetDim(static_cast<size_t>(DIM_INDEX::ONE));
    auto headK = vShape.GetDim(static_cast<size_t>(DIM_INDEX::ONE));
    auto dimQK = qShape.GetDim(static_cast<size_t>(DIM_INDEX::TWO));
    auto dimV = vShape.GetDim(static_cast<size_t>(DIM_INDEX::TWO));

    auto dimChecker = [](int64_t dim) -> bool {
        return ((dim % 16) == 0);
    };

    auto headChecker = [](int64_t head) -> bool {
        return head >= supportHeadRange[0] && head <= supportHeadRange[1];
    };

    OPS_CHECK(headQ != headK, OPS_LOG_E("", "tiling failed current only support MHA.\n"), return false);

    OPS_CHECK(!dimChecker(dimQK), OPS_LOG_E("", "tiling failed dimQK must mutiple of 16.\n"), return false);

    OPS_CHECK(!dimChecker(dimV), OPS_LOG_E("", "tiling failed dimGV must mutiple of 16.\n"), return false);
    OPS_CHECK(!headChecker(headQ), OPS_LOG_E("", "tiling failed headQ is not in support range[1~16].\n"), return false);

    tilingData.set_batch(batch);
    tilingData.set_heads(headQ);
    tilingData.set_dimQK(dimQK);
    tilingData.set_dimV(dimV);
    tilingData.set_totalSeqLenQ(totalSeqLenQ);
    tilingData.set_totalSeqLenK(totalSeqLenK);
    return true;
}

static bool ParseAttr(gert::TilingContext* context, HstuForwardV2TilingData& tilingData)
{
    const auto* attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, false);

    const auto* maxSeqQ = attrs->GetAttrPointer<uint32_t>(static_cast<size_t>(ATTR_INDEX::MAX_SEQLEN_Q));
    OPS_CHECK_PTR_NULL(maxSeqQ, false);

    const auto* maxSeqK = attrs->GetAttrPointer<uint32_t>(static_cast<size_t>(ATTR_INDEX::MAX_SEQLEN_K));
    OPS_CHECK_PTR_NULL(maxSeqK, false);

    const auto* targetGroupSize = attrs->GetAttrPointer<int32_t>(static_cast<size_t>(ATTR_INDEX::TARGET_GROUP_SIZE));
    OPS_CHECK_PTR_NULL(targetGroupSize, false);

    const auto* scale = attrs->GetAttrPointer<float>(static_cast<size_t>(ATTR_INDEX::SCALE));
    OPS_CHECK_PTR_NULL(scale, false);

    const auto* alpha = attrs->GetAttrPointer<float>(static_cast<size_t>(ATTR_INDEX::ALPHA));
    OPS_CHECK_PTR_NULL(alpha, false);

    tilingData.set_maxSeqLenQ(*maxSeqQ);
    tilingData.set_maxSeqLenK(*maxSeqK);
    tilingData.set_targetGroupSize(*targetGroupSize);
    tilingData.set_scale(*scale);
    tilingData.set_alpha(*alpha);
    return true;
}

static bool TilingKeySet(gert::TilingContext* context, HstuForwardV2TilingData& tilingData)
{
    const auto* rab = context->GetInputTensor(static_cast<size_t>(IN_INDEX::RAB));
    bool hasRab = (nullptr != rab);

    auto dimQK = tilingData.get_dimQK();
    auto dimV = tilingData.get_dimV();

    uint32_t tilingDim = 128;
    if ((dimQK > 128) || (dimV > 128)) {
        tilingDim = 256;
    }

    ASCENDC_TPL_SEL_PARAM(context, hasRab, tilingDim);
    return true;
}

static void ShareMemorySet(gert::TilingContext* context, HstuForwardV2TilingData& tilingData)
{
    auto dimQK = tilingData.get_dimQK();
    auto dimV = tilingData.get_dimV();

    uint32_t BLOCK_N = 128;
    if ((dimQK > 128) || (dimV > 128)) {
        BLOCK_N = 64;
    }

    auto totalKSeqLens = tilingData.get_batch() * tilingData.get_heads() * tilingData.get_maxSeqLenQ();
    auto totalKBlockCnt = (totalKSeqLens + BLOCK_N - 1) / BLOCK_N;
    int64_t shareMemorySizes = totalKBlockCnt * sizeof(uint32_t);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t systemWorkspaceSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = shareMemorySizes + systemWorkspaceSize;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    HstuForwardV2TilingData tilingData;

    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    OPS_CHECK(!ParseShape(context, tilingData), OPS_LOG_E("", "parse shape failed.\n"), return ge::GRAPH_FAILED);
    OPS_CHECK(!ParseAttr(context, tilingData), OPS_LOG_E("", "parse attr failed.\n"), return ge::GRAPH_FAILED);
    OPS_CHECK(!TilingKeySet(context, tilingData), OPS_LOG_E("", "tiling keyset failed.\n"), return ge::GRAPH_FAILED);
    ShareMemorySet(context, tilingData);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t coreNum = ascendPlatform.GetCoreNumAic();
    context->SetBlockDim(coreNum);

    OPS_LOG_E_IF_NULL("RawTilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferAttenOutputShape(gert::InferShapeContext* context)
{
    const gert::Shape* queryShape = context->GetInputShape(static_cast<size_t>(IN_INDEX::Q));
    OPS_CHECK_PTR_NULL(queryShape, return ge::GRAPH_FAILED);

    gert::Shape* attenOutputShape = context->GetOutputShape(static_cast<size_t>(OUT_INDEX::ATTN_OUTPUT));
    OPS_CHECK_PTR_NULL(attenOutputShape, return ge::GRAPH_FAILED);
    *attenOutputShape = *queryShape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    auto result = InferAttenOutputShape(context);
    OPS_CHECK(result == ge::GRAPH_FAILED, OPS_LOG_E("", "InferAttenOutputShape failed.\n"), return ge::GRAPH_FAILED);

    return result;
}
}  // namespace ge

namespace ops {
class HstuForwardV2 : public OpDef {
public:
    explicit HstuForwardV2(const char* name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).Follow("q", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).Follow("q", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Input("mask").ParamType(OPTIONAL).Follow("q", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Input("rab").ParamType(OPTIONAL).Follow("q", FollowType::DTYPE).FormatList({ge::FORMAT_ND});
        this->Input("seq_offset_q")  // 规避optional类型无法正常生成json文件的问题
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("seq_offset_k")
            .ParamType(OPTIONAL)
            .Follow("seq_offset_q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("num_context")
            .ParamType(OPTIONAL)
            .Follow("seq_offset_q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("num_target")
            .ParamType(OPTIONAL)
            .Follow("seq_offset_q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        // 可选: flash_attn_metadata 分核输出(int32,HEAD+FA+FD 布局)。未传 → kernel 收到 nullptr →
        // 旧设备现算分核(零回归)
        this->Input("metadata").ParamType(OPTIONAL).DataType({ge::DT_INT32, ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("attn_output").ParamType(REQUIRED).Follow("q", FollowType::DTYPE).FormatList({ge::FORMAT_ND});

        this->Attr("max_seqlen_q").Int();
        this->Attr("max_seqlen_k").Int();
        this->Attr("silu_scale").Float();
        this->Attr("target_group_size").AttrType(OPTIONAL).Int(0);
        this->Attr("alpha").Float();

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(HstuForwardV2);
}  // namespace ops
