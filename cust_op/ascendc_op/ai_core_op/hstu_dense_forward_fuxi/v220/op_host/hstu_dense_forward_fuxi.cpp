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

#include "register/op_def_registry.h"
#include "tiling_policy_factory.h"

using namespace HstuDenseForwardFuxi;

template<typename T>
const char *GetLayoutHelpFunc(T* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL("attrs", attrs, return nullptr);

    const char *layout = attrs->GetAttrPointer<char>(INDEX_T::INDEX_3);
    OPS_LOG_E_IF_NULL("layout", layout, return nullptr);

    return layout;
}

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    auto layout = GetLayoutHelpFunc<gert::TilingContext>(context);
    OPS_LOG_E_IF_NULL("layout", layout, return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion == platform_ascendc::SocVersion::ASCEND310P) {
        layout = "normalv200";
    }

    auto tilingPolicy = TilingPolicyFactory::CreatePolicy(layout);
    OPS_LOG_E_IF_NULL("tilingPolicy", tilingPolicy, return ge::GRAPH_FAILED);

    return tilingPolicy->TilingProcess(context);
}
}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    
    auto layout = GetLayoutHelpFunc<gert::InferShapeContext>(context);
    OPS_LOG_E_IF_NULL("layout", layout, return ge::GRAPH_FAILED);

    auto tilingPolicy = TilingPolicyFactory::CreatePolicy(layout);
    OPS_LOG_E_IF_NULL("tilingPolicy", tilingPolicy, return ge::GRAPH_FAILED);

    return tilingPolicy->InferShape(context);
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    
    auto layout = GetLayoutHelpFunc<gert::InferDataTypeContext>(context);
    OPS_LOG_E_IF_NULL("layout", layout, return ge::GRAPH_FAILED);

    auto tilingPolicy = TilingPolicyFactory::CreatePolicy(layout);
    OPS_LOG_E_IF_NULL("tilingPolicy", tilingPolicy, return ge::GRAPH_FAILED);

    return tilingPolicy->InferDtype(context);
}
}  // namespace ge

namespace ops {
class HstuDenseForwardFuxi : public OpDef {
public:
    explicit HstuDenseForwardFuxi(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("timestamp_bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("position_bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("attn_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("maskType").Int();
        this->Attr("max_seq_len").Int();
        this->Attr("silu_scale").Float();
        this->Attr("layout").AttrType(OPTIONAL).String("normal");
        this->Attr("seq_offset").AttrType(OPTIONAL).ListInt();

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDtype);

        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend310p", aicore_config);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend910_95", aicore_config);
    }
};

OP_ADD(HstuDenseForwardFuxi);
}  // namespace ops
