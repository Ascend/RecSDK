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

using namespace HstuDenseForward;

template<typename T>
const char *GetLayoutHelpFunc(T* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_PTR_NULL(attrs, nullptr);

    const char *layout = attrs->GetAttrPointer<char>(ATTR_INDEX_T::LAYOUT_INDEX);
    OPS_CHECK_PTR_NULL(layout, nullptr);

    return layout;
}

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_CHECK_PTR_NULL(context, return ge::GRAPH_FAILED);

    auto layout = GetLayoutHelpFunc<gert::TilingContext>(context);
    OPS_CHECK_PTR_NULL(layout, return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion == platform_ascendc::SocVersion::ASCEND310P) {
        layout = "normalv200";
    }

    auto tilingPolicy = TilingPolicyFactory::CreatePolicy(layout);
    OPS_CHECK_PTR_NULL(tilingPolicy, return ge::GRAPH_FAILED);

    return tilingPolicy->TilingProcess(context);
}
}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_CHECK_PTR_NULL(context, return ge::GRAPH_FAILED);
    
    auto layout = GetLayoutHelpFunc<gert::InferShapeContext>(context);
    OPS_CHECK_PTR_NULL(layout, return ge::GRAPH_FAILED);

    auto tilingPolicy = TilingPolicyFactory::CreatePolicy(layout);
    OPS_CHECK_PTR_NULL(tilingPolicy, return ge::GRAPH_FAILED);

    return tilingPolicy->InferShape(context);
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    OPS_CHECK_PTR_NULL(context, return ge::GRAPH_FAILED);
    
    auto layout = GetLayoutHelpFunc<gert::InferDataTypeContext>(context);
    OPS_CHECK_PTR_NULL(layout, return ge::GRAPH_FAILED);

    auto tilingPolicy = TilingPolicyFactory::CreatePolicy(layout);
    OPS_CHECK_PTR_NULL(tilingPolicy, return ge::GRAPH_FAILED);

    return tilingPolicy->InferDtype(context);
}
}  // namespace ge

namespace ops {
class HstuDenseForward : public OpDef {
public:
    explicit HstuDenseForward(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .Follow("q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .Follow("q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("mask")
            .ParamType(OPTIONAL)
            .Follow("q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("attn_bias")
            .ParamType(OPTIONAL)
            .Follow("q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("seq_offset_q") // 规避optional类型无法正常生成json文件的问题
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("seq_offset_k")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("seq_offset_t")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("kv_cache")
            .ParamType(OPTIONAL)
            .Follow("q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Input("page_offsets")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("page_ids")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("last_page_len")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("num_context")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("num_target")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Output("attn_output")
            .ParamType(REQUIRED)
            .Follow("q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Attr("mask_type").Int();
        this->Attr("max_seqlen_q").Int();
        this->Attr("max_seqlen_k").Int();
        this->Attr("silu_scale").Float();
        this->Attr("layout").AttrType(OPTIONAL).String("normal");
        this->Attr("target_group_size").AttrType(OPTIONAL).Int(0);
        this->Attr("is_delta_qk").AttrType(OPTIONAL).Int(0);
        this->Attr("alpha").Float();
        this->Attr("deterministic").AttrType(OPTIONAL).Bool(false);

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

OP_ADD(HstuDenseForward);
}  // namespace ops
