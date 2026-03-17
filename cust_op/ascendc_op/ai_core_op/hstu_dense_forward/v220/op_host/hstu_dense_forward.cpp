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
#ifdef SUPPORT_V200
    #include "tiling_policy_dense_v200.h"
#else
    #include "tiling_policy_dense.h"
#endif

using namespace HstuForward;

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_CHECK_PTR_NULL(context, return ge::GRAPH_FAILED);

#ifdef SUPPORT_V200
    HstuDenseForward::TilingPolicyNormalv200 tilingPolicy;
#else
    HstuDenseForward::TilingPolicyDense tilingPolicy;
#endif

    return tilingPolicy.TilingProcess(context);
}
}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_CHECK_PTR_NULL(context, return ge::GRAPH_FAILED);
    
#ifdef SUPPORT_V200
    HstuDenseForward::TilingPolicyNormalv200 tilingPolicy;
#else
    HstuDenseForward::TilingPolicyDense tilingPolicy;
#endif

    return tilingPolicy.InferShape(context);
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    OPS_CHECK_PTR_NULL(context, return ge::GRAPH_FAILED);

#ifdef SUPPORT_V200
    HstuDenseForward::TilingPolicyNormalv200 tilingPolicy;
#else
    HstuDenseForward::TilingPolicyDense tilingPolicy;
#endif

    return tilingPolicy.InferDtype(context);
}
}  // namespace ge

namespace ops {
class HstuDenseForward : public OpDef {
public:
    explicit HstuDenseForward(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
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
        this->Output("attn_output")
            .ParamType(REQUIRED)
            .Follow("q", FollowType::DTYPE)
            .FormatList({ge::FORMAT_ND});
        this->Attr("mask_type").Int();
        this->Attr("max_seqlen").Int();
        this->Attr("silu_scale").Float();

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
#ifdef SUPPORT_950
        this->AICore().AddConfig("ascend950", aicore_config);
#endif
    }
};

OP_ADD(HstuDenseForward);
}  // namespace ops
