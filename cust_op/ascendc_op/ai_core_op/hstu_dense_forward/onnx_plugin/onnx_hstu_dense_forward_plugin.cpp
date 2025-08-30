/* Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "graph/operator.h"
#include "register/register.h"
#include "json.hpp"

using namespace ge;
using json = nlohmann::json;

namespace domi {
Status ParseOnnxParamsHstuDenseForward(const ge::Operator& op_src, ge::Operator& op_dest)
{
    // trans op_src to op_dest
    AscendString attrs_string;
    float siluScale = 1.0;
    int32_t maxSeqLen = 768;
    int32_t maskType = 3;
    std::vector<int> seqOffset;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("attribute", attrs_string)) {
        json attrs = json::parse(attrs_string.GetString());
        for (json attr : attrs["attribute"]) {
            if (attr["name"] == "maxSeqLen" || attr["name"] == "max_seq_len") {
                maxSeqLen = attr["i"];
            } else if (attr["name"] == "siluScale" || attr["name"] == "silu_scale") {
                siluScale = attr["f"];
            }
        }
    }

    op_dest.SetAttr("maskType", maskType);
    op_dest.SetAttr("max_seq_len", maxSeqLen);
    op_dest.SetAttr("silu_scale", siluScale);
    op_dest.SetAttr("layout", "normal");
    op_dest.SetAttr("seq_offets", seqOffset);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("HstuDenseForward")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::9::HstuDenseForward"),
                   ge::AscendString("ai.onnx::10::HstuDenseForward"),
                   ge::AscendString("ai.onnx::11::HstuDenseForward"),
                   ge::AscendString("ai.onnx::12::HstuDenseForward"),
                   ge::AscendString("ai.onnx::13::HstuDenseForward"),
                   ge::AscendString("ai.onnx::14::HstuDenseForward"),
                   ge::AscendString("ai.onnx::15::HstuDenseForward"),
                   ge::AscendString("ai.onnx::17::HstuDenseForward")})
    .ParseParamsByOperatorFn(ParseOnnxParamsHstuDenseForward)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
