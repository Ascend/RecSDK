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
Status ParseOnnxParamsGatherForRank1(const ge::Operator& op_src, ge::Operator& op_dest)
{
    return SUCCESS;
}

REGISTER_CUSTOM_OP("GatherForRank1")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::9::GatherForRank1"),
                ge::AscendString("ai.onnx::10::GatherForRank1"),
                ge::AscendString("ai.onnx::11::GatherForRank1"),
                ge::AscendString("ai.onnx::12::GatherForRank1"),
                ge::AscendString("ai.onnx::13::GatherForRank1"),
                ge::AscendString("ai.onnx::14::GatherForRank1"),
                ge::AscendString("ai.onnx::15::GatherForRank1"),
                ge::AscendString("ai.onnx::17::GatherForRank1")})
    .ParseParamsByOperatorFn(ParseOnnxParamsGatherForRank1)
    .ImplyType(ImplyType::TVM);
}  // namespace domi