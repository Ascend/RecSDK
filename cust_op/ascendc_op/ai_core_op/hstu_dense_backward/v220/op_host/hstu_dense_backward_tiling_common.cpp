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


#include <cctype>
#include <string>
#include <unordered_map>

#include "register/op_def_registry.h"

#include "hstu_dense_backward_tiling_common.h"

ShapeRange::ShapeRange(int64_t lbound, int64_t ubound, int64_t multiple, const char *name)
{
    this->lbound = lbound;
    this->ubound = ubound;
    this->multiple = multiple;
    this->name = name;
}

bool ShapeRange::Check(int64_t val) const
{
    OPS_CHECK((val < lbound || val > ubound || val % multiple != 0),
        OPS_LOG_E("", "%s must meet range[%lld %lld] and multiple of [%lld]. but get value %lld\n",
            name, lbound, ubound, multiple, val),
        return false);
    return true;
}

ge::graphStatus GetInputLayout(const gert::RuntimeAttrs *attrs, InputLayout &layout)
{
    OPS_CHECK_PTR_NULL(attrs, return ge::GRAPH_FAILED);

    const char *inputLayout = attrs->GetAttrPointer<char>(ATTR_INDEX_T::LAYOUT_INDEX);
    OPS_CHECK_PTR_NULL(inputLayout, return ge::GRAPH_FAILED);

    std::string inputLayoutStr = std::string(inputLayout);
    for (auto &c : inputLayoutStr) {
        c = tolower(c);
    }

    if (inputLayoutStr == "normal") {
        layout = InputLayout::NORMAL;
    } else if (inputLayoutStr == "jagged") {
        layout = InputLayout::JAGGED;
    } else {
        OPS_LOG_E("", "the input layout should be normal/jagged");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool IfMask(const int32_t &maskType, MaskType maskTypeEnum)
{
    return static_cast<int32_t>(maskTypeEnum) == maskType;
}

bool IsSameShape(const gert::Shape &shape0, const gert::Shape &shape1, int dim)
{
    if (shape0.GetDimNum() != dim || shape1.GetDimNum() != dim) {
        return false;
    }

    for (int i = 0; i < dim; i++) {
        if (shape0.GetDim(i) != shape1.GetDim(i)) {
            return false;
        }
    }

    return true;
}