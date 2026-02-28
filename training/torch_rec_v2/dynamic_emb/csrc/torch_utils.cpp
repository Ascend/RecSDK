/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "torch_utils.h"

namespace dyn_emb {
DataType scalartype_to_datatype(at::ScalarType scalar_type)
{
    switch (scalar_type) {
        case at::kFloat:
            return DataType::Float32;
        case at::kHalf:
            return DataType::Float16;
        case at::kBFloat16:
            return DataType::BFloat16;
        case at::kLong:
            return DataType::Int64;
        case at::kInt:
            return DataType::Int32;
        case at::kUInt64:
            return DataType::UInt64;
        case at::kUInt32:
            return DataType::UInt32;
        default:
            throw std::invalid_argument("Unsupported scalar_type");
    }
}

at::ScalarType convertTypeMetaToScalarType(const caffe2::TypeMeta &typeMeta)
{
    if (typeMeta == caffe2::TypeMeta::Make<float>()) {
        return at::kFloat;
    } else if (typeMeta == caffe2::TypeMeta::Make<at::Half>()) {
        return at::kHalf;
    } else if (typeMeta == caffe2::TypeMeta::Make<at::BFloat16>()) {
        return at::kBFloat16;
    } else if (typeMeta == caffe2::TypeMeta::Make<int64_t>()) {
        return at::kLong;
    } else if (typeMeta == caffe2::TypeMeta::Make<int>()) {
        return at::kInt;
    } else if (typeMeta == caffe2::TypeMeta::Make<uint64_t>()) {
        return at::kUInt64;
    } else if (typeMeta == caffe2::TypeMeta::Make<uint32_t>()) {
        return at::kUInt32;
    } else {
        throw std::invalid_argument("Unsupported typeMeta");
    }
}
}