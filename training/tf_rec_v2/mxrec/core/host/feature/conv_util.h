/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef REC_SDK_CONV_UTIL_H
#define REC_SDK_CONV_UTIL_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "spdlog/spdlog.h"
#include "tensorflow/core/framework/tensor.h"

#include "common/types.h"

namespace rec_sdk {
namespace feature {

template <typename T>
std::vector<T> ConvertTensorToVector1D(const tensorflow::Tensor& tensor)
{
    if (tensor.NumElements() == 0) {
        spdlog::error("Cannot convert an empty tensorflow::tensor.");
        throw std::invalid_argument("empty tensor");
    }

    if (tensor.dims() != 1) {
        spdlog::error("Tensor should be 1-dimensional, but got {} dimensions.", tensor.dims());
        throw std::invalid_argument("not 1-dimensional tensor");
    }

    auto tensorData = tensor.flat<T>();
    auto vec = std::vector<T>(tensorData.size());

    for (size_t i = 0; i < tensorData.size(); i++) {
        vec[i] = tensorData(i);
    }

    return vec;
}

template <typename T>
tensorflow::Tensor ConvertVectorToTensor1D(const std::vector<T>& vec)
{
    if (vec.empty()) {
        spdlog::error("Cannot convert an empty std::vector.");
        throw std::invalid_argument("empty vector");
    }

    auto tensor =
        tensorflow::Tensor(common::GetTFDataType<T>(), tensorflow::TensorShape({static_cast<int64_t>(vec.size())}));
    auto tensorData = tensor.flat<T>();

    for (size_t i = 0; i < vec.size(); i++) {
        tensorData(i) = vec[i];
    }

    return tensor;
}

}  // namespace feature
}  // namespace rec_sdk

#endif // REC_SDK_CONV_UTIL_H
