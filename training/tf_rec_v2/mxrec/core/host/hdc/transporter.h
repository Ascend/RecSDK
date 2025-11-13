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

#ifndef REC_SDK_TRANSPORTER_H
#define REC_SDK_TRANSPORTER_H

#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace rec_sdk {
namespace hdc {

class Transporter {
public:
    virtual ~Transporter() noexcept = default;

    virtual void SendTensors(std::vector<tensorflow::Tensor>&& tensors) const = 0;
    virtual std::vector<tensorflow::Tensor> RecvTensors() const = 0;

    bool hdcRunning_ = true;
};

}  // namespace hdc
}  // namespace rec_sdk

#endif // REC_SDK_TRANSPORTER_H
