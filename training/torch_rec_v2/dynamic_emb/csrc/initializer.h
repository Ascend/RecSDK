/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#pragma once

#include <cstdint>
#include <ATen/ATen.h>

#include "dynamic_variable_base.h"

namespace dyn_emb {

class CurandStateContext {
public:
    CurandStateContext();
    ~CurandStateContext();

    CurandStateContext(const CurandStateContext&) = delete;
    CurandStateContext& operator=(const CurandStateContext&) = delete;

    curandState* ptr();

private:
    curandState* states_{nullptr};
};

void normal_init(at::Tensor buffer, at::Tensor indices, CurandStateContext& curand_state_context, float mean,
                 float std_dev);

void truncated_normal_init(at::Tensor buffer, at::Tensor indices, CurandStateContext& curand_state_context, float mean,
                           float std_dev, float lower, float upper);

void uniform_init(at::Tensor buffer, at::Tensor indices, CurandStateContext& curand_state_context, float lower,
                  float upper);

void const_init(at::Tensor buffer, at::Tensor indices, float value);

void debug_init(at::Tensor buffer, at::Tensor indices, at::Tensor keys);

}  // namespace dyn_emb
