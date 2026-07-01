/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#ifndef TILING_POLICY_DEFINE_H
#define TILING_POLICY_DEFINE_H

#include <cstdio>
#include <cstdint>

// Maximum supported batch size for variable-length sequence batching
constexpr int HSTU_MAX_BATCH_SIZE = 2048;
// Maximum number of AIV cores per NPU (910B has 48 AIV cores)
constexpr int HSTU_MAX_AIV_NUM = 48;

#endif  // TILING_POLICY_DEFINE_H
