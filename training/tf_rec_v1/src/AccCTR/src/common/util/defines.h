/* Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
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

#ifndef OCK_DEFINES_H
#define OCK_DEFINES_H

#include <cstdint>

namespace ock {
namespace ctr {
using HResult = int32_t;
constexpr int FACTOR_BIT = 2;
constexpr int FACTOR = 4;
constexpr int HASH_L_L = 16;
constexpr int HASH_L = 32;
constexpr int HASH_H = 48;
constexpr int DEFAULT_NUM = 256;
constexpr int MAX_ID_COUNT = 1 << 29;
constexpr int MAX_DESIRED_SIZE = 1431655765; // (2^32 -1)/2/1.5
}
}

#endif // OCK_DEFINES_H
