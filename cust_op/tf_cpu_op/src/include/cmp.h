/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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

#ifndef RECBASE_CMP_H
#define RECBASE_CMP_H

#include <cstdlib>

namespace ock {

// Less
template <typename T> int Less(T *input0, T *input1, bool *output, size_t length);

// Less right
template <typename T> int Less(T *input0, T input1, bool *output, size_t length);

// Less left
template <typename T> int Less(T input0, T *input1, bool *output, size_t length);

// Greater
template <typename T> int Greater(T *input0, T *input1, bool *output, size_t length);

// Greater right
template <typename T> int Greater(T *input0, T input1, bool *output, size_t length);

// Greater left
template <typename T> int Greater(T input0, T *input1, bool *output, size_t length);


} // namespace ock

#endif // RECBASE_CMP_H