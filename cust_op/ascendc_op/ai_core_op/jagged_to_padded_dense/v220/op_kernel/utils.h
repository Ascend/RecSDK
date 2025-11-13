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

#ifndef ATTENTION_FUSION_GRAD_UTILS_H
#define ATTENTION_FUSION_GRAD_UTILS_H
#include "kernel_operator.h"

namespace JaggedToPaddedDense {
constexpr int DATA_TYPE_INT64 = 8;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int DATA_COPY_ALIGN_BYTES = 16;

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T1, typename T2>
__aicore__ inline T1 AlignUp(T1 value, T2 align)
{
    return CeilDiv<T1, T2>(value, align) * align;
}

#ifdef __CCE_KT_TEST__

#define LOG(X...) Log(X)
__global__ __aicore__ void printArgs() {}

template <typename T, typename... Args>
__global__ __aicore__ void printArgs(T t, Args&&... args)
{
    std::cout << t << " ";
    printArgs(args...);
}

template <typename... Args>
__global__ __aicore__ void Log(Args&&... args)
{
    // don't log when using npu
    std::cout << "[AttentionFusion LOG][" << AscendC::GetBlockIdx() << "]  ";
    printArgs(args...);
    std::cout << std::endl;
}

#else
#define LOG(X...)
#endif
}

#endif