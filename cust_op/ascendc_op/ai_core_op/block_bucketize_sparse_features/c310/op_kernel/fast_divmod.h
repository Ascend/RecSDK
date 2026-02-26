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

#ifndef FAST_DIVMOD_H
#define FAST_DIVMOD_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

// 无符号快除法
template <typename UnsignedT>
class FastDivmod {
public:
    __aicore__ inline FastDivmod(UnsignedT magic, uint32_t shift, UnsignedT divisor)
        : magic_(magic), shift_(shift), divisor_(divisor)
    {
    }

    __aicore__ inline UnsignedT Div(UnsignedT n) const
    {
        if (divisor_ <= 1) {
            return (divisor_ == 1) ? n : static_cast<UnsignedT>(0);
        }
        UnsignedT q = MulHigh(n, magic_);
        UnsignedT t = ((n - q) >> 1) + q;
        return t >> (shift_ - 1);
    }

    __aicore__ inline UnsignedT Mod(UnsignedT n) const
    {
        if (divisor_ <= 1) {
            return (divisor_ == 1) ? static_cast<UnsignedT>(0) : n;
        }
        return n - Div(n) * divisor_;
    }

private:
    // MulHigh: 返回 (a * b) 的高 N 位，其中 N = sizeof(UnsignedT) * 8
    __aicore__ static inline UnsignedT MulHigh(UnsignedT a, UnsignedT b)
    {
        // 4字节处理
        if constexpr (sizeof(UnsignedT) == 4) {
            return static_cast<uint32_t>(
                (static_cast<uint64_t>(a) * static_cast<uint64_t>(b)) >> 32);
        } else {
            // 8字节处理
            return static_cast<UnsignedT>(__umul64hi(
                static_cast<uint64_t>(a), static_cast<uint64_t>(b)));
        }
    }

    UnsignedT magic_;    // 预计算的 magic number
    uint32_t shift_;     // 预计算的位移量，= ceil(log2(divisor))
    UnsignedT divisor_;  // 原始除数
};

#endif // FAST_DIVMOD_H
