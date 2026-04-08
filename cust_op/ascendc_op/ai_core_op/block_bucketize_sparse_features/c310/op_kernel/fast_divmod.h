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
#include <limits>
#include <type_traits>

#include "kernel_operator.h"

using namespace AscendC;

template <typename UnsignedT>
class FastDivmod {
public:
    static constexpr UnsignedT UINT_DIV_MAX_DIVIDEND =
        static_cast<UnsignedT>(std::numeric_limits<typename std::make_signed<UnsignedT>::type>::max());

    __aicore__ inline FastDivmod(UnsignedT magic, uint32_t shift, UnsignedT divisor)
        : magic_(magic), shift_(shift), divisor_(divisor)
    {
    }

    __aicore__ inline UnsignedT Div(UnsignedT n) const
    {
        if (divisor_ <= 1) {
            return (divisor_ == 1) ? n : static_cast<UnsignedT>(0);
        }
        if (n > UINT_DIV_MAX_DIVIDEND) {
            return n / divisor_;
        }
        return AscendC::Simt::UintDiv<UnsignedT>(n, magic_, static_cast<UnsignedT>(shift_));
    }

    __aicore__ inline UnsignedT Mod(UnsignedT n) const
    {
        if (divisor_ <= 1) {
            return (divisor_ == 1) ? static_cast<UnsignedT>(0) : n;
        }
        if (n > UINT_DIV_MAX_DIVIDEND) {
            return n % divisor_;
        }
        const UnsignedT q = AscendC::Simt::UintDiv<UnsignedT>(n, magic_, static_cast<UnsignedT>(shift_));
        return n - q * divisor_;
    }

private:
    UnsignedT magic_;
    uint32_t shift_;
    UnsignedT divisor_;
};

#endif // FAST_DIVMOD_H
