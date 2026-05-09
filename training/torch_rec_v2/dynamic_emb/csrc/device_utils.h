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

#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H

#include "asc_fp16.h"
#include "asc_bf16.h"

namespace dyn_emb {

#define DEVICE_INLINE __simt_callee__ __forceinline__

template <typename TOUT, typename TIN>
struct TypeConvertFunc;

template <>
struct TypeConvertFunc<half, float> {
    static DEVICE_INLINE half convert(float val)
    {
        return __float2half(val);
    }
};

template <>
struct TypeConvertFunc<float, half> {
    static DEVICE_INLINE float convert(half val)
    {
        return __half2float(val);
    }
};

template <>
struct TypeConvertFunc<bfloat16_t, float> {
    static DEVICE_INLINE bfloat16_t convert(float val)
    {
        return __float2bfloat16(val);
    }
};

template <>
struct TypeConvertFunc<float, bfloat16_t> {
    static DEVICE_INLINE float convert(bfloat16_t val)
    {
        return __bfloat162float(val);
    }
};

template <>
struct TypeConvertFunc<bfloat16_t, half> {
    static DEVICE_INLINE bfloat16_t convert(half val)
    {
        float temp = __half2float(val);
        return __float2bfloat16(temp);
    }
};

template <>
struct TypeConvertFunc<half, bfloat16_t> {
    static DEVICE_INLINE half convert(bfloat16_t val)
    {
        float temp = __bfloat162float(val);
        return __float2half(temp);
    }
};

template <>
struct TypeConvertFunc<float, float> {
    static DEVICE_INLINE float convert(float val)
    {
        return val;
    }
};

template <>
struct TypeConvertFunc<half, half> {
    static DEVICE_INLINE half convert(half val)
    {
        return val;
    }
};

template <>
struct TypeConvertFunc<bfloat16_t, bfloat16_t> {
    static DEVICE_INLINE bfloat16_t convert(bfloat16_t val)
    {
        return val;
    }
};

template <>
struct TypeConvertFunc<float, long long> {
    static DEVICE_INLINE float convert(long long val)
    {
        return static_cast<float>(val);
    }
};

template <>
struct TypeConvertFunc<float, unsigned int> {
    static DEVICE_INLINE float convert(unsigned int val)
    {
        return static_cast<float>(val);
    }
};

template <>
struct TypeConvertFunc<int, long long> {
    static DEVICE_INLINE int convert(long long val)
    {
        return static_cast<int>(val);
    }
};

template <>
struct TypeConvertFunc<int, unsigned int> {
    static DEVICE_INLINE int convert(unsigned int val)
    {
        return static_cast<int>(val);
    }
};

}  // namespace dyn_emb

#endif  // DEVICE_UTILS_H
