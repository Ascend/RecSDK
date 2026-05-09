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

#ifndef LOOKUP_KERNEL_H
#define LOOKUP_KERNEL_H

#include "utils.h"
#include "device_utils.h"

namespace dyn_emb {

template <typename T>
struct Vec4T {};

template <>
struct Vec4T<half> {
    union U {
        float2 f;
        half2 h[2];
    } value;

    DEVICE_INLINE Vec4T()
    {
        value.h[0].x = 0.f;
        value.h[0].y = 0.f;
        value.h[1].x = 0.f;
        value.h[1].y = 0.f;
    }

    DEVICE_INLINE void reset()
    {
        value.h[0].x = 0.f;
        value.h[0].y = 0.f;
        value.h[1].x = 0.f;
        value.h[1].y = 0.f;
    }

    DEVICE_INLINE void reset(const half initial_value)
    {
        value.h[0].x = initial_value;
        value.h[0].y = initial_value;
        value.h[1].x = initial_value;
        value.h[1].y = initial_value;
    }

    DEVICE_INLINE void load(const float* p, int n)
    {
        if (n == 4) {
            float4 f = *(reinterpret_cast<const float4*>(p));
            float2 firstf{f.x, f.y};
            float2 secondf{f.z, f.w};
            value.h[0] = __float22half2_rn(firstf);
            value.h[1] = __float22half2_rn(secondf);
        } else {
            if (n > 0) {
                value.h[0].x = __float2half(p[0]);
            }
            if (n > 1) {
                value.h[0].y = __float2half(p[1]);
            }
            if (n > 2) {
                value.h[1].x = __float2half(p[2]);
            }
        }
    }

    DEVICE_INLINE void load(const half* p, int n)
    {
        if (n == 4) {
            value.f = *(reinterpret_cast<const float2*>(p));
        } else {
            if (n > 0) {
                value.h[0].x = p[0];
            }
            if (n > 1) {
                value.h[0].y = p[1];
            }
            if (n > 2) {
                value.h[1].x = p[2];
            }
        }
    }

    DEVICE_INLINE void load(const bfloat16_t* p, int n)
    {
        if (n == 4) {
            float2 f = *(reinterpret_cast<const float2*>(p));
            bfloat16x2_t first = __float2bfloat162_rn(f.x);
            bfloat16x2_t second = __float2bfloat162_rn(f.y);
            value.h[0] = make_half2(TypeConvertFunc<half, bfloat16_t>::convert(first.x),
                                    TypeConvertFunc<half, bfloat16_t>::convert(first.y));
            value.h[1] = make_half2(TypeConvertFunc<half, bfloat16_t>::convert(second.x),
                                    TypeConvertFunc<half, bfloat16_t>::convert(second.y));
        } else {
            if (n > 0) {
                value.h[0].x = TypeConvertFunc<half, bfloat16_t>::convert(p[0]);
            }
            if (n > 1) {
                value.h[0].y = TypeConvertFunc<half, bfloat16_t>::convert(p[1]);
            }
            if (n > 2) {
                value.h[1].x = TypeConvertFunc<half, bfloat16_t>::convert(p[2]);
            }
        }
    }

    DEVICE_INLINE void load(const float* p)
    {
        float4 f = *(reinterpret_cast<const float4*>(p));
        float2 firstf{f.x, f.y};
        float2 secondf{f.z, f.w};
        value.h[0] = __float22half2_rn(firstf);
        value.h[1] = __float22half2_rn(secondf);
    }

    DEVICE_INLINE void load(const half* p)
    {
        value.f = *(reinterpret_cast<const float2*>(p));
    }

    DEVICE_INLINE void load(const bfloat16_t* p)
    {
        float2 f = *(reinterpret_cast<const float2*>(p));
        bfloat16x2_t first = __float2bfloat162_rn(f.x);
        bfloat16x2_t second = __float2bfloat162_rn(f.y);
        value.h[0] = make_half2(TypeConvertFunc<half, bfloat16_t>::convert(first.x),
                                TypeConvertFunc<half, bfloat16_t>::convert(first.y));
        value.h[1] = make_half2(TypeConvertFunc<half, bfloat16_t>::convert(second.x),
                                TypeConvertFunc<half, bfloat16_t>::convert(second.y));
    }

    DEVICE_INLINE void atomic_store_accum(__gm__ float* dst, int n)
    {
        if (n > 0) {
            asc_atomic_add(dst, __half2float(value.h[0].x));
        }
        if (n > 1) {
            asc_atomic_add(dst + 1, __half2float(value.h[0].y));
        }
        if (n > 2) {
            asc_atomic_add(dst + 2, __half2float(value.h[1].x));
        }
        if (n > 3) {
            asc_atomic_add(dst + 3, __half2float(value.h[1].y));
        }
    }

    DEVICE_INLINE void atomic_store_accum(__gm__ half* dst, int n)
    {
        if (n == 4) {
            asc_atomic_add((reinterpret_cast<__gm__ half2*>(dst)), value.h[0]);
            asc_atomic_add((reinterpret_cast<__gm__ half2*>(dst + 2)), value.h[1]);
        } else {
            if (n > 0) {
                asc_atomic_add(dst, value.h[0].x);
            }
            if (n > 1) {
                asc_atomic_add(dst + 1, value.h[0].y);
            }
            if (n > 2) {
                asc_atomic_add(dst + 2, value.h[1].x);
            }
        }
    }

    DEVICE_INLINE void atomic_store_accum(__gm__ bfloat16_t* dst, int n)
    {
        if (n == 4) {
            bfloat16x2_t h0 = make_bfloat162(TypeConvertFunc<bfloat16_t, half>::convert(value.h[0].x),
                                             TypeConvertFunc<bfloat16_t, half>::convert(value.h[0].y));
            bfloat16x2_t h1 = make_bfloat162(TypeConvertFunc<bfloat16_t, half>::convert(value.h[1].x),
                                             TypeConvertFunc<bfloat16_t, half>::convert(value.h[1].y));
            asc_atomic_add(reinterpret_cast<__gm__ bfloat16x2_t*>(dst), h0);
            asc_atomic_add(reinterpret_cast<__gm__ bfloat16x2_t*>(dst + 2), h1);
        } else {
            if (n > 0) {
                asc_atomic_add(dst, TypeConvertFunc<bfloat16_t, half>::convert(value.h[0].x));
            }
            if (n > 1) {
                asc_atomic_add(dst + 1, TypeConvertFunc<bfloat16_t, half>::convert(value.h[0].y));
            }
            if (n > 2) {
                asc_atomic_add(dst + 2, TypeConvertFunc<bfloat16_t, half>::convert(value.h[1].x));
            }
        }
    }

    DEVICE_INLINE void store(float* dst, int n)
    {
        if (n == 4) {
            float4 f;
            f.x = __half2float(value.h[0].x);
            f.y = __half2float(value.h[0].y);
            f.z = __half2float(value.h[1].x);
            f.w = __half2float(value.h[1].y);
            *(reinterpret_cast<float4*>(dst)) = f;
        } else {
            if (n > 0) {
                dst[0] = __half2float(value.h[0].x);
            }
            if (n > 1) {
                dst[1] = __half2float(value.h[0].y);
            }
            if (n > 2) {
                dst[2] = __half2float(value.h[1].x);
            }
        }
    }

    DEVICE_INLINE void store(half* dst, int n)
    {
        if (n == 4) {
            *(reinterpret_cast<float2*>(dst)) = value.f;
        } else {
            if (n > 0) {
                dst[0] = value.h[0].x;
            }
            if (n > 1) {
                dst[1] = value.h[0].y;
            }
            if (n > 2) {
                dst[2] = value.h[1].x;
            }
        }
    }

    DEVICE_INLINE void store(bfloat16_t* dst, int n)
    {
        if (n == 4) {
            union {
                float2 f;
                bfloat16x2_t h[2];
            } tmp;
            tmp.h[0].x = TypeConvertFunc<bfloat16_t, half>::convert(value.h[0].x);
            tmp.h[0].y = TypeConvertFunc<bfloat16_t, half>::convert(value.h[0].y);
            tmp.h[1].x = TypeConvertFunc<bfloat16_t, half>::convert(value.h[1].x);
            tmp.h[1].y = TypeConvertFunc<bfloat16_t, half>::convert(value.h[1].y);
            *(reinterpret_cast<float2*>(dst)) = tmp.f;
        } else {
            if (n > 0) {
                dst[0] = TypeConvertFunc<bfloat16_t, half>::convert(value.h[0].x);
            }
            if (n > 1) {
                dst[1] = TypeConvertFunc<bfloat16_t, half>::convert(value.h[0].y);
            }
            if (n > 2) {
                dst[2] = TypeConvertFunc<bfloat16_t, half>::convert(value.h[1].x);
            }
        }
    }

    DEVICE_INLINE void store(float* dst)
    {
        float4 f;
        f.x = __half2float(value.h[0].x);
        f.y = __half2float(value.h[0].y);
        f.z = __half2float(value.h[1].x);
        f.w = __half2float(value.h[1].y);
        *(reinterpret_cast<float4*>(dst)) = f;
    }

    DEVICE_INLINE void store(half* dst)
    {
        *(reinterpret_cast<float2*>(dst)) = value.f;
    }

    DEVICE_INLINE void store(bfloat16_t* dst)
    {
        union {
            float2 f;
            bfloat16x2_t h[2];
        } tmp;
        tmp.h[0].x = TypeConvertFunc<bfloat16_t, half>::convert(value.h[0].x);
        tmp.h[0].y = TypeConvertFunc<bfloat16_t, half>::convert(value.h[0].y);
        tmp.h[1].x = TypeConvertFunc<bfloat16_t, half>::convert(value.h[1].x);
        tmp.h[1].y = TypeConvertFunc<bfloat16_t, half>::convert(value.h[1].y);
        *(reinterpret_cast<float2*>(dst)) = tmp.f;
    }
};

}  // namespace dyn_emb

#endif  // LOOKUP_KERNEL_H
