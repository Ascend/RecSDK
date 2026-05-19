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
            (void)asc_atomic_add(dst, __half2float(value.h[0].x));
        }
        if (n > 1) {
            (void)asc_atomic_add(dst + 1, __half2float(value.h[0].y));
        }
        if (n > 2) {
            (void)asc_atomic_add(dst + 2, __half2float(value.h[1].x));
        }
        if (n > 3) {
            (void)asc_atomic_add(dst + 3, __half2float(value.h[1].y));
        }
    }

    DEVICE_INLINE void atomic_store_accum(__gm__ half* dst, int n)
    {
        if (n == 4) {
            (void)asc_atomic_add((reinterpret_cast<__gm__ half2*>(dst)), value.h[0]);
            (void)asc_atomic_add((reinterpret_cast<__gm__ half2*>(dst + 2)), value.h[1]);
        } else {
            if (n > 0) {
                (void)asc_atomic_add(dst, value.h[0].x);
            }
            if (n > 1) {
                (void)asc_atomic_add(dst + 1, value.h[0].y);
            }
            if (n > 2) {
                (void)asc_atomic_add(dst + 2, value.h[1].x);
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
            (void)asc_atomic_add(reinterpret_cast<__gm__ bfloat16x2_t*>(dst), h0);
            (void)asc_atomic_add(reinterpret_cast<__gm__ bfloat16x2_t*>(dst + 2), h1);
        } else {
            if (n > 0) {
                (void)asc_atomic_add(dst, TypeConvertFunc<bfloat16_t, half>::convert(value.h[0].x));
            }
            if (n > 1) {
                (void)asc_atomic_add(dst + 1, TypeConvertFunc<bfloat16_t, half>::convert(value.h[0].y));
            }
            if (n > 2) {
                (void)asc_atomic_add(dst + 2, TypeConvertFunc<bfloat16_t, half>::convert(value.h[1].x));
            }
        }
    }

    DEVICE_INLINE void store(__gm__ float* dst, int n)
    {
        if (n == 4) {
            float4 f;
            f.x = __half2float(value.h[0].x);
            f.y = __half2float(value.h[0].y);
            f.z = __half2float(value.h[1].x);
            f.w = __half2float(value.h[1].y);
            *(reinterpret_cast<__gm__ float4*>(dst)) = f;
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

    DEVICE_INLINE void store(__gm__ half* dst, int n)
    {
        if (n == 4) {
            *(reinterpret_cast<__gm__ float2*>(dst)) = value.f;
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

    DEVICE_INLINE void store(__gm__ bfloat16_t* dst, int n)
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
            *(reinterpret_cast<__gm__ float2*>(dst)) = tmp.f;
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

    DEVICE_INLINE void store(__gm__ float* dst)
    {
        float4 f;
        f.x = __half2float(value.h[0].x);
        f.y = __half2float(value.h[0].y);
        f.z = __half2float(value.h[1].x);
        f.w = __half2float(value.h[1].y);
        *(reinterpret_cast<__gm__ float4*>(dst)) = f;
    }

    DEVICE_INLINE void store(__gm__ half* dst)
    {
        *(reinterpret_cast<__gm__ float2*>(dst)) = value.f;
    }

    DEVICE_INLINE void store(__gm__ bfloat16_t* dst)
    {
        union {
            float2 f;
            bfloat16x2_t h[2];
        } tmp;
        tmp.h[0].x = TypeConvertFunc<bfloat16_t, half>::convert(value.h[0].x);
        tmp.h[0].y = TypeConvertFunc<bfloat16_t, half>::convert(value.h[0].y);
        tmp.h[1].x = TypeConvertFunc<bfloat16_t, half>::convert(value.h[1].x);
        tmp.h[1].y = TypeConvertFunc<bfloat16_t, half>::convert(value.h[1].y);
        *(reinterpret_cast<__gm__ float2*>(dst)) = tmp.f;
    }
};

template <>
struct Vec4T<bfloat16_t> {
    union U {
        float2 f;
        bfloat16x2_t h[2];

        DEVICE_INLINE U() : f{} {}
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

    DEVICE_INLINE void reset(const bfloat16_t initial_value)
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
            value.h[0] = __float22bfloat162_rn(firstf);
            value.h[1] = __float22bfloat162_rn(secondf);
        } else {
            if (n > 0) {
                value.h[0].x = __float2bfloat16(p[0]);
            }
            if (n > 1) {
                value.h[0].y = __float2bfloat16(p[1]);
            }
            if (n > 2) {
                value.h[1].x = __float2bfloat16(p[2]);
            }
        }
    }

    DEVICE_INLINE void load(const half* p, int n)
    {
        if (n == 4) {
            float2 f = *(reinterpret_cast<const float2*>(p));
            value.h[0] = __float2bfloat162_rn(f.x);
            value.h[1] = __float2bfloat162_rn(f.y);
        } else {
            if (n > 0) {
                value.h[0].x = TypeConvertFunc<bfloat16_t, half>::convert(p[0]);
            }
            if (n > 1) {
                value.h[0].y = TypeConvertFunc<bfloat16_t, half>::convert(p[1]);
            }
            if (n > 2) {
                value.h[1].x = TypeConvertFunc<bfloat16_t, half>::convert(p[2]);
            }
        }
    }

    DEVICE_INLINE void load(const bfloat16_t* p, int n)
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

    DEVICE_INLINE void load(const float* p)
    {
        float4 f = *(reinterpret_cast<const float4*>(p));
        float2 firstf{f.x, f.y};
        float2 secondf{f.z, f.w};
        value.h[0] = __float22bfloat162_rn(firstf);
        value.h[1] = __float22bfloat162_rn(secondf);
    }

    DEVICE_INLINE void load(const half* p)
    {
        float2 f = *(reinterpret_cast<const float2*>(p));
        value.h[0] = __float2bfloat162_rn(f.x);
        value.h[1] = __float2bfloat162_rn(f.y);
    }

    DEVICE_INLINE void load(const bfloat16_t* p)
    {
        value.f = *(reinterpret_cast<const float2*>(p));
    }

    DEVICE_INLINE void atomic_store_accum(__gm__ float* dst, int n)
    {
        if (n > 0) {
            (void)asc_atomic_add(dst, __bfloat162float(value.h[0].x));
        }
        if (n > 1) {
            (void)asc_atomic_add(dst + 1, __bfloat162float(value.h[0].y));
        }
        if (n > 2) {
            (void)asc_atomic_add(dst + 2, __bfloat162float(value.h[1].x));
        }
        if (n > 3) {
            (void)asc_atomic_add(dst + 3, __bfloat162float(value.h[1].y));
        }
    }

    DEVICE_INLINE void atomic_store_accum(__gm__ half* dst, int n)
    {
        if (n == 4) {
            half2 h0 = make_half2(TypeConvertFunc<half, bfloat16_t>::convert(value.h[0].x),
                                  TypeConvertFunc<half, bfloat16_t>::convert(value.h[0].y));
            half2 h1 = make_half2(TypeConvertFunc<half, bfloat16_t>::convert(value.h[1].x),
                                  TypeConvertFunc<half, bfloat16_t>::convert(value.h[1].y));
            (void)asc_atomic_add(reinterpret_cast<__gm__ half2*>(dst), h0);
            (void)asc_atomic_add(reinterpret_cast<__gm__ half2*>(dst + 2), h1);
        } else {
            if (n > 0) {
                (void)asc_atomic_add(dst, TypeConvertFunc<half, bfloat16_t>::convert(value.h[0].x));
            }
            if (n > 1) {
                (void)asc_atomic_add(dst + 1, TypeConvertFunc<half, bfloat16_t>::convert(value.h[0].y));
            }
            if (n > 2) {
                (void)asc_atomic_add(dst + 2, TypeConvertFunc<half, bfloat16_t>::convert(value.h[1].x));
            }
        }
    }

    DEVICE_INLINE void atomic_store_accum(__gm__ bfloat16_t* dst, int n)
    {
        if (n == 4) {
            (void)asc_atomic_add(reinterpret_cast<__gm__ bfloat16x2_t*>(dst), value.h[0]);
            (void)asc_atomic_add(reinterpret_cast<__gm__ bfloat16x2_t*>(dst + 2), value.h[1]);
        } else {
            if (n > 0) {
                (void)asc_atomic_add(dst, value.h[0].x);
            }
            if (n > 1) {
                (void)asc_atomic_add(dst + 1, value.h[0].y);
            }
            if (n > 2) {
                (void)asc_atomic_add(dst + 2, value.h[1].x);
            }
        }
    }

    DEVICE_INLINE void store(__gm__ float* dst, int n)
    {
        if (n == 4) {
            float4 f;
            f.x = __bfloat162float(value.h[0].x);
            f.y = __bfloat162float(value.h[0].y);
            f.z = __bfloat162float(value.h[1].x);
            f.w = __bfloat162float(value.h[1].y);
            *(reinterpret_cast<__gm__ float4*>(dst)) = f;
        } else {
            if (n > 0) {
                dst[0] = __bfloat162float(value.h[0].x);
            }
            if (n > 1) {
                dst[1] = __bfloat162float(value.h[0].y);
            }
            if (n > 2) {
                dst[2] = __bfloat162float(value.h[1].x);
            }
        }
    }

    DEVICE_INLINE void store(__gm__ half* dst, int n)
    {
        if (n == 4) {
            union {
                float2 f;
                half2 h[2];
            } tmp;
            tmp.h[0].x = TypeConvertFunc<half, bfloat16_t>::convert(value.h[0].x);
            tmp.h[0].y = TypeConvertFunc<half, bfloat16_t>::convert(value.h[0].y);
            tmp.h[1].x = TypeConvertFunc<half, bfloat16_t>::convert(value.h[1].x);
            tmp.h[1].y = TypeConvertFunc<half, bfloat16_t>::convert(value.h[1].y);
            *(reinterpret_cast<__gm__ float2*>(dst)) = tmp.f;
        } else {
            if (n > 0) {
                dst[0] = TypeConvertFunc<half, bfloat16_t>::convert(value.h[0].x);
            }
            if (n > 1) {
                dst[1] = TypeConvertFunc<half, bfloat16_t>::convert(value.h[0].y);
            }
            if (n > 2) {
                dst[2] = TypeConvertFunc<half, bfloat16_t>::convert(value.h[1].x);
            }
        }
    }

    DEVICE_INLINE void store(__gm__ bfloat16_t* dst, int n)
    {
        if (n == 4) {
            *(reinterpret_cast<__gm__ float2*>(dst)) = value.f;
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

    DEVICE_INLINE void store(__gm__ float* dst)
    {
        float4 f;
        f.x = __bfloat162float(value.h[0].x);
        f.y = __bfloat162float(value.h[0].y);
        f.z = __bfloat162float(value.h[1].x);
        f.w = __bfloat162float(value.h[1].y);
        *(reinterpret_cast<__gm__ float4*>(dst)) = f;
    }

    DEVICE_INLINE void store(__gm__ half* dst)
    {
        union {
            float2 f;
            half2 h[2];
        } tmp;
        tmp.h[0].x = TypeConvertFunc<half, bfloat16_t>::convert(value.h[0].x);
        tmp.h[0].y = TypeConvertFunc<half, bfloat16_t>::convert(value.h[0].y);
        tmp.h[1].x = TypeConvertFunc<half, bfloat16_t>::convert(value.h[1].x);
        tmp.h[1].y = TypeConvertFunc<half, bfloat16_t>::convert(value.h[1].y);
        *(reinterpret_cast<__gm__ float2*>(dst)) = tmp.f;
    }

    DEVICE_INLINE void store(__gm__ bfloat16_t* dst)
    {
        *(reinterpret_cast<__gm__ float2*>(dst)) = value.f;
    }
};

template <>
struct Vec4T<float> {
    float4 val;

    DEVICE_INLINE Vec4T()
    {
        val.x = 0.f;
        val.y = 0.f;
        val.z = 0.f;
        val.w = 0.f;
    }

    DEVICE_INLINE void reset()
    {
        val.x = 0.f;
        val.y = 0.f;
        val.z = 0.f;
        val.w = 0.f;
    }

    DEVICE_INLINE void reset(const float initial_value)
    {
        val.x = initial_value;
        val.y = initial_value;
        val.z = initial_value;
        val.w = initial_value;
    }

    DEVICE_INLINE void load(const float* p, int n)
    {
        if (n == 4) {
            val = *(reinterpret_cast<const float4*>(p));
        } else {
            if (n > 0) {
                val.x = p[0];
            }
            if (n > 1) {
                val.y = p[1];
            }
            if (n > 2) {
                val.z = p[2];
            }
        }
    }

    DEVICE_INLINE void load(const half* p, int n)
    {
        if (n == 4) {
            Vec4T<half> h;
            h.load(p, n);
            val.x = __half2float(h.value.h[0].x);
            val.y = __half2float(h.value.h[0].y);
            val.z = __half2float(h.value.h[1].x);
            val.w = __half2float(h.value.h[1].y);
        } else {
            if (n > 0) {
                val.x = __half2float(p[0]);
            }
            if (n > 1) {
                val.y = __half2float(p[1]);
            }
            if (n > 2) {
                val.z = __half2float(p[2]);
            }
        }
    }

    DEVICE_INLINE void load(const bfloat16_t* p, int n)
    {
        if (n == 4) {
            Vec4T<bfloat16_t> h;
            h.load(p);
            val.x = __bfloat162float(h.value.h[0].x);
            val.y = __bfloat162float(h.value.h[0].y);
            val.z = __bfloat162float(h.value.h[1].x);
            val.w = __bfloat162float(h.value.h[1].y);
        } else {
            if (n > 0) {
                val.x = __bfloat162float(p[0]);
            }
            if (n > 1) {
                val.y = __bfloat162float(p[1]);
            }
            if (n > 2) {
                val.z = __bfloat162float(p[2]);
            }
        }
    }

    DEVICE_INLINE void load(const float* p)
    {
        val = *(reinterpret_cast<const float4*>(p));
    }

    DEVICE_INLINE void load(const half* p)
    {
        Vec4T<half> h;
        h.load(p);
        val.x = __half2float(h.value.h[0].x);
        val.y = __half2float(h.value.h[0].y);
        val.z = __half2float(h.value.h[1].x);
        val.w = __half2float(h.value.h[1].y);
    }

    DEVICE_INLINE void load(const bfloat16_t* p)
    {
        Vec4T<bfloat16_t> h;
        h.load(p);
        val.x = __bfloat162float(h.value.h[0].x);
        val.y = __bfloat162float(h.value.h[0].y);
        val.z = __bfloat162float(h.value.h[1].x);
        val.w = __bfloat162float(h.value.h[1].y);
    }

    DEVICE_INLINE void store(__gm__ float* dst, int n)
    {
        if (n == 4) {
            *(reinterpret_cast<__gm__ float4*>(dst)) = val;
        } else {
            if (n > 0) {
                dst[0] = val.x;
            }
            if (n > 1) {
                dst[1] = val.y;
            }
            if (n > 2) {
                dst[2] = val.z;
            }
        }
    }

    DEVICE_INLINE void store(__gm__ half* dst, int n)
    {
        if (n == 4) {
            Vec4T<half> h;
            h.load(reinterpret_cast<float*>(&val), 4);
            h.store(dst, 4);
        } else {
            if (n > 0) {
                dst[0] = __float2half(val.x);
            }
            if (n > 1) {
                dst[1] = __float2half(val.y);
            }
            if (n > 2) {
                dst[2] = __float2half(val.z);
            }
        }
    }

    DEVICE_INLINE void store(__gm__ bfloat16_t* dst, int n)
    {
        if (n == 4) {
            Vec4T<bfloat16_t> h;
            h.load(reinterpret_cast<float*>(&val), 4);
            h.store(dst, 4);
        } else {
            if (n > 0) {
                dst[0] = __float2bfloat16(val.x);
            }
            if (n > 1) {
                dst[1] = __float2bfloat16(val.y);
            }
            if (n > 2) {
                dst[2] = __float2bfloat16(val.z);
            }
        }
    }

    DEVICE_INLINE void atomic_store_accum(__gm__ float* dst, int n)
    {
        if (n > 0) {
            (void)asc_atomic_add(dst, val.x);
        }
        if (n > 1) {
            (void)asc_atomic_add(dst + 1, val.y);
        }
        if (n > 2) {
            (void)asc_atomic_add(dst + 2, val.z);
        }
        if (n > 3) {
            (void)asc_atomic_add(dst + 3, val.w);
        }
    }

    DEVICE_INLINE void atomic_store_accum(__gm__ half* dst, int n)
    {
        if (n == 4) {
            half2 tmp1;
            half2 tmp2;
            tmp1.x = __float2half(val.x);
            tmp1.y = __float2half(val.y);
            tmp2.x = __float2half(val.z);
            tmp2.y = __float2half(val.w);
            (void)asc_atomic_add(reinterpret_cast<__gm__ half2*>(dst), tmp1);
            (void)asc_atomic_add(reinterpret_cast<__gm__ half2*>(dst + 2), tmp2);
        } else {
            if (n > 0) {
                (void)asc_atomic_add(dst, __float2half(val.x));
            }
            if (n > 1) {
                (void)asc_atomic_add(dst + 1, __float2half(val.y));
            }
            if (n > 2) {
                (void)asc_atomic_add(dst + 2, __float2half(val.z));
            }
        }
    }

    DEVICE_INLINE void atomic_store_accum(__gm__ bfloat16_t* dst, int n)
    {
        if (n == 4) {
            bfloat16x2_t tmp1;
            bfloat16x2_t tmp2;
            tmp1.x = __float2bfloat16(val.x);
            tmp1.y = __float2bfloat16(val.y);
            tmp2.x = __float2bfloat16(val.z);
            tmp2.y = __float2bfloat16(val.w);
            (void)asc_atomic_add(reinterpret_cast<__gm__ bfloat16x2_t*>(dst), tmp1);
            (void)asc_atomic_add(reinterpret_cast<__gm__ bfloat16x2_t*>(dst + 2), tmp2);
        } else {
            if (n > 0) {
                (void)asc_atomic_add(dst, __float2bfloat16(val.x));
            }
            if (n > 1) {
                (void)asc_atomic_add(dst + 1, __float2bfloat16(val.y));
            }
            if (n > 2) {
                (void)asc_atomic_add(dst + 2, __float2bfloat16(val.z));
            }
        }
    }

    DEVICE_INLINE void store(__gm__ float* dst)
    {
        *(reinterpret_cast<__gm__ float4*>(dst)) = val;
    }

    DEVICE_INLINE void store(__gm__ half* dst)
    {
        Vec4T<half> h;
        h.load(reinterpret_cast<float*>(&val), 4);
        h.store(dst, 4);
    }

    DEVICE_INLINE void store(__gm__ bfloat16_t* dst)
    {
        Vec4T<bfloat16_t> h;
        h.load(reinterpret_cast<float*>(&val), 4);
        h.store(dst, 4);
    }

    DEVICE_INLINE void accumulate(const Vec4T<float>& other)
    {
        val.x += other.val.x;
        val.y += other.val.y;
        val.z += other.val.z;
        val.w += other.val.w;
    }

    DEVICE_INLINE void accumulate(const Vec4T<half>& other)
    {
        val.x += __half2float(other.value.h[0].x);
        val.y += __half2float(other.value.h[0].y);
        val.z += __half2float(other.value.h[1].x);
        val.w += __half2float(other.value.h[1].y);
    }

    DEVICE_INLINE void accumulate(const Vec4T<bfloat16_t>& other)
    {
        val.x += __bfloat162float(other.value.h[0].x);
        val.y += __bfloat162float(other.value.h[0].y);
        val.z += __bfloat162float(other.value.h[1].x);
        val.w += __bfloat162float(other.value.h[1].y);
    }

    DEVICE_INLINE void accumulate_multiply(const Vec4T<float>& other, float weight)
    {
        val.x += (other.val.x * weight);
        val.y += (other.val.y * weight);
        val.z += (other.val.z * weight);
        val.w += (other.val.w * weight);
    }

    DEVICE_INLINE void accumulate_multiply(const Vec4T<float>& other, half weight)
    {
        val.x += (other.val.x * __half2float(weight));
        val.y += (other.val.y * __half2float(weight));
        val.z += (other.val.z * __half2float(weight));
        val.w += (other.val.w * __half2float(weight));
    }

    DEVICE_INLINE void accumulate_multiply(const Vec4T<float>& other, bfloat16_t weight)
    {
        val.x += (other.val.x * __bfloat162float(weight));
        val.y += (other.val.y * __bfloat162float(weight));
        val.z += (other.val.z * __bfloat162float(weight));
        val.w += (other.val.w * __bfloat162float(weight));
    }

    DEVICE_INLINE void accumulate_multiply(const Vec4T<half>& other, float weight)
    {
        val.x += (__half2float(other.value.h[0].x) * weight);
        val.y += (__half2float(other.value.h[0].y) * weight);
        val.z += (__half2float(other.value.h[1].x) * weight);
        val.w += (__half2float(other.value.h[1].y) * weight);
    }

    DEVICE_INLINE void accumulate_multiply(const Vec4T<half>& other, half weight)
    {
        val.x += (__half2float(other.value.h[0].x) * __half2float(weight));
        val.y += (__half2float(other.value.h[0].y) * __half2float(weight));
        val.z += (__half2float(other.value.h[1].x) * __half2float(weight));
        val.w += (__half2float(other.value.h[1].y) * __half2float(weight));
    }

    DEVICE_INLINE void accumulate_multiply(const Vec4T<half>& other, bfloat16_t weight)
    {
        val.x += (__half2float(other.value.h[0].x) * __bfloat162float(weight));
        val.y += (__half2float(other.value.h[0].y) * __bfloat162float(weight));
        val.z += (__half2float(other.value.h[1].x) * __bfloat162float(weight));
        val.w += (__half2float(other.value.h[1].y) * __bfloat162float(weight));
    }

    DEVICE_INLINE void accumulate_multiply(const Vec4T<bfloat16_t>& other, float weight)
    {
        val.x += (__bfloat162float(other.value.h[0].x) * weight);
        val.y += (__bfloat162float(other.value.h[0].y) * weight);
        val.z += (__bfloat162float(other.value.h[1].x) * weight);
        val.w += (__bfloat162float(other.value.h[1].y) * weight);
    }

    DEVICE_INLINE void accumulate_multiply(const Vec4T<bfloat16_t>& other, half weight)
    {
        val.x += (__bfloat162float(other.value.h[0].x) * __half2float(weight));
        val.y += (__bfloat162float(other.value.h[0].y) * __half2float(weight));
        val.z += (__bfloat162float(other.value.h[1].x) * __half2float(weight));
        val.w += (__bfloat162float(other.value.h[1].y) * __half2float(weight));
    }

    DEVICE_INLINE void accumulate_multiply(const Vec4T<bfloat16_t>& other, bfloat16_t weight)
    {
        val.x += (__bfloat162float(other.value.h[0].x) * __bfloat162float(weight));
        val.y += (__bfloat162float(other.value.h[0].y) * __bfloat162float(weight));
        val.z += (__bfloat162float(other.value.h[1].x) * __bfloat162float(weight));
        val.w += (__bfloat162float(other.value.h[1].y) * __bfloat162float(weight));
    }
};

}  // namespace dyn_emb

#endif  // LOOKUP_KERNEL_H
