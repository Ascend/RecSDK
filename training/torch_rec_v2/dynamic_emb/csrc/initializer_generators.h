/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <simt_api/common_functions.h>
#include <simt_api/math_functions.h>
#include "kernel_operator.h"
#include "dynamic_variable_base.h"
#include "device_utils.h"
#include "lookup_kernel.h"

using namespace AscendC;

namespace dyn_emb {

constexpr uint32_t EMBEDDING_GENERATOR_MAX_THREADS_PER_BLOCK = 2048;

__forceinline__ __simt_callee__ uint64_t GlobalThreadId()
{
    uint64_t id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    return id;
}

__forceinline__ __simt_callee__ void curand_init(uint64_t seed, uint64_t sequence, uint64_t offset,
                                                 __gm__ curandState* state)
{
    // 1. 基础播种：使用传入的种子初始化最核心的位移寄存器
    if (seed == 0) {
        seed = 123456789;  // 规避全0死循环, 当seed为0时，默认初始化为123456789
    }
    state->x = static_cast<uint32_t>(seed);
    // 简单的线性同余法衍生其他寄存器，参数与 glibc rand 一致
    state->y = static_cast<uint32_t>(seed * 1103515245ULL + 12345);
    state->z = static_cast<uint32_t>(state->y * 1103515245ULL + 12345);
    state->w = static_cast<uint32_t>(state->z * 1103515245ULL + 12345);
    state->v = static_cast<uint32_t>(state->w * 1103515245ULL + 12345);

    // 2. 引入序列号(sequence)和偏移量(offset)的影响
    // 362437 XORWOW 算法中加法计数器 d 的步进常数，与 NVIDIA curandStateXORWOW 一致。
    state->d = 362437 + static_cast<uint32_t>(sequence);

    // 3. 预迭代（热身）：跳过 offset 指定的步数，让状态充分混合，
    // 并且精准定位到序列的指定偏移位置(XORWOW 状态转移的固定位移)
    for (uint64_t i = 0; i < offset; ++i) {
        uint32_t t = (state->x ^ (state->x >> 2));
        state->x = state->y;
        state->y = state->z;
        state->z = state->w;
        state->w = state->v;
        state->v = (state->v ^ (state->v << 4)) ^ (t ^ (t << 1));
        state->d += 362437;
    }
}

__forceinline__ __simt_callee__ uint32_t curand_next(curandState& state)
{
    uint32_t t = (state.x ^ (state.x >> 2));
    state.x = state.y;
    state.y = state.z;
    state.z = state.w;
    state.w = state.v;
    state.v = (state.v ^ (state.v << 4)) ^ (t ^ (t << 1));
    state.d += 362437;
    return state.d + state.v;
}

__forceinline__ __simt_callee__ float curand_uniform_float(curandState& state)
{
    // 1. 获取一个 32 位的伪随机整数
    uint32_t r = curand_next(state);

    // 2. 提取高 23 位作为 float 的尾数 (Mantissa)
    // IEEE 754 单精度浮点数中，尾数占 23 位。
    // 右移 9 位可以丢弃低 9 位，保留最具随机性的高 23 位。
    r >>= 9;

    // 3. 拼接 IEEE 754 格式的二进制位
    // float 的结构：[1位符号位(0)] [8位指数位(127)] [23位尾数(r)]
    // 指数位设为 127 (即 0x3F800000 中的 0x7F << 23)，代表 2^0 = 1.0
    // 这样构造出的浮点数范围天然落在 [1.0, 2.0) 之间
    uint32_t ieee_bits = 0x3F800000 | r;

    // 4. 将uint32_t二进制位解释为float
    float result = __uint_as_float(ieee_bits);

    // 5. 减去 1.0f，将范围从 [1.0, 2.0) 映射到 [0.0, 1.0)
    // 由于 r 不会全为0（XORWOW算法特性），实际结果通常为 (0.0, 1.0]
    return result - 1.0f;
}

__forceinline__ __simt_callee__ float curand_normal_float(curandState& state)
{
    if (state.has_spare) {
        state.has_spare = 0;
        return state.normal_spare;
    }

    float u1 = 0.0f;
    float u2 = 0.0f;
    float mag = 0.0f;
    float z0 = 0.0f;
    float z1 = 0.0f;

    // Box-Muller 变换核心逻辑
    // 规避 u1 = 0 的情况，防止 logf(0) 导致负无穷
    do {
        u1 = curand_uniform_float(state);
        u2 = curand_uniform_float(state);
    } while (u1 <= 1e-7f);

    // 使用单精度数学函数 (sqrtf, logf, cosf, sinf) 保证 GPU/CPU 性能
    mag = sqrtf(-2.0f * logf(u1));
    z0 = mag * cosf(2.0f * 3.14159265358979323846f * u2);
    z1 = mag * sinf(2.0f * 3.14159265358979323846f * u2);

    // 返回第一个数，将第二个数存入 state 的缓存字段
    state.normal_spare = z1;
    state.has_spare = 1;
    return z0;
}

__forceinline__ __simt_callee__ void set_local_state(__gm__ curandState* global_state, curandState& local_state)
{
    // 当前毕昇编译器不支持localState_ = state_[GlobalThreadId()]操作，使用成员变量依次赋值规避
    auto global_thread_id = GlobalThreadId();
    local_state.x = global_state[global_thread_id].x;
    local_state.y = global_state[global_thread_id].y;
    local_state.z = global_state[global_thread_id].z;
    local_state.w = global_state[global_thread_id].w;
    local_state.v = global_state[global_thread_id].v;
    local_state.d = global_state[global_thread_id].d;
    local_state.normal_spare = global_state[global_thread_id].normal_spare;
    local_state.has_spare = global_state[global_thread_id].has_spare;
}

__forceinline__ __simt_callee__ void set_global_state(const curandState& local_state, __gm__ curandState* global_state)
{
    // 当前毕昇编译器不支持state_[GlobalThreadId()] = localState_操作，使用成员变量依次赋值规避
    auto global_thread_id = GlobalThreadId();
    global_state[global_thread_id].x = local_state.x;
    global_state[global_thread_id].y = local_state.y;
    global_state[global_thread_id].z = local_state.z;
    global_state[global_thread_id].w = local_state.w;
    global_state[global_thread_id].v = local_state.v;
    global_state[global_thread_id].d = local_state.d;
    global_state[global_thread_id].normal_spare = local_state.normal_spare;
    global_state[global_thread_id].has_spare = local_state.has_spare;
}

template <typename T, typename EmbeddingGenerator>
__simt_vf__ __aicore__
LAUNCH_BOUND(EMBEDDING_GENERATOR_MAX_THREADS_PER_BLOCK) inline void generate_local_tensor_kernel_vf(
    __ubuf__ T* local, uint32_t current_tile_size, int64_t vec_id, typename EmbeddingGenerator::Args generator_args)
{
    EmbeddingGenerator emb_gen(generator_args);

    for (uint32_t i = threadIdx.x; i < current_tile_size; i += blockDim.x) {
        auto tmp = emb_gen.generate(vec_id);
        local[i] = dyn_emb::TypeConvertFunc<T, float>::convert(tmp);
    }

    emb_gen.destroy();
}

struct UniformEmbeddingGenerator {
    struct Args {
        __gm__ curandState* state;
        float lower;
        float upper;
    };

    __forceinline__ __simt_callee__ UniformEmbeddingGenerator(Args args)
        : load_(false),
          state_(args.state),
          lower(args.lower),
          upper(args.upper)
    {
    }

    __forceinline__ __aicore__ UniformEmbeddingGenerator() : load_(false) {}

    __forceinline__ __aicore__ void init_simd(Args args)
    {
        state_ = args.state;
        lower = args.lower;
        upper = args.upper;
    }

    __forceinline__ __simt_callee__ float generate(int64_t vec_id)
    {
        if (!load_) {
            set_local_state(state_, localState_);
            load_ = true;
        }
        auto tmp = curand_uniform_float(this->localState_);
        return (upper - lower) * tmp + lower;
    }

    template <typename T>
    __forceinline__ __aicore__ void generate_simd_tensor(LocalTensor<T>& local, int64_t vec_id,
                                                         uint32_t current_tile_size, uint32_t tile_size,
                                                         bool use_zero_fill)
    {
        if (use_zero_fill) {
            T value = dyn_emb::SimdTypeConvertFunc<T, float>::convert(0.0f);
            Duplicate(local, value, static_cast<int32_t>(tile_size));
            return;
        }

        auto generator_args = Args{state_, lower, upper};
        auto local_ptr = reinterpret_cast<__ubuf__ T*>(local.GetPhyAddr());
        asc_vf_call<generate_local_tensor_kernel_vf<T, UniformEmbeddingGenerator>>(
            dim3{EMBEDDING_GENERATOR_MAX_THREADS_PER_BLOCK}, local_ptr, current_tile_size, vec_id, generator_args);
    }

    __forceinline__ __simt_callee__ void destroy()
    {
        if (load_) {
            set_global_state(localState_, state_);
        }
    }

    __forceinline__ __aicore__ void destroy_simd() {}

    bool load_;
    curandState localState_;
    __gm__ curandState* state_;
    float lower;
    float upper;
};

struct NormalEmbeddingGenerator {
    struct Args {
        __gm__ curandState* state;
        float mean;
        float std_dev;
    };

    __forceinline__ __simt_callee__ NormalEmbeddingGenerator(Args args)
        : load_(false),
          state_(args.state),
          mean(args.mean),
          std_dev(args.std_dev)
    {
    }

    __forceinline__ __aicore__ NormalEmbeddingGenerator() : load_(false) {}

    __forceinline__ __aicore__ void init_simd(Args args)
    {
        state_ = args.state;
        mean = args.mean;
        std_dev = args.std_dev;
    }

    __forceinline__ __simt_callee__ float generate(int64_t vec_id)
    {
        if (!load_) {
            set_local_state(state_, localState_);
            load_ = true;
        }
        auto tmp = curand_normal_float(this->localState_);
        return std_dev * tmp + mean;
    }

    template <typename T>
    __forceinline__ __aicore__ void generate_simd_tensor(LocalTensor<T>& local, int64_t vec_id,
                                                         uint32_t current_tile_size, uint32_t tile_size,
                                                         bool use_zero_fill)
    {
        if (use_zero_fill) {
            T value = dyn_emb::SimdTypeConvertFunc<T, float>::convert(0.0f);
            Duplicate(local, value, static_cast<int32_t>(tile_size));
            return;
        }

        auto generator_args = Args{state_, mean, std_dev};
        auto local_ptr = reinterpret_cast<__ubuf__ T*>(local.GetPhyAddr());
        asc_vf_call<generate_local_tensor_kernel_vf<T, NormalEmbeddingGenerator>>(
            dim3{EMBEDDING_GENERATOR_MAX_THREADS_PER_BLOCK}, local_ptr, current_tile_size, vec_id, generator_args);
    }

    __forceinline__ __simt_callee__ void destroy()
    {
        if (load_) {
            set_global_state(localState_, state_);
        }
    }

    __forceinline__ __aicore__ void destroy_simd() {}

    bool load_;
    curandState localState_;
    __gm__ curandState* state_;
    float mean;
    float std_dev;
};

struct TruncatedNormalEmbeddingGenerator {
    struct Args {
        __gm__ curandState* state;
        float mean;
        float std_dev;
        float lower;
        float upper;
    };

    __forceinline__ __simt_callee__ TruncatedNormalEmbeddingGenerator(Args args)
        : load_(false),
          state_(args.state),
          mean(args.mean),
          std_dev(args.std_dev),
          lower(args.lower),
          upper(args.upper)
    {
    }

    __forceinline__ __aicore__ TruncatedNormalEmbeddingGenerator() : load_(false) {}

    __forceinline__ __aicore__ void init_simd(Args args)
    {
        state_ = args.state;
        mean = args.mean;
        std_dev = args.std_dev;
        lower = args.lower;
        upper = args.upper;
    }

    __forceinline__ __simt_callee__ float generate(int64_t vec_id)
    {
        if (!load_) {
            set_local_state(state_, localState_);
            load_ = true;
        }
        auto l = normcdff((lower - mean) / std_dev);
        auto u = normcdff((upper - mean) / std_dev);
        u = 2 * u - 1;
        l = 2 * l - 1;

        float tmp = curand_uniform_float(this->localState_);
        tmp = tmp * (u - l) + l;
        tmp = erfinvf(tmp);
        tmp *= scale * std_dev;
        tmp += mean;
        tmp = __fmaxf(tmp, lower);
        tmp = __fminf(tmp, upper);
        return tmp;
    }

    template <typename T>
    __forceinline__ __aicore__ void generate_simd_tensor(LocalTensor<T>& local, int64_t vec_id,
                                                         uint32_t current_tile_size, uint32_t tile_size,
                                                         bool use_zero_fill)
    {
        if (use_zero_fill) {
            T value = dyn_emb::SimdTypeConvertFunc<T, float>::convert(0.0f);
            Duplicate(local, value, static_cast<int32_t>(tile_size));
            return;
        }

        auto generator_args = Args{state_, mean, std_dev, lower, upper};
        auto local_ptr = reinterpret_cast<__ubuf__ T*>(local.GetPhyAddr());
        asc_vf_call<generate_local_tensor_kernel_vf<T, TruncatedNormalEmbeddingGenerator>>(
            dim3{EMBEDDING_GENERATOR_MAX_THREADS_PER_BLOCK}, local_ptr, current_tile_size, vec_id, generator_args);
    }

    __forceinline__ __simt_callee__ void destroy()
    {
        if (load_) {
            set_global_state(localState_, state_);
        }
    }

    __forceinline__ __aicore__ void destroy_simd() {}

    bool load_;
    curandState localState_;
    __gm__ curandState* state_;
    float mean;
    float std_dev;
    float lower;
    float upper;
    // 初始化scale = 1.4142136，该值等于sqrtf(2.0f)，避免sqrtf计算，同时规避调用sqrtf产生的链接报错
    float scale = 1.4142136;
};

template <typename K>
struct MappingEmbeddingGenerator {
    struct Args {
        const __gm__ K* keys;
        uint64_t mod;
    };

    __forceinline__ __simt_callee__ MappingEmbeddingGenerator(Args args) : mod(args.mod), keys(args.keys) {}

    __forceinline__ __aicore__ MappingEmbeddingGenerator() {}

    __forceinline__ __aicore__ void init_simd(Args args)
    {
        mod = args.mod;
        keys = args.keys;
    }

    __forceinline__ __simt_callee__ float generate(int64_t vec_id)
    {
        K key = keys[vec_id];
        return static_cast<float>(key % mod);
    }

    __forceinline__ __aicore__ float generate_simd(int64_t vec_id)
    {
        K key = keys[vec_id];
        return static_cast<float>(key % mod);
    }

    template <typename T>
    __forceinline__ __aicore__ void generate_simd_tensor(LocalTensor<T>& local, int64_t vec_id,
                                                         uint32_t current_tile_size, uint32_t tile_size,
                                                         bool use_zero_fill)
    {
        (void)current_tile_size;
        float tmp = use_zero_fill ? 0.0f : generate_simd(vec_id);
        T value = dyn_emb::SimdTypeConvertFunc<T, float>::convert(tmp);
        Duplicate(local, value, static_cast<int32_t>(tile_size));
    }

    __forceinline__ __simt_callee__ void destroy() {}
    __forceinline__ __aicore__ void destroy_simd() {}
    uint64_t mod;
    const __gm__ K* keys;
};

struct ConstEmbeddingGenerator {
    struct Args {
        float val;
    };

    __forceinline__ __simt_callee__ ConstEmbeddingGenerator(Args args) : val(args.val) {}

    __forceinline__ __aicore__ ConstEmbeddingGenerator() {}

    __forceinline__ __aicore__ void init_simd(Args args)
    {
        val = args.val;
    }

    __forceinline__ __simt_callee__ float generate(int64_t vec_id)
    {
        (void)vec_id;
        return val;
    }

    __forceinline__ __aicore__ float generate_simd(int64_t vec_id)
    {
        (void)vec_id;
        return val;
    }

    template <typename T>
    __forceinline__ __aicore__ void generate_simd_tensor(LocalTensor<T>& local, int64_t vec_id,
                                                         uint32_t current_tile_size, uint32_t tile_size,
                                                         bool use_zero_fill)
    {
        (void)vec_id;
        (void)current_tile_size;
        float tmp = use_zero_fill ? 0.0f : val;
        T value = dyn_emb::SimdTypeConvertFunc<T, float>::convert(tmp);
        Duplicate(local, value, static_cast<int32_t>(tile_size));
    }

    __forceinline__ __simt_callee__ void destroy() {}
    __forceinline__ __aicore__ void destroy_simd() {}
    float val;
};
}  // namespace dyn_emb
