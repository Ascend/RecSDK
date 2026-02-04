/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
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

#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <acl/acl.h>
namespace dyn_emb {

// 计算大于等于value的最小2的幂
inline int64_t next_power_of_two(int64_t value)
{
    int64_t result = 1;
    while (result < value) {
        result *= 2;
    }
    return result;
}

enum class DataType : uint32_t {
    Float32 = 0,
    Float16,
    BFloat16,
    Int64,
    UInt64,
    Int32,
    UInt32,
    Size_t,
};

enum class EvictStrategy : uint32_t {
    kLru = 0,
    kLfu = 1,      // dynamicemb don't use
    kEpochLru = 2, // dynamicemb don't use
    kEpochLfu = 3, // dynamicemb don't use
    kCustomized = 4,
};

enum class SafeCheckMode : int { ERROR = 0, WARNING = 1, IGNORE = 2 };

enum class OptimizerType : int {
    Null = 0, // used in inference mode.
    SGD,
    Adam,
    AdamW,
    AdaGrad,
    RowWiseAdaGrad,
};

// host侧没有half和bfloat16类型的定义，这里定义为uint16_t以保持相同的类型占位宽度
struct half {
    uint16_t value;
};

struct bfloat16_t {
    uint16_t value;
};

#define CASE_TYPE_USING_HINT(enum_type, type, HINT, ...)                       \
    case (enum_type): {                                                          \
        using HINT = type;                                                         \
        __VA_ARGS__();                                                             \
        break;                                                                     \
    }

#define CASE_ENUM_USING_HINT(enum_type, HINT, ...)                             \
    case (enum_type): {                                                          \
        constexpr auto HINT = enum_type;                                           \
        __VA_ARGS__();                                                             \
        break;                                                                     \
    }

#define DISPATCH_INTEGER_DATATYPE_FUNCTION(DATA_TYPE, HINT, ...)               \
    switch (DATA_TYPE) {                                                         \
        CASE_TYPE_USING_HINT(DataType::Int64, int64_t, HINT, __VA_ARGS__)          \
        CASE_TYPE_USING_HINT(DataType::UInt64, uint64_t, HINT, __VA_ARGS__)        \
    default:                                                                     \
        exit(EXIT_FAILURE);                                                        \
    }

#define DISPATCH_FLOAT_DATATYPE_FUNCTION(DATA_TYPE, HINT, ...)                 \
    switch (DATA_TYPE) {                                                         \
        CASE_TYPE_USING_HINT(DataType::Float32, float, HINT, __VA_ARGS__)          \
        CASE_TYPE_USING_HINT(DataType::Float16, half, HINT, __VA_ARGS__)         \
        CASE_TYPE_USING_HINT(DataType::BFloat16, bfloat16_t, HINT, __VA_ARGS__) \
    default:                                                                     \
       exit(EXIT_FAILURE);                                                        \
    }

#define DISPATCH_EVICTYPE_FUNCTION(EVICT_TYPE, HINT, ...)                      \
    switch (EVICT_TYPE) {                                                        \
        CASE_ENUM_USING_HINT(EvictStrategy::kLru, HINT, __VA_ARGS__)               \
        CASE_ENUM_USING_HINT(EvictStrategy::kCustomized, HINT, __VA_ARGS__)        \
        CASE_ENUM_USING_HINT(EvictStrategy::kLfu, HINT, __VA_ARGS__)               \
    default:                                                                     \
        exit(EXIT_FAILURE);                                                        \
    }

} // namespace dyn_emb

#endif // UTILS_H
