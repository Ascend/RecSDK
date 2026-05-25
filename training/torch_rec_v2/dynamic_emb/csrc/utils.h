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

namespace dyn_emb {

constexpr uint32_t kDoubleBuffer = 2;

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
    kLfu = 1,       // dynamicemb don't use
    kEpochLru = 2,  // dynamicemb don't use
    kEpochLfu = 3,  // dynamicemb don't use
    kCustomized = 4,
};

enum class SafeCheckMode : int {
    ERROR = 0,
    WARNING = 1,
    IGNORE = 2
};

enum class OptimizerType : int {
    Null = 0,  // used in inference mode.
    SGD,
    Adam,
    AdamW,
    AdaGrad,
    RowWiseAdaGrad,
};

#define CASE_TYPE_USING_HINT(enum_type, type, HINT, ...)                       \
    case (enum_type): {                                                          \
        using HINT = type;                                                         \
        __VA_ARGS__();                                                             \
        break;                                                                     \
    }

#define CASE_ENUM_USING_HINT(enum_type, HINT, ...) \
    case (enum_type): {                            \
        constexpr auto HINT = enum_type;           \
        __VA_ARGS__();                             \
        break;                                     \
    }

#define DISPATCH_INTEGER_DATATYPE_FUNCTION(DATA_TYPE, HINT, ...)            \
    switch (DATA_TYPE) {                                                    \
        CASE_TYPE_USING_HINT(DataType::Int64, int64_t, HINT, __VA_ARGS__)   \
        CASE_TYPE_USING_HINT(DataType::UInt64, uint64_t, HINT, __VA_ARGS__) \
        default:                                                            \
            exit(EXIT_FAILURE);                                             \
    }

#define DISPATCH_FLOAT_DATATYPE_FUNCTION(DATA_TYPE, HINT, ...)                  \
    switch (DATA_TYPE) {                                                        \
        CASE_TYPE_USING_HINT(DataType::Float32, float, HINT, __VA_ARGS__)       \
        CASE_TYPE_USING_HINT(DataType::Float16, half, HINT, __VA_ARGS__)        \
        CASE_TYPE_USING_HINT(DataType::BFloat16, bfloat16_t, HINT, __VA_ARGS__) \
        default:                                                                \
            exit(EXIT_FAILURE);                                                 \
    }

#define DISPATCH_EVICTYPE_FUNCTION(EVICT_TYPE, HINT, ...)                   \
    switch (EVICT_TYPE) {                                                   \
        CASE_ENUM_USING_HINT(EvictStrategy::kLru, HINT, __VA_ARGS__)        \
        CASE_ENUM_USING_HINT(EvictStrategy::kCustomized, HINT, __VA_ARGS__) \
        CASE_ENUM_USING_HINT(EvictStrategy::kLfu, HINT, __VA_ARGS__)        \
        default:                                                            \
            exit(EXIT_FAILURE);                                             \
    }


// SIMD 向量搬运类算子的启动分核与 tiling 参数。
struct SimdValueMoveLaunchTiling {
    uint32_t block_dim;              // 实际启动核数
    uint32_t former_num;             // 多搬运 1 条数据的核数
    uint64_t former_core_move_num;   // 上述每个核搬运的条数
    uint64_t tail_core_move_num;       // 其余核每个核搬运的条数
    uint64_t valid_ub_size;            // 内核 UB 可用空间，用于 <<<block_dim, valid_ub_size, stream>>>
    uint32_t tile_size;                // 每条数据按 dim 切分时的 tile 大小
    uint32_t num_tiles;                // 每条数据的 tile 个数
};

/**
 * @brief 计算 SIMD 向量 value-move 类算子的启动分核与 tiling 参数。
 *
 * 当 n < max_block_dim 时，仅 former_num（等于 n）个核有搬运任务，会将 block_dim 缩减为 n，
 * 避免多余核心空转。
 *
 * @param n              待搬运的数据条数（如指针个数）
 * @param max_block_dim  平台最大可用核数（通常取 AclSingleton::GetMaxCores()）
 * @param dim            每条数据的元素个数（embedding 维度）
 * @param element_size   单个元素字节数（如 sizeof(float)）
 * @param valid_ub_size  单核可用 UB 大小，由调用方根据场景选择：
 *                       - 纯 SIMD 算子：AclSingleton::GetInstance().GetTotalUbSize()
 *                       - 混合 SIMT+SIMD 算子：AclSingleton::GetInstance().GetMixedOpUbSize()
 * @param buffer_num     UB 双缓冲份数，默认 kDoubleBuffer（2）
 *
 * @return SimdValueMoveLaunchTiling 启动参数，用于内核 launch 及内核内分核逻辑。
 *
 * 调用示例（load_from_pointer_hybrid）：
 * @code
 *   uint32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
 *   uint64_t ubSize = AclSingleton::GetInstance().GetTotalUbSize();  // 或 GetMixedOpUbSize()
 *   auto tiling = ComputeSimdValueMoveLaunchTiling(num, maxCores, dim, sizeof(T), ubSize);
 *   kernel<T><<<tiling.block_dim, tiling.valid_ub_size, stream>>>(
 *       tiling.former_num, tiling.former_core_move_num, tiling.tail_core_move_num,
 *       tiling.tile_size, tiling.num_tiles, dim, dst, num, src_ptrs);
 * @endcode
 */
inline SimdValueMoveLaunchTiling ComputeSimdValueMoveLaunchTiling(uint32_t n, uint32_t max_block_dim, uint32_t dim,
                                                                uint32_t element_size, uint64_t valid_ub_size,
                                                                uint32_t buffer_num = kDoubleBuffer)
{
    SimdValueMoveLaunchTiling info;
    info.block_dim = max_block_dim;
    if (n > 0 && n < max_block_dim) {
        info.block_dim = n;
    }
    if (info.block_dim == 0) {
        info.block_dim = 1;
    }

    info.tail_core_move_num = n / info.block_dim;
    info.former_core_move_num = info.tail_core_move_num + 1;
    info.former_num = n - info.tail_core_move_num * info.block_dim;
    info.valid_ub_size = valid_ub_size;

    uint32_t max_tile_size = static_cast<uint32_t>(info.valid_ub_size / (buffer_num * element_size));
    if (max_tile_size == 0) {
        max_tile_size = 1;
    }
    info.tile_size = (dim <= max_tile_size) ? dim : max_tile_size;
    info.num_tiles = (dim + info.tile_size - 1) / info.tile_size;
    return info;
}

class DeviceProp {
public:
    static DeviceProp& getDeviceProp(int device_id = 0);
    // DeviceProp(const DeviceProp&) = delete; //TODO: whether to remove
    DeviceProp& operator=(const DeviceProp&) = delete;

    int num_sms = 0;
    int warp_size = 0;
    int max_thread_per_sm = 0;
    int max_thread_per_block = 0;
    int total_threads = 0;

private:
    explicit DeviceProp(int device_id);
    ~DeviceProp() = default;
};
}  // namespace dyn_emb

#endif  // UTILS_H
