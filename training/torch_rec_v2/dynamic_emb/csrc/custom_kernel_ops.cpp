/*
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
#include "custom_kernel_ops.h"
#include "./ops/load_from_pointer/load_from_pointer_kernel_hybrid.h"
#include "./ops/maskselect_op/maskselect_op_kernel.h"
#include "./ops/look_backward/lookup_backward_v2_kernel.h"
#include "./ops/ops_utils.h"
#include "utils.h"

namespace dyn_emb {
namespace {

constexpr int32_t kMaskSelectMaxThreadsPerBlock = 1024;
constexpr int32_t kMaskSelectMaxElementsPerThread = 4;
constexpr int32_t kMaskSelectElementsPerBlock = kMaskSelectMaxThreadsPerBlock * kMaskSelectMaxElementsPerThread;
constexpr int32_t kMaskSelectCumsumDivisor = 4;
constexpr int32_t kMaskSelectCumsumMultiplier = 2;
constexpr int32_t kMaskSelectBufferNum = 2;
constexpr int32_t kMaskSelectUbOverheadBytes = 288;

int32_t ComputeMaskSelectMaxElementsPerBlock(uint64_t ub_size, bool select_index)
{
    if (ub_size <= kMaskSelectUbOverheadBytes) {
        return 1;
    }
    const uint64_t available = ub_size - kMaskSelectUbOverheadBytes;
    // flags(2*epb) + prefix(16*epb) + [keys(16*epb)] + reduceTmp(8*epb)
    const uint64_t bytesPerElement = select_index ? 26ULL : 42ULL;
    int32_t maxElements = static_cast<int32_t>(available / bytesPerElement);
    if (maxElements > kMaskSelectElementsPerBlock) {
        maxElements = kMaskSelectElementsPerBlock;
    }
    if (maxElements < kMaskSelectMaxThreadsPerBlock) {
        maxElements = kMaskSelectMaxThreadsPerBlock;
    }
    return maxElements;
}

struct MaskSelectLaunchConfig {
    int32_t isSmall;
    int32_t isFullCore;
    int32_t totalBlocks;
    int32_t blocksPerCore;
    int32_t remainderBlocks;
    int32_t elementsPerBlock;
    int32_t coreNum;
};

MaskSelectLaunchConfig ComputeMaskSelectLaunchConfig(int64_t num_total, uint32_t max_cores, uint64_t ub_size,
                                                     bool select_index)
{
    MaskSelectLaunchConfig config{};
    config.coreNum = 1;
    if (num_total <= 0) {
        return config;
    }

    int32_t smallThreshold = static_cast<int32_t>(max_cores) * kMaskSelectMaxThreadsPerBlock / kMaskSelectCumsumDivisor;
    int32_t smallThreshold64 =
        static_cast<int32_t>(max_cores) * kMaskSelectMaxThreadsPerBlock * kMaskSelectCumsumMultiplier + smallThreshold;

    config.elementsPerBlock = kMaskSelectElementsPerBlock;
    config.isSmall = 0;
    if (num_total <= smallThreshold64) {
        config.isSmall = 1;
        config.elementsPerBlock = kMaskSelectMaxThreadsPerBlock;
    } else {
        const int32_t maxElementsPerBlock = ComputeMaskSelectMaxElementsPerBlock(ub_size, select_index);
        if (config.elementsPerBlock > maxElementsPerBlock) {
            config.elementsPerBlock = maxElementsPerBlock;
        }
    }
    config.totalBlocks = static_cast<int32_t>((num_total + config.elementsPerBlock - 1) / config.elementsPerBlock);
    config.isFullCore = (config.totalBlocks > static_cast<int32_t>(max_cores)) ? 1 : 0;
    config.coreNum = config.isFullCore ? static_cast<int32_t>(max_cores) : config.totalBlocks;
    if (config.coreNum <= 0) {
        config.coreNum = 1;
    }
    config.blocksPerCore = config.totalBlocks / config.coreNum;
    config.remainderBlocks = config.totalBlocks % config.coreNum;
    return config;
}

}  // namespace

void load_from_pointer_hybrid_ops(void* pointers, void* dst, uint32_t dim, uint32_t num, aclrtStream stream,
                                  uint32_t coreNum, uint32_t oType, uint64_t totalUbSize)
{
    dyn_emb::DataType outType = static_cast<dyn_emb::DataType>(oType);
    FLOAT_TYPE_DISPATCH(outType, DataType, {
        auto launchTiling = dyn_emb::ComputeSimdValueMoveLaunchTiling(num, coreNum, dim, sizeof(DataType), totalUbSize);
        load_from_pointer_kernel_hybrid<DataType><<<launchTiling.block_dim, launchTiling.valid_ub_size, stream>>>(
            launchTiling.former_num, launchTiling.former_core_move_num, launchTiling.tail_core_move_num,
            launchTiling.tile_size, launchTiling.num_tiles, dim, reinterpret_cast<DataType*>(dst), num,
            reinterpret_cast<DataType**>(pointers));
    });
}

#define LOOKUP_BACKWARD_V2_KERNEL_LAUNCH(IS_FLOAT2, VALUE_T)                                                         \
    do {                                                                                                             \
        if (is_mean) {                                                                                               \
            lookup_backward_v2_kernel<DTYPE_X, VALUE_T, true, IS_FLOAT2><<<core_num, 0, stream>>>(                   \
                reinterpret_cast<VALUE_T*>(grad), reinterpret_cast<VALUE_T*>(unique_buffer),                         \
                reinterpret_cast<DTYPE_X*>(inverse_indices), reinterpret_cast<DTYPE_X*>(biased_offsets), launch_dim, \
                num_slots, total_blocks, blocks_per_core, remainder_blocks, is_small);                               \
        } else {                                                                                                     \
            lookup_backward_v2_kernel<DTYPE_X, VALUE_T, false, IS_FLOAT2><<<core_num, 0, stream>>>(                  \
                reinterpret_cast<VALUE_T*>(grad), reinterpret_cast<VALUE_T*>(unique_buffer),                         \
                reinterpret_cast<DTYPE_X*>(inverse_indices), reinterpret_cast<DTYPE_X*>(biased_offsets), launch_dim, \
                num_slots, total_blocks, blocks_per_core, remainder_blocks, is_small);                               \
        }                                                                                                            \
    } while (0)

void lookup_backward_v2_launch(void* grad, void* unique_buffer, void* inverse_indices, void* biased_offsets,
                               int32_t launch_dim, int32_t num_slots, int32_t combiner, int32_t total_blocks,
                               int32_t blocks_per_core, int32_t remainder_blocks, uint32_t index_type, bool is_small,
                               bool is_float2, uint32_t value_type, int32_t core_num, aclrtStream stream)
{
    const bool is_mean = (combiner == 1);
    const dyn_emb::DataType index_enum = static_cast<dyn_emb::DataType>(index_type);
    const dyn_emb::DataType value_enum = static_cast<dyn_emb::DataType>(value_type);

    if (is_float2) {
        INDEX_DTYPE_DISPATCH(index_enum, DTYPE_X, { LOOKUP_BACKWARD_V2_KERNEL_LAUNCH(true, float2); });
        return;
    }

    INDEX_DTYPE_DISPATCH(index_enum, DTYPE_X, {
        FLOAT_TYPE_DISPATCH(value_enum, value_t, { LOOKUP_BACKWARD_V2_KERNEL_LAUNCH(false, value_t); });
    });
}

void maskselect_ops(void* flags, uint8_t* inputs, void* outputs, void* num_selected, void* workspace, int64_t num_total,
                    uint32_t is_uint64, uint32_t select_index, aclrtStream stream, uint32_t max_cores, uint64_t ub_size)
{
    auto flags_ptr = reinterpret_cast<bool*>(flags);
    auto num_selected_ptr = reinterpret_cast<int64_t*>(num_selected);
    auto workspace_ptr = reinterpret_cast<int64_t*>(workspace);
    const dyn_emb::DataType key_type = (is_uint64 != 0) ? dyn_emb::DataType::UInt64 : dyn_emb::DataType::Int64;
    MaskSelectLaunchConfig config = ComputeMaskSelectLaunchConfig(num_total, max_cores, ub_size, select_index != 0);

    INT_TYPE_DISPATCH(key_type, KeyType, {
        static_assert(sizeof(KeyType) == sizeof(int64_t));
        auto inputs_ptr = reinterpret_cast<int64_t*>(inputs);
        auto outputs_ptr = reinterpret_cast<int64_t*>(outputs);

        if (select_index != 0) {
            maskselect_index_op<<<config.coreNum, ub_size, stream>>>(
                flags_ptr, outputs_ptr, num_selected_ptr, workspace_ptr, num_total, static_cast<int32_t>(is_uint64),
                config.isSmall, config.isFullCore, config.totalBlocks, config.blocksPerCore, config.remainderBlocks,
                config.elementsPerBlock);
            return;
        }

        maskselect_op<<<config.coreNum, ub_size, stream>>>(
            flags_ptr, inputs_ptr, outputs_ptr, num_selected_ptr, workspace_ptr, num_total,
            static_cast<int32_t>(is_uint64), config.isSmall, config.isFullCore, config.totalBlocks,
            config.blocksPerCore, config.remainderBlocks, config.elementsPerBlock);
    });
}

int64_t GetMaskSelectWorkspaceElems(int64_t num_total, uint32_t max_cores, uint64_t ub_size, bool select_index)
{
    MaskSelectLaunchConfig config = ComputeMaskSelectLaunchConfig(num_total, max_cores, ub_size, select_index);
    return num_total + static_cast<int64_t>(config.totalBlocks);
}

}  // namespace dyn_emb
