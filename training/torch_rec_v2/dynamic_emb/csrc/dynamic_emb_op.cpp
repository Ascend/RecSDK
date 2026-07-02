/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
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

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include "securec.h"
#include <iostream>
#include <string>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclnn/aclnn_base.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_unique2.h"

#include "acl_singleton.h"
#include "aclrtlaunch_Adam_update.h"
#include "aclrtlaunch_Adam_update_float2.h"
#include "aclrtlaunch_update.h"
#include "aclrtlaunch_update_float2.h"
#include "aclrtlaunch_adagrad_fused_simd.h"
#include "aclrtlaunch_adagrad_simd.h"
#include "aclrtlaunch_adamw_fused_simd.h"
#include "aclrtlaunch_adamw_simd.h"
#include "aclrtlaunch_sgd_fused_simd.h"
#include "aclrtlaunch_sgd_simd.h"
#include "aclrtlaunch_update_fused.h"
#include "aclrtlaunch_update_float2_fused.h"
#include "optimizer_kind.h"
#include "aclrtlaunch_block_bucketsize_sparse_features.h"
#include "aclrtlaunch_device_timestamp.h"
#include "aclrtlaunch_gather_dim0.h"
#include "aclrtlaunch_get_new_length_and_offsets_op.h"
#include "aclrtlaunch_get_table_range_op.h"
#include "aclrtlaunch_load_from_pointer.h"
#include "aclrtlaunch_reduce_grad_op.h"
#include "aclrtlaunch_unique_op.h"
#include "aclrtlaunch_select_op.h"
#include "aclrtlaunch_select_index_op.h"
#include "aclrtlaunch_pooling_embeddings.h"
#include "aclrtlaunch_pooling_embeddings_simd.h"
#include "aclrtlaunch_lookup_backward.h"
#include "aclrtlaunch_rowwise_adagrad_update.h"
#include "aclrtlaunch_rowwise_adagrad_update_float2.h"
#include "aclrtlaunch_rowwise_adagrad_fused.h"
#include "aclrtlaunch_rowwise_adagrad_fused_float2.h"
#include "aclrtlaunch_rowwise_adagrad_simd.h"
#include "aclrtlaunch_rowwise_adagrad_fused_simd.h"
#include "./ops/Rowwise_adagrad_update/rowwise_adagrad_simd_tiling.h"
#include "dynamic_variable_base.h"
#include "initializer.h"
#include "torch_utils.h"
#include "utils.h"
#include "./custom_kernel_ops.h"
#include "./ops/unique_op/cpu_unique.h"
#include "./ops/Adagrad_update/adagrad_simd_tiling.h"
#include "./ops/AdamW_update/adamw_simd_tiling.h"
#include "./ops/sgd_update/sgd_simd_tiling.h"
#include "./ops/pooling_embeddings/pooling_embeddings_simd_tiling.h"
#define LOG_ERROR(msg) std::cout << "[INFO]" << msg << std::endl
namespace py = pybind11;
namespace dyn_emb {
constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
constexpr int32_t SMALL_DATA_THRESHOLD = 44 * MAX_THREADS_PER_BLOCK;
constexpr int32_t SMALL_DATA_THRESHOLD_32 = 24 * MAX_THREADS_PER_BLOCK;
constexpr int32_t ELEMENTS_PER_BLOCK = MAX_THREADS_PER_BLOCK * MAX_ELEMENTS_PER_THREAD;

// lookup_backward: inline binary search + per-thread ResolveGradSource cache in SIMT scatter.
// Profiled on A5 via tests/perf/dynamic_emb_op/look_backward.py.
//
// lookup_backward_v2: slot-path SIMT scatter, template kernel <<<>>> direct invoke.
// Host: lookup_backward_v2_launch() in custom_kernel_ops; empty slots skipped in kernel.
// Falls back to lookup_backward when empty_ratio < 5% and num_key > 2 * num_active_slots.
// Active-slot stats (CPU) run only when num_key > num_slots; otherwise fallback is impossible.
//
// SIMT tiling / block scheduling: num_key * launch_dim (launch_dim = dim or dim/2 for float2).
constexpr int32_t LOOKUP_BACKWARD_FLOAT2_DIM_THRESHOLD = 8;
constexpr float LOOKUP_BACKWARD_V2_EMPTY_RATIO_THRESHOLD = 0.05f;
constexpr int64_t LOOKUP_BACKWARD_V2_KEYS_PER_ACTIVE_FACTOR = 2;
constexpr float HASH_TABLE_FACTOR = 1.5f;
constexpr int64_t MIN_HASH_TABLE_CAPACITY = 1;
constexpr int32_t CACHE_ALIGN = 64;
constexpr size_t MIN_CORE_NUM = 1;
constexpr int32_t CPU_KEY_THRESHOLD = 10000;
using ReturnType =
    std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>>;

static bool LaunchUpdateKernelCommon(aclrtStream stream, uint8_t* gradsPtr, uint8_t* valuesPtr, uint8_t* foundsPtr,
                                     uint32_t gradDim, int32_t inLength, int32_t maxCores, bool isSmall, float beta1,
                                     float beta2, float oneMinusBeta1, float oneMinusBeta2, float stepSize,
                                     float invVHatDenom, float decayFactor, float eps, DataType gradType,
                                     DataType valType, uint32_t optimizerKind)
{
    constexpr int32_t maxElementsPerThread = 2;
    const int32_t elementsPerBlock = MAX_THREADS_PER_BLOCK * maxElementsPerThread;
    if (gradDim % 2 == 0 && gradType == DataType::Float32 && valType == DataType::Float32) {
        int32_t vecLength = inLength / 2;
        const int32_t vecPerBlock = elementsPerBlock / 2;
        int32_t totalBlocks = (vecLength + vecPerBlock - 1) / vecPerBlock;
        int32_t coreNum = std::min(maxCores, totalBlocks);
        if (coreNum == 0) {
            LOG_ERROR("core_num is zero!");
            return false;
        }
        int32_t blocksPerCore = totalBlocks / coreNum;
        int32_t remainderBlocks = totalBlocks % coreNum;
        ACLRT_LAUNCH_KERNEL(update_float2)
        (coreNum, stream, gradsPtr, valuesPtr, foundsPtr, gradDim, vecLength, beta1, beta2, oneMinusBeta1,
         oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, totalBlocks, blocksPerCore, remainderBlocks, isSmall,
         static_cast<uint32_t>(gradType), static_cast<uint32_t>(valType), optimizerKind, maxElementsPerThread);
    } else {
        int32_t totalBlocks = (inLength + elementsPerBlock - 1) / elementsPerBlock;
        int32_t coreNum = std::min(maxCores, totalBlocks);
        if (coreNum == 0) {
            LOG_ERROR("core_num is zero!");
            return false;
        }
        int32_t blocksPerCore = totalBlocks / coreNum;
        int32_t remainderBlocks = totalBlocks % coreNum;
        ACLRT_LAUNCH_KERNEL(update)
        (coreNum, stream, gradsPtr, valuesPtr, foundsPtr, gradDim, inLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2,
         stepSize, invVHatDenom, decayFactor, eps, totalBlocks, blocksPerCore, remainderBlocks, isSmall,
         static_cast<uint32_t>(gradType), static_cast<uint32_t>(valType), optimizerKind, maxElementsPerThread);
    }
    return true;
}

static bool LaunchUpdateFusedKernelCommon(aclrtStream stream, uint8_t* gradsPtr, uint8_t* valuesPtr, uint32_t gradDim,
                                          uint32_t valDim, int32_t inLength, int32_t maxCores, bool isSmall,
                                          float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2,
                                          float stepSize, float invVHatDenom, float decayFactor, float eps,
                                          DataType gradType, DataType valType, uint32_t optimizerKind)
{
    constexpr int32_t maxElementsPerThread = 2;
    const int32_t elementsPerBlock = MAX_THREADS_PER_BLOCK * maxElementsPerThread;
    if (gradDim % 2 == 0 && gradType == DataType::Float32 && valType == DataType::Float32) {
        int32_t vecLength = inLength / 2;
        const int32_t vecPerBlock = elementsPerBlock / 2;
        int32_t totalBlocks = (vecLength + vecPerBlock - 1) / vecPerBlock;
        int32_t coreNum = std::min(maxCores, totalBlocks);
        if (coreNum == 0) {
            LOG_ERROR("core_num is zero!");
            return false;
        }
        int32_t blocksPerCore = totalBlocks / coreNum;
        int32_t remainderBlocks = totalBlocks % coreNum;
        ACLRT_LAUNCH_KERNEL(update_float2_fused)
        (coreNum, stream, gradsPtr, valuesPtr, nullptr, gradDim, valDim, vecLength, beta1, beta2, oneMinusBeta1,
         oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, totalBlocks, blocksPerCore, remainderBlocks, isSmall,
         static_cast<uint32_t>(gradType), static_cast<uint32_t>(valType), optimizerKind, maxElementsPerThread);
    } else {
        int32_t totalBlocks = (inLength + elementsPerBlock - 1) / elementsPerBlock;
        int32_t coreNum = std::min(maxCores, totalBlocks);
        if (coreNum == 0) {
            LOG_ERROR("core_num is zero!");
            return false;
        }
        int32_t blocksPerCore = totalBlocks / coreNum;
        int32_t remainderBlocks = totalBlocks % coreNum;
        ACLRT_LAUNCH_KERNEL(update_fused)
        (coreNum, stream, gradsPtr, valuesPtr, nullptr, gradDim, valDim, inLength, beta1, beta2, oneMinusBeta1,
         oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, totalBlocks, blocksPerCore, remainderBlocks, isSmall,
         static_cast<uint32_t>(gradType), static_cast<uint32_t>(valType), optimizerKind, maxElementsPerThread);
    }
    return true;
}
static constexpr int64_t kAdagradSimdElemThreshold = 4000000LL;
static constexpr int64_t kOptimizerSimdElemThreshold = 4000000LL;

static bool IsAdagradSimdDtype(DataType dtype)
{
    return dtype == DataType::Float32 || dtype == DataType::Float16 || dtype == DataType::BFloat16;
}

static bool ShouldUseRowwiseAdagradSimd(uint32_t gradDim, DataType gradType, DataType valType,
                                        std::shared_ptr<dyn_emb::DynamicVariableBase> ht = nullptr)
{
    if (ht != nullptr && !ht->is_pure_hbm_mode()) {
        return true;
    }
    return (gradDim >= 4096U) && (gradDim % 8U == 0U);
}

static bool LaunchRowwiseAdagradSimdTiling(const at::Tensor& gradsContinuous, uint32_t gradDim, uint32_t valDim,
                                           int32_t numRows, int32_t maxCores, float lr, float eps,
                                           uint32_t rowsPerGroup, DataType gradType, DataType weightType,
                                           int32_t& coreNumOut, at::Tensor& tilingNpuOut)
{
    if (!IsAdagradSimdDtype(gradType) || !IsAdagradSimdDtype(weightType)) {
        LOG_ERROR("LaunchRowwiseAdagradSimdTiling: unsupported grad/weight dtype for Rowwise Adagrad SIMD. "
                  "Supported: Float32, Float16, BFloat16.");
        return false;
    }

    const int32_t numGroups = (numRows + static_cast<int32_t>(rowsPerGroup) - 1) / static_cast<int32_t>(rowsPerGroup);
    coreNumOut = static_cast<int32_t>(
        std::max<int64_t>(1, std::min<int64_t>(static_cast<int64_t>(maxCores), static_cast<int64_t>(numGroups))));

    RowwiseAdagradSimdTilingData tilingData{};
    tilingData.gradDim = gradDim;
    tilingData.valDim = valDim;
    tilingData.numRows = numRows;
    tilingData.lr = lr;
    tilingData.eps = eps;
    tilingData.needCoreNum = coreNumOut;
    tilingData.rowsPerGroup = rowsPerGroup;
    tilingData.gradType = static_cast<uint32_t>(gradType);
    tilingData.weightType = static_cast<uint32_t>(weightType);

    at::Tensor tilingHost = at::empty({static_cast<int64_t>(sizeof(RowwiseAdagradSimdTilingData))},
                                      at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    if (memcpy_s(tilingHost.data_ptr(), tilingHost.nbytes(), &tilingData, sizeof(tilingData)) != EOK) {
        return false;
    }
    tilingNpuOut = tilingHost.to(gradsContinuous.device()).contiguous();
    return true;
}

static bool LaunchRowwiseAdagradFusedSimd(aclrtStream stream, uint8_t* gradsPtr, uint8_t* valuesPtr,
                                          const at::Tensor& gradsContinuous, uint32_t gradDim, uint32_t valDim,
                                          int32_t numRows, int32_t maxCores, float lr, float eps, DataType gradType,
                                          DataType weightType)
{
    constexpr uint32_t kRowsPerGroup = 1U;
    int32_t coreNum = 0;
    at::Tensor tilingNpu;
    if (!LaunchRowwiseAdagradSimdTiling(gradsContinuous, gradDim, valDim, numRows, maxCores, lr, eps, kRowsPerGroup,
                                        gradType, weightType, coreNum, tilingNpu)) {
        return false;
    }
    ACLRT_LAUNCH_KERNEL(rowwise_adagrad_fused_simd)
    (coreNum, stream, gradsPtr, valuesPtr, static_cast<uint8_t*>(tilingNpu.data_ptr()));
    return true;
}

static bool LaunchRowwiseAdagradSimd(aclrtStream stream, uint8_t* gradsPtr, uint8_t* rowPtrsPtr, uint8_t* foundsPtr,
                                     const at::Tensor& gradsContinuous, uint32_t gradDim, uint32_t valDim,
                                     int32_t numRows, int32_t maxCores, float lr, float eps, uint32_t rowsPerGroup,
                                     DataType gradType, DataType weightType)
{
    int32_t coreNum = 0;
    at::Tensor tilingNpu;
    if (!LaunchRowwiseAdagradSimdTiling(gradsContinuous, gradDim, valDim, numRows, maxCores, lr, eps, rowsPerGroup,
                                        gradType, weightType, coreNum, tilingNpu)) {
        return false;
    }
    ACLRT_LAUNCH_KERNEL(rowwise_adagrad_simd)
    (coreNum, stream, gradsPtr, rowPtrsPtr, foundsPtr, static_cast<uint8_t*>(tilingNpu.data_ptr()));
    return true;
}

static bool IsSupportedOptimizerSimdDtype(DataType gradType, DataType valType)
{
    const auto isFloatLike = [](DataType t) {
        return t == DataType::Float32 || t == DataType::Float16 || t == DataType::BFloat16;
    };
    return isFloatLike(gradType) && isFloatLike(valType);
}

static bool ShouldUseOptimizerSimd(uint32_t gradDim, int32_t inLength, DataType gradType, DataType valType)
{
    return (gradDim > 512U) && (gradDim % 8U == 0U) &&
           (static_cast<int64_t>(inLength) >= kOptimizerSimdElemThreshold) &&
           IsSupportedOptimizerSimdDtype(gradType, valType);
}

static uint32_t AdagradSimdRowsPerGroup(DataType gradType, DataType valType)
{
    return (gradType == DataType::Float32 && valType == DataType::Float32) ? 2U : 1U;
}

static bool ShouldUseAdagradSimd(uint32_t gradDim, int32_t inLength, DataType gradType, DataType valType,
                                 const std::shared_ptr<dyn_emb::DynamicVariableBase> ht = nullptr)
{
    if (ht != nullptr && !ht->is_pure_hbm_mode()) {
        return true;
    }
    return (gradDim > 512U) && (gradDim % 8U == 0U) && (static_cast<int64_t>(inLength) >= kAdagradSimdElemThreshold);
}

static bool LaunchAdagradSimdTiling(const at::Tensor& gradsContinuous, uint32_t gradDim, uint32_t valDim,
                                    int32_t numRows, int32_t maxCores, float lr, float eps, uint32_t rowsPerGroup,
                                    DataType gradType, DataType weightType, int32_t& coreNumOut,
                                    at::Tensor& tilingNpuOut)
{
    if (!IsAdagradSimdDtype(gradType) || !IsAdagradSimdDtype(weightType)) {
        LOG_ERROR("LaunchAdagradSimdTiling: unsupported grad/weight dtype for Adagrad SIMD. "
                  "Supported: Float32, Float16, BFloat16.");
        return false;
    }

    const int32_t numGroups = (numRows + static_cast<int32_t>(rowsPerGroup) - 1) / static_cast<int32_t>(rowsPerGroup);
    coreNumOut = static_cast<int32_t>(
        std::max<int64_t>(1, std::min<int64_t>(static_cast<int64_t>(maxCores), static_cast<int64_t>(numGroups))));

    AdagradSimdTilingData tilingData{};
    tilingData.gradDim = gradDim;
    tilingData.valDim = valDim;
    tilingData.numRows = numRows;
    tilingData.lr = lr;
    tilingData.eps = eps;
    tilingData.needCoreNum = coreNumOut;
    tilingData.rowsPerGroup = rowsPerGroup;
    tilingData.gradType = static_cast<uint32_t>(gradType);
    tilingData.weightType = static_cast<uint32_t>(weightType);

    at::Tensor tilingHost = at::empty({static_cast<int64_t>(sizeof(AdagradSimdTilingData))},
                                      at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    if (memcpy_s(tilingHost.data_ptr(), tilingHost.nbytes(), &tilingData, sizeof(tilingData)) != EOK) {
        return false;
    }
    tilingNpuOut = tilingHost.to(gradsContinuous.device()).contiguous();
    return true;
}

static bool LaunchAdagradFusedSimd(aclrtStream stream, uint8_t* gradsPtr, uint8_t* valuesPtr,
                                   const at::Tensor& gradsContinuous, uint32_t gradDim, uint32_t valDim,
                                   int32_t numRows, int32_t maxCores, float lr, float eps, DataType gradType,
                                   DataType weightType)
{
    const uint32_t rowsPerGroup = AdagradSimdRowsPerGroup(gradType, weightType);
    int32_t coreNum = 0;
    at::Tensor tilingNpu;
    if (!LaunchAdagradSimdTiling(gradsContinuous, gradDim, valDim, numRows, maxCores, lr, eps, rowsPerGroup, gradType,
                                 weightType, coreNum, tilingNpu)) {
        return false;
    }
    ACLRT_LAUNCH_KERNEL(adagrad_fused_simd)
    (coreNum, stream, gradsPtr, valuesPtr, static_cast<uint8_t*>(tilingNpu.data_ptr()));
    return true;
}

static bool LaunchAdagradSimd(aclrtStream stream, uint8_t* gradsPtr, uint8_t* rowPtrsPtr, uint8_t* foundsPtr,
                              const at::Tensor& gradsContinuous, uint32_t gradDim, int32_t numRows, int32_t maxCores,
                              float lr, float eps, uint32_t rowsPerGroup, DataType gradType, DataType weightType)
{
    const uint32_t valDim = gradDim * 2U;
    int32_t coreNum = 0;
    at::Tensor tilingNpu;
    TORCH_CHECK(LaunchAdagradSimdTiling(gradsContinuous, gradDim, valDim, numRows, maxCores, lr, eps, rowsPerGroup,
                                        gradType, weightType, coreNum, tilingNpu),
                "LaunchAdagradSimd: LaunchAdagradSimdTiling failed.");
    ACLRT_LAUNCH_KERNEL(adagrad_simd)
    (coreNum, stream, gradsPtr, rowPtrsPtr, foundsPtr, static_cast<uint8_t*>(tilingNpu.data_ptr()));
    return true;
}

static bool LaunchAdamWSimdTiling(const at::Tensor& gradsContinuous, uint32_t gradDim, uint32_t valDim, int32_t numRows,
                                  int32_t maxCores, float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2,
                                  float stepSize, float invVHatDenom, float decayFactor, float eps,
                                  uint32_t rowsPerGroup, DataType gradType, DataType weightType, int32_t& coreNumOut,
                                  at::Tensor& tilingNpuOut)
{
    if (!IsSupportedOptimizerSimdDtype(gradType, weightType)) {
        LOG_ERROR("LaunchAdamWSimdTiling: unsupported grad/weight dtype for AdamW SIMD.");
        return false;
    }

    const int32_t numGroups = (numRows + static_cast<int32_t>(rowsPerGroup) - 1) / static_cast<int32_t>(rowsPerGroup);
    coreNumOut = static_cast<int32_t>(
        std::max<int64_t>(1, std::min<int64_t>(static_cast<int64_t>(maxCores), static_cast<int64_t>(numGroups))));

    AdamWSimdTilingData tilingData{};
    tilingData.gradDim = gradDim;
    tilingData.valDim = valDim;
    tilingData.numRows = numRows;
    tilingData.beta1 = beta1;
    tilingData.beta2 = beta2;
    tilingData.oneMinusBeta1 = oneMinusBeta1;
    tilingData.oneMinusBeta2 = oneMinusBeta2;
    tilingData.stepSize = stepSize;
    tilingData.invVHatDenom = invVHatDenom;
    tilingData.decayFactor = decayFactor;
    tilingData.eps = eps;
    tilingData.needCoreNum = coreNumOut;
    tilingData.rowsPerGroup = rowsPerGroup;
    tilingData.gradType = static_cast<uint32_t>(gradType);
    tilingData.weightType = static_cast<uint32_t>(weightType);

    at::Tensor tilingHost = at::empty({static_cast<int64_t>(sizeof(AdamWSimdTilingData))},
                                      at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    if (memcpy_s(tilingHost.data_ptr(), tilingHost.nbytes(), &tilingData, sizeof(tilingData)) != EOK) {
        return false;
    }
    tilingNpuOut = tilingHost.to(gradsContinuous.device()).contiguous();
    return true;
}

static bool LaunchAdamWFusedSimd(aclrtStream stream, uint8_t* gradsPtr, uint8_t* valuesPtr,
                                 const at::Tensor& gradsContinuous, uint32_t gradDim, uint32_t valDim, int32_t numRows,
                                 int32_t maxCores, float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2,
                                 float stepSize, float invVHatDenom, float decayFactor, float eps, DataType gradType,
                                 DataType weightType)
{
    constexpr uint32_t kRowsPerGroup = 1U;
    int32_t coreNum = 0;
    at::Tensor tilingNpu;
    if (!LaunchAdamWSimdTiling(gradsContinuous, gradDim, valDim, numRows, maxCores, beta1, beta2, oneMinusBeta1,
                               oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, kRowsPerGroup, gradType,
                               weightType, coreNum, tilingNpu)) {
        return false;
    }
    ACLRT_LAUNCH_KERNEL(adamw_fused_simd)
    (coreNum, stream, gradsPtr, valuesPtr, static_cast<uint8_t*>(tilingNpu.data_ptr()));
    return true;
}

static bool LaunchAdamWSimd(aclrtStream stream, uint8_t* gradsPtr, uint8_t* rowPtrsPtr, uint8_t* foundsPtr,
                            const at::Tensor& gradsContinuous, uint32_t gradDim, int32_t numRows, int32_t maxCores,
                            float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2, float stepSize,
                            float invVHatDenom, float decayFactor, float eps, uint32_t rowsPerGroup, DataType gradType,
                            DataType weightType)
{
    const uint32_t valDim = gradDim * 3U;
    int32_t coreNum = 0;
    at::Tensor tilingNpu;
    TORCH_CHECK(LaunchAdamWSimdTiling(gradsContinuous, gradDim, valDim, numRows, maxCores, beta1, beta2, oneMinusBeta1,
                                      oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, rowsPerGroup, gradType,
                                      weightType, coreNum, tilingNpu),
                "LaunchAdamWSimd: LaunchAdamWSimdTiling failed.");
    ACLRT_LAUNCH_KERNEL(adamw_simd)
    (coreNum, stream, gradsPtr, rowPtrsPtr, foundsPtr, static_cast<uint8_t*>(tilingNpu.data_ptr()));
    return true;
}

static bool LaunchSgdSimdTiling(const at::Tensor& gradsContinuous, uint32_t gradDim, uint32_t valDim, int32_t numRows,
                                int32_t maxCores, float lr, uint32_t rowsPerGroup, DataType gradType,
                                DataType weightType, int32_t& coreNumOut, at::Tensor& tilingNpuOut)
{
    if (!IsSupportedOptimizerSimdDtype(gradType, weightType)) {
        LOG_ERROR("LaunchSgdSimdTiling: unsupported grad/weight dtype for SGD SIMD.");
        return false;
    }

    const int32_t numGroups = (numRows + static_cast<int32_t>(rowsPerGroup) - 1) / static_cast<int32_t>(rowsPerGroup);
    coreNumOut = static_cast<int32_t>(
        std::max<int64_t>(1, std::min<int64_t>(static_cast<int64_t>(maxCores), static_cast<int64_t>(numGroups))));

    SgdSimdTilingData tilingData{};
    tilingData.gradDim = gradDim;
    tilingData.valDim = valDim;
    tilingData.numRows = numRows;
    tilingData.lr = lr;
    tilingData.needCoreNum = coreNumOut;
    tilingData.rowsPerGroup = rowsPerGroup;
    tilingData.gradType = static_cast<uint32_t>(gradType);
    tilingData.weightType = static_cast<uint32_t>(weightType);

    at::Tensor tilingHost = at::empty({static_cast<int64_t>(sizeof(SgdSimdTilingData))},
                                      at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    if (memcpy_s(tilingHost.data_ptr(), tilingHost.nbytes(), &tilingData, sizeof(tilingData)) != EOK) {
        return false;
    }
    tilingNpuOut = tilingHost.to(gradsContinuous.device()).contiguous();
    return true;
}

static bool LaunchSgdFusedSimd(aclrtStream stream, uint8_t* gradsPtr, uint8_t* valuesPtr,
                               const at::Tensor& gradsContinuous, uint32_t gradDim, uint32_t valDim, int32_t numRows,
                               int32_t maxCores, float lr, DataType gradType, DataType weightType)
{
    constexpr uint32_t kRowsPerGroup = 2U;
    int32_t coreNum = 0;
    at::Tensor tilingNpu;
    if (!LaunchSgdSimdTiling(gradsContinuous, gradDim, valDim, numRows, maxCores, lr, kRowsPerGroup, gradType,
                             weightType, coreNum, tilingNpu)) {
        return false;
    }
    ACLRT_LAUNCH_KERNEL(sgd_fused_simd)
    (coreNum, stream, gradsPtr, valuesPtr, static_cast<uint8_t*>(tilingNpu.data_ptr()));
    return true;
}

static bool LaunchSgdSimd(aclrtStream stream, uint8_t* gradsPtr, uint8_t* rowPtrsPtr, uint8_t* foundsPtr,
                          const at::Tensor& gradsContinuous, uint32_t gradDim, int32_t numRows, int32_t maxCores,
                          float lr, uint32_t rowsPerGroup, DataType gradType, DataType weightType)
{
    const uint32_t valDim = gradDim;
    int32_t coreNum = 0;
    at::Tensor tilingNpu;
    TORCH_CHECK(LaunchSgdSimdTiling(gradsContinuous, gradDim, valDim, numRows, maxCores, lr, rowsPerGroup, gradType,
                                    weightType, coreNum, tilingNpu),
                "LaunchSgdSimd: LaunchSgdSimdTiling failed.");
    ACLRT_LAUNCH_KERNEL(sgd_simd)
    (coreNum, stream, gradsPtr, rowPtrsPtr, foundsPtr, static_cast<uint8_t*>(tilingNpu.data_ptr()));
    return true;
}

ReturnType block_bucketsize_sparse_features_npu(const at::Tensor& lengths, const at::Tensor& indices, bool bucketizePos,
                                                bool sequence, const at::Tensor& distTypePerFeature,
                                                const at::Tensor& blockSizes, int32_t mySize,
                                                const c10::optional<at::Tensor>& weights)
{
    if (mySize == 0) {
        throw std::runtime_error("MySize is zero");
    }
    const at::OptionalDeviceGuard guard(device_of(lengths));
    at::ScalarType targetIntDtype = indices.scalar_type();
    auto lengthsContig = lengths.to(targetIntDtype).contiguous();
    auto indicesContig = indices.to(targetIntDtype).contiguous();
    auto blockSizesContig = blockSizes.to(targetIntDtype).contiguous();
    auto distTypeContig = distTypePerFeature.to(targetIntDtype).contiguous();

    int32_t lengthSize = lengths.size(0);
    int32_t indicesSize = indices.size(0);
    int32_t blockSize = blockSizes.size(0);
    if (lengthSize == 0 || blockSize == 0) {
        throw std::runtime_error("LengthSize or blockSize is zero");
    }
    int32_t B = lengthSize / blockSize;
    // offsetsContig dtype is torch.int64
    auto offsetsContig = at::cumsum(lengthsContig, 0).contiguous();
    auto intOptions = at::TensorOptions().dtype(targetIntDtype).device(lengthsContig.device());
    auto newLengths = at::zeros({lengthSize * mySize}, intOptions);
    auto newIndices = at::empty({indicesSize}, intOptions);

    // 权重
    c10::optional<at::Tensor> newWeights = c10::nullopt;
    at::Tensor weightsContig;
    if (weights.has_value()) {
        weightsContig = weights.value().contiguous();
        newWeights = at::empty_like(weightsContig);
    } else {
        weightsContig = at::Tensor();
    }

    // 分桶位置
    c10::optional<at::Tensor> newPos = c10::nullopt;
    at::Tensor newPosTensor;
    if (bucketizePos) {
        newPos = at::empty({indicesSize}, intOptions);
        newPosTensor = newPos.value();
    } else {
        newPosTensor = at::Tensor();
    }

    // 反桶化
    c10::optional<at::Tensor> unbucketizePermute = c10::nullopt;
    at::Tensor unbucketizePermuteTensor;
    if (sequence) {
        unbucketizePermute = at::empty({indicesSize}, intOptions);
        unbucketizePermuteTensor = unbucketizePermute.value();
    } else {
        unbucketizePermuteTensor = at::Tensor();
    }

    // 分配工作空间（current_offsets）
    bool isInt32 = (targetIntDtype == torch::kInt32);
    size_t elemSize = isInt32 ? sizeof(int32_t) : sizeof(int64_t);
    size_t currentOffsetsSize = lengthSize * mySize * elemSize;
    size_t alignedWorkSpaceSize = ((currentOffsetsSize + CACHE_ALIGN - 1) / CACHE_ALIGN) * CACHE_ALIGN;

    // blocksum的累加和的空间
    size_t newLengthsTotalSize = lengthSize * mySize;
    bool isSmall = (newLengthsTotalSize <= (isInt32 ? SMALL_DATA_THRESHOLD_32 : SMALL_DATA_THRESHOLD));
    size_t elementsPerBlock4NewLength = isSmall ? MAX_THREADS_PER_BLOCK : ELEMENTS_PER_BLOCK;
    size_t totalBlocksForCusum = (newLengthsTotalSize + elementsPerBlock4NewLength - 1) / elementsPerBlock4NewLength;
    int32_t stride = CACHE_ALIGN / (isInt32 ? sizeof(int32_t) : sizeof(int64_t));
    size_t blockSumsSize = totalBlocksForCusum * stride * elemSize;
    alignedWorkSpaceSize += ((blockSumsSize + CACHE_ALIGN - 1) / CACHE_ALIGN) * CACHE_ALIGN;
    at::Tensor workspace =
        at::empty({static_cast<int64_t>(alignedWorkSpaceSize)}, lengthsContig.options().dtype(torch::kUInt8));
    void* currentOffsetsPtr = workspace.data_ptr<uint8_t>();

    // 计算核心数
    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    size_t coreNum = std::min(totalBlocksForCusum, static_cast<size_t>(maxCores));
    coreNum = std::max(coreNum, static_cast<size_t>(MIN_CORE_NUM));

    void* lengthsPtr = lengthsContig.data_ptr();
    void* indicesPtr = indicesContig.data_ptr();
    void* weightsPtr = weightsContig.defined() ? weightsContig.data_ptr() : nullptr;
    void* offsetsPtr = offsetsContig.data_ptr();
    void* distTypePtr = distTypeContig.data_ptr();
    void* blockSizesPtr = blockSizesContig.data_ptr();
    void* newlengthsPtr = newLengths.data_ptr();
    void* newIndicesPtr = newIndices.data_ptr();
    void* newweightsPtr = newWeights.has_value() ? newWeights.value().data_ptr() : nullptr;
    void* newPosPtr = newPosTensor.defined() ? newPosTensor.data_ptr() : nullptr;
    void* unbucketizePermutePtr = unbucketizePermuteTensor.defined() ? unbucketizePermuteTensor.data_ptr() : nullptr;

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(true);
    ACLRT_LAUNCH_KERNEL(block_bucketsize_sparse_features)
    (coreNum, stream, lengthsPtr, indicesPtr, weightsPtr, offsetsPtr, distTypePtr, blockSizesPtr, newlengthsPtr,
     newIndicesPtr, newweightsPtr, newPosPtr, unbucketizePermutePtr, currentOffsetsPtr,
     static_cast<int32_t>(lengthSize), static_cast<int32_t>(indicesSize), static_cast<int32_t>(mySize),
     static_cast<int32_t>(B), static_cast<int32_t>(isInt32), static_cast<int32_t>(weights.has_value()),
     static_cast<int32_t>(sequence), static_cast<int32_t>(bucketizePos), static_cast<int32_t>(isSmall),
     static_cast<int32_t>(newLengthsTotalSize), static_cast<int32_t>(totalBlocksForCusum));

    return std::make_tuple(newLengths, newIndices, newWeights, newPos, unbucketizePermute);
}

torch::Tensor gather_embedding(const torch::Tensor& inputs, const torch::Tensor& indices)
{
    TORCH_CHECK(indices.dtype() == torch::kInt64 || indices.dtype() == torch::kUInt64,
                "gather_embedding: indices must be int64 or uint64");
    uint32_t eleSz = inputs.element_size();
    TORCH_CHECK(eleSz == 4 or eleSz == 2, "inputs element size must be 4 or 2");

    uint64_t indicesLen = indices.size(0);
    uint32_t dim = inputs.size(1);

    if (indicesLen == 0) {
        return torch::empty({0, dim}, inputs.options());
    }

    uint64_t outLen = indicesLen * dim;
    constexpr uint32_t GATHER_THRESHOLD = 100000;

    if (outLen > GATHER_THRESHOLD && indices.scalar_type() == torch::kInt64) {
        torch::Tensor indicesExpand = indices.unsqueeze(-1).expand({indicesLen, dim});
        return torch::gather(inputs, 0, indicesExpand);
    }

    auto inType = scalartype_to_datatype(inputs.scalar_type());
    auto indexType = scalartype_to_datatype(indices.scalar_type());
    torch::Tensor output = torch::empty({indicesLen, dim}, inputs.options());
    void* inData = inputs.contiguous().data_ptr();
    void* indicesData = indices.contiguous().data_ptr();
    void* outData = output.data_ptr();
    constexpr uint32_t THREAD_NUM = 1024;

    if (dim % 4 == 0) {
        // 使用float2读写内存
        if (eleSz == 4) {
            outLen >>= 1;
        } else {
            outLen >>= 2;
        }
    }

    uint32_t totalBlocks = (outLen + THREAD_NUM - 1) / THREAD_NUM;
    uint32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    uint32_t coreNum = (totalBlocks < maxCores) ? totalBlocks : maxCores;

    uint32_t blocksPerCore = totalBlocks / coreNum;
    uint32_t remainderBlocks = totalBlocks % coreNum;
    auto stream = c10_npu::getCurrentNPUStream().stream(true);

    ACLRT_LAUNCH_KERNEL(gather_dim0)
    (coreNum, stream, inData, indicesData, outData, dim, outLen, blocksPerCore, remainderBlocks, THREAD_NUM,
     static_cast<uint32_t>(inType), static_cast<uint32_t>(indexType), eleSz);
    return output;
}

torch::Tensor load_from_pointer_imp(const torch::Tensor& pointers, torch::Tensor& output)
{
    uint32_t eleSz = output.element_size();
    TORCH_CHECK(eleSz == 4 or eleSz == 2, "output element size must be 4 or 2");

    uint64_t inLen = pointers.size(0);
    if (inLen == 0) {
        return output;
    }

    auto outType = scalartype_to_datatype(output.scalar_type());
    uint32_t dim = output.size(1);
    uint64_t outLen = inLen * dim;
    TORCH_CHECK(outLen == output.numel(), "output.numel() must equal pointers.numel() * output.size(1)");

    void* pData = pointers.contiguous().data_ptr();
    void* outData = output.contiguous().data_ptr();
    constexpr uint32_t THREAD_NUM = 1024;

    if (dim % 4 == 0) {
        // 使用float2读写内存
        if (eleSz == 4) {
            outLen >>= 1;
        } else {
            outLen >>= 2;
        }
    }

    uint32_t totalBlocks = (outLen + THREAD_NUM - 1) / THREAD_NUM;
    uint32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    uint32_t coreNum = (totalBlocks < maxCores) ? totalBlocks : maxCores;

    uint32_t blocksPerCore = totalBlocks / coreNum;    // 每核基础块数
    uint32_t remainderBlocks = totalBlocks % coreNum;  // 余数块数
    auto stream = c10_npu::getCurrentNPUStream().stream(true);

    if (outLen < UINT32_MAX) {
        ACLRT_LAUNCH_KERNEL(load_from_pointer)
        (coreNum, stream, pData, outData, dim, outLen, blocksPerCore, remainderBlocks, THREAD_NUM, 1,
         static_cast<uint32_t>(outType), eleSz);
    } else {
        ACLRT_LAUNCH_KERNEL(load_from_pointer)
        (coreNum, stream, pData, outData, dim, outLen, blocksPerCore, remainderBlocks, THREAD_NUM, 0,
         static_cast<uint32_t>(outType), eleSz);
    }

    return output;
}

torch::Tensor load_from_pointer_hybrid_imp(const torch::Tensor& pointers, torch::Tensor& output)
{
    uint32_t eleSz = output.element_size();
    TORCH_CHECK(eleSz == 4 or eleSz == 2, "output element size must be 4 or 2");

    // 传入数据量不会超过uint32_t类型上限
    uint32_t inLen = pointers.size(0);
    if (inLen == 0) {
        LOG_ERROR("[DYNAMIC_EMB] load_from_pointer_hybrid: pointers is empty");
        return output;
    }

    auto outType = scalartype_to_datatype(output.scalar_type());
    uint32_t dim = output.size(1);
    uint64_t outLen = static_cast<uint64_t>(inLen) * dim;
    TORCH_CHECK(outLen == output.numel(), "output.numel() must equal pointers.numel() * output.size(1)");

    void* pData = pointers.is_contiguous() ? pointers.data_ptr() : pointers.contiguous().data_ptr();
    void* outData = output.is_contiguous() ? output.data_ptr() : output.contiguous().data_ptr();

    uint32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    uint64_t totalUbSize = AclSingleton::GetInstance().GetTotalUbSize();
    auto stream = c10_npu::getCurrentNPUStream().stream(true);

    dyn_emb::load_from_pointer_hybrid_ops(pData, outData, dim, inLen, stream, maxCores, static_cast<uint32_t>(outType),
                                          totalUbSize);

    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> unique_op_impl(const at::Tensor& key)
{
    const at::OptionalDeviceGuard guard(device_of(key));
    TORCH_CHECK(key.dtype() == torch::kInt64, "unique_op only supports int64 dtype for key");

    auto input_key = key.contiguous();
    int64_t key_numel = input_key.numel();
    // 计算哈希表容量和分配workspace
    int64_t hashTableCapacity = static_cast<int64_t>(std::ceil(key_numel * HASH_TABLE_FACTOR));
    hashTableCapacity = std::max(hashTableCapacity, MIN_HASH_TABLE_CAPACITY);
    size_t keySlotSize = hashTableCapacity * sizeof(int64_t);
    size_t countSlotSize = hashTableCapacity * sizeof(int64_t);
    size_t globalCounterSize = sizeof(int64_t);
    size_t userWorkspaceSize = keySlotSize + countSlotSize + globalCounterSize;
    // 按64B对齐
    userWorkspaceSize = ((userWorkspaceSize + CACHE_ALIGN - 1) / CACHE_ALIGN) * CACHE_ALIGN;
    at::Tensor workspace =
        at::empty({static_cast<int64_t>(userWorkspaceSize)}, input_key.options().dtype(torch::kUInt8));

    // 计算corenum的实际要用到的核心数
    size_t elementsPerBlock = MAX_THREADS_PER_BLOCK;
    size_t totalBlocks = (key_numel + elementsPerBlock - 1) / elementsPerBlock;
    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    size_t coreNum = std::min(totalBlocks, static_cast<size_t>(maxCores));
    coreNum = std::max(coreNum, static_cast<size_t>(MIN_CORE_NUM));

    auto unique_key = at::empty_like(input_key);
    auto uniqueSrcIndices = at::empty(input_key.sizes(), input_key.options().dtype(torch::kInt32));
    auto restore_index = at::empty(input_key.sizes(), input_key.options().dtype(torch::kLong));
    auto count = at::zeros_like(unique_key, input_key.options().dtype(torch::kLong));

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    ACLRT_LAUNCH_KERNEL(unique_op)
    (coreNum, stream, input_key.data_ptr<int64_t>(), unique_key.data_ptr<int64_t>(), restore_index.data_ptr<int64_t>(),
     count.data_ptr<int64_t>(), workspace.data_ptr<uint8_t>(), hashTableCapacity, static_cast<int32_t>(key_numel),
     uniqueSrcIndices.data_ptr<int32_t>());

    auto valid_count = at::nonzero(count > 0).numel();
    unique_key = unique_key.narrow(0, 0, valid_count);
    uniqueSrcIndices = uniqueSrcIndices.narrow(0, 0, valid_count);
    count = count.narrow(0, 0, valid_count);

    return std::make_tuple(unique_key, restore_index, count, uniqueSrcIndices);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> hash_unique(const at::Tensor& key)
{
    at::Tensor cpuKey = key.to(at::kCPU).contiguous();

    DedupResult res;
    if (cpuKey.dtype() == torch::kInt64) {
        HashDeduplicator<int64_t> hashDeduplicator;
        res = hashDeduplicator.deduplicate(cpuKey);
    } else {
        HashDeduplicator<uint64_t> hashDeduplicator;
        res = hashDeduplicator.deduplicate(cpuKey);
    }

    int64_t uniqueSize = res.uniqueElements.size(0);
    int64_t uniqueByteSize = uniqueSize * sizeof(int64_t);
    int64_t reverseSize = key.size(0);
    int64_t reverseByteSize = reverseSize * sizeof(int64_t);

    at::Tensor uniqueKeysNpu = at::empty({uniqueSize}, key.options());
    at::Tensor reverseIndicesNpu = at::empty({reverseSize}, key.options().dtype(torch::kInt64));
    at::Tensor countsNpu = at::empty({uniqueSize}, key.options().dtype(torch::kInt64));

    aclrtMemcpy(uniqueKeysNpu.data_ptr(), uniqueByteSize, res.uniqueElements.data_ptr(), uniqueByteSize,
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(reverseIndicesNpu.data_ptr(), reverseByteSize, res.reverseIndices.data_ptr(), reverseByteSize,
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(countsNpu.data_ptr(), uniqueByteSize, res.counts.data_ptr(), uniqueByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    return std::make_tuple(uniqueKeysNpu, reverseIndicesNpu, countsNpu);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sort_unique_op(const at::Tensor& key)
{
    auto [uniquekey, restoreIndex, count] = torch::_unique2(key, false, true, true);
    return std::make_tuple(uniquekey, restoreIndex, count);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_op_npu(const at::Tensor& keys)
{
    const at::OptionalDeviceGuard guard(at::device_of(keys));
    TORCH_CHECK(keys.dtype() == torch::kInt64 || keys.dtype() == torch::kUInt64,
                "unique_op only supports int64 (kInt64) or uint64 (kUInt64)");
    TORCH_CHECK(keys.dim() == 1, "unique_op only supports 1-dimensional tensor for key, but got ", keys.dim(),
                "-dimensional tensor");

    int keySize = keys.size(0);
    if (keySize > CPU_KEY_THRESHOLD) {
        return sort_unique_op(keys);
    } else {
        return hash_unique(keys);
    }
}

at::Tensor get_table_range_npu(const at::Tensor& offsets, const at::Tensor& featureOffsets)
{
    const at::OptionalDeviceGuard guard(device_of(offsets));
    TORCH_CHECK(offsets.dtype() == featureOffsets.dtype(), "offsets and featureOffsets must have the same dtype");
    TORCH_CHECK(offsets.dtype() == torch::kInt32 || offsets.dtype() == torch::kInt64 ||
                    offsets.dtype() == torch::kUInt32 || offsets.dtype() == torch::kUInt64,
                "get_table_range_op only supports int32/int64/uint32/uint64 dtype");
    TORCH_CHECK(offsets.dim() == 1 && featureOffsets.dim() == 1, "offsets and featureOffsets must be 1D tensor");

    TORCH_CHECK(offsets.numel() > 0 && featureOffsets.numel() > 0,
                "get_table_range_op: offsets / featureOffsets tensor cannot be empty (numel=0)! "
                "Current status - offsets: ",
                ((offsets.numel() == 0) ? "empty " : "non-empty "),
                "featureOffsets: ", ((featureOffsets.numel() == 0) ? "empty" : "non-empty"));

    auto inputOffsets = offsets.contiguous();
    auto inputFeatureOffsets = featureOffsets.contiguous();
    auto tableRange = at::empty_like(inputFeatureOffsets);
    int64_t tableNum = inputFeatureOffsets.size(0) - 1;
    int64_t featureNumXBatch = inputOffsets.size(0) - 1;

    if (tableNum == 0) {
        tableRange.zero_();
        return tableRange;
    }

    const int64_t numFeature = inputFeatureOffsets.select(0, tableNum).cpu().item<int64_t>();
    int64_t batch = 0;
    if (numFeature != 0) {
        batch = featureNumXBatch / numFeature;
    }
    if (batch == 0) {
        tableRange.fill_(inputOffsets.select(0, 0).cpu().item());
        return tableRange;
    }

    auto offsetType = scalartype_to_datatype(convertTypeMetaToScalarType(inputOffsets.dtype()));

    // 配置内核启动参数
    size_t totalTasks = tableNum + 1;
    size_t elementsPerBlock = MAX_THREADS_PER_BLOCK;
    size_t totalBlocks = (totalTasks + elementsPerBlock - 1) / elementsPerBlock;

    // 获取设备最大可用AIV核心数
    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    size_t coreNum = std::min(totalBlocks, static_cast<size_t>(maxCores));
    coreNum = std::max(coreNum, static_cast<size_t>(MIN_CORE_NUM));

    // 算子执行阶段
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    ACLRT_LAUNCH_KERNEL(get_table_range_op)
    (coreNum, stream, inputOffsets.data_ptr(), inputFeatureOffsets.data_ptr(), tableRange.data_ptr(), featureNumXBatch,
     tableNum, static_cast<uint32_t>(offsetType));

    return tableRange;
}

static void launch_select_kernel(bool selectIndex, const at::Tensor& flags, const c10::optional<at::Tensor>& inputs,
                                 at::Tensor& outputs, at::Tensor& numSelected)
{
    int64_t numTotal = selectIndex ? outputs.size(0) : inputs->size(0);
    if (numTotal == 0) {
        numSelected.zero_();
        return;
    }

    auto inputFlags = flags.contiguous();
    auto inputOutputs = outputs.contiguous();
    auto inputNumSelected = numSelected.contiguous();

    int32_t isUInt64 = 0;
    if (selectIndex) {
        TORCH_CHECK(outputs.dtype() == torch::kInt64 || outputs.dtype() == torch::kUInt64,
                    "select_index only supports int64/uint64 dtype for output_indices");
        isUInt64 = (outputs.dtype() == torch::kUInt64) ? 1 : 0;
    } else {
        TORCH_CHECK(inputs->dtype() == torch::kInt64 || inputs->dtype() == torch::kUInt64,
                    "select only supports int64/uint64 dtype for inputs");
        isUInt64 = (inputs->dtype() == torch::kUInt64) ? 1 : 0;
    }

    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    int32_t stride = CACHE_ALIGN / static_cast<int32_t>(sizeof(int64_t));
    int32_t isSmall = (numTotal <= static_cast<int64_t>(SMALL_DATA_THRESHOLD)) ? 1 : 0;
    int32_t elementsPerBlock = isSmall ? MAX_THREADS_PER_BLOCK : ELEMENTS_PER_BLOCK;
    int32_t totalBlocks = static_cast<int32_t>((numTotal + elementsPerBlock - 1) / elementsPerBlock);
    int32_t coreNum = std::min(totalBlocks, maxCores);
    coreNum = std::max(coreNum, static_cast<int32_t>(MIN_CORE_NUM));

    int64_t workspaceElems = numTotal + static_cast<int64_t>(totalBlocks) * stride;
    auto workspace = at::empty({workspaceElems}, inputFlags.options().dtype(torch::kInt64));

    auto stream = c10_npu::getCurrentNPUStream().stream(true);

    if (selectIndex) {
        ACLRT_LAUNCH_KERNEL(select_index_op)
        (coreNum, stream, inputFlags.data_ptr<bool>(), inputOutputs.data_ptr(), inputNumSelected.data_ptr(),
         workspace.data_ptr(), numTotal, isUInt64, isSmall, totalBlocks);
    } else {
        auto inputInputs = inputs->contiguous();
        ACLRT_LAUNCH_KERNEL(select_op)
        (coreNum, stream, inputFlags.data_ptr<bool>(), inputInputs.data_ptr(), inputOutputs.data_ptr(),
         inputNumSelected.data_ptr(), workspace.data_ptr(), numTotal, isUInt64, isSmall, totalBlocks);
    }
}

void select_npu(const at::Tensor& flags, const at::Tensor& inputs, at::Tensor& outputs, at::Tensor& numSelected)
{
    const at::OptionalDeviceGuard guard(device_of(flags));
    TORCH_CHECK(flags.dtype() == torch::kBool, "select: flags must be bool tensor");
    TORCH_CHECK(flags.dim() == 1 && inputs.dim() == 1 && outputs.dim() == 1,
                "select: flags, inputs and outputs must be 1D tensors");
    TORCH_CHECK(flags.size(0) == inputs.size(0), "select: flags and inputs must have the same length");
    TORCH_CHECK(numSelected.numel() == 1, "select: num_selected must be a scalar tensor");
    launch_select_kernel(false, flags, inputs, outputs, numSelected);
}

void select_index_npu(const at::Tensor& flags, at::Tensor& outputIndices, at::Tensor& numSelected)
{
    const at::OptionalDeviceGuard guard(device_of(flags));
    TORCH_CHECK(flags.dtype() == torch::kBool, "select_index: flags must be bool tensor");
    TORCH_CHECK(flags.dim() == 1 && outputIndices.dim() == 1,
                "select_index: flags and output_indices must be 1D tensors");
    TORCH_CHECK(flags.size(0) == outputIndices.size(0),
                "select_index: flags and output_indices must have the same length");
    TORCH_CHECK(numSelected.numel() == 1, "select_index: num_selected must be a scalar tensor");
    launch_select_kernel(true, flags, c10::nullopt, outputIndices, numSelected);
}

void get_new_length_and_offsets_npu(const at::Tensor& dUniqueOffsets, const at::Tensor& dTableOffsetsInFeature,
                                    at::Tensor& newOffsets, at::Tensor& newLengths, int localBatchSize)
{
    const at::OptionalDeviceGuard guard(device_of(dUniqueOffsets));

    TORCH_CHECK(newOffsets.dtype() == torch::kInt32 || newOffsets.dtype() == torch::kInt64,
                "newOffsets only supports int32/int64 dtype (got ", newOffsets.dtype().name(), ")");
    TORCH_CHECK(newLengths.dtype() == newOffsets.dtype(),
                "newLengths dtype must match newOffsets (newOffsets: ", newOffsets.dtype().name(),
                ", newLengths: ", newLengths.dtype().name(), ")");

    int64_t newLengthsSize = newLengths.size(0);
    int32_t isInt32 = (newOffsets.dtype() == torch::kInt32) ? 1 : 0;
    auto inputUniqueOffsets = dUniqueOffsets.contiguous();
    auto inputTableOffsets = dTableOffsetsInFeature.contiguous();

    int64_t tableNum = inputTableOffsets.size(0) - 1;
    TORCH_CHECK(tableNum >= 0, "dTableOffsetsInFeature must have at least 1 element (size >=1)");
    // 配置内核启动参数
    size_t totalTasks = static_cast<size_t>(newLengthsSize);
    size_t elementsPerBlock = MAX_THREADS_PER_BLOCK;
    size_t totalBlocks = (totalTasks + elementsPerBlock - 1) / elementsPerBlock;
    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    size_t coreNum = std::min(totalBlocks, static_cast<size_t>(maxCores));
    coreNum = std::max(coreNum, static_cast<size_t>(MIN_CORE_NUM));

    // 算子执行阶段
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    ACLRT_LAUNCH_KERNEL(get_new_length_and_offsets_op)
    (coreNum, stream, inputUniqueOffsets.data_ptr<int64_t>(), inputTableOffsets.data_ptr<int64_t>(),
     newOffsets.data_ptr(), newLengths.data_ptr(), tableNum, newLengthsSize, localBatchSize, isInt32);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> segmented_unique_op(
    const at::Tensor& keys, const at::Tensor& segmentRange)
{
    TORCH_CHECK(keys.dtype() == torch::kInt64 || keys.dtype() == torch::kUInt64,
                "segmented_unique_op_npu only supports int64 or uint64 dtype for keys");
    int64_t keysNum = keys.size(0);

    at::Tensor hSegmentRange =
        at::empty(segmentRange.sizes(), segmentRange.options().device(at::kCPU).pinned_memory(true));
    hSegmentRange.copy_(segmentRange);
    int64_t* hSegmentRangePtr = hSegmentRange.data_ptr<int64_t>();
    int tableNum = segmentRange.size(0) - 1;

    std::vector<at::Tensor> tmpUniqueIndices(tableNum);
    std::vector<at::Tensor> uniqueCntVec(tableNum);
    std::vector<int64_t> hUniqueIndicesRange(tableNum + 1, 0);

    at::Tensor dUniqueNums = at::empty(tableNum, segmentRange.options());
    at::Tensor inverseIdx = at::empty(keysNum, segmentRange.options());
    int64_t globalOffset = 0;

    // 分段去重
    for (int i = 0; i < tableNum; ++i) {
        int64_t indicesBegin = hSegmentRangePtr[i];
        int64_t indicesEnd = hSegmentRangePtr[i + 1];
        int64_t indicesLength = indicesEnd - indicesBegin;

        if (indicesLength == 0) {
            dUniqueNums.slice(0, i, i + 1).fill_(0);
            hUniqueIndicesRange[i + 1] = hUniqueIndicesRange[i];
        } else {
            at::Tensor tmpIndices = keys.slice(0, indicesBegin, indicesEnd).contiguous();
            at::Tensor tmpInverseIdx = inverseIdx.slice(0, indicesBegin, indicesEnd);
            auto [tmpUnique, tmpRestoreIndex, tmpCount] = unique_op_npu(tmpIndices);
            at::Tensor gloabalRestoreIndex = tmpRestoreIndex + globalOffset;
            tmpInverseIdx.copy_(gloabalRestoreIndex);

            int64_t tmpUniqueNum = tmpUnique.size(0);
            tmpUniqueIndices[i] = at::empty(tmpUniqueNum, keys.options());
            tmpUniqueIndices[i].copy_(tmpUnique);
            uniqueCntVec[i] = tmpCount;
            dUniqueNums.slice(0, i, i + 1).fill_(tmpUniqueNum);
            hUniqueIndicesRange[i + 1] = hUniqueIndicesRange[i] + tmpUniqueNum;
            globalOffset += tmpUniqueNum;
        }
    }

    at::Tensor hUniqueIndicesTableRange =
        at::empty(tableNum + 1, segmentRange.options().device(at::kCPU).pinned_memory(true));
    size_t data_size = static_cast<size_t>((tableNum + 1) * sizeof(int64_t));
    errno_t ret =
        memcpy_s(hUniqueIndicesTableRange.data_ptr<int64_t>(), data_size, hUniqueIndicesRange.data(), data_size);
    TORCH_CHECK(ret == EOK, "UniqueIndicesRange memcpy_s failed, ret = ", ret);
    at::Tensor dUniqueIndicesTableRange = at::zeros(tableNum + 1, segmentRange.options());
    dUniqueIndicesTableRange.copy_(hUniqueIndicesTableRange);

    // 合并所有分段的唯一 keys
    int64_t numUniqueTotal = hUniqueIndicesRange[tableNum];
    at::Tensor uniqueKeys = at::empty(numUniqueTotal, keys.options());
    at::Tensor uniqueSrcIndices = at::empty({0}, keys.options());  // 接口保留
    at::Tensor uniqueCnts = at::empty(numUniqueTotal, keys.options().dtype(torch::kLong));
    int64_t uniqueEmbsOffset = 0;

    for (int i = 0; i < tableNum; ++i) {
        int64_t tmpUniqueNum = hUniqueIndicesRange[i + 1] - hUniqueIndicesRange[i];
        if (tmpUniqueNum != 0) {
            uniqueKeys.slice(0, uniqueEmbsOffset, uniqueEmbsOffset + tmpUniqueNum).copy_(tmpUniqueIndices[i]);
            uniqueCnts.slice(0, uniqueEmbsOffset, uniqueEmbsOffset + tmpUniqueNum)
                .copy_(uniqueCntVec[i].slice(0, 0, tmpUniqueNum));
            uniqueEmbsOffset += tmpUniqueNum;
        }
    }

    return std::make_tuple(uniqueKeys, inverseIdx, dUniqueIndicesTableRange, hUniqueIndicesTableRange, uniqueCnts,
                           uniqueSrcIndices);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> segmented_unique_op_npu(const at::Tensor& keys,
                                                                                   const at::Tensor& segmentRange)
{
    auto result = segmented_unique_op(keys, segmentRange);
    return std::make_tuple(std::get<0>(result), std::get<1>(result), std::get<2>(result), std::get<3>(result));
}

void dedup_input_indices_npu(const at::Tensor indices, const at::Tensor offsets,
                             const at::Tensor dTableOffsetsInFeature, int tableNum, int localBatchSize,
                             const at::Tensor reverseIdx, const at::Tensor dUniqueNums, const at::Tensor dUniqueOffsets,
                             std::vector<at::Tensor> uniqueIdx, at::Tensor& newOffsets, at::Tensor& newLengths)
{
    int64_t indicesShape = indices.size(0);
    at::Tensor segmentRange = get_table_range_npu(offsets, dTableOffsetsInFeature);
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> segUniqueResult =
        segmented_unique_op(indices, segmentRange);
    at::Tensor uniqueKeys = std::get<0>(segUniqueResult);
    at::Tensor inverseIdx = std::get<1>(segUniqueResult);
    at::Tensor dUniqueIndicesRange = std::get<2>(segUniqueResult);
    at::Tensor hUniqueIndicesRange = std::get<3>(segUniqueResult);

    size_t reverseIdxSize = inverseIdx.numel() * inverseIdx.element_size();
    if (reverseIdxSize > 0) {
        TORCH_CHECK(aclrtMemcpy(reverseIdx.data_ptr(), reverseIdxSize, inverseIdx.data_ptr(), reverseIdxSize,
                                ACL_MEMCPY_DEVICE_TO_DEVICE) == ACL_SUCCESS,
                    "Sync MemCpy failed for inverse idx.");
    }

    at::Tensor lengths = hUniqueIndicesRange.narrow(0, 1, tableNum) - hUniqueIndicesRange.narrow(0, 0, tableNum);
    dUniqueNums.copy_(lengths);
    const int64_t* hUniqueIndicesRangePtr = hUniqueIndicesRange.data_ptr<int64_t>();
    for (int i = 0; i < tableNum; ++i) {
        int64_t start = hUniqueIndicesRangePtr[i];
        int64_t end = hUniqueIndicesRangePtr[i + 1];
        int64_t len = end - start;

        if (len > 0) {
            auto src_slice = uniqueKeys.narrow(0, start, len);
            auto dst_slice = uniqueIdx[i].narrow(0, 0, len);
            dst_slice.copy_(src_slice);
        }
    }

    dUniqueOffsets.copy_(dUniqueIndicesRange);

    get_new_length_and_offsets_npu(dUniqueOffsets, dTableOffsetsInFeature, newOffsets, newLengths, localBatchSize);
}

std::tuple<at::Tensor, at::Tensor> reduce_grads(at::Tensor& grad, at::Tensor& uniqueIdx, at::Tensor& inverse)
{
    TORCH_CHECK(grad.dim() == 2, "only support 2D grad");
    TORCH_CHECK(inverse.element_size() == 8, "inverse element must be 8 bytes");

    int num = uniqueIdx.size(0);
    int dim = grad.size(1);
    at::Tensor uniqueGrad = at::zeros({num, dim}, grad.options());

    if (num == 0) {
        return std::make_tuple(uniqueIdx, uniqueGrad);
    }

    const int DIM_THRESHOLD = 8;
    if (dim > DIM_THRESHOLD) {
        uniqueGrad.scatter_add_(0, inverse.unsqueeze(1).expand_as(grad), grad);
        c10_npu::getCurrentNPUStream().synchronize();
        return std::make_tuple(uniqueIdx, uniqueGrad);
    }

    int maxCore = AclSingleton::GetInstance().GetMaxCores();
    auto stream = c10_npu::getCurrentNPUStream().stream(true);

    at::Tensor cGrad = grad.contiguous();
    at::Tensor cInverse = inverse.contiguous();
    at::Tensor cUniqueGrad = uniqueGrad.contiguous();

    int blkNum = (cGrad.numel() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    int coreNum = (blkNum < maxCore) ? blkNum : maxCore;
    int baseBlk = blkNum / coreNum;
    int remainBlk = blkNum % coreNum;

    ACLRT_LAUNCH_KERNEL(reduce_grad_op)
    (coreNum, stream, cGrad.data_ptr(), cInverse.data_ptr(), cGrad.size(0), cGrad.size(1), baseBlk, remainBlk,
     cUniqueGrad.data_ptr());
    c10_npu::getCurrentNPUStream().synchronize();
    if (!uniqueGrad.is_contiguous()) {
        // 如何内存不连续，contiguous会分配新内存
        uniqueGrad.copy_(cUniqueGrad);
    }

    return std::make_tuple(uniqueIdx, uniqueGrad);
}

void find_pointers(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n, const at::Tensor keys,
                   at::Tensor values, at::Tensor founds, const std::optional<uint64_t> score = std::nullopt)
{
    if (n == 0) {
        return;
    }

    TORCH_CHECK(n == keys.size(0) && n == values.size(0) && n == founds.size(0),
                "find_pointers: n must equal to the size of keys, values and founds tensors.");

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto values_data_ptr = reinterpret_cast<void**>(values.data_ptr<int64_t>());
    auto found_tensor_data_ptr = founds.data_ptr<bool>();

    // update score.
    if (score.has_value()) {
        void* score_ptr = nullptr;
        if (table->get_evict_strategy() == EvictStrategy::kCustomized ||
            table->get_evict_strategy() == EvictStrategy::kLfu) {
            auto&& option = at::TensorOptions().dtype(torch::kInt64).device(keys.device());
            // broadcast scores
            at::Tensor bc_scores = at::empty({static_cast<int64_t>(n)}, option);
            // tensor类型是uint64时，调用fill接口会报错
            bc_scores.fill_(score.value());
            score_ptr = bc_scores.data_ptr();
        }
        table->find_pointers(n, keys.data_ptr(), values_data_ptr, found_tensor_data_ptr, score_ptr, stream);
    } else {
        std::shared_ptr<const dyn_emb::DynamicVariableBase> const_table = table;
        const_table->find_pointers(n, keys.data_ptr(), values_data_ptr, found_tensor_data_ptr, nullptr, stream);
    }
}

void find_pointers_with_scores(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n,
                               const at::Tensor keys, at::Tensor values, at::Tensor founds,
                               const std::optional<at::Tensor>& scores = std::nullopt)
{
    if (n == 0) {
        return;
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto values_data_ptr = reinterpret_cast<void**>(values.data_ptr<int64_t>());
    auto found_tensor_data_ptr = founds.data_ptr<bool>();

    // update score.
    if (scores.has_value()) {
        if (table->get_evict_strategy() == EvictStrategy::kCustomized ||
            table->get_evict_strategy() == EvictStrategy::kLfu) {
            table->find_pointers(n, keys.data_ptr(), values_data_ptr, found_tensor_data_ptr, scores.value().data_ptr(),
                                 stream);
        } else {
            table->find_pointers(n, keys.data_ptr(), values_data_ptr, found_tensor_data_ptr, nullptr, stream);
        }
    } else {
        std::shared_ptr<const dyn_emb::DynamicVariableBase> const_table = table;
        const_table->find_pointers(n, keys.data_ptr(), values_data_ptr, found_tensor_data_ptr, nullptr, stream);
    }
}

void find_and_initialize(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n, const at::Tensor keys,
                         at::Tensor value_ptrs, at::Tensor values, at::Tensor founds,
                         const c10::optional<dyn_emb::InitializerArgs>& initializer_args = c10::nullopt)
{
    if (n == 0) {
        return;
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto values_data_ptr = reinterpret_cast<void**>(value_ptrs.data_ptr<int64_t>());
    auto found_tensor_data_ptr = founds.data_ptr<bool>();
    table->find_and_initialize(n, keys.data_ptr(), values_data_ptr, values.data_ptr(), found_tensor_data_ptr,
                               initializer_args, stream);
}

int64_t dyn_emb_rows(std::shared_ptr<dyn_emb::DynamicVariableBase> table)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    return table->rows(stream);
}

int64_t dyn_emb_cols(std::shared_ptr<dyn_emb::DynamicVariableBase> table)
{
    return table->cols();
}

bool dyn_emb_is_pure_hbm_mode(std::shared_ptr<dyn_emb::DynamicVariableBase> table)
{
    return table->is_pure_hbm_mode();
}

void count_matched(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const uint64_t threshold,
                   at::Tensor num_matched)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    table->count_matched(threshold, num_matched, stream);
}

void export_batch_matched(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const uint64_t threshold,
                          const uint64_t n, const uint64_t offset, at::Tensor num_matched, at::Tensor keys,
                          at::Tensor values)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    table->export_batch_matched(threshold, n, offset, num_matched, keys, values, c10::nullopt, stream);
}

void insert_and_evict(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n, const at::Tensor keys,
                      const at::Tensor values, const std::optional<uint64_t> score, at::Tensor evicted_keys,
                      at::Tensor evicted_values, at::Tensor evicted_score, at::Tensor d_evicted_counter,
                      bool unique_key = true, bool ignore_evict_strategy = false)
{
    if (not score and
        (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu)) {
        throw std::invalid_argument("Must specify the score when evict strategy is customized or LFU.");
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    if (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu) {
        auto&& option = at::TensorOptions().dtype(torch::kInt64).device(keys.device());
        // broadcast scores
        at::Tensor bc_scores = at::empty({static_cast<int64_t>(n)}, option);
        // fill_接口不支持uint64_t
        bc_scores.fill_(static_cast<int64_t>(score.value()));
        table->insert_and_evict(n, keys.data_ptr(), values.data_ptr(), bc_scores.data_ptr(), evicted_keys.data_ptr(),
                                evicted_values.data_ptr(), evicted_score.data_ptr(),
                                reinterpret_cast<uint64_t*>(d_evicted_counter.data_ptr()), stream, unique_key,
                                ignore_evict_strategy);
    } else {
        table->insert_and_evict(n, keys.data_ptr(), values.data_ptr(), nullptr, evicted_keys.data_ptr(),
                                evicted_values.data_ptr(), evicted_score.data_ptr(),
                                reinterpret_cast<uint64_t*>(d_evicted_counter.data_ptr()), stream, unique_key,
                                ignore_evict_strategy);
    }
}

void find(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n, const at::Tensor keys,
          const at::Tensor values, const at::Tensor founds, const c10::optional<at::Tensor>& score = c10::nullopt)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    if (score.has_value()) {
        at::Tensor score_ = score.value();
        table->find(n, keys.data_ptr(), values.data_ptr(), founds.data_ptr<bool>(), score_.data_ptr(), stream);
    } else {
        table->find(n, keys.data_ptr(), values.data_ptr(), founds.data_ptr<bool>(), nullptr, stream);
    }
}

void erase(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n, const at::Tensor keys)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    table->erase(n, keys.data_ptr(), stream);
}

void clear(std::shared_ptr<dyn_emb::DynamicVariableBase> table)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    table->clear(stream);
}

void reserve(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t new_capacity)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    table->reserve(new_capacity, stream);
}

void accum_or_assign(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n, const at::Tensor keys,
                     const at::Tensor value_or_deltas, const at::Tensor accum_or_assigns,
                     const c10::optional<at::Tensor>& score = c10::nullopt, bool ignore_evict_strategy = false)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    if (score.has_value()) {
        at::Tensor score_ = score.value();
        table->accum_or_assign(n, keys.data_ptr(), value_or_deltas.data_ptr(), accum_or_assigns.data_ptr<bool>(),
                               score_.data_ptr(), stream, ignore_evict_strategy);
    } else {
        table->accum_or_assign(n, keys.data_ptr(), value_or_deltas.data_ptr(), accum_or_assigns.data_ptr<bool>(),
                               nullptr, stream, ignore_evict_strategy);
    }
}

void find_or_insert(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n, const at::Tensor keys,
                    const at::Tensor values, const std::optional<uint64_t> score = std::nullopt, bool unique_key = true,
                    bool ignore_evict_strategy = false)
{
    if (not score and
        (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu)) {
        throw std::invalid_argument("Must specify the score when evict strategy is customized or LFU.");
    }
    if (n == 0) {
        return;
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    at::Tensor new_tensor =
        at::empty({static_cast<int64_t>(n)}, at::TensorOptions().dtype(at::kLong).device(values.device()));

    auto new_tensor_data_ptr = reinterpret_cast<void**>(new_tensor.data_ptr<int64_t>());

    at::Tensor found_tensor =
        at::empty({static_cast<int64_t>(n)}, at::TensorOptions().dtype(at::kBool).device(keys.device()));

    auto found_tensor_data_ptr = found_tensor.data_ptr<bool>();

    if (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu) {
        auto&& option = at::TensorOptions().dtype(torch::kInt64).device(keys.device());
        // broadcast scores
        at::Tensor bc_scores = at::empty({static_cast<int64_t>(n)}, option);
        // fill_接口不支持uint64_t
        bc_scores.fill_(static_cast<int64_t>(score.value()));
        table->find_or_insert(n, keys.data_ptr(), new_tensor_data_ptr, values.data_ptr(), found_tensor_data_ptr,
                              bc_scores.data_ptr(), stream, unique_key, ignore_evict_strategy);

    } else {
        table->find_or_insert(n, keys.data_ptr(), new_tensor_data_ptr, values.data_ptr(), found_tensor_data_ptr,
                              nullptr, stream, unique_key, ignore_evict_strategy);
    }
}

void find_or_insert_pointers(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n, const at::Tensor keys,
                             at::Tensor values, at::Tensor founds, const std::optional<uint64_t> score = c10::nullopt,
                             bool unique_key = true, bool ignore_evict_strategy = false)
{
    if (not score and
        (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu)) {
        throw std::invalid_argument("Must specify the score when evict strategy is customized or LFU.");
    }
    if (n == 0) {
        return;
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto values_data_ptr = reinterpret_cast<void**>(values.data_ptr<int64_t>());
    auto found_tensor_data_ptr = founds.data_ptr<bool>();

    if (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu) {
        auto&& option = at::TensorOptions().dtype(torch::kInt64).device(keys.device());
        // broadcast scores
        at::Tensor bc_scores = at::empty({static_cast<int64_t>(n)}, option);
        // fill_接口不支持uint64_t
        bc_scores.fill_(static_cast<int64_t>(score.value()));
        table->find_or_insert_pointers(n, keys.data_ptr(), values_data_ptr, found_tensor_data_ptr, bc_scores.data_ptr(),
                                       stream, unique_key, ignore_evict_strategy);
    } else {
        table->find_or_insert_pointers(n, keys.data_ptr(), values_data_ptr, found_tensor_data_ptr, nullptr, stream,
                                       unique_key, ignore_evict_strategy);
    }
}

void assign(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n, const at::Tensor keys,
            const at::Tensor values, const c10::optional<at::Tensor>& score = c10::nullopt, bool unique_key = true)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    if (score.has_value()) {
        at::Tensor score_ = score.value();
        table->assign(n, keys.data_ptr(), values.data_ptr(), score_.data_ptr(), stream, unique_key);
    } else {
        table->assign(n, keys.data_ptr(), values.data_ptr(), nullptr, stream, unique_key);
    }
}

void lookup_forward(const at::Tensor& src, const at::Tensor& dst, const at::Tensor& offset, const at::Tensor& inverse,
                    int32_t combiner, int32_t totalDims, int32_t accumDims, int32_t evSize, int32_t numVec,
                    int32_t batchSize)
{
    // data type
    TORCH_CHECK(offset.dtype() == inverse.dtype(), "offset and inverse must have the same dtype");
    TORCH_CHECK(offset.dim() == 1 && inverse.dim() == 1, "offset and inverse must be 1D tensor");
    auto srcType = scalartype_to_datatype(convertTypeMetaToScalarType(src.dtype()));
    auto dstType = scalartype_to_datatype(convertTypeMetaToScalarType(dst.dtype()));
    auto offsetType = scalartype_to_datatype(convertTypeMetaToScalarType(offset.dtype()));

    // data info
    uint8_t* srcData = src.is_contiguous() ? static_cast<uint8_t*>(src.data_ptr())
                                           : static_cast<uint8_t*>(src.contiguous().data_ptr());
    uint8_t* offsetData = offset.is_contiguous() ? static_cast<uint8_t*>(offset.data_ptr())
                                                 : static_cast<uint8_t*>(offset.contiguous().data_ptr());
    uint8_t* inverseData = inverse.is_contiguous() ? static_cast<uint8_t*>(inverse.data_ptr())
                                                   : static_cast<uint8_t*>(inverse.contiguous().data_ptr());
    uint8_t* dstData = static_cast<uint8_t*>(dst.data_ptr());

    constexpr uint32_t THREAD_NUM = 1024;
    constexpr int32_t EMBEDDING_THRESHOLD = 8;
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    bool useFloat2 = (evSize % 2 == 0 && srcType == DataType::Float32 && evSize > EMBEDDING_THRESHOLD);
    int32_t evSizeVec = evSize;
    if (useFloat2) {
        evSizeVec >>= 1;
    }
    int32_t outLen = evSizeVec * numVec;
    int32_t totalBlocks = (outLen + THREAD_NUM - 1) / THREAD_NUM;
    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    int32_t coreNum = std::min(maxCores, totalBlocks);
    TORCH_CHECK(coreNum > 0, "coreNum must be greater than 0");
    int32_t blocksPerCore = totalBlocks / coreNum;
    int32_t remainderBlocks = totalBlocks % coreNum;
    // 参考其他算子在A5机器上的经验值，offset dtype为int32时，小表阈值为24*1024，其他为44*1024
    bool isInt32 = offset.dtype() == torch::kInt32;
    bool isSmall = (outLen <= (isInt32 ? SMALL_DATA_THRESHOLD_32 : SMALL_DATA_THRESHOLD));

    ACLRT_LAUNCH_KERNEL(pooling_embeddings)
    (coreNum, aclStream, srcData, dstData, offsetData, inverseData, combiner, totalDims, accumDims, evSize, numVec,
     batchSize, totalBlocks, blocksPerCore, remainderBlocks, isSmall, static_cast<uint32_t>(srcType),
     static_cast<uint32_t>(dstType), static_cast<uint32_t>(offsetType), THREAD_NUM, outLen, useFloat2, evSizeVec);
}

class DeviceTimestamp {
public:
    DeviceTimestamp()
    {
        CheckAclRet(aclrtMalloc(reinterpret_cast<void**>(&d_timestamp), sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST) ==
                        ACL_SUCCESS,
                    "aclrtMalloc d_timestamp failed.");
        // 分配页锁定主机内存
        CheckAclRet(aclrtMallocHost(reinterpret_cast<void**>(&h_timestamp_pinned), sizeof(int64_t)) == ACL_SUCCESS,
                    "aclrtMallocHost h_timestamp_pinned failed.");
    }

    ~DeviceTimestamp()
    {
        CheckAclRet(aclrtFree(d_timestamp) == ACL_SUCCESS, "aclrtFree d_timestamp failed.");
        CheckAclRet(aclrtFreeHost(h_timestamp_pinned) == ACL_SUCCESS, "aclrtFreeHost h_timestamp_pinned failed.");
    }

    int64_t get(const aclrtStream& stream)
    {
        ACLRT_LAUNCH_KERNEL(device_timestamp)(1, stream, d_timestamp);
        CheckAclRet(aclrtMemcpyAsync(h_timestamp_pinned, sizeof(int64_t), d_timestamp, sizeof(int64_t),
                                     ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS,
                    "aclrtMemcpyAsync h_timestamp_pinned failed.");
        CheckAclRet(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "aclrtSynchronizeStream failed.");
        // 从页锁定内存读取值
        return *h_timestamp_pinned;
    }

private:
    int64_t* d_timestamp{nullptr};
    int64_t* h_timestamp_pinned{nullptr};  // 指向页锁定主机内存
};

int64_t device_timestamp()
{
    // 获取当前流
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    // 创建线程内共享对象，每个线程一个，每个线程只创建一次
    thread_local DeviceTimestamp deviceTimestamp;
    return deviceTimestamp.get(stream);
}
void dynamic_emb_Adam_with_pointer(const torch::Tensor& grads, const torch::Tensor& val_pointers, DataType val_type,
                                   int64_t state_dim, const float lr, const float beta1, const float beta2,
                                   const float eps, const float weight_decay, const uint32_t iter_num)
{
    int32_t in_length = grads.numel();
    if (in_length == 0) {
        return;
    }
    uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto grads_continous = grads.contiguous();
    auto val_pointers_continous = val_pointers.contiguous();

    float* grads_ptr = grads_continous.data_ptr<float>();
    void* values_ptr = val_pointers_continous.data_ptr();

    // 将重复计算的常量提取到 Host 侧仅计算一次
    float oneMinusBeta1 = 1.0f - beta1;
    float oneMinusBeta2 = 1.0f - beta2;
    float mHatDenom = 1.0f - std::pow(beta1, iter_num);
    float vHatDenom = 1.0f - std::pow(beta2, iter_num);

    float step_size = lr / mHatDenom;
    float inv_vHatDenom = 1.0f / vHatDenom;

    // 获取最大核数
    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    bool is_small = (in_length <= SMALL_DATA_THRESHOLD);

    // === 分支 1：偶数维度且 float32，使用 float2 算子===
    if (grad_dim % 2 == 0) {
        int32_t vec_length = in_length / 2;
        const int32_t VEC_PER_BLOCK = ELEMENTS_PER_BLOCK / 2;

        int32_t total_blocks = (vec_length + VEC_PER_BLOCK - 1) / VEC_PER_BLOCK;
        int32_t core_num = std::min(max_cores, total_blocks);
        if (core_num == 0) {
            return;
        }
        int32_t blocks_per_core = total_blocks / core_num;
        int32_t remainder_blocks = total_blocks % core_num;

        ACLRT_LAUNCH_KERNEL(Adam_update_float2)
        (core_num, stream, grads_ptr, values_ptr, grad_dim, vec_length, beta1, beta2, oneMinusBeta1, oneMinusBeta2,
         step_size, inv_vHatDenom, weight_decay, eps, total_blocks, blocks_per_core, remainder_blocks, is_small);
    } else {  // === 分支 2：奇数维度，使用标量 float 算子 ===
        int32_t total_blocks = (in_length + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
        int32_t core_num = std::min(max_cores, total_blocks);
        if (core_num == 0) {
            return;
        }

        int32_t blocks_per_core = total_blocks / core_num;
        int32_t remainder_blocks = total_blocks % core_num;

        ACLRT_LAUNCH_KERNEL(Adam_update)
        (core_num, stream, grads_ptr, values_ptr, grad_dim, in_length, beta1, beta2, oneMinusBeta1, oneMinusBeta2,
         step_size, inv_vHatDenom, weight_decay, eps, total_blocks, blocks_per_core, remainder_blocks, is_small);
    }
}

void dynamic_emb_adamW_with_pointer(const torch::Tensor& grads, const torch::Tensor& val_pointers, DataType val_type,
                                    int64_t state_dim, const float lr, const float beta1, const float beta2,
                                    const float eps, const float weight_decay, const uint32_t iter_num)
{
    int32_t in_length = grads.numel();
    if (in_length == 0) {
        LOG_ERROR("dynamic_emb_adamW_with_pointer: in_length is zero!");
        return;
    }
    const uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));
    if (state_dim != static_cast<int64_t>(grad_dim) * 2) {
        LOG_ERROR("dynamic_emb_adamW_with_pointer: state_dim must be 2 * embedding dim (m and v per dim).");
        return;
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto grads_continuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto val_pointers_continuous = val_pointers.is_contiguous() ? val_pointers : val_pointers.contiguous();

    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continuous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(val_pointers_continuous.data_ptr());
    const float base_float = 1.0f;
    const float one_minus_beta1 = base_float - beta1;
    const float one_minus_beta2 = base_float - beta2;
    const float m_hat_denom = base_float - std::pow(beta1, iter_num);
    const float v_hat_denom = base_float - std::pow(beta2, iter_num);
    const float step_size = lr / m_hat_denom;
    const float inv_v_hat_denom = base_float / v_hat_denom;
    const float decay_factor = base_float - lr * weight_decay;

    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    const int32_t num_rows = in_length / static_cast<int32_t>(grad_dim);
    if (ShouldUseOptimizerSimd(grad_dim, in_length, grad_type, val_type)) {
        constexpr uint32_t kRowsPerGroup = 1U;
        if (LaunchAdamWSimd(stream, grads_ptr, values_ptr, nullptr, grads_continuous, grad_dim, num_rows, max_cores,
                            beta1, beta2, one_minus_beta1, one_minus_beta2, step_size, inv_v_hat_denom, decay_factor,
                            eps, kRowsPerGroup, grad_type, val_type)) {
            return;
        }
    }

    if (!IsSupportedOptimizerSimdDtype(grad_type, val_type)) {
        LOG_ERROR("dynamic_emb_adamW_with_pointer: unsupported grad/weight dtype for AdamW update.");
        return;
    }

    bool is_small = (in_length <= SMALL_DATA_THRESHOLD);
    constexpr uint32_t optimizer_kind = static_cast<uint32_t>(OptimizerKind::AdamW);
    if (!LaunchUpdateKernelCommon(stream, grads_ptr, values_ptr, nullptr, grad_dim, in_length, max_cores, is_small,
                                  beta1, beta2, one_minus_beta1, one_minus_beta2, step_size, inv_v_hat_denom,
                                  decay_factor, eps, grad_type, val_type, optimizer_kind)) {
        LOG_ERROR("dynamic_emb_adamW_with_pointer: LaunchUpdateKernelCommon failed!");
        return;
    }
}

void dynamic_emb_adamW_with_pointer_hybrid(const torch::Tensor& grads, const torch::Tensor& val_pointers,
                                           DataType val_type, int64_t state_dim, const float lr, const float beta1,
                                           const float beta2, const float eps, const float weight_decay,
                                           const uint32_t iter_num)
{
    int32_t in_length = grads.numel();
    if (in_length == 0) {
        LOG_ERROR("dynamic_emb_adamW_with_pointer_hybrid: in_length is zero!");
        return;
    }
    const uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));
    if (state_dim != static_cast<int64_t>(grad_dim) * 2) {
        LOG_ERROR("dynamic_emb_adamW_with_pointer_hybrid: state_dim must be 2 * embedding dim (m and v per dim).");
        return;
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto grads_continuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto val_pointers_continuous = val_pointers.is_contiguous() ? val_pointers : val_pointers.contiguous();

    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continuous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(val_pointers_continuous.data_ptr());
    const float base_float = 1.0f;
    const float one_minus_beta1 = base_float - beta1;
    const float one_minus_beta2 = base_float - beta2;
    const float m_hat_denom = base_float - std::pow(beta1, iter_num);
    const float v_hat_denom = base_float - std::pow(beta2, iter_num);
    const float step_size = lr / m_hat_denom;
    const float inv_v_hat_denom = base_float / v_hat_denom;
    const float decay_factor = base_float - lr * weight_decay;

    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    const int32_t num_rows = in_length / static_cast<int32_t>(grad_dim);

    if (!IsSupportedOptimizerSimdDtype(grad_type, val_type)) {
        LOG_ERROR("dynamic_emb_adamW_with_pointer_hybrid: unsupported grad/weight dtype for AdamW update.");
        return;
    }

    constexpr uint32_t kRowsPerGroup = 1U;
    if (!LaunchAdamWSimd(stream, grads_ptr, values_ptr, nullptr, grads_continuous, grad_dim, num_rows, max_cores, beta1,
                         beta2, one_minus_beta1, one_minus_beta2, step_size, inv_v_hat_denom, decay_factor, eps,
                         kRowsPerGroup, grad_type, val_type)) {
        LOG_ERROR("dynamic_emb_adamW_with_pointer_hybrid: LaunchAdamWSimd failed!");
    }
}

void dynamic_emb_adagrad_with_pointer(const torch::Tensor& grads, const torch::Tensor& valPointers, DataType valType,
                                      int64_t stateDim, const float lr, const float eps)
{
    int32_t inLength = static_cast<int32_t>(grads.numel());
    if (inLength == 0) {
        LOG_ERROR("dynamic_emb_adagrad_with_pointer: inLength is zero!");
        return;
    }
    const uint32_t gradDim = static_cast<uint32_t>(grads.size(1));
    if (stateDim != static_cast<int64_t>(gradDim)) {
        LOG_ERROR("dynamic_emb_adagrad_with_pointer: stateDim must equal embedding dim (sum of squared grads state per "
                  "dim).");
        return;
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto gradsContinuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto valPointersContinuous = valPointers.is_contiguous() ? valPointers : valPointers.contiguous();

    uint8_t* gradsPtr = static_cast<uint8_t*>(gradsContinuous.data_ptr());
    uint8_t* valuesPtr = static_cast<uint8_t*>(valPointersContinuous.data_ptr());
    const float beta1 = 0.0f;
    const float beta2 = 0.0f;
    const float oneMinusBeta1 = 0.0f;
    const float oneMinusBeta2 = 0.0f;
    const float stepSize = lr;
    const float invVHatDenom = 1.0f;
    const float decayFactor = 1.0f;

    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    const int32_t numRows = inLength / static_cast<int32_t>(gradDim);
    if (ShouldUseAdagradSimd(gradDim, inLength, gradType, valType)) {
        const uint32_t rowsPerGroup = AdagradSimdRowsPerGroup(gradType, valType);
        if (LaunchAdagradSimd(stream, gradsPtr, valuesPtr, nullptr, gradsContinuous, gradDim, numRows, maxCores, lr,
                              eps, rowsPerGroup, gradType, valType)) {
            return;
        }
    }

    bool isSmall = (inLength <= SMALL_DATA_THRESHOLD);
    constexpr uint32_t optimizerKind = static_cast<uint32_t>(OptimizerKind::AdaGrad);

    if (!LaunchUpdateKernelCommon(stream, gradsPtr, valuesPtr, nullptr, gradDim, inLength, maxCores, isSmall, beta1,
                                  beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                                  gradType, valType, optimizerKind)) {
        return;
    }
}

void dynamic_emb_rowwise_adagrad_with_pointer(const torch::Tensor& grads, const torch::Tensor& valPointers,
                                              DataType valType, int64_t stateDim, const float lr, const float eps)
{
    int32_t inLength = static_cast<int32_t>(grads.numel());
    if (inLength <= 0) {
        LOG_ERROR("dynamic_emb_rowwise_adagrad_with_pointer: grads.numel() must be positive.");
        return;
    }
    const uint32_t gradDim = static_cast<uint32_t>(grads.size(1));
    if (stateDim <= 0 || gradDim <= 0) {
        LOG_ERROR("dynamic_emb_rowwise_adagrad_with_pointer: stateDim and grads.size(1) must be positive.");
        return;
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto gradsContinuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto valPointersContinuous = valPointers.is_contiguous() ? valPointers : valPointers.contiguous();

    uint8_t* gradsPtr = static_cast<uint8_t*>(gradsContinuous.data_ptr());
    uint8_t* valuesPtr = static_cast<uint8_t*>(valPointersContinuous.data_ptr());
    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));

    const int32_t numRows = inLength / static_cast<int32_t>(gradDim);
    const int32_t coreNum = std::max<int32_t>(1, std::min<int32_t>(maxCores, numRows));
    const uint32_t valDim = gradDim + static_cast<uint32_t>(stateDim);

    if (ShouldUseRowwiseAdagradSimd(gradDim, gradType, valType)) {
        constexpr uint32_t kRowsPerGroup = 1U;
        if (LaunchRowwiseAdagradSimd(stream, gradsPtr, valuesPtr, nullptr, gradsContinuous, gradDim, valDim, numRows,
                                     maxCores, lr, eps, kRowsPerGroup, gradType, valType)) {
            return;
        }
    }

    // 偶数 embedding 维且 float32 时走 float2 算子
    if (gradDim % 2 == 0 && gradType == DataType::Float32 && valType == DataType::Float32) {
        ACLRT_LAUNCH_KERNEL(rowwise_adagrad_update_float2)
        (coreNum, stream, gradsPtr, valuesPtr, nullptr, gradDim, inLength, lr, eps);
        return;
    }

    ACLRT_LAUNCH_KERNEL(rowwise_adagrad_update)
    (coreNum, stream, gradsPtr, valuesPtr, nullptr, gradDim, inLength, lr, eps, static_cast<uint32_t>(gradType),
     static_cast<uint32_t>(valType));
}

void dynamic_emb_rowwise_adagrad_with_pointer_hybrid(const torch::Tensor& grads, const torch::Tensor& valPointers,
                                                     DataType valType, int64_t stateDim, const float lr,
                                                     const float eps)
{
    int32_t inLength = static_cast<int32_t>(grads.numel());
    if (inLength <= 0) {
        LOG_ERROR("dynamic_emb_rowwise_adagrad_with_pointer_hybrid: grads.numel() must be positive.");
        return;
    }
    const uint32_t gradDim = static_cast<uint32_t>(grads.size(1));
    if (stateDim <= 0 || gradDim <= 0) {
        LOG_ERROR("dynamic_emb_rowwise_adagrad_with_pointer_hybrid: stateDim and grads.size(1) must be positive.");
        return;
    }

    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto gradsContinuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto valPointersContinuous = valPointers.is_contiguous() ? valPointers : valPointers.contiguous();

    uint8_t* gradsPtr = static_cast<uint8_t*>(gradsContinuous.data_ptr());
    uint8_t* valuesPtr = static_cast<uint8_t*>(valPointersContinuous.data_ptr());
    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();

    const int32_t numRows = inLength / static_cast<int32_t>(gradDim);
    const uint32_t valDim = gradDim + static_cast<uint32_t>(stateDim);
    constexpr uint32_t kRowsPerGroup = 1U;
    if (!LaunchRowwiseAdagradSimd(stream, gradsPtr, valuesPtr, nullptr, gradsContinuous, gradDim, valDim, numRows,
                                  maxCores, lr, eps, kRowsPerGroup, gradType, valType)) {
        LOG_ERROR("dynamic_emb_rowwise_adagrad_with_pointer_hybrid: LaunchRowwiseAdagradSimd failed.");
    }
}

void dynamic_emb_adamW_with_table(std::shared_ptr<dyn_emb::DynamicVariableBase> ht, const uint64_t n,
                                  const torch::Tensor& indices, const torch::Tensor& grads, const float lr,
                                  const float beta1, const float beta2, const float eps, const float weight_decay,
                                  const uint32_t iter_num, DataType weight_type)
{
    int32_t in_length = grads.numel();
    if (n == 0 || in_length == 0) {
        LOG_ERROR("n or in_length is zero!");
        return;
    }
    at::Tensor founds =
        at::empty({static_cast<int64_t>(n)}, at::TensorOptions().dtype(at::kBool).device(indices.device()));
    at::Tensor vector_ptrs =
        at::empty({static_cast<int64_t>(n)}, at::TensorOptions().dtype(at::kLong).device(indices.device()));

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    find_pointers(ht, n, indices, vector_ptrs, founds);

    const uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));
    auto grads_continuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto vector_pointers_continuous = vector_ptrs.is_contiguous() ? vector_ptrs : vector_ptrs.contiguous();
    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continuous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(vector_pointers_continuous.data_ptr());
    auto founds_continuous = founds.contiguous();
    uint8_t* founds_ptr = static_cast<uint8_t*>(founds_continuous.data_ptr());
    const float base_float = 1.0f;
    const float one_m_beta1 = base_float - beta1;
    const float one_m_beta2 = base_float - beta2;
    const float m_hat_denom = base_float - std::pow(beta1, iter_num);
    const float v_hat_denom = base_float - std::pow(beta2, iter_num);
    const float step_size = lr / m_hat_denom;
    const float inv_v_hat_denom = base_float / v_hat_denom;
    const float decay_factor = base_float - lr * weight_decay;

    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    const int32_t num_rows = in_length / static_cast<int32_t>(grad_dim);
    if (ShouldUseOptimizerSimd(grad_dim, in_length, grad_type, weight_type) || !ht->is_pure_hbm_mode()) {
        constexpr uint32_t kRowsPerGroup = 1U;
        if (LaunchAdamWSimd(stream, grads_ptr, values_ptr, founds_ptr, grads_continuous, grad_dim, num_rows, max_cores,
                            beta1, beta2, one_m_beta1, one_m_beta2, step_size, inv_v_hat_denom, decay_factor, eps,
                            kRowsPerGroup, grad_type, weight_type)) {
            return;
        } else {
            LOG_ERROR("dynamic_emb_adamW_with_table: LaunchAdamWSimd failed!");
            return;
        }
    }

    if (!IsSupportedOptimizerSimdDtype(grad_type, weight_type)) {
        LOG_ERROR("dynamic_emb_adamW_with_table: unsupported grad/weight dtype for AdamW update.");
        return;
    }

    bool is_small = (in_length <= SMALL_DATA_THRESHOLD);
    constexpr uint32_t optimizer_kind = static_cast<uint32_t>(OptimizerKind::AdamW);
    if (!LaunchUpdateKernelCommon(stream, grads_ptr, values_ptr, founds_ptr, grad_dim, in_length, max_cores, is_small,
                                  beta1, beta2, one_m_beta1, one_m_beta2, step_size, inv_v_hat_denom, decay_factor, eps,
                                  grad_type, weight_type, optimizer_kind)) {
        LOG_ERROR("dynamic_emb_adamW_with_table: LaunchUpdateKernelCommon failed!");
        return;
    }
}

void dynamic_emb_adagrad_with_table(std::shared_ptr<dyn_emb::DynamicVariableBase> ht, const uint64_t n,
                                    const torch::Tensor& indices, const torch::Tensor& grads, const float lr,
                                    const float eps, DataType weightType)
{
    int32_t inLength = static_cast<int32_t>(grads.numel());
    if (n == 0 || inLength == 0) {
        LOG_ERROR("n or inLength is zero!");
        return;
    }

    at::Tensor founds =
        at::empty({static_cast<int64_t>(n)}, at::TensorOptions().dtype(at::kBool).device(indices.device()));
    at::Tensor vectorPtrs =
        at::empty({static_cast<int64_t>(n)}, at::TensorOptions().dtype(at::kLong).device(indices.device()));

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    find_pointers(ht, n, indices, vectorPtrs, founds);

    uint32_t gradDim = static_cast<uint32_t>(grads.size(1));
    auto gradsContinuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto vectorPtrsContinuous = vectorPtrs.is_contiguous() ? vectorPtrs : vectorPtrs.contiguous();
    uint8_t* gradsPtr = static_cast<uint8_t*>(gradsContinuous.data_ptr());
    uint8_t* valuesPtr = static_cast<uint8_t*>(vectorPtrsContinuous.data_ptr());
    auto foundsContinuous = founds.contiguous();
    uint8_t* foundsPtr = static_cast<uint8_t*>(foundsContinuous.data_ptr());

    const float beta1 = 0.0f;
    const float beta2 = 0.0f;
    const float oneMinusBeta1 = 0.0f;
    const float oneMinusBeta2 = 0.0f;
    const float stepSize = lr;
    const float invVHatDenom = 1.0f;
    const float decayFactor = 1.0f;

    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    const int32_t numRows = inLength / static_cast<int32_t>(gradDim);
    if (ShouldUseAdagradSimd(gradDim, inLength, gradType, weightType, ht)) {
        constexpr uint32_t kRowsPerGroup = 1U;
        if (LaunchAdagradSimd(stream, gradsPtr, valuesPtr, foundsPtr, gradsContinuous, gradDim, numRows, maxCores, lr,
                              eps, kRowsPerGroup, gradType, weightType)) {
            return;
        }
    }

    bool isSmall = (inLength <= SMALL_DATA_THRESHOLD);
    constexpr uint32_t optimizerKind = static_cast<uint32_t>(OptimizerKind::AdaGrad);

    if (!LaunchUpdateKernelCommon(stream, gradsPtr, valuesPtr, foundsPtr, gradDim, inLength, maxCores, isSmall, beta1,
                                  beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                                  gradType, weightType, optimizerKind)) {
        return;
    }
}

void dynamic_emb_rowwise_adagrad_with_table(std::shared_ptr<dyn_emb::DynamicVariableBase> ht, const uint64_t n,
                                            const torch::Tensor& indices, const torch::Tensor& grads, const float lr,
                                            const float eps, DataType weightType)
{
    int32_t inLength = static_cast<int32_t>(grads.numel());
    if (n == 0 || inLength <= 0) {
        LOG_ERROR("dynamic_emb_rowwise_adagrad_with_table: n and grads.numel() must be positive.");
        return;
    }

    at::Tensor founds =
        at::empty({static_cast<int64_t>(n)}, at::TensorOptions().dtype(at::kBool).device(indices.device()));
    at::Tensor vectorPtrs =
        at::empty({static_cast<int64_t>(n)}, at::TensorOptions().dtype(at::kLong).device(indices.device()));

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    find_pointers(ht, n, indices, vectorPtrs, founds);

    uint32_t gradDim = static_cast<uint32_t>(grads.size(1));
    auto gradsContinuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto vectorPtrsContinuous = vectorPtrs.is_contiguous() ? vectorPtrs : vectorPtrs.contiguous();
    uint8_t* gradsPtr = static_cast<uint8_t*>(gradsContinuous.data_ptr());
    uint8_t* valuesPtr = static_cast<uint8_t*>(vectorPtrsContinuous.data_ptr());
    auto foundsContinuous = founds.contiguous();
    uint8_t* foundsPtr = static_cast<uint8_t*>(foundsContinuous.data_ptr());
    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    if (gradDim == 0) {
        LOG_ERROR("dynamic_emb_rowwise_adagrad_with_table: grads.size(1) must be positive.");
        return;
    }
    const int32_t numRows = inLength / static_cast<int32_t>(gradDim);
    const int32_t coreNum = std::max<int32_t>(1, std::min<int32_t>(maxCores, numRows));
    const uint32_t valDim = gradDim + static_cast<uint32_t>(ht->optstate_dim());

    if (ShouldUseRowwiseAdagradSimd(gradDim, gradType, weightType, ht)) {
        constexpr uint32_t kRowsPerGroup = 1U;
        if (LaunchRowwiseAdagradSimd(stream, gradsPtr, valuesPtr, foundsPtr, gradsContinuous, gradDim, valDim, numRows,
                                     maxCores, lr, eps, kRowsPerGroup, gradType, weightType)) {
            return;
        }
    }

    // 偶数 embedding 维且 float32 时走 float2 算子
    if (gradDim % 2 == 0 && gradType == DataType::Float32 && weightType == DataType::Float32) {
        ACLRT_LAUNCH_KERNEL(rowwise_adagrad_update_float2)
        (coreNum, stream, gradsPtr, valuesPtr, foundsPtr, gradDim, inLength, lr, eps);
        return;
    }

    ACLRT_LAUNCH_KERNEL(rowwise_adagrad_update)
    (coreNum, stream, gradsPtr, valuesPtr, foundsPtr, gradDim, inLength, lr, eps, static_cast<uint32_t>(gradType),
     static_cast<uint32_t>(weightType));
}

void dynamic_emb_adamW_fused(const torch::Tensor& grads, const torch::Tensor& values, const float lr, const float beta1,
                             const float beta2, const float eps, const float weight_decay, const uint32_t iter_num)
{
    int32_t in_length = grads.numel();
    if (in_length == 0) {
        LOG_ERROR("dynamic_emb_adamW_fused: in_length is zero!");
        return;
    }
    const uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));
    const uint32_t val_dim = static_cast<uint32_t>(values.size(1));
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto grads_continuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto values_continuous = values.is_contiguous() ? values : values.contiguous();
    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continuous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(values_continuous.data_ptr());
    const float base_float = 1.0f;
    const float one_m_beta1 = base_float - beta1;
    const float one_m_beta2 = base_float - beta2;
    const float m_hat_denom = base_float - std::pow(beta1, iter_num);
    const float v_hat_denom = base_float - std::pow(beta2, iter_num);
    const float step_size = lr / m_hat_denom;
    const float inv_v_hat_denom = base_float / v_hat_denom;
    const float decay_factor = base_float - lr * weight_decay;

    auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    auto val_type = scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));
    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    const int32_t num_rows = in_length / static_cast<int32_t>(grad_dim);
    if (ShouldUseOptimizerSimd(grad_dim, in_length, grad_type, val_type)) {
        if (LaunchAdamWFusedSimd(stream, grads_ptr, values_ptr, grads_continuous, grad_dim, val_dim, num_rows,
                                 max_cores, beta1, beta2, one_m_beta1, one_m_beta2, step_size, inv_v_hat_denom,
                                 decay_factor, eps, grad_type, val_type)) {
            return;
        } else {
            LOG_ERROR("dynamic_emb_adamW_fused: LaunchAdamWFusedSimd failed!");
            return;
        }
    }

    if (!IsSupportedOptimizerSimdDtype(grad_type, val_type)) {
        LOG_ERROR("dynamic_emb_adamW_fused: unsupported grad/weight dtype for AdamW update.");
        return;
    }

    bool is_small = (in_length <= SMALL_DATA_THRESHOLD);
    constexpr uint32_t optimizer_kind = static_cast<uint32_t>(OptimizerKind::AdamW);
    if (!LaunchUpdateFusedKernelCommon(stream, grads_ptr, values_ptr, grad_dim, val_dim, in_length, max_cores, is_small,
                                       beta1, beta2, one_m_beta1, one_m_beta2, step_size, inv_v_hat_denom, decay_factor,
                                       eps, grad_type, val_type, optimizer_kind)) {
        LOG_ERROR("dynamic_emb_adamW_fused: LaunchUpdateFusedKernelCommon failed!");
        return;
    }
}

void dynamic_emb_adamW_fused_hybrid(const torch::Tensor& grads, const torch::Tensor& values, const float lr,
                                    const float beta1, const float beta2, const float eps, const float weight_decay,
                                    const uint32_t iter_num)
{
    int32_t in_length = grads.numel();
    if (in_length == 0) {
        LOG_ERROR("dynamic_emb_adamW_fused_hybrid: in_length is zero!");
        return;
    }
    const uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));
    const uint32_t val_dim = static_cast<uint32_t>(values.size(1));
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto grads_continuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto values_continuous = values.is_contiguous() ? values : values.contiguous();
    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continuous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(values_continuous.data_ptr());
    const float base_float = 1.0f;
    const float one_m_beta1 = base_float - beta1;
    const float one_m_beta2 = base_float - beta2;
    const float m_hat_denom = base_float - std::pow(beta1, iter_num);
    const float v_hat_denom = base_float - std::pow(beta2, iter_num);
    const float step_size = lr / m_hat_denom;
    const float inv_v_hat_denom = base_float / v_hat_denom;
    const float decay_factor = base_float - lr * weight_decay;

    auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    auto val_type = scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));
    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    const int32_t num_rows = in_length / static_cast<int32_t>(grad_dim);

    if (!IsSupportedOptimizerSimdDtype(grad_type, val_type)) {
        LOG_ERROR("dynamic_emb_adamW_fused_hybrid: unsupported grad/weight dtype for AdamW update.");
        return;
    }

    if (!LaunchAdamWFusedSimd(stream, grads_ptr, values_ptr, grads_continuous, grad_dim, val_dim, num_rows, max_cores,
                              beta1, beta2, one_m_beta1, one_m_beta2, step_size, inv_v_hat_denom, decay_factor, eps,
                              grad_type, val_type)) {
        LOG_ERROR("dynamic_emb_adamW_fused_hybrid: LaunchAdamWFusedSimd failed!");
    }
}

void dynamic_emb_adagrad_fused(const torch::Tensor& grads, const torch::Tensor& values, const float lr, const float eps)
{
    int32_t inLength = static_cast<int32_t>(grads.numel());
    if (inLength == 0) {
        LOG_ERROR("inLength is zero!");
        return;
    }
    uint32_t gradDim = static_cast<uint32_t>(grads.size(1));
    uint32_t valDim = static_cast<uint32_t>(values.size(1));
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto gradsContinuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto valuesContinuous = values.is_contiguous() ? values : values.contiguous();
    uint8_t* gradsPtr = static_cast<uint8_t*>(gradsContinuous.data_ptr());
    uint8_t* valuesPtr = static_cast<uint8_t*>(valuesContinuous.data_ptr());

    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    auto valType = scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));
    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();

    const int32_t numRows = inLength / static_cast<int32_t>(gradDim);
    if (ShouldUseAdagradSimd(gradDim, inLength, gradType, valType)) {
        if (LaunchAdagradFusedSimd(stream, gradsPtr, valuesPtr, gradsContinuous, gradDim, valDim, numRows, maxCores, lr,
                                   eps, gradType, valType)) {
            return;
        }
    }

    const float beta1 = 0.0f;
    const float beta2 = 0.0f;
    const float oneMinusBeta1 = 0.0f;
    const float oneMinusBeta2 = 0.0f;
    const float stepSize = lr;
    const float invVHatDenom = 1.0f;
    const float decayFactor = 1.0f;

    bool isSmall = (inLength <= SMALL_DATA_THRESHOLD);
    constexpr uint32_t optimizerKind = static_cast<uint32_t>(OptimizerKind::AdaGrad);

    if (!LaunchUpdateFusedKernelCommon(stream, gradsPtr, valuesPtr, gradDim, valDim, inLength, maxCores, isSmall, beta1,
                                       beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                                       gradType, valType, optimizerKind)) {
        return;
    }
}

void dynamic_emb_rowwise_adagrad_fused(const torch::Tensor& grads, const torch::Tensor& values, const float lr,
                                       const float eps)
{
    int32_t inLength = static_cast<int32_t>(grads.numel());
    if (inLength <= 0) {
        LOG_ERROR("dynamic_emb_rowwise_adagrad_fused: grads.numel() must be positive.");
        return;
    }
    uint32_t gradDim = static_cast<uint32_t>(grads.size(1));
    uint32_t valDim = static_cast<uint32_t>(values.size(1));
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto gradsContinuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto valuesContinuous = values.is_contiguous() ? values : values.contiguous();
    uint8_t* gradsPtr = static_cast<uint8_t*>(gradsContinuous.data_ptr());
    uint8_t* valuesPtr = static_cast<uint8_t*>(valuesContinuous.data_ptr());
    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    auto valType = scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));
    if (gradDim == 0U || valDim <= gradDim) {
        LOG_ERROR("dynamic_emb_rowwise_adagrad_fused: grads.size(1) must be positive and values.size(1) must be "
                  "greater than grads.size(1).");
        return;
    }
    const int32_t numRows = inLength / static_cast<int32_t>(gradDim);
    const int32_t coreNum = std::max<int32_t>(1, std::min<int32_t>(maxCores, numRows));

    if (ShouldUseRowwiseAdagradSimd(gradDim, gradType, valType)) {
        if (LaunchRowwiseAdagradFusedSimd(stream, gradsPtr, valuesPtr, gradsContinuous, gradDim, valDim, numRows,
                                          maxCores, lr, eps, gradType, valType)) {
            return;
        }
    }

    // 偶数 embedding 维且 float32 时走 float2 算子
    if (gradDim % 2 == 0 && gradType == DataType::Float32 && valType == DataType::Float32) {
        ACLRT_LAUNCH_KERNEL(rowwise_adagrad_fused_float2)
        (coreNum, stream, gradsPtr, valuesPtr, gradDim, valDim, inLength, lr, eps);
        return;
    }

    ACLRT_LAUNCH_KERNEL(rowwise_adagrad_fused)
    (coreNum, stream, gradsPtr, valuesPtr, gradDim, valDim, inLength, lr, eps, static_cast<uint32_t>(gradType),
     static_cast<uint32_t>(valType));
}

void dynamic_emb_rowwise_adagrad_fused_hybrid(const torch::Tensor& grads, const torch::Tensor& values, const float lr,
                                              const float eps)
{
    int32_t inLength = static_cast<int32_t>(grads.numel());
    if (inLength <= 0) {
        LOG_ERROR("dynamic_emb_rowwise_adagrad_fused_hybrid: grads.numel() must be positive.");
        return;
    }
    uint32_t gradDim = static_cast<uint32_t>(grads.size(1));
    uint32_t valDim = static_cast<uint32_t>(values.size(1));
    if (gradDim == 0U || valDim <= gradDim) {
        LOG_ERROR("dynamic_emb_rowwise_adagrad_fused_hybrid: grads.size(1) must be positive and values.size(1) must be "
                  "greater than grads.size(1).");
        return;
    }

    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    auto valType = scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto gradsContinuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto valuesContinuous = values.is_contiguous() ? values : values.contiguous();
    uint8_t* gradsPtr = static_cast<uint8_t*>(gradsContinuous.data_ptr());
    uint8_t* valuesPtr = static_cast<uint8_t*>(valuesContinuous.data_ptr());
    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    const int32_t numRows = inLength / static_cast<int32_t>(gradDim);

    if (!LaunchRowwiseAdagradFusedSimd(stream, gradsPtr, valuesPtr, gradsContinuous, gradDim, valDim, numRows, maxCores,
                                       lr, eps, gradType, valType)) {
        LOG_ERROR("dynamic_emb_rowwise_adagrad_fused_hybrid: LaunchRowwiseAdagradFusedSimd failed.");
    }
}

void dynamic_emb_sgd_with_pointer(const torch::Tensor& grads, const torch::Tensor& val_pointers, DataType val_type,
                                  const float lr)
{
    int32_t in_length = grads.numel();
    if (in_length == 0) {
        LOG_ERROR("dynamic_emb_sgd_with_pointer: in_length is zero!");
        return;
    }
    const uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto grads_continuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto val_pointers_continuous = val_pointers.is_contiguous() ? val_pointers : val_pointers.contiguous();

    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continuous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(val_pointers_continuous.data_ptr());

    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    const int32_t num_rows = in_length / static_cast<int32_t>(grad_dim);
    if (ShouldUseOptimizerSimd(grad_dim, in_length, grad_type, val_type)) {
        constexpr uint32_t kRowsPerGroup = 2U;
        if (LaunchSgdSimd(stream, grads_ptr, values_ptr, nullptr, grads_continuous, grad_dim, num_rows, max_cores, lr,
                          kRowsPerGroup, grad_type, val_type)) {
            return;
        } else {
            LOG_ERROR("dynamic_emb_sgd_with_pointer: LaunchSgdSimd failed!");
            return;
        }
    }

    if (!IsSupportedOptimizerSimdDtype(grad_type, val_type)) {
        LOG_ERROR("dynamic_emb_sgd_with_pointer: unsupported grad/weight dtype for SGD update.");
        return;
    }

    bool is_small = (in_length <= SMALL_DATA_THRESHOLD);
    constexpr uint32_t optimizer_kind = static_cast<uint32_t>(OptimizerKind::SGD);
    if (!LaunchUpdateKernelCommon(stream, grads_ptr, values_ptr, nullptr, grad_dim, in_length, max_cores, is_small,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, lr, 0.0f, grad_type, val_type, optimizer_kind)) {
        LOG_ERROR("dynamic_emb_sgd_with_pointer: LaunchUpdateKernelCommon failed!");
        return;
    }
}

void dynamic_emb_sgd_with_pointer_hybrid(const torch::Tensor& grads, const torch::Tensor& val_pointers,
                                         DataType val_type, const float lr)
{
    int32_t in_length = grads.numel();
    if (in_length == 0) {
        LOG_ERROR("dynamic_emb_sgd_with_pointer_hybrid: in_length is zero!");
        return;
    }
    const uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto grads_continuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto val_pointers_continuous = val_pointers.is_contiguous() ? val_pointers : val_pointers.contiguous();

    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continuous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(val_pointers_continuous.data_ptr());

    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    const int32_t num_rows = in_length / static_cast<int32_t>(grad_dim);

    if (!IsSupportedOptimizerSimdDtype(grad_type, val_type)) {
        LOG_ERROR("dynamic_emb_sgd_with_pointer_hybrid: unsupported grad/weight dtype for SGD update.");
        return;
    }

    constexpr uint32_t kRowsPerGroup = 2U;
    if (!LaunchSgdSimd(stream, grads_ptr, values_ptr, nullptr, grads_continuous, grad_dim, num_rows, max_cores, lr,
                       kRowsPerGroup, grad_type, val_type)) {
        LOG_ERROR("dynamic_emb_sgd_with_pointer_hybrid: LaunchSgdSimd failed!");
    }
}

void dynamic_emb_sgd_with_table(std::shared_ptr<dyn_emb::DynamicVariableBase> ht, const uint64_t n,
                                const torch::Tensor& indices, const torch::Tensor& grads, const float lr,
                                DataType weight_type)
{
    int32_t in_length = grads.numel();
    if (n == 0 || in_length == 0) {
        LOG_ERROR("n or in_length is zero!");
        return;
    }
    at::Tensor founds =
        at::empty({static_cast<int64_t>(n)}, at::TensorOptions().dtype(at::kBool).device(indices.device()));
    at::Tensor vector_ptrs =
        at::empty({static_cast<int64_t>(n)}, at::TensorOptions().dtype(at::kLong).device(indices.device()));

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    find_pointers(ht, n, indices, vector_ptrs, founds);

    const uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));
    auto grads_continuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto vector_pointers_continuous = vector_ptrs.is_contiguous() ? vector_ptrs : vector_ptrs.contiguous();
    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continuous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(vector_pointers_continuous.data_ptr());
    auto founds_continuous = founds.contiguous();
    uint8_t* founds_ptr = static_cast<uint8_t*>(founds_continuous.data_ptr());

    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    const int32_t num_rows = in_length / static_cast<int32_t>(grad_dim);
    if (ShouldUseOptimizerSimd(grad_dim, in_length, grad_type, weight_type) || !ht->is_pure_hbm_mode()) {
        constexpr uint32_t kRowsPerGroup = 1U;
        if (LaunchSgdSimd(stream, grads_ptr, values_ptr, founds_ptr, grads_continuous, grad_dim, num_rows, max_cores,
                          lr, kRowsPerGroup, grad_type, weight_type)) {
            return;
        } else {
            LOG_ERROR("dynamic_emb_sgd_with_table: LaunchSgdSimd failed!");
            return;
        }
    }

    if (!IsSupportedOptimizerSimdDtype(grad_type, weight_type)) {
        LOG_ERROR("dynamic_emb_sgd_with_table: unsupported grad/weight dtype for SGD update.");
        return;
    }

    bool is_small = (in_length <= SMALL_DATA_THRESHOLD);
    constexpr uint32_t optimizer_kind = static_cast<uint32_t>(OptimizerKind::SGD);
    if (!LaunchUpdateKernelCommon(stream, grads_ptr, values_ptr, founds_ptr, grad_dim, in_length, max_cores, is_small,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, lr, 0.0f, grad_type, weight_type,
                                  optimizer_kind)) {
        LOG_ERROR("dynamic_emb_sgd_with_table: LaunchUpdateKernelCommon failed!");
        return;
    }
}

void dynamic_emb_sgd_fused(const torch::Tensor& grads, const torch::Tensor& values, const float lr)
{
    int32_t in_length = grads.numel();
    if (in_length == 0) {
        LOG_ERROR("dynamic_emb_sgd_fused: in_length is zero!");
        return;
    }
    const uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));
    const uint32_t val_dim = static_cast<uint32_t>(values.size(1));
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto grads_continuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto values_continuous = values.is_contiguous() ? values : values.contiguous();
    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continuous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(values_continuous.data_ptr());

    auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    auto val_type = scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));
    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    const int32_t num_rows = in_length / static_cast<int32_t>(grad_dim);
    if (ShouldUseOptimizerSimd(grad_dim, in_length, grad_type, val_type)) {
        if (LaunchSgdFusedSimd(stream, grads_ptr, values_ptr, grads_continuous, grad_dim, val_dim, num_rows, max_cores,
                               lr, grad_type, val_type)) {
            return;
        } else {
            LOG_ERROR("dynamic_emb_sgd_fused: LaunchSgdFusedSimd failed!");
            return;
        }
    }

    if (!IsSupportedOptimizerSimdDtype(grad_type, val_type)) {
        LOG_ERROR("dynamic_emb_sgd_fused: unsupported grad/weight dtype for SGD update.");
        return;
    }

    bool is_small = (in_length <= SMALL_DATA_THRESHOLD);
    constexpr uint32_t optimizer_kind = static_cast<uint32_t>(OptimizerKind::SGD);
    if (!LaunchUpdateFusedKernelCommon(stream, grads_ptr, values_ptr, grad_dim, val_dim, in_length, max_cores, is_small,
                                       0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, lr, 0.0f, grad_type, val_type,
                                       optimizer_kind)) {
        LOG_ERROR("dynamic_emb_sgd_fused: LaunchUpdateFusedKernelCommon failed!");
        return;
    }
}

void dynamic_emb_sgd_fused_hybrid(const torch::Tensor& grads, const torch::Tensor& values, const float lr)
{
    int32_t in_length = grads.numel();
    if (in_length == 0) {
        LOG_ERROR("dynamic_emb_sgd_fused_hybrid: in_length is zero!");
        return;
    }
    const uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));
    const uint32_t val_dim = static_cast<uint32_t>(values.size(1));
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto grads_continuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto values_continuous = values.is_contiguous() ? values : values.contiguous();
    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continuous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(values_continuous.data_ptr());

    auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    auto val_type = scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));
    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    const int32_t num_rows = in_length / static_cast<int32_t>(grad_dim);

    if (!IsSupportedOptimizerSimdDtype(grad_type, val_type)) {
        LOG_ERROR("dynamic_emb_sgd_fused_hybrid: unsupported grad/weight dtype for SGD update.");
        return;
    }

    if (!LaunchSgdFusedSimd(stream, grads_ptr, values_ptr, grads_continuous, grad_dim, val_dim, num_rows, max_cores, lr,
                            grad_type, val_type)) {
        LOG_ERROR("dynamic_emb_sgd_fused_hybrid: LaunchSgdFusedSimd failed!");
    }
}

static int32_t count_active_slots(const at::Tensor& biased_offsets)
{
    at::Tensor offsets_cpu = biased_offsets.contiguous().cpu();
    const int64_t num_slots = offsets_cpu.size(0) - 1;
    if (num_slots <= 0) {
        return 0;
    }
    at::Tensor lengths = offsets_cpu.slice(0, 1) - offsets_cpu.slice(0, 0, -1);
    return static_cast<int32_t>(lengths.gt(0).sum().item<int64_t>());
}

static bool should_fallback_to_lookup_backward_key_path(int64_t num_key, int32_t num_slots,
                                                        const at::Tensor& biased_offsets)
{
    if (num_key > LOOKUP_BACKWARD_V2_KEYS_PER_ACTIVE_FACTOR * static_cast<int64_t>(num_slots)) {
        return true;
    }
    if (num_key <= static_cast<int64_t>(num_slots)) {
        return false;
    }
    const int32_t num_active_slots = count_active_slots(biased_offsets);
    if (num_active_slots <= 0 || num_slots <= 0) {
        return false;
    }
    const float empty_ratio = static_cast<float>(num_slots - num_active_slots) / static_cast<float>(num_slots);
    return empty_ratio < LOOKUP_BACKWARD_V2_EMPTY_RATIO_THRESHOLD &&
           num_key > LOOKUP_BACKWARD_V2_KEYS_PER_ACTIVE_FACTOR * static_cast<int64_t>(num_active_slots);
}

static void lookup_backward_key_path(const at::Tensor grad, const at::Tensor unique_buffer,
                                     const at::Tensor unique_indices, const at::Tensor inverse_indices,
                                     const at::Tensor biased_offsets, int32_t dim, int32_t table_num,
                                     int32_t batch_size, int32_t feature_num, int32_t num_key, int32_t combiner)
{
    (void)unique_indices;
    (void)table_num;
    (void)batch_size;
    (void)feature_num;

    const int64_t num_pooling_outputs = biased_offsets.size(0) - 1;

    auto grad_contin = grad.is_contiguous() ? grad : grad.contiguous();
    auto unique_buffer_contin = unique_buffer.is_contiguous() ? unique_buffer : unique_buffer.contiguous();
    auto biased_offsets_contin = biased_offsets.is_contiguous() ? biased_offsets : biased_offsets.contiguous();
    auto inverse_indices_contin = inverse_indices.is_contiguous() ? inverse_indices : inverse_indices.contiguous();
    auto value_type = scalartype_to_datatype(convertTypeMetaToScalarType(grad.dtype()));
    auto index_type = scalartype_to_datatype(convertTypeMetaToScalarType(inverse_indices.dtype()));
    void* grad_ptr = grad_contin.data_ptr();
    void* unique_buffer_ptr = unique_buffer_contin.data_ptr();
    void* inverse_indices_ptr = inverse_indices_contin.data_ptr();
    void* biased_offsets_ptr = biased_offsets_contin.data_ptr();

    const int32_t small_data_threshold =
        (inverse_indices.dtype() == torch::kInt32) ? SMALL_DATA_THRESHOLD_32 : SMALL_DATA_THRESHOLD;
    const bool is_float2 =
        (grad.dtype() == torch::kFloat32 && dim % 2 == 0 && dim > LOOKUP_BACKWARD_FLOAT2_DIM_THRESHOLD);
    const int32_t launch_dim = is_float2 ? (dim >> 1) : dim;
    const int64_t launch_elements = static_cast<int64_t>(num_key) * static_cast<int64_t>(launch_dim);

    const bool is_small = (launch_elements <= small_data_threshold);
    const int32_t total_blocks = static_cast<int32_t>((launch_elements + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK);

    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    int32_t core_num = std::min(max_cores, total_blocks);
    if (core_num == 0) {
        LOG_ERROR("core_num is zero!");
        return;
    }
    int32_t blocks_per_core = total_blocks / core_num;
    int32_t remainder_blocks = total_blocks % core_num;
    auto stream = c10_npu::getCurrentNPUStream().stream(true);

    ACLRT_LAUNCH_KERNEL(lookup_backward)
    (core_num, stream, grad_ptr, unique_buffer_ptr, inverse_indices_ptr, biased_offsets_ptr, launch_dim, num_key,
     static_cast<int32_t>(num_pooling_outputs), combiner, total_blocks, blocks_per_core, remainder_blocks,
     static_cast<uint32_t>(index_type), is_small, is_float2, static_cast<uint32_t>(value_type));
}

void lookup_backward_v2(const at::Tensor grad, const at::Tensor unique_buffer, const at::Tensor unique_indices,
                        const at::Tensor inverse_indices, const at::Tensor biased_offsets, int32_t dim,
                        int32_t table_num, int32_t batch_size, int32_t feature_num, int32_t num_key, int32_t combiner)
{
    (void)unique_indices;
    (void)table_num;
    (void)batch_size;
    (void)feature_num;
    TORCH_CHECK(biased_offsets.dtype() == inverse_indices.dtype(),
                "lookup_backward_v2: biased_offsets and inverse_indices must have the same type ");
    TORCH_CHECK(unique_indices.dtype() == inverse_indices.dtype(),
                "lookup_backward_v2: unique_indices must match inverse_indices dtype");
    TORCH_CHECK(inverse_indices.dtype() == torch::kInt32 || inverse_indices.dtype() == torch::kInt64 ||
                    inverse_indices.dtype() == torch::kUInt64,
                "lookup_backward_v2: index tensors must be int32, int64, or uint64");
    TORCH_CHECK(grad.scalar_type() == unique_buffer.scalar_type(),
                "lookup_backward_v2: grad and unique_buffer must have the same dtype");
    TORCH_CHECK(grad.dtype() == torch::kFloat32 || grad.dtype() == torch::kFloat16 || grad.dtype() == torch::kBFloat16,
                "lookup_backward_v2: grad and unique_buffer must be float32, float16, or bfloat16");
    TORCH_CHECK(combiner == 0 || combiner == 1, "lookup_backward_v2: combiner must be 0 (SUM) or 1 (MEAN)");
    TORCH_CHECK(biased_offsets.dim() == 1, "lookup_backward_v2: biased_offsets must be 1D");
    TORCH_CHECK(inverse_indices.dim() == 1, "lookup_backward_v2: inverse_indices must be 1D");
    TORCH_CHECK(grad.dim() == 2, "lookup_backward_v2: grad must be 2D");
    TORCH_CHECK(grad.size(1) == static_cast<int64_t>(dim), "lookup_backward_v2: grad cols must equal dim");

    const int64_t num_pooling_outputs = biased_offsets.size(0) - 1;
    TORCH_CHECK(num_pooling_outputs > 0, "lookup_backward_v2: biased_offsets must contain at least one sample");
    TORCH_CHECK(grad.size(0) == num_pooling_outputs,
                "lookup_backward_v2: grad rows must equal len(biased_offsets) - 1, got grad_rows=", grad.size(0),
                ", num_pooling_outputs=", num_pooling_outputs);
    TORCH_CHECK(inverse_indices.size(0) == static_cast<int64_t>(num_key),
                "lookup_backward_v2: inverse_indices length must equal num_key");

    if (num_key == 0 || dim == 0) {
        return;
    }

    const int32_t num_slots = static_cast<int32_t>(num_pooling_outputs);
    if (should_fallback_to_lookup_backward_key_path(num_key, num_slots, biased_offsets)) {
        lookup_backward_key_path(grad, unique_buffer, unique_indices, inverse_indices, biased_offsets, dim, table_num,
                                 batch_size, feature_num, num_key, combiner);
        return;
    }

    auto grad_contin = grad.is_contiguous() ? grad : grad.contiguous();
    auto unique_buffer_contin = unique_buffer.is_contiguous() ? unique_buffer : unique_buffer.contiguous();
    auto biased_offsets_contin = biased_offsets.is_contiguous() ? biased_offsets : biased_offsets.contiguous();
    auto inverse_indices_contin = inverse_indices.is_contiguous() ? inverse_indices : inverse_indices.contiguous();
    auto value_type = scalartype_to_datatype(convertTypeMetaToScalarType(grad.dtype()));
    auto index_type = scalartype_to_datatype(convertTypeMetaToScalarType(inverse_indices.dtype()));
    void* grad_ptr = grad_contin.data_ptr();
    void* unique_buffer_ptr = unique_buffer_contin.data_ptr();
    void* inverse_indices_ptr = inverse_indices_contin.data_ptr();
    void* biased_offsets_ptr = biased_offsets_contin.data_ptr();

    const int32_t small_data_threshold =
        (inverse_indices.dtype() == torch::kInt32) ? SMALL_DATA_THRESHOLD_32 : SMALL_DATA_THRESHOLD;
    const bool is_float2 =
        (grad.dtype() == torch::kFloat32 && dim % 2 == 0 && dim > LOOKUP_BACKWARD_FLOAT2_DIM_THRESHOLD);
    const int32_t launch_dim = is_float2 ? (dim >> 1) : dim;
    const int64_t launch_elements = static_cast<int64_t>(num_slots) * static_cast<int64_t>(launch_dim);

    const bool is_small = (launch_elements <= small_data_threshold);
    const int32_t total_blocks = static_cast<int32_t>((launch_elements + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK);

    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    int32_t core_num = std::min(max_cores, total_blocks);
    if (core_num == 0) {
        LOG_ERROR("core_num is zero!");
        return;
    }
    int32_t blocks_per_core = total_blocks / core_num;
    int32_t remainder_blocks = total_blocks % core_num;
    auto stream = c10_npu::getCurrentNPUStream().stream(true);

    lookup_backward_v2_launch(grad_ptr, unique_buffer_ptr, inverse_indices_ptr, biased_offsets_ptr, launch_dim,
                              num_slots, combiner, total_blocks, blocks_per_core, remainder_blocks,
                              static_cast<uint32_t>(index_type), is_small, is_float2, static_cast<uint32_t>(value_type),
                              core_num, stream);
}

void dynamic_emb_adagrad_with_pointer_hybrid(const torch::Tensor& grads, const torch::Tensor& valPointers,
                                             DataType valType, int64_t stateDim, const float lr, const float eps)
{
    int32_t inLength = static_cast<int32_t>(grads.numel());
    if (inLength == 0) {
        LOG_ERROR("dynamic_emb_adagrad_with_pointer_hybrid: grads.numel is zero!");
        return;
    }
    const uint32_t gradDim = static_cast<uint32_t>(grads.size(1));
    if (stateDim != static_cast<int64_t>(gradDim)) {
        LOG_ERROR("dynamic_emb_adagrad_with_pointer_hybrid: stateDim must equal embedding dim (sum of squared grads "
                  "state per dim), "
                  << "stateDim=" << stateDim << ", gradDim=" << gradDim);
        return;
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto gradsContinuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto valPointersContinuous = valPointers.is_contiguous() ? valPointers : valPointers.contiguous();

    uint8_t* gradsPtr = static_cast<uint8_t*>(gradsContinuous.data_ptr());
    uint8_t* valuesPtr = static_cast<uint8_t*>(valPointersContinuous.data_ptr());

    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    const int32_t numRows = inLength / static_cast<int32_t>(gradDim);
    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    const uint32_t rowsPerGroup = AdagradSimdRowsPerGroup(gradType, valType);
    LaunchAdagradSimd(stream, gradsPtr, valuesPtr, nullptr, gradsContinuous, gradDim, numRows, maxCores, lr, eps,
                      rowsPerGroup, gradType, valType);
}

void dynamic_emb_adagrad_fused_hybrid(const torch::Tensor& grads, const torch::Tensor& values, const float lr,
                                      const float eps)
{
    int32_t inLength = static_cast<int32_t>(grads.numel());
    if (inLength == 0) {
        LOG_ERROR("dynamic_emb_adagrad_fused_hybrid: grads.numel is zero!");
        return;
    }
    uint32_t gradDim = static_cast<uint32_t>(grads.size(1));
    uint32_t valDim = static_cast<uint32_t>(values.size(1));
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto gradsContinuous = grads.is_contiguous() ? grads : grads.contiguous();
    auto valuesContinuous = values.is_contiguous() ? values : values.contiguous();
    uint8_t* gradsPtr = static_cast<uint8_t*>(gradsContinuous.data_ptr());
    uint8_t* valuesPtr = static_cast<uint8_t*>(valuesContinuous.data_ptr());

    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    auto valType = scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));

    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    const int32_t numRows = inLength / static_cast<int32_t>(gradDim);
    LaunchAdagradFusedSimd(stream, gradsPtr, valuesPtr, gradsContinuous, gradDim, valDim, numRows, maxCores, lr, eps,
                           gradType, valType);
}

static int64_t ReadOffsetBaseScalar(const at::Tensor& offsetContinuous)
{
    const at::Tensor elem = offsetContinuous.index({0}).cpu().contiguous();
    if (elem.scalar_type() == at::kUInt64) {
        return static_cast<int64_t>(*elem.data_ptr<uint64_t>());
    }
    return *elem.data_ptr<int64_t>();
}

static uint32_t PoolingSimdElemBytes(DataType dataType)
{
    if (dataType == DataType::Float16 || dataType == DataType::BFloat16) {
        return 2U;
    }
    return 4U;
}

static uint64_t PoolingSimdUbBytes(int32_t evSize, DataType srcType, DataType dstType)
{
    // 32字节对齐，用于SIMD UB的计算
    const uint32_t evSizeAligned = ((static_cast<uint32_t>(evSize) + 7U) / 8U) * 8U;
    uint64_t ubNeed = 2ULL * static_cast<uint64_t>(evSizeAligned) * sizeof(float);
    // 如果srcType或dstType不是Float32，则需要额外分配UB空间做float的类型转换
    if (srcType != DataType::Float32 || dstType != DataType::Float32) {
        const uint32_t stagingElemBytes = std::max(PoolingSimdElemBytes(srcType), PoolingSimdElemBytes(dstType));
        ubNeed += static_cast<uint64_t>(evSizeAligned) * static_cast<uint64_t>(stagingElemBytes);
    }
    return ubNeed;
}

static bool LaunchPoolingEmbeddingsSimdTiling(const at::Tensor& refTensor, int32_t combiner, int32_t totalDims,
                                              int32_t accumDims, int32_t evSize, int32_t numVec, int32_t batchSize,
                                              int32_t srcNumRows, int32_t inverseLen, int64_t offsetBase,
                                              uint32_t offsetType, uint32_t srcType, uint32_t dstType, int32_t maxCores,
                                              int32_t& coreNumOut, at::Tensor& tilingNpuOut)
{
    coreNumOut = std::max<int32_t>(1, std::min<int32_t>(maxCores, numVec));

    PoolingEmbeddingsSimdTilingData tilingData{};
    tilingData.combiner = combiner;
    tilingData.totalDims = totalDims;
    tilingData.accumDims = accumDims;
    tilingData.evSize = evSize;
    tilingData.numVec = numVec;
    tilingData.batchSize = batchSize;
    tilingData.srcNumRows = srcNumRows;
    tilingData.inverseLen = inverseLen;
    tilingData.needCoreNum = coreNumOut;
    tilingData.offsetBase = offsetBase;
    tilingData.offsetType = offsetType;
    tilingData.srcType = srcType;
    tilingData.dstType = dstType;

    at::Tensor tilingHost = at::empty({static_cast<int64_t>(sizeof(PoolingEmbeddingsSimdTilingData))},
                                      at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    if (memcpy_s(tilingHost.data_ptr(), tilingHost.nbytes(), &tilingData, sizeof(tilingData)) != EOK) {
        LOG_ERROR("LaunchPoolingEmbeddingsSimdTiling: memcpy_s tiling data failed. evSize="
                  << evSize << ", numVec=" << numVec << ", needCoreNum=" << coreNumOut);
        return false;
    }
    tilingNpuOut = tilingHost.to(refTensor.device()).contiguous();
    return true;
}

static bool LaunchPoolingEmbeddingsSimd(aclrtStream stream, uint8_t* srcData, uint8_t* dstData, uint8_t* offsetData,
                                        uint8_t* inverseData, const at::Tensor& refTensor, int32_t combiner,
                                        int32_t totalDims, int32_t accumDims, int32_t evSize, int32_t numVec,
                                        int32_t batchSize, int32_t srcNumRows, int32_t inverseLen, int64_t offsetBase,
                                        uint32_t offsetType, uint32_t srcType, uint32_t dstType, int32_t maxCores)
{
    int32_t coreNum = 0;
    at::Tensor tilingNpu;
    if (!LaunchPoolingEmbeddingsSimdTiling(refTensor, combiner, totalDims, accumDims, evSize, numVec, batchSize,
                                           srcNumRows, inverseLen, offsetBase, offsetType, srcType, dstType, maxCores,
                                           coreNum, tilingNpu)) {
        LOG_ERROR("LaunchPoolingEmbeddingsSimd: LaunchPoolingEmbeddingsSimdTiling failed. evSize="
                  << evSize << ", numVec=" << numVec << ", batchSize=" << batchSize << ", maxCores=" << maxCores);
        return false;
    }
    ACLRT_LAUNCH_KERNEL(pooling_embeddings_simd)
    (coreNum, stream, srcData, dstData, offsetData, inverseData, static_cast<uint8_t*>(tilingNpu.data_ptr()));
    return true;
}

void lookup_forward_hybrid(const at::Tensor& src, const at::Tensor& dst, const at::Tensor& offset,
                           const at::Tensor& inverse, int32_t combiner, int32_t totalDims, int32_t accumDims,
                           int32_t evSize, int32_t numVec, int32_t batchSize)
{
    TORCH_CHECK(offset.dtype() == inverse.dtype(), "offset and inverse must have the same dtype");
    TORCH_CHECK(offset.dim() == 1 && inverse.dim() == 1,
                "offset and inverse must be 1D tensor. offset.dim()=", offset.dim(), ", inverse.dim()=", inverse.dim());

    auto srcType = scalartype_to_datatype(convertTypeMetaToScalarType(src.dtype()));
    auto dstType = scalartype_to_datatype(convertTypeMetaToScalarType(dst.dtype()));
    auto offsetType = scalartype_to_datatype(convertTypeMetaToScalarType(offset.dtype()));

    const uint64_t ubNeed = PoolingSimdUbBytes(evSize, srcType, dstType);
    const uint64_t ubSize = static_cast<uint64_t>(AclSingleton::GetInstance().GetTotalUbSize());
    TORCH_CHECK(ubNeed <= ubSize, "lookup_forward_hybrid: evSize is too large for SIMD UB. evSize=", evSize,
                ", ubNeed=", ubNeed, ", ubSize=", ubSize);
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto srcContinuous = src.is_contiguous() ? src : src.contiguous();
    auto offsetContinuous = offset.is_contiguous() ? offset : offset.contiguous();
    auto inverseContinuous = inverse.is_contiguous() ? inverse : inverse.contiguous();

    const int64_t offsetBase = ReadOffsetBaseScalar(offsetContinuous);

    uint8_t* srcData = static_cast<uint8_t*>(srcContinuous.data_ptr());
    uint8_t* offsetData = static_cast<uint8_t*>(offsetContinuous.data_ptr());
    uint8_t* inverseData = static_cast<uint8_t*>(inverseContinuous.data_ptr());
    uint8_t* dstData = static_cast<uint8_t*>(dst.data_ptr());

    const int32_t srcNumRows = static_cast<int32_t>(src.size(0));
    const int32_t inverseLen = static_cast<int32_t>(inverse.numel());
    const int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();

    TORCH_CHECK(LaunchPoolingEmbeddingsSimd(stream, srcData, dstData, offsetData, inverseData, srcContinuous, combiner,
                                            totalDims, accumDims, evSize, numVec, batchSize, srcNumRows, inverseLen,
                                            offsetBase, static_cast<uint32_t>(offsetType),
                                            static_cast<uint32_t>(srcType), static_cast<uint32_t>(dstType), maxCores),
                "lookup_forward_hybrid: LaunchPoolingEmbeddingsSimd failed.");
}
// PYTHON WARP
void bind_dyn_emb_op(py::module& m)
{
    py::class_<dyn_emb::InitializerArgs>(m, "InitializerArgs")
        .def(py::init<>())
        // 相等比较
        .def("__eq__",
             [](const dyn_emb::InitializerArgs& a, const dyn_emb::InitializerArgs& b) {
                 return a.mode_ == b.mode_ && a.mean_ == b.mean_ && a.std_dev_ == b.std_dev_ && a.lower_ == b.lower_ &&
                        a.upper_ == b.upper_ && a.value_ == b.value_;
             })
        .def(py::init([](const std::string& mode, float mean, float std_dev, float lower, float upper, float value) {
            return dyn_emb::InitializerArgs(mode, mean, std_dev, lower, upper, value);
        }))
        .def(py::pickle(
            [](const InitializerArgs& p) {  // __getstate__
                return py::make_tuple(p.mode_, p.mean_, p.std_dev_, p.lower_, p.upper_, p.value_);
            },
            [](py::tuple t) {  // __setstate__
                constexpr int num_of_args = 6;
                if (t.size() != num_of_args)
                    throw std::runtime_error("Invalid number args of InitializerArgs!");
                InitializerArgs p(t[0].cast<std::string>(), t[1].cast<float>(), t[2].cast<float>(), t[3].cast<float>(),
                                  t[4].cast<float>(), t[5].cast<float>());
                return p;
            }));

    py::class_<dyn_emb::DynamicVariableBase, std::shared_ptr<dyn_emb::DynamicVariableBase>>(m, "DynamicEmbTable")
        .def(py::init([](dyn_emb::DataType key_type, dyn_emb::DataType value_type, dyn_emb::EvictStrategy evict_type,
                         int64_t dim = 128, int64_t init_capaity = 1024, int64_t max_capaity = 2048,
                         size_t max_hbm_for_vectors = 0, size_t max_bucket_size = 128, float max_load_factor = 0.5,
                         int block_size = 128, int io_block_size = 1024, int device_id = -1, bool io_by_cpu = false,
                         bool use_constant_memory = false, int reserved_key_start_bit = 0,
                         size_t num_of_buckets_per_alloc = 1,
                         const dyn_emb::InitializerArgs& initializer_args = dyn_emb::InitializerArgs(),
                         const int safe_check_mode = static_cast<int>(SafeCheckMode::IGNORE),
                         const int optimizer_type = static_cast<int>(OptimizerType::Null)) {
            int64_t pow2init_capaity = next_power_of_two(init_capaity);
            int64_t pow2max_capaity = next_power_of_two(max_capaity);
            auto table = dyn_emb::VariableFactory::Create(
                key_type, value_type, evict_type, dim, pow2init_capaity, pow2max_capaity, max_hbm_for_vectors,
                max_bucket_size, max_load_factor, block_size, io_block_size, device_id, io_by_cpu, use_constant_memory,
                reserved_key_start_bit, num_of_buckets_per_alloc, initializer_args,
                static_cast<SafeCheckMode>(safe_check_mode), static_cast<OptimizerType>(optimizer_type));
            return table;
        }))
        .def("get_key_type", &dyn_emb::DynamicVariableBase::get_key_type, "Get Dynamic Emb Table key type")
        .def("get_value_type", &dyn_emb::DynamicVariableBase::get_value_type, "Get Dynamic Emb Table value type")
        .def("get_evict_strategy", &dyn_emb::DynamicVariableBase::get_evict_strategy,
             "Get evict strategy of Dynamic Emb Table.")
        .def("get_max_capacity", &dyn_emb::DynamicVariableBase::get_max_capacity,
             "Get max capacity of Dynamic Emb Table.")
        .def("get_initializer_args", &dyn_emb::DynamicVariableBase::get_initializer_args, "Get initializer arguments.")
        .def("optstate_dim", &dyn_emb::DynamicVariableBase::optstate_dim, "Get dim of all optimizer states.")
        .def("set_initial_optstate", &dyn_emb::DynamicVariableBase::set_initial_optstate,
             "Set initial optimizer state value.", py::arg("value"))
        .def("get_initial_optstate", &dyn_emb::DynamicVariableBase::get_initial_optstate,
             "Get initial optimizer state value.")
        .def("get_emb_cols", &dyn_emb::DynamicVariableBase::get_emb_cols, "Get the number of columns in the table.")
        .def("load", &dyn_emb::DynamicVariableBase::load, "Load a key-value pair in the table", py::arg("n"),
             py::arg("keys"), py::arg("values"), py::arg("score") = c10::nullopt, py::arg("unique_key") = true,
             py::arg("ignore_evict_strategy") = false)
        .def("update", &dyn_emb::DynamicVariableBase::update, "Update a key-value pair in the table", py::arg("n"),
             py::arg("keys"), py::arg("values"), py::arg("score") = c10::nullopt, py::arg("unique_key") = true,
             py::arg("ignore_evict_strategy") = false)
        .def("export_batch", &dyn_emb::DynamicVariableBase::export_batch, "export key value from table", py::arg("n"),
             py::arg("offset"), py::arg("d_counter"), py::arg("keys"), py::arg("values"),
             py::arg("score") = c10::nullopt)
        .def("evict_strategy", &dyn_emb::DynamicVariableBase::evict_strategy,
             "Get evict strategy of Dynamic Emb Table.");

    py::enum_<dyn_emb::DataType>(m, "DynamicEmbDataType")
        .value("Float32", dyn_emb::DataType::Float32)
        .value("BFloat16", dyn_emb::DataType::BFloat16)
        .value("Float16", dyn_emb::DataType::Float16)
        .value("Int64", dyn_emb::DataType::Int64)
        .value("UInt64", dyn_emb::DataType::UInt64)
        .value("Int32", dyn_emb::DataType::Int32)
        .value("UInt32", dyn_emb::DataType::UInt32)
        .value("Size_t", dyn_emb::DataType::Size_t)
        .export_values();

    py::enum_<dyn_emb::EvictStrategy>(m, "EvictStrategy")
        .value("kLru", dyn_emb::EvictStrategy::kLru)
        .value("kLfu", dyn_emb::EvictStrategy::kLfu)
        .value("kEpochLru", dyn_emb::EvictStrategy::kEpochLru)
        .value("kEpochLfu", dyn_emb::EvictStrategy::kEpochLfu)
        .value("kCustomized", dyn_emb::EvictStrategy::kCustomized)
        .export_values();

    py::enum_<dyn_emb::SafeCheckMode>(m, "SafeCheckMode")
        .value("ERROR", dyn_emb::SafeCheckMode::ERROR)
        .value("WARNING", dyn_emb::SafeCheckMode::WARNING)
        .value("IGNORE", dyn_emb::SafeCheckMode::IGNORE)
        .export_values();

    py::enum_<dyn_emb::OptimizerType>(m, "OptimizerType")
        .value("Null", dyn_emb::OptimizerType::Null)
        .value("SGD", dyn_emb::OptimizerType::SGD)
        .value("Adam", dyn_emb::OptimizerType::Adam)
        .value("AdamW", dyn_emb::OptimizerType::AdamW)
        .value("AdaGrad", dyn_emb::OptimizerType::AdaGrad)
        .value("RowWiseAdaGrad", dyn_emb::OptimizerType::RowWiseAdaGrad)
        .export_values();

    m.def("get_table_range_op", &get_table_range_npu, "Calculate table range based on offsets and feature offsets",
          py::arg("offsets"), py::arg("featureOffsets"));
    m.def("select", &select_npu, "Select items in inputs where flags are true.", py::arg("flags"), py::arg("inputs"),
          py::arg("outputs"), py::arg("num_selected"));
    m.def("select_index", &select_index_npu, "Select indices where flags are true.", py::arg("flags"),
          py::arg("output_indices"), py::arg("num_selected"));
    m.def("unique_op", &unique_op_npu, "NPU-accelerated unique operation for int64 tensor", py::arg("key"));
    m.def("segmented_unique_op", &segmented_unique_op_npu,
          "NPU-accelerated segmented unique operation for int64 tensor", py::arg("keys"), py::arg("segmentRange"));

    m.def("block_bucketize_sparse_features", &block_bucketsize_sparse_features_npu,
          "NPU-accelerated bucketize operation for sparse features", py::arg("lengths"), py::arg("indices"),
          py::arg("bucketizePos"), py::arg("sequence"), py::arg("distTypePerFeature"), py::arg("blockSizes"),
          py::arg("mySize"), py::arg("weights") = c10::nullopt);

    m.def("get_new_length_and_offsets_op", &get_new_length_and_offsets_npu,
          "NPU-accelerated calculation of new offsets and lengths", py::arg("d_unique_offsets"),
          py::arg("d_table_offsets_in_feature"), py::arg("new_offsets"), py::arg("new_lengths"),
          py::arg("local_batch_size"));

    m.def("gather_embedding", &gather_embedding, "gather_embedding", py::arg("inputs"), py::arg("indices"));

    m.def("find_pointers", &find_pointers,
          "Find a key-value pair in the table , and return every "
          "value's ptr",
          py::arg("table"), py::arg("n"), py::arg("keys"), py::arg("values"), py::arg("founds"),
          py::arg("score") = py::none());

    m.def("find_pointers_with_scores", &find_pointers_with_scores,
          "Find a key-value pair in the table , and return every "
          "value's ptr",
          py::arg("table"), py::arg("n"), py::arg("keys"), py::arg("values"), py::arg("founds"),
          py::arg("scores") = py::none());

    m.def("find_and_initialize", &find_and_initialize, "Find and initialize embeddings with given initializer",
          py::arg("table"), py::arg("n"), py::arg("keys"), py::arg("value_ptrs"), py::arg("values"), py::arg("founds"),
          py::arg("initializer_args") = py::none());

    m.def("dyn_emb_rows", &dyn_emb_rows, "Get the number of rows in the table", py::arg("table"));

    m.def("dyn_emb_cols", &dyn_emb_cols, "Get the number of columns in the table", py::arg("table"));

    m.def("dyn_emb_is_pure_hbm_mode", &dyn_emb_is_pure_hbm_mode,
          "Check whether all value storage currently resides in HBM", py::arg("table"));

    m.def("export_batch_matched", &export_batch_matched,
          "Export KV-pairs within [offset, offset + n) whose score > threshold", py::arg("table"), py::arg("threshold"),
          py::arg("n"), py::arg("offset"), py::arg("num_matched"), py::arg("keys"), py::arg("values"));

    m.def("count_matched", &count_matched, "Count the KV-pairs whose score > threshold in the whole table.",
          py::arg("table"), py::arg("threshold"), py::arg("num_matched"));

    m.def("insert_and_evict", &insert_and_evict, "Insert keys and values, evicting if necessary", py::arg("table"),
          py::arg("n"), py::arg("keys"), py::arg("values"), py::arg("score"), py::arg("evicted_keys"),
          py::arg("evicted_values"), py::arg("evicted_score"), py::arg("d_evicted_counter"),
          py::arg("unique_key") = true, py::arg("ignore_evict_strategy") = false);

    m.def("find", &find, "Find values in the table based on keys", py::arg("table"), py::arg("n"), py::arg("keys"),
          py::arg("values"), py::arg("founds"), py::arg("score") = c10::nullopt);

    m.def("erase", &erase, "Erase values from the table based on keys", py::arg("table"), py::arg("n"),
          py::arg("keys"));

    m.def("clear", &clear, "Clear all keys in the table", py::arg("table"));

    m.def("reserve", &reserve, "reserve hash table capacity", py::arg("table"), py::arg("new_capacity"));

    m.def("accum_or_assign", &accum_or_assign, "Accumulate or assign values to the table", py::arg("table"),
          py::arg("n"), py::arg("keys"), py::arg("value_or_deltas"), py::arg("accum_or_assigns"),
          py::arg("score") = c10::nullopt, py::arg("ignore_evict_strategy") = false);

    m.def("find_or_insert", &find_or_insert, "Find or insert a key-value pair in the table", py::arg("table"),
          py::arg("n"), py::arg("keys"), py::arg("values"), py::arg("score") = py::none(), py::arg("unique_key") = true,
          py::arg("ignore_evict_strategy") = false);

    m.def("find_or_insert_pointers", &find_or_insert_pointers,
          "Find or insert a key-value pair in the table , and return every "
          "value's ptr",
          py::arg("table"), py::arg("n"), py::arg("keys"), py::arg("values"), py::arg("founds"),
          py::arg("score") = py::none(), py::arg("unique_key") = true, py::arg("ignore_evict_strategy") = false);

    m.def("assign", &assign, "Assign values to the table based on keys", py::arg("table"), py::arg("n"),
          py::arg("keys"), py::arg("values"), py::arg("score") = c10::nullopt, py::arg("unique_key") = true);

    m.def("device_timestamp", &device_timestamp, "device_timestamp");

    m.def("load_from_pointer", &load_from_pointer_imp, "load_from_pointer", py::arg("pointers"), py::arg("dst"));

    m.def("load_from_pointer_hybrid", &load_from_pointer_hybrid_imp, "load_from_pointer_hybrid", py::arg("pointers"),
          py::arg("dst"));

    m.def("reduce_grads", &reduce_grads, "reduce grads", py::arg("grad"), py::arg("unique"), py::arg("inverse"));

    m.def("dynamic_emb_Adam_with_pointer", &dyn_emb::dynamic_emb_Adam_with_pointer,
          "Adam optimizer for dynamic embedding", py::arg("grads"), py::arg("val_pointers"), py::arg("val_type"),
          py::arg("state_dim"), py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
          py::arg("weight_decay"), py::arg("iter_num"));

    m.def("dynamic_emb_adamW_with_table", &dyn_emb::dynamic_emb_adamW_with_table,
          "AdamW optimizer for dynamic embedding", py::arg("ht"), py::arg("n"), py::arg("indices"), py::arg("grads"),
          py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"),
          py::arg("iter_num"), py::arg("weight_type"));
    m.def("dynamic_emb_adagrad_with_table", &dyn_emb::dynamic_emb_adagrad_with_table,
          "AdaGrad optimizer for dynamic embedding", py::arg("ht"), py::arg("n"), py::arg("indices"), py::arg("grads"),
          py::arg("lr"), py::arg("eps"), py::arg("weightType"));
    m.def("dynamic_emb_rowwise_adagrad_with_table", &dyn_emb::dynamic_emb_rowwise_adagrad_with_table,
          "RowWise AdaGrad optimizer for dynamic embedding", py::arg("ht"), py::arg("n"), py::arg("indices"),
          py::arg("grads"), py::arg("lr"), py::arg("eps"), py::arg("weightType"));
    m.def("dynamic_emb_sgd_with_table", &dyn_emb::dynamic_emb_sgd_with_table, "SGD optimizer for dynamic embedding",
          py::arg("ht"), py::arg("n"), py::arg("indices"), py::arg("grads"), py::arg("lr"), py::arg("weight_type"));

    m.def("dynamic_emb_adamW_with_pointer", &dyn_emb::dynamic_emb_adamW_with_pointer,
          "AdamW optimizer for dynamic embedding", py::arg("grads"), py::arg("val_pointers"), py::arg("val_type"),
          py::arg("state_dim"), py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
          py::arg("weight_decay"), py::arg("iter_num"));
    m.def("dynamic_emb_adagrad_with_pointer", &dyn_emb::dynamic_emb_adagrad_with_pointer,
          "AdaGrad optimizer for dynamic embedding", py::arg("grads"), py::arg("valPointers"), py::arg("valType"),
          py::arg("stateDim"), py::arg("lr"), py::arg("eps"));
    m.def("dynamic_emb_rowwise_adagrad_with_pointer", &dyn_emb::dynamic_emb_rowwise_adagrad_with_pointer,
          "RowWise AdaGrad optimizer for dynamic embedding", py::arg("grads"), py::arg("valPointers"),
          py::arg("valType"), py::arg("stateDim"), py::arg("lr"), py::arg("eps"));
    m.def("dynamic_emb_rowwise_adagrad_with_pointer_hybrid", &dyn_emb::dynamic_emb_rowwise_adagrad_with_pointer_hybrid,
          "RowWise AdaGrad optimizer for dynamic embedding (SIMD only)", py::arg("grads"), py::arg("valPointers"),
          py::arg("valType"), py::arg("stateDim"), py::arg("lr"), py::arg("eps"));
    m.def("dynamic_emb_sgd_with_pointer", &dyn_emb::dynamic_emb_sgd_with_pointer, "SGD optimizer for dynamic embedding",
          py::arg("grads"), py::arg("val_pointers"), py::arg("val_type"), py::arg("lr"));
    m.def("dynamic_emb_adamW_with_pointer_hybrid", &dyn_emb::dynamic_emb_adamW_with_pointer_hybrid,
          "AdamW optimizer for dynamic embedding", py::arg("grads"), py::arg("val_pointers"), py::arg("val_type"),
          py::arg("state_dim"), py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
          py::arg("weight_decay"), py::arg("iter_num"));
    m.def("dynamic_emb_sgd_with_pointer_hybrid", &dyn_emb::dynamic_emb_sgd_with_pointer_hybrid,
          "SGD optimizer for dynamic embedding", py::arg("grads"), py::arg("val_pointers"), py::arg("val_type"),
          py::arg("lr"));
    m.def("dynamic_emb_adamW_fused", &dyn_emb::dynamic_emb_adamW_fused, "AdamW optimizer for dynamic embedding",
          py::arg("grads"), py::arg("values"), py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
          py::arg("weight_decay"), py::arg("iter_num"));
    m.def("dynamic_emb_adagrad_fused", &dyn_emb::dynamic_emb_adagrad_fused, "AdaGrad optimizer for dynamic embedding",
          py::arg("grads"), py::arg("values"), py::arg("lr"), py::arg("eps"));
    m.def("dynamic_emb_rowwise_adagrad_fused", &dyn_emb::dynamic_emb_rowwise_adagrad_fused,
          "RowWise AdaGrad optimizer for dynamic embedding", py::arg("grads"), py::arg("values"), py::arg("lr"),
          py::arg("eps"));
    m.def("dynamic_emb_rowwise_adagrad_fused_hybrid", &dyn_emb::dynamic_emb_rowwise_adagrad_fused_hybrid,
          "RowWise AdaGrad optimizer for dynamic embedding (SIMD only)", py::arg("grads"), py::arg("values"),
          py::arg("lr"), py::arg("eps"));
    m.def("dynamic_emb_sgd_fused", &dyn_emb::dynamic_emb_sgd_fused, "SGD optimizer for dynamic embedding",
          py::arg("grads"), py::arg("values"), py::arg("lr"));
    m.def("dynamic_emb_sgd_fused_hybrid", &dyn_emb::dynamic_emb_sgd_fused_hybrid, "SGD optimizer for dynamic embedding",
          py::arg("grads"), py::arg("values"), py::arg("lr"));
    m.def("dynamic_emb_adamW_fused_hybrid", &dyn_emb::dynamic_emb_adamW_fused_hybrid,
          "AdamW optimizer for dynamic embedding", py::arg("grads"), py::arg("values"), py::arg("lr"), py::arg("beta1"),
          py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"), py::arg("iter_num"));

    m.def("dedup_input_indices_op", &dedup_input_indices_npu,
          "NPU-accelerated deduplication for input indices (dedup input indices on NPU)", py::arg("indices"),
          py::arg("offsets"), py::arg("d_table_offsets_in_feature"), py::arg("table_num"), py::arg("local_batch_size"),
          py::arg("reverse_idx"), py::arg("d_unique_nums"), py::arg("d_unique_offsets"), py::arg("unique_idx"),
          py::arg("new_offsets"), py::arg("new_lengths"));

    m.def("lookup_forward", &lookup_forward, "lookup_forward", py::arg("src"), py::arg("dst"), py::arg("offset"),
          py::arg("inverse"), py::arg("combiner"), py::arg("total_dims"), py::arg("accum_dims"), py::arg("ev_size"),
          py::arg("num_vec"), py::arg("batch_size"));
    m.def("lookup_backward", &lookup_backward_v2,
          "backward with slot-path kernel; dense multi-key slots fall back to lookup_backward with key-path",
          py::arg("grad"), py::arg("unique_buffer"), py::arg("unique_indices"), py::arg("inverse_indices"),
          py::arg("biased_offsets"), py::arg("dim"), py::arg("tables_num"), py::arg("batch_size"),
          py::arg("num_feature"), py::arg("num_key"), py::arg("combiner"));
    m.def("dynamic_emb_adagrad_fused_hybrid", &dyn_emb::dynamic_emb_adagrad_fused_hybrid,
          "AdaGrad optimizer for dynamic embedding (SIMD only)", py::arg("grads"), py::arg("values"), py::arg("lr"),
          py::arg("eps"));
    m.def("dynamic_emb_adagrad_with_pointer_hybrid", &dyn_emb::dynamic_emb_adagrad_with_pointer_hybrid,
          "AdaGrad optimizer for dynamic embedding (SIMD only)", py::arg("grads"), py::arg("valPointers"),
          py::arg("valType"), py::arg("stateDim"), py::arg("lr"), py::arg("eps"));
    m.def("lookup_forward_hybrid", &lookup_forward_hybrid, "lookup_forward (hybrid vector path)", py::arg("src"),
          py::arg("dst"), py::arg("offset"), py::arg("inverse"), py::arg("combiner"), py::arg("total_dims"),
          py::arg("accum_dims"), py::arg("ev_size"), py::arg("num_vec"), py::arg("batch_size"));

    py::class_<dyn_emb::CurandStateContext>(m, "CurandStateContext").def(py::init<>());

    m.def("normal_init", &dyn_emb::normal_init, "Normal initializer", py::arg("buffer"), py::arg("indices"),
          py::arg("curand_state_context"), py::arg("mean"), py::arg("std_dev"));

    m.def("truncated_normal_init", &dyn_emb::truncated_normal_init, "Truncated normal initializer", py::arg("buffer"),
          py::arg("indices"), py::arg("curand_state_context"), py::arg("mean"), py::arg("std_dev"), py::arg("lower"),
          py::arg("upper"));

    m.def("uniform_init", &dyn_emb::uniform_init, "Uniform initializer", py::arg("buffer"), py::arg("indices"),
          py::arg("curand_state_context"), py::arg("lower"), py::arg("upper"));

    m.def("const_init", &dyn_emb::const_init, "Const initializer", py::arg("buffer"), py::arg("indices"),
          py::arg("value"));

    m.def("debug_init", &dyn_emb::debug_init, "Debug initializer", py::arg("buffer"), py::arg("indices"),
          py::arg("keys"));
}
}  // namespace dyn_emb
