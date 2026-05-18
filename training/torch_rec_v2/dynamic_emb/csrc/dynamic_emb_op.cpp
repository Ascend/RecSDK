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
#include "aclrtlaunch_pooling_embeddings.h"
#include "aclrtlaunch_lookup_backward.h"
#include "dynamic_variable_base.h"
#include "torch_utils.h"
#include "utils.h"
#include "./ops/unique_op/cpu_unique.h"
#define LOG_ERROR(msg) std::cout << "[INFO]" << msg << std::endl
namespace py = pybind11;
namespace dyn_emb {
constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
constexpr int32_t SMALL_DATA_THRESHOLD = 44 * MAX_THREADS_PER_BLOCK;
constexpr int32_t SMALL_DATA_THRESHOLD_32 = 24 * MAX_THREADS_PER_BLOCK;
constexpr int32_t ELEMENTS_PER_BLOCK = MAX_THREADS_PER_BLOCK * MAX_ELEMENTS_PER_THREAD;
constexpr float HASH_TABLE_FACTOR = 1.5f;
constexpr int64_t MIN_HASH_TABLE_CAPACITY = 1;
constexpr int32_t CACHE_ALIGN = 64;
constexpr size_t MIN_CORE_NUM = 1;
constexpr int32_t CPU_KEY_THRESHOLD = 10000;
using ReturnType =
    std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>>;

static bool LaunchUpdateKernelCommon(
    aclrtStream stream,
    uint8_t* gradsPtr,
    uint8_t* valuesPtr,
    uint8_t* foundsPtr,
    uint32_t gradDim,
    int32_t inLength,
    int32_t maxCores,
    bool isSmall,
    float beta1,
    float beta2,
    float oneMinusBeta1,
    float oneMinusBeta2,
    float stepSize,
    float invVHatDenom,
    float decayFactor,
    float eps,
    DataType gradType,
    DataType valType,
    uint32_t optimizerKind)
{
    if (gradDim % 2 == 0 && gradType == DataType::Float32 && valType == DataType::Float32) {
        int32_t vecLength = inLength / 2;
        const int32_t vecPerBlock = ELEMENTS_PER_BLOCK / 2;
        int32_t totalBlocks = (vecLength + vecPerBlock - 1) / vecPerBlock;
        int32_t coreNum = std::min(maxCores, totalBlocks);
        if (coreNum == 0) {
            LOG_ERROR("core_num is zero!");
            return false;
        }
        int32_t blocksPerCore = totalBlocks / coreNum;
        int32_t remainderBlocks = totalBlocks % coreNum;
        ACLRT_LAUNCH_KERNEL(update_float2)(
            coreNum, stream, gradsPtr, valuesPtr, foundsPtr, gradDim, vecLength,
            beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom,
            decayFactor, eps, totalBlocks, blocksPerCore, remainderBlocks, isSmall,
            static_cast<uint32_t>(gradType), static_cast<uint32_t>(valType), optimizerKind);
    } else {
        int32_t totalBlocks = (inLength + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
        int32_t coreNum = std::min(maxCores, totalBlocks);
        if (coreNum == 0) {
            LOG_ERROR("core_num is zero!");
            return false;
        }
        int32_t blocksPerCore = totalBlocks / coreNum;
        int32_t remainderBlocks = totalBlocks % coreNum;
        ACLRT_LAUNCH_KERNEL(update)(
            coreNum, stream, gradsPtr, valuesPtr, foundsPtr, gradDim, inLength,
            beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom,
            decayFactor, eps, totalBlocks, blocksPerCore, remainderBlocks, isSmall,
            static_cast<uint32_t>(gradType), static_cast<uint32_t>(valType), optimizerKind);
    }
    return true;
}

static bool LaunchUpdateFusedKernelCommon(
    aclrtStream stream,
    uint8_t* gradsPtr,
    uint8_t* valuesPtr,
    uint32_t gradDim,
    uint32_t valDim,
    int32_t inLength,
    int32_t maxCores,
    bool isSmall,
    float beta1,
    float beta2,
    float oneMinusBeta1,
    float oneMinusBeta2,
    float stepSize,
    float invVHatDenom,
    float decayFactor,
    float eps,
    DataType gradType,
    DataType valType,
    uint32_t optimizerKind)
{
    if (gradDim % 2 == 0 && gradType == DataType::Float32 && valType == DataType::Float32) {
        int32_t vecLength = inLength / 2;
        const int32_t vecPerBlock = ELEMENTS_PER_BLOCK / 2;
        int32_t totalBlocks = (vecLength + vecPerBlock - 1) / vecPerBlock;
        int32_t coreNum = std::min(maxCores, totalBlocks);
        if (coreNum == 0) {
            LOG_ERROR("core_num is zero!");
            return false;
        }
        int32_t blocksPerCore = totalBlocks / coreNum;
        int32_t remainderBlocks = totalBlocks % coreNum;
        ACLRT_LAUNCH_KERNEL(update_float2_fused)(
            coreNum, stream, gradsPtr, valuesPtr, nullptr, gradDim, valDim, vecLength,
            beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom,
            decayFactor, eps, totalBlocks, blocksPerCore, remainderBlocks, isSmall,
            static_cast<uint32_t>(gradType), static_cast<uint32_t>(valType), optimizerKind);
    } else {
        int32_t totalBlocks = (inLength + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
        int32_t coreNum = std::min(maxCores, totalBlocks);
        if (coreNum == 0) {
            LOG_ERROR("core_num is zero!");
            return false;
        }
        int32_t blocksPerCore = totalBlocks / coreNum;
        int32_t remainderBlocks = totalBlocks % coreNum;
        ACLRT_LAUNCH_KERNEL(update_fused)(
            coreNum, stream, gradsPtr, valuesPtr, nullptr, gradDim, valDim, inLength,
            beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom,
            decayFactor, eps, totalBlocks, blocksPerCore, remainderBlocks, isSmall,
            static_cast<uint32_t>(gradType), static_cast<uint32_t>(valType), optimizerKind);
    }
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
    TORCH_CHECK(indices.element_size() == 8, "indices element size must be 8");
    uint32_t eleSz = inputs.element_size();
    TORCH_CHECK(eleSz == 4 or eleSz == 2, "inputs element size must be 4 or 2");

    uint64_t indicesLen = indices.size(0);
    uint32_t dim = inputs.size(1);

    if (indicesLen == 0) {
        return torch::empty({0, dim}, inputs.options());
    }

    uint64_t outLen = indicesLen * dim;
    constexpr uint32_t GATHER_THRESHOLD = 100000;

    if (outLen > GATHER_THRESHOLD && indices.scalar_type() != at::kUInt64) {
        torch::Tensor indicesExpand = indices.unsqueeze(-1).expand({indicesLen, dim});
        return torch::gather(inputs, 0, indicesExpand);
    }

    auto inType = scalartype_to_datatype(inputs.scalar_type());
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

    ACLRT_LAUNCH_KERNEL(gather_dim0)(coreNum, stream, inData, indicesData, outData,
        dim, outLen, blocksPerCore, remainderBlocks, THREAD_NUM, static_cast<uint32_t>(inType), eleSz);
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
        ACLRT_LAUNCH_KERNEL(load_from_pointer)(coreNum, stream, pData, outData, dim, outLen,
            blocksPerCore, remainderBlocks, THREAD_NUM, 1, static_cast<uint32_t>(outType), eleSz);
    } else {
        ACLRT_LAUNCH_KERNEL(load_from_pointer)(coreNum, stream, pData, outData, dim, outLen,
            blocksPerCore, remainderBlocks, THREAD_NUM, 0, static_cast<uint32_t>(outType), eleSz);
    }

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
    ACLRT_LAUNCH_KERNEL(unique_op)(coreNum, stream, input_key.data_ptr<int64_t>(), unique_key.data_ptr<int64_t>(),
                                   restore_index.data_ptr<int64_t>(), count.data_ptr<int64_t>(),
                                   workspace.data_ptr<uint8_t>(), hashTableCapacity, static_cast<int32_t>(key_numel),
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
    TORCH_CHECK(offsets.dtype() == torch::kInt32 || offsets.dtype() == torch::kInt64,
                "get_table_range_op only supports int32/int64 dtype");
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
    int32_t isInt32 = (inputOffsets.dtype() == torch::kInt32) ? 1 : 0;

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
    ACLRT_LAUNCH_KERNEL(get_table_range_op)(coreNum, stream, inputOffsets.data_ptr(), inputFeatureOffsets.data_ptr(),
                                            tableRange.data_ptr(), featureNumXBatch, tableNum, isInt32);

    return tableRange;
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
    ACLRT_LAUNCH_KERNEL(get_new_length_and_offsets_op)(
        coreNum, stream, inputUniqueOffsets.data_ptr<int64_t>(), inputTableOffsets.data_ptr<int64_t>(),
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

    ACLRT_LAUNCH_KERNEL(reduce_grad_op)(coreNum, stream, cGrad.data_ptr(), cInverse.data_ptr(), cGrad.size(0),
        cGrad.size(1), baseBlk, remainBlk, cUniqueGrad.data_ptr());
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
            // hkv要求score的tensor类型为uint64
            bc_scores.to(at::kUInt64);
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

int64_t dyn_emb_rows(std::shared_ptr<dyn_emb::DynamicVariableBase> table)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    return table->rows(stream);
}
  
void count_matched(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
                   const uint64_t threshold,
                   at::Tensor num_matched)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    table->count_matched(threshold, num_matched, stream);
}

void export_batch_matched(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
                          const uint64_t threshold, const uint64_t n,
                          const uint64_t offset, at::Tensor num_matched,
                          at::Tensor keys, at::Tensor values)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    table->export_batch_matched(threshold, n, offset, num_matched, keys, values, c10::nullopt, stream);
}

void insert_and_evict(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
                      const size_t n, const at::Tensor keys, const at::Tensor values,
                      const std::optional<uint64_t> score, at::Tensor evicted_keys,
                      at::Tensor evicted_values, at::Tensor evicted_score,
                      at::Tensor d_evicted_counter, bool unique_key = true,
                      bool ignore_evict_strategy = false)
{
    if (not score and (table->evict_strategy() == EvictStrategy::kCustomized ||
        table->evict_strategy() == EvictStrategy::kLfu)) {
        throw std::invalid_argument("Must specify the score when evict strategy is customized or LFU.");
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    if (table->evict_strategy() == EvictStrategy::kCustomized ||
        table->evict_strategy() == EvictStrategy::kLfu) {
        auto&& option = at::TensorOptions().dtype(torch::kInt64).device(keys.device());
        // broadcast scores
        at::Tensor bc_scores = at::empty({static_cast<int64_t>(n)}, option);
        // fill_接口不支持uint64_t
        bc_scores.fill_(static_cast<int64_t>(score.value()));
        // hkv要求score的tensor类型为uint64
        bc_scores.to(at::kUInt64);
        table->insert_and_evict(n, keys.data_ptr(), values.data_ptr(), bc_scores.data_ptr(),
            evicted_keys.data_ptr(), evicted_values.data_ptr(), evicted_score.data_ptr(),
            reinterpret_cast<uint64_t*>(d_evicted_counter.data_ptr()), stream, unique_key, ignore_evict_strategy);
    } else {
        table->insert_and_evict(n, keys.data_ptr(), values.data_ptr(), nullptr, 
            evicted_keys.data_ptr(), evicted_values.data_ptr(), evicted_score.data_ptr(),
            reinterpret_cast<uint64_t*>(d_evicted_counter.data_ptr()), stream, unique_key, ignore_evict_strategy);
    }
}

void find(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n,
          const at::Tensor keys, const at::Tensor values, const at::Tensor founds,
          const c10::optional<at::Tensor> &score = c10::nullopt)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    if (score.has_value()) {
        at::Tensor score_ = score.value();
        table->find(n, keys.data_ptr(), values.data_ptr(), founds.data_ptr<bool>(),
                    score_.data_ptr(), stream);
    } else {
        table->find(n, keys.data_ptr(), values.data_ptr(), founds.data_ptr<bool>(),
                    nullptr, stream);
    }
}

void erase(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n,
           const at::Tensor keys)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    table->erase(n, keys.data_ptr(), stream);
}

void clear(std::shared_ptr<dyn_emb::DynamicVariableBase> table)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    table->clear(stream);
}

void reserve(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
            const size_t new_capacity)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    table->reserve(new_capacity, stream);
}

void accum_or_assign(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
                     const size_t n, const at::Tensor keys,
                     const at::Tensor value_or_deltas,
                     const at::Tensor accum_or_assigns,
                     const c10::optional<at::Tensor> &score = c10::nullopt,
                     bool ignore_evict_strategy = false)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    if (score.has_value()) {
        at::Tensor score_ = score.value();
        table->accum_or_assign(n, keys.data_ptr(), value_or_deltas.data_ptr(),
                               accum_or_assigns.data_ptr<bool>(), score_.data_ptr(),
                               stream, ignore_evict_strategy);
    } else {
        table->accum_or_assign(n, keys.data_ptr(), value_or_deltas.data_ptr(),
                               accum_or_assigns.data_ptr<bool>(), nullptr, stream,
                               ignore_evict_strategy);
    }
}

void assign(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n,
            const at::Tensor keys, const at::Tensor values,
            const c10::optional<at::Tensor> &score = c10::nullopt,
            bool unique_key = true)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    if (score.has_value()) {
        at::Tensor score_ = score.value();
        table->assign(n, keys.data_ptr(), values.data_ptr(), score_.data_ptr(),
                      stream, unique_key);
    } else {
        table->assign(n, keys.data_ptr(), values.data_ptr(), nullptr, stream,
                      unique_key);
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

    bool isFloat2 = (evSize % 2 == 0 && evSize > EMBEDDING_THRESHOLD);
    int32_t evSizeVec = evSize;
    if (isFloat2) {
        evSizeVec >>= 1;
    }
    int32_t outLen = evSizeVec * numVec;
    int32_t totalBlocks = (outLen + THREAD_NUM - 1) / THREAD_NUM;
    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    int32_t coreNum = std::min(maxCores, totalBlocks);
    TORCH_CHECK(coreNum > 0, "coreNum must be greater than 0");
    int32_t blocksPerCore = totalBlocks / coreNum;
    int32_t remainderBlocks = totalBlocks % coreNum;
    bool isSmall = (maxCores >= totalBlocks) ? true : false;

    ACLRT_LAUNCH_KERNEL(pooling_embeddings)(
        coreNum, aclStream, srcData, dstData, offsetData, inverseData, combiner, totalDims,
        accumDims, evSize, numVec, batchSize, totalBlocks, blocksPerCore, remainderBlocks, isSmall,
        static_cast<uint32_t>(srcType), static_cast<uint32_t>(dstType), static_cast<uint32_t>(offsetType),
        THREAD_NUM, outLen, isFloat2, evSizeVec);
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

        ACLRT_LAUNCH_KERNEL(Adam_update_float2)(core_num, stream, grads_ptr, values_ptr, grad_dim, vec_length,
                        beta1, beta2, oneMinusBeta1, oneMinusBeta2,
                        step_size, inv_vHatDenom, weight_decay, eps,
                        total_blocks, blocks_per_core, remainder_blocks, is_small);
    } else { // === 分支 2：奇数维度，使用标量 float 算子 ===
        int32_t total_blocks = (in_length + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
        int32_t core_num = std::min(max_cores, total_blocks);
        if (core_num == 0) {
            return;
        }

        int32_t blocks_per_core = total_blocks / core_num;
        int32_t remainder_blocks = total_blocks % core_num;

        ACLRT_LAUNCH_KERNEL(Adam_update)(core_num, stream, grads_ptr, values_ptr, grad_dim, in_length,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2,
                step_size, inv_vHatDenom, weight_decay, eps,
                total_blocks, blocks_per_core, remainder_blocks, is_small);
    }
}

void dynamic_emb_adamW_with_pointer(const torch::Tensor& grads, const torch::Tensor& val_pointers, DataType val_type,
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
    
    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(val_pointers_continous.data_ptr());
    float base_float = 1.0f;
    // 将重复计算的常量提取到 Host 侧仅计算一次
    float oneMinusBeta1 = base_float - beta1;
    float oneMinusBeta2 = base_float - beta2;
    float mHatDenom = base_float - std::pow(beta1, iter_num);
    float vHatDenom = base_float - std::pow(beta2, iter_num);

    float step_size = lr / mHatDenom;
    float inv_vHatDenom = base_float / vHatDenom;
    float decay_factor = base_float - lr * weight_decay;

    // 获取最大核数
    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    bool is_small = (in_length <= SMALL_DATA_THRESHOLD);
    auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    constexpr uint32_t optimizer_kind = static_cast<uint32_t>(OptimizerKind::AdamW);
    // === 分支 1：偶数维度，使用 float2 算子,且 grad_type 和 val_type 为 float32===
    if (grad_dim % 2 == 0 && grad_type == DataType::Float32 && val_type == DataType::Float32) {
        int32_t vec_length = in_length / 2;
        const int32_t VEC_PER_BLOCK = ELEMENTS_PER_BLOCK / 2;

        int32_t total_blocks = (vec_length + VEC_PER_BLOCK - 1) / VEC_PER_BLOCK;
        int32_t core_num = std::min(max_cores, total_blocks);
        if (core_num == 0) {
            LOG_ERROR("core_num is zero!");
            return;
        }

        int32_t blocks_per_core = total_blocks / core_num;
        int32_t remainder_blocks = total_blocks % core_num;

        ACLRT_LAUNCH_KERNEL(update_float2)(core_num, stream, grads_ptr, values_ptr, nullptr, grad_dim, vec_length,
            beta1, beta2, oneMinusBeta1, oneMinusBeta2, step_size, inv_vHatDenom, 
            decay_factor, eps, total_blocks, blocks_per_core, remainder_blocks, is_small, 
            static_cast<uint32_t>(grad_type), static_cast<uint32_t>(val_type), optimizer_kind);

    } else { // === 分支 2：奇数维度，使用标量 float 算子 ===
        int32_t total_blocks = (in_length + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
        int32_t core_num = std::min(max_cores, total_blocks);
        if (core_num == 0) {
            LOG_ERROR("core_num is zero!");
            return;
        }

        int32_t blocks_per_core = total_blocks / core_num;
        int32_t remainder_blocks = total_blocks % core_num;

        ACLRT_LAUNCH_KERNEL(update)(core_num, stream, grads_ptr, values_ptr, nullptr, grad_dim, in_length,
            beta1, beta2, oneMinusBeta1, oneMinusBeta2, step_size, inv_vHatDenom, 
            decay_factor, eps, total_blocks, blocks_per_core, remainder_blocks, is_small, 
            static_cast<uint32_t>(grad_type), static_cast<uint32_t>(val_type), optimizer_kind);
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
        LOG_ERROR("dynamic_emb_adagrad_with_pointer: stateDim must equal embedding dim (sum of squared grads state per dim).");
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
    bool isSmall = (inLength <= SMALL_DATA_THRESHOLD);
    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    constexpr uint32_t optimizerKind = static_cast<uint32_t>(OptimizerKind::AdaGrad);

    if (!LaunchUpdateKernelCommon(stream, gradsPtr, valuesPtr, nullptr, gradDim, inLength, maxCores, isSmall, beta1, beta2,
        oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, gradType, valType, optimizerKind)) {
        return;
    }
}

void dynamic_emb_rowwise_adagrad_with_pointer(const torch::Tensor& grads, const torch::Tensor& valPointers, DataType valType,
    int64_t stateDim, const float lr, const float eps)
{
    int32_t inLength = static_cast<int32_t>(grads.numel());
    if (inLength == 0) {
        LOG_ERROR("dynamic_emb_rowwise_adagrad_with_pointer: inLength is zero!");
        return;
    }
    const uint32_t gradDim = static_cast<uint32_t>(grads.size(1));
    if (stateDim <= 0) {
        LOG_ERROR("dynamic_emb_rowwise_adagrad_with_pointer: stateDim must be positive.");
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
    bool isSmall = (inLength <= SMALL_DATA_THRESHOLD);
    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    constexpr uint32_t optimizerKind = static_cast<uint32_t>(OptimizerKind::RowWiseAdaGrad);

    if (!LaunchUpdateKernelCommon(stream, gradsPtr, valuesPtr, nullptr, gradDim, inLength, maxCores, isSmall, beta1, beta2,
        oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, gradType, valType, optimizerKind)) {
        return;
    }
}

void dynamic_emb_adamW_with_table( std::shared_ptr<dyn_emb::DynamicVariableBase> ht,
    const uint64_t n, const torch::Tensor& indices, const torch::Tensor& grads, 
    const float lr, const float beta1, const float beta2,
    const float eps, const float weight_decay, const uint32_t iter_num, DataType weight_type)
{
    int32_t in_length = grads.numel();
    if (n == 0 || in_length == 0) {
        LOG_ERROR("n or in_length is zero!");
        return;
    }
    at::Tensor founds = at::empty({static_cast<int64_t>(n)}, 
                                  at::TensorOptions().dtype(at::kBool).device(indices.device()));
    at::Tensor vector_ptrs = at::empty({static_cast<int64_t>(n)}, 
                                    at::TensorOptions().dtype(at::kLong).device(indices.device()));

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    find_pointers(ht, n, indices, vector_ptrs, founds);

    uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));
    auto grads_continous = grads.contiguous();
    auto vector_pointers_continous = vector_ptrs.contiguous();
    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(vector_pointers_continous.data_ptr());
    auto founds_continous = founds.contiguous();
    uint8_t* founds_ptr = static_cast<uint8_t*>(founds_continous.data_ptr());
    float base_float = 1.0f;
    // 将重复计算的常量提取到 Host 侧仅计算一次
    float one_m_beta1 = base_float - beta1;
    float one_m_beta2 = base_float - beta2;
    float m_hat_denom = base_float - std::pow(beta1, iter_num);
    float v_hat_denom = base_float - std::pow(beta2, iter_num);
    float step_size = lr / m_hat_denom;
    float inv_v_hat_denom = base_float / v_hat_denom;
    float decay_factor = base_float - lr * weight_decay;

    // 获取最大核数
    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();

    bool is_small = (in_length <= SMALL_DATA_THRESHOLD);
    auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    constexpr uint32_t optimizer_kind = static_cast<uint32_t>(OptimizerKind::AdamW);
    // === 分支 1：偶数维度，使用 float2 算子,且 grad_type 和 weight_type 为 float32===
    if (grad_dim % 2 == 0 && grad_type == DataType::Float32 && weight_type == DataType::Float32) {
        int32_t vec_length = in_length / 2;
        const int32_t VEC_PER_BLOCK = ELEMENTS_PER_BLOCK / 2;

        int32_t total_blocks = (vec_length + VEC_PER_BLOCK - 1) / VEC_PER_BLOCK;
        int32_t core_num = std::min(max_cores, total_blocks);
        if (core_num == 0) {
            LOG_ERROR("core_num is zero!");
            return;
        }
        int32_t blocks_per_core = total_blocks / core_num;
        int32_t remainder_blocks = total_blocks % core_num;
        ACLRT_LAUNCH_KERNEL(update_float2)(core_num, stream, grads_ptr, values_ptr, founds_ptr, grad_dim, vec_length,
            beta1, beta2, one_m_beta1, one_m_beta2, step_size, inv_v_hat_denom, 
            decay_factor, eps, total_blocks, blocks_per_core, remainder_blocks, is_small, 
            static_cast<uint32_t>(grad_type), static_cast<uint32_t>(weight_type), optimizer_kind);
    } else { // === 分支 2：奇数维度，使用标量 float 算子 ===
        int32_t total_blocks = (in_length + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
        int32_t core_num = std::min(max_cores, total_blocks);
        if (core_num == 0) {
            LOG_ERROR("core_num is zero!");
            return;
        }
        int32_t blocks_per_core = total_blocks / core_num;
        int32_t remainder_blocks = total_blocks % core_num;
        ACLRT_LAUNCH_KERNEL(update)(core_num, stream, grads_ptr, values_ptr, founds_ptr, grad_dim, in_length,
            beta1, beta2, one_m_beta1, one_m_beta2, step_size, inv_v_hat_denom, 
            decay_factor, eps, total_blocks, blocks_per_core, remainder_blocks, is_small, 
            static_cast<uint32_t>(grad_type), static_cast<uint32_t>(weight_type), optimizer_kind);
    }
}

void dynamic_emb_adagrad_with_table(std::shared_ptr<dyn_emb::DynamicVariableBase> ht,
    const uint64_t n, const torch::Tensor& indices, const torch::Tensor& grads,
    const float lr, const float eps, DataType weightType)
{
    int32_t inLength = static_cast<int32_t>(grads.numel());
    if (n == 0 || inLength == 0) {
        LOG_ERROR("n or inLength is zero!");
        return;
    }

    at::Tensor founds = at::empty({static_cast<int64_t>(n)},
                                  at::TensorOptions().dtype(at::kBool).device(indices.device()));
    at::Tensor vectorPtrs = at::empty({static_cast<int64_t>(n)},
                                      at::TensorOptions().dtype(at::kLong).device(indices.device()));

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
    bool isSmall = (inLength <= SMALL_DATA_THRESHOLD);
    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    constexpr uint32_t optimizerKind = static_cast<uint32_t>(OptimizerKind::AdaGrad);

    if (!LaunchUpdateKernelCommon(stream, gradsPtr, valuesPtr, foundsPtr, gradDim, inLength, maxCores, isSmall, beta1, beta2,
        oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, gradType, weightType, optimizerKind)) {
        return;
    }
}

void dynamic_emb_rowwise_adagrad_with_table(std::shared_ptr<dyn_emb::DynamicVariableBase> ht,
    const uint64_t n, const torch::Tensor& indices, const torch::Tensor& grads,
    const float lr, const float eps, DataType weightType)
{
    int32_t inLength = static_cast<int32_t>(grads.numel());
    if (n == 0 || inLength == 0) {
        LOG_ERROR("n or inLength is zero!");
        return;
    }

    at::Tensor founds = at::empty({static_cast<int64_t>(n)},
                                  at::TensorOptions().dtype(at::kBool).device(indices.device()));
    at::Tensor vectorPtrs = at::empty({static_cast<int64_t>(n)},
                                      at::TensorOptions().dtype(at::kLong).device(indices.device()));

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
    bool isSmall = (inLength <= SMALL_DATA_THRESHOLD);
    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    constexpr uint32_t optimizerKind = static_cast<uint32_t>(OptimizerKind::RowWiseAdaGrad);

    if (!LaunchUpdateKernelCommon(stream, gradsPtr, valuesPtr, foundsPtr, gradDim, inLength, maxCores, isSmall, beta1, beta2,
        oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, gradType, weightType, optimizerKind)) {
        return;
    }
}

void dynamic_emb_adamW_fused(const torch::Tensor& grads, const torch::Tensor& values, const float lr, const float beta1, const float beta2,
                                    const float eps, const float weight_decay, const uint32_t iter_num)
{
    int32_t in_length = grads.numel();
    if (in_length == 0) {
        LOG_ERROR("in_length is zero!");
        return;
    }
    uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));
    uint32_t val_dim = static_cast<uint32_t>(values.size(1));
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto grads_continous = grads.contiguous();
    auto values_continous = values.contiguous();
    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(values_continous.data_ptr());
    float base_float = 1.0f;
    // 将重复计算的常量提取到 Host 侧仅计算一次
    float one_m_beta1 = base_float - beta1;
    float one_m_beta2 = base_float - beta2;
    float m_hat_denom = base_float - std::pow(beta1, iter_num);
    float v_hat_denom = base_float - std::pow(beta2, iter_num);

    float step_size = lr / m_hat_denom;
    float inv_v_hat_denom = base_float / v_hat_denom;
    float decay_factor = base_float - lr * weight_decay;

    // 获取最大核数
    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    bool is_small = (in_length <= SMALL_DATA_THRESHOLD);
    auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    auto val_type = scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));
    constexpr uint32_t optimizer_kind = static_cast<uint32_t>(OptimizerKind::AdamW);
    // === 分支 1：偶数维度，使用 float2 算子,且 grad_type 和 val_type 为 float32===
    if (grad_dim % 2 == 0 && grad_type == DataType::Float32 && val_type == DataType::Float32) {
        int32_t vec_length = in_length / 2;
        const int32_t VEC_PER_BLOCK = ELEMENTS_PER_BLOCK / 2;
        int32_t total_blocks = (vec_length + VEC_PER_BLOCK - 1) / VEC_PER_BLOCK;
        int32_t core_num = std::min(max_cores, total_blocks);
        if (core_num == 0) {
            LOG_ERROR("core_num is zero!");
            return;
        }
        int32_t blocks_per_core = total_blocks / core_num;
        int32_t remainder_blocks = total_blocks % core_num;
        ACLRT_LAUNCH_KERNEL(update_float2_fused)(core_num, stream, grads_ptr, values_ptr, nullptr, grad_dim, val_dim, vec_length,
            beta1, beta2, one_m_beta1, one_m_beta2, step_size, inv_v_hat_denom, decay_factor, eps, total_blocks, blocks_per_core, 
            remainder_blocks, is_small, static_cast<uint32_t>(grad_type), static_cast<uint32_t>(val_type), optimizer_kind);
    } else { // === 分支 2：奇数维度，使用标量 float 算子 ===
        int32_t total_blocks = (in_length + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
        int32_t core_num = std::min(max_cores, total_blocks);
        if (core_num == 0) {
            LOG_ERROR("core_num is zero!");
            return;
        }
        int32_t blocks_per_core = total_blocks / core_num;
        int32_t remainder_blocks = total_blocks % core_num;
        ACLRT_LAUNCH_KERNEL(update_fused)(core_num, stream, grads_ptr, values_ptr, nullptr, grad_dim, val_dim, in_length,
            beta1, beta2, one_m_beta1, one_m_beta2, step_size, inv_v_hat_denom, decay_factor, eps, total_blocks,
            blocks_per_core, remainder_blocks, is_small, static_cast<uint32_t>(grad_type), 
            static_cast<uint32_t>(val_type), optimizer_kind);
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

    const float beta1 = 0.0f;
    const float beta2 = 0.0f;
    const float oneMinusBeta1 = 0.0f;
    const float oneMinusBeta2 = 0.0f;
    const float stepSize = lr;
    const float invVHatDenom = 1.0f;
    const float decayFactor = 1.0f;

    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    bool isSmall = (inLength <= SMALL_DATA_THRESHOLD);
    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    auto valType = scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));
    constexpr uint32_t optimizerKind = static_cast<uint32_t>(OptimizerKind::AdaGrad);

    if (!LaunchUpdateFusedKernelCommon(stream, gradsPtr, valuesPtr, gradDim, valDim, inLength, maxCores, isSmall, beta1, beta2,
        oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, gradType, valType, optimizerKind)) {
        return;
    }
}

void dynamic_emb_rowwise_adagrad_fused(const torch::Tensor& grads, const torch::Tensor& values, const float lr, const float eps)
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

    const float beta1 = 0.0f;
    const float beta2 = 0.0f;
    const float oneMinusBeta1 = 0.0f;
    const float oneMinusBeta2 = 0.0f;
    const float stepSize = lr;
    const float invVHatDenom = 1.0f;
    const float decayFactor = 1.0f;

    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    bool isSmall = (inLength <= SMALL_DATA_THRESHOLD);
    auto gradType = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    auto valType = scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));
    constexpr uint32_t optimizerKind = static_cast<uint32_t>(OptimizerKind::RowWiseAdaGrad);

    if (!LaunchUpdateFusedKernelCommon(stream, gradsPtr, valuesPtr, gradDim, valDim, inLength, maxCores, isSmall, beta1, beta2,
        oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, gradType, valType, optimizerKind)) {
        return;
    }
}

void dynamic_emb_sgd_with_pointer(const torch::Tensor& grads, const torch::Tensor& val_pointers, DataType val_type,
                                  const float lr)
{
    int32_t in_length = grads.numel();
    if (in_length == 0) {
        LOG_ERROR("in_length is zero!");
        return;
    }
    uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto grads_continous = grads.contiguous();
    auto val_pointers_continous = val_pointers.contiguous();

    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(val_pointers_continous.data_ptr());

    float decay_factor = lr;

    // 获取最大核数
    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    bool is_small = (in_length <= SMALL_DATA_THRESHOLD);
    auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    constexpr uint32_t optimizer_kind = static_cast<uint32_t>(OptimizerKind::SGD);
    // === 分支 1：偶数维度，使用 float2 算子,且 grad_type 和 val_type 为 float32===
    if (grad_dim % 2 == 0 && grad_type == DataType::Float32 && val_type == DataType::Float32) {
        int32_t vec_length = in_length / 2;
        const int32_t VEC_PER_BLOCK = ELEMENTS_PER_BLOCK / 2;

        int32_t total_blocks = (vec_length + VEC_PER_BLOCK - 1) / VEC_PER_BLOCK;
        int32_t core_num = std::min(max_cores, total_blocks);
        if (core_num == 0) {
            LOG_ERROR("core_num is zero!");
            return;
        }

        int32_t blocks_per_core = total_blocks / core_num;
        int32_t remainder_blocks = total_blocks % core_num;

        ACLRT_LAUNCH_KERNEL(update_float2)(core_num, stream, grads_ptr, values_ptr, nullptr, grad_dim, vec_length, 0, 0,
                                           0, 0, 0, 0, decay_factor, 0, total_blocks, blocks_per_core, remainder_blocks,
                                           is_small, static_cast<uint32_t>(grad_type), static_cast<uint32_t>(val_type),
                                           optimizer_kind);

    } else {  // === 分支 2：奇数维度，使用标量 float 算子 ===
        int32_t total_blocks = (in_length + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
        int32_t core_num = std::min(max_cores, total_blocks);
        if (core_num == 0) {
            LOG_ERROR("core_num is zero!");
            return;
        }

        int32_t blocks_per_core = total_blocks / core_num;
        int32_t remainder_blocks = total_blocks % core_num;

        ACLRT_LAUNCH_KERNEL(update)(core_num, stream, grads_ptr, values_ptr, nullptr, grad_dim, in_length, 0, 0, 0, 0,
                                    0, 0, decay_factor, 0, total_blocks, blocks_per_core, remainder_blocks, is_small,
                                    static_cast<uint32_t>(grad_type), static_cast<uint32_t>(val_type), optimizer_kind);
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
    at::Tensor founds = at::empty({static_cast<int64_t>(n)},
                                  at::TensorOptions().dtype(at::kBool).device(indices.device()));
    at::Tensor vector_ptrs = at::empty({static_cast<int64_t>(n)},
                                    at::TensorOptions().dtype(at::kLong).device(indices.device()));

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    find_pointers(ht, n, indices, vector_ptrs, founds);

    uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));
    auto grads_continous = grads.contiguous();
    auto vector_pointers_continous = vector_ptrs.contiguous();
    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(vector_pointers_continous.data_ptr());
    auto founds_continous = founds.contiguous();
    uint8_t* founds_ptr = static_cast<uint8_t*>(founds_continous.data_ptr());

    float decay_factor = lr;

    // 获取最大核数
    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();

    bool is_small = (in_length <= SMALL_DATA_THRESHOLD);
    auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    constexpr uint32_t optimizer_kind = static_cast<uint32_t>(OptimizerKind::SGD);
    // === 分支 1：偶数维度，使用 float2 算子,且 grad_type 和 weight_type 为 float32===
    if (grad_dim % 2 == 0 && grad_type == DataType::Float32 && weight_type == DataType::Float32) {
        int32_t vec_length = in_length / 2;
        const int32_t VEC_PER_BLOCK = ELEMENTS_PER_BLOCK / 2;

        int32_t total_blocks = (vec_length + VEC_PER_BLOCK - 1) / VEC_PER_BLOCK;
        int32_t core_num = std::min(max_cores, total_blocks);
        if (core_num == 0) {
            LOG_ERROR("core_num is zero!");
            return;
        }
        int32_t blocks_per_core = total_blocks / core_num;
        int32_t remainder_blocks = total_blocks % core_num;
        ACLRT_LAUNCH_KERNEL(update_float2)(core_num, stream, grads_ptr, values_ptr, founds_ptr, grad_dim, vec_length, 0,
                                           0, 0, 0, 0, 0, decay_factor, 0, total_blocks, blocks_per_core,
                                           remainder_blocks, is_small, static_cast<uint32_t>(grad_type),
                                           static_cast<uint32_t>(weight_type), optimizer_kind);
    } else { // === 分支 2：奇数维度，使用标量 float 算子 ===
        int32_t total_blocks = (in_length + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
        int32_t core_num = std::min(max_cores, total_blocks);
        if (core_num == 0) {
            LOG_ERROR("core_num is zero!");
            return;
        }
        int32_t blocks_per_core = total_blocks / core_num;
        int32_t remainder_blocks = total_blocks % core_num;
        ACLRT_LAUNCH_KERNEL(update)(core_num, stream, grads_ptr, values_ptr, founds_ptr, grad_dim, in_length, 0, 0, 0,
                                    0, 0, 0, decay_factor, 0, total_blocks, blocks_per_core, remainder_blocks, is_small,
                                    static_cast<uint32_t>(grad_type), static_cast<uint32_t>(weight_type),
                                    optimizer_kind);
    }
}

void dynamic_emb_sgd_fused(const torch::Tensor& grads, const torch::Tensor& values, const float lr)
{
    int32_t in_length = grads.numel();
    if (in_length == 0) {
        LOG_ERROR("in_length is zero!");
        return;
    }
    uint32_t grad_dim = static_cast<uint32_t>(grads.size(1));
    uint32_t val_dim = static_cast<uint32_t>(values.size(1));
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto grads_continous = grads.contiguous();
    auto values_continous = values.contiguous();
    uint8_t* grads_ptr = static_cast<uint8_t*>(grads_continous.data_ptr());
    uint8_t* values_ptr = static_cast<uint8_t*>(values_continous.data_ptr());

    float decay_factor = lr;

    // 获取最大核数
    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    bool is_small = (in_length <= SMALL_DATA_THRESHOLD);
    auto grad_type = scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
    auto val_type = scalartype_to_datatype(convertTypeMetaToScalarType(values.dtype()));
    constexpr uint32_t optimizer_kind = static_cast<uint32_t>(OptimizerKind::SGD);
    // === 分支 1：偶数维度，使用 float2 算子,且 grad_type 和 val_type 为 float32===
    if (grad_dim % 2 == 0 && grad_type == DataType::Float32 && val_type == DataType::Float32) {
        int32_t vec_length = in_length / 2;
        const int32_t VEC_PER_BLOCK = ELEMENTS_PER_BLOCK / 2;
        int32_t total_blocks = (vec_length + VEC_PER_BLOCK - 1) / VEC_PER_BLOCK;
        int32_t core_num = std::min(max_cores, total_blocks);
        if (core_num == 0) {
            LOG_ERROR("core_num is zero!");
            return;
        }
        int32_t blocks_per_core = total_blocks / core_num;
        int32_t remainder_blocks = total_blocks % core_num;
        ACLRT_LAUNCH_KERNEL(update_float2_fused)(
            core_num, stream, grads_ptr, values_ptr, nullptr, grad_dim, val_dim, vec_length, 0, 0, 0, 0, 0, 0,
            decay_factor, 0, total_blocks, blocks_per_core, remainder_blocks, is_small,
            static_cast<uint32_t>(grad_type), static_cast<uint32_t>(val_type), optimizer_kind);
    } else {  // === 分支 2：奇数维度，使用标量 float 算子 ===
        int32_t total_blocks = (in_length + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
        int32_t core_num = std::min(max_cores, total_blocks);
        if (core_num == 0) {
            LOG_ERROR("core_num is zero!");
            return;
        }
        int32_t blocks_per_core = total_blocks / core_num;
        int32_t remainder_blocks = total_blocks % core_num;
        ACLRT_LAUNCH_KERNEL(update_fused)(core_num, stream, grads_ptr, values_ptr, nullptr, grad_dim, val_dim,
                                          in_length, 0, 0, 0, 0, 0, 0, decay_factor, 0, total_blocks, blocks_per_core,
                                          remainder_blocks, is_small, static_cast<uint32_t>(grad_type),
                                          static_cast<uint32_t>(val_type), optimizer_kind);
    }
}

void lookup_backward(const at::Tensor grad, const at::Tensor unique_buffer,
                     const at::Tensor unique_indices,
                     const at::Tensor inverse_indices,
                     const at::Tensor biased_offsets,  int32_t dim,
                     int32_t table_num, int32_t batch_size, int32_t feature_num,
                     int32_t num_key, int32_t combiner)
{   //数据检查
    TORCH_CHECK(biased_offsets.dtype() == inverse_indices.dtype(), "biased_offsets and inverse_indices must have the same type ");

    //数据准备
    auto grad_contin = grad.contiguous();
    auto unique_indices_contin=unique_indices.contiguous();
    auto biased_offsets_contin=biased_offsets.contiguous();
    auto inverse_indices_contin=inverse_indices.contiguous();
    float* grad_ptr = grad_contin.data_ptr<float>();
    float* unique_buffer_ptr = unique_buffer.data_ptr<float>();
    void* biased_offsets_ptr = biased_offsets_contin.data_ptr();
    void* inverse_indices_ptr = inverse_indices_contin.data_ptr();
    void* unique_indices_ptr = unique_indices_contin.data_ptr();
    // tiling
	bool isInt32 = biased_offsets.dtype() == torch::kInt32;
    bool is_small = (num_key * dim  <= (isInt32 ? SMALL_DATA_THRESHOLD_32:SMALL_DATA_THRESHOLD));
    int32_t total_blocks = (num_key * dim + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    int32_t core_num = std::min(max_cores, total_blocks);
    if (core_num == 0) {
        LOG_ERROR("core_num is zero!");
        return;
    }
    int32_t blocks_per_core = total_blocks / core_num;
    int32_t remainder_blocks = total_blocks % core_num;
    auto stream = c10_npu::getCurrentNPUStream().stream(true);

    at::Tensor kernelStatus = at::ones({1}, grad_contin.options().dtype(at::kBool));

    ACLRT_LAUNCH_KERNEL(lookup_backward)(core_num, stream, grad_ptr, unique_buffer_ptr,
        unique_indices_ptr, inverse_indices_ptr, biased_offsets_ptr,
        dim, table_num, batch_size, feature_num, num_key, combiner,
        total_blocks, blocks_per_core, remainder_blocks, isInt32, is_small, kernelStatus.data_ptr());

    TORCH_CHECK(
        kernelStatus.item<bool>(),
        "lookup_backward kernel failed: src_index == -1 detected in kernel (kernelStatus == false).");
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
    
    m.def("dyn_emb_rows", &dyn_emb_rows, "Get the number of rows in the table",
          py::arg("table"));
  
    m.def("export_batch_matched", &export_batch_matched,
          "Export KV-pairs within [offset, offset + n) whose score > threshold", py::arg("table"),
          py::arg("threshold"), py::arg("n"), py::arg("offset"), py::arg("num_matched"),
          py::arg("keys"), py::arg("values"));

    m.def("count_matched", &count_matched,
          "Count the KV-pairs whose score > threshold in the whole table.", py::arg("table"),
          py::arg("threshold"), py::arg("num_matched"));

    m.def("insert_and_evict", &insert_and_evict,
          "Insert keys and values, evicting if necessary", py::arg("table"),
          py::arg("n"), py::arg("keys"), py::arg("values"), py::arg("score"),
          py::arg("evicted_keys"), py::arg("evicted_values"),
          py::arg("evicted_score"), py::arg("d_evicted_counter"),
          py::arg("unique_key") = true, py::arg("ignore_evict_strategy") = false);

    m.def("find", &find, "Find values in the table based on keys",
          py::arg("table"), py::arg("n"), py::arg("keys"), py::arg("values"),
          py::arg("founds"), py::arg("score") = c10::nullopt);

    m.def("erase", &erase, "Erase values from the table based on keys",
          py::arg("table"), py::arg("n"), py::arg("keys"));
        
    m.def("clear", &clear, "Clear all keys in the table", py::arg("table"));

    m.def("reserve", &reserve, "reserve hash table capacity", py::arg("table"),
          py::arg("new_capacity"));
      
    m.def("accum_or_assign", &accum_or_assign,
          "Accumulate or assign values to the table", py::arg("table"),
          py::arg("n"), py::arg("keys"), py::arg("value_or_deltas"),
          py::arg("accum_or_assigns"), py::arg("score") = c10::nullopt,
          py::arg("ignore_evict_strategy") = false);
  
    m.def("assign", &assign, "Assign values to the table based on keys",
          py::arg("table"), py::arg("n"), py::arg("keys"), py::arg("values"),
          py::arg("score") = c10::nullopt, py::arg("unique_key") = true);

    m.def("device_timestamp", &device_timestamp, "device_timestamp");

    m.def("load_from_pointer", &load_from_pointer_imp, "load_from_pointer", py::arg("pointers"), py::arg("dst"));

    m.def("reduce_grads", &reduce_grads, "reduce grads", py::arg("grad"), py::arg("unique"), py::arg("inverse"));

    m.def("dynamic_emb_Adam_with_pointer", &dyn_emb::dynamic_emb_Adam_with_pointer,
          "Adam optimizer for dynamic embedding", py::arg("grads"), py::arg("val_pointers"), py::arg("val_type"),
          py::arg("state_dim"), py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
          py::arg("weight_decay"), py::arg("iter_num"));

    m.def("dynamic_emb_adamW_with_table", &dyn_emb::dynamic_emb_adamW_with_table,   
          "AdamW optimizer for dynamic embedding", py::arg("ht"), py::arg("n"), py::arg("indices"), py::arg("grads"),
          py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"), py::arg("iter_num"),
          py::arg("weight_type"));
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
          "RowWise AdaGrad optimizer for dynamic embedding", py::arg("grads"), py::arg("valPointers"), py::arg("valType"),
          py::arg("stateDim"), py::arg("lr"), py::arg("eps"));
    m.def("dynamic_emb_sgd_with_pointer", &dyn_emb::dynamic_emb_sgd_with_pointer, "SGD optimizer for dynamic embedding",
          py::arg("grads"), py::arg("val_pointers"), py::arg("val_type"), py::arg("lr"));

    m.def("dynamic_emb_adamW_fused", &dyn_emb::dynamic_emb_adamW_fused,
          "AdamW optimizer for dynamic embedding", py::arg("grads"), py::arg("values"), 
          py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
          py::arg("weight_decay"), py::arg("iter_num"));
    m.def("dynamic_emb_adagrad_fused", &dyn_emb::dynamic_emb_adagrad_fused,
          "AdaGrad optimizer for dynamic embedding", py::arg("grads"), py::arg("values"),
          py::arg("lr"), py::arg("eps"));
    m.def("dynamic_emb_rowwise_adagrad_fused", &dyn_emb::dynamic_emb_rowwise_adagrad_fused,
          "RowWise AdaGrad optimizer for dynamic embedding", py::arg("grads"), py::arg("values"),
          py::arg("lr"), py::arg("eps"));
    m.def("dynamic_emb_sgd_fused", &dyn_emb::dynamic_emb_sgd_fused, "SGD optimizer for dynamic embedding",
          py::arg("grads"), py::arg("values"), py::arg("lr"));

    m.def("dedup_input_indices_op", &dedup_input_indices_npu,
          "NPU-accelerated deduplication for input indices (dedup input indices on NPU)", py::arg("indices"),
          py::arg("offsets"), py::arg("d_table_offsets_in_feature"), py::arg("table_num"), py::arg("local_batch_size"),
          py::arg("reverse_idx"), py::arg("d_unique_nums"), py::arg("d_unique_offsets"), py::arg("unique_idx"),
          py::arg("new_offsets"), py::arg("new_lengths"));

    m.def("lookup_forward", &lookup_forward, "lookup_forward", py::arg("src"), py::arg("dst"), py::arg("offset"),
          py::arg("inverse"), py::arg("combiner"), py::arg("total_dims"), py::arg("accum_dims"), py::arg("ev_size"),
          py::arg("num_vec"), py::arg("batch_size"));
    m.def("lookup_backward", &lookup_backward, "backward", py::arg("grad"),
      py::arg("unique_buffer"), py::arg("unique_indices"),
      py::arg("inverse_indices"), py::arg("biased_offsets"), py::arg("dim"),
      py::arg("tables_num"), py::arg("batch_size"), py::arg("num_feature"),
      py::arg("num_key"), py::arg("combiner"));
}
}  // namespace dyn_emb
