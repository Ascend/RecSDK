/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
* Copyright (c) huawei Platforms, Inc. and affiliates.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree.
*/
#include <ATen/ops/empty_like.h>
#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>

#include <cstdint>
#include <vector>

#include "torch/torch.h"
#include "common_utils.h"

using BucketResult = std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>,
                     std::optional<at::Tensor>,
                     std::optional<at::Tensor>, std::optional<at::Tensor>, at::Tensor>;
namespace hybrid {

// 计算前缀和，preSum数组长度应为length+1
template <typename T>
void PrefixSum(const int length, const T* array, T* preSum)
{
    preSum[0] = 0;
    for (const auto i : c10::irange(length)) {
        preSum[i + 1] = array[i] + preSum[i];
    }
}

template <typename OffsetT, typename IndexT>
void ComputeNewLengths(const OffsetT* offsetsData, const IndexT* indicesData, OffsetT* newLengthsData,
                       int32_t numFeatures, int32_t batchSize, int64_t bucketSize, int64_t lengthsSize)
{
    for (const auto featureIdx : c10::irange(numFeatures)) {
        const auto blockSize = bucketSize;
        for (const auto batchIdx : c10::irange(batchSize)) {
            const auto linearIndex = featureIdx * batchSize + batchIdx;
            const OffsetT start = offsetsData[linearIndex];
            const OffsetT end = offsetsData[linearIndex + 1];
            for (const auto i : c10::irange(start, end)) {
                const IndexT idx = indicesData[i];
                const IndexT bucket = idx % blockSize;
                newLengthsData[bucket * lengthsSize + linearIndex]++;
            }
        }
    }
}

template <bool Sequence, typename OffsetT, typename IndexT>
void FillNewIndices(const OffsetT* offsetsData, const IndexT* indicesData, OffsetT* newOffsetsData,
                    IndexT* newIndicesData, IndexT* unbucketizePermuteData, int32_t numFeatures,
                    int32_t batchSize, int64_t bucketSize, int64_t lengthsSize)
{
    for (const auto featureIdx : c10::irange(numFeatures)) {
        const auto blockSize = bucketSize;
        for (const auto batchIdx : c10::irange(batchSize)) {
            const auto linearIndex = featureIdx * batchSize + batchIdx;
            const OffsetT start = offsetsData[linearIndex];
            const OffsetT end = offsetsData[linearIndex + 1];
            for (const auto i : c10::irange(start, end)) {
                const IndexT idx = indicesData[i];
                const IndexT bucket = idx % blockSize;
                const IndexT pos = newOffsetsData[bucket * lengthsSize + linearIndex];
                newIndicesData[pos] = idx;
                if constexpr (Sequence) {
                    unbucketizePermuteData[i] = pos;
                }
                newOffsetsData[bucket * lengthsSize + linearIndex]++;
            }
        }
    }
}

template <typename OffsetT, typename IndexT, bool ReturnCount>
int64_t Deduplicate(OffsetT* newLengthsData, const OffsetT* newOffsetsData, const OffsetT* offsetsData,
                    const IndexT* indicesData, IndexT* newIndicesData, IndexT* unbucketizePermuteData,
                    int32_t numFeatures, int32_t batchSize, int64_t bucketSize, IndexT* idsCountData)
{
    int32_t uniqueOffset = 0;
    OffsetT curOffset = 0;
    std::vector<ska::flat_hash_map<IndexT, int32_t>> uniqueMaps(numFeatures * bucketSize);
    IndexT lastOffset = 0;
    for (const auto featureBucketIdx : c10::irange(numFeatures * bucketSize)) {
        auto& uniqueMap = uniqueMaps[featureBucketIdx];
        for (const auto batchIdx : c10::irange(batchSize)) {
            const auto linearIndex = featureBucketIdx * batchSize + batchIdx;
            const OffsetT start = lastOffset;
            const OffsetT end = newOffsetsData[linearIndex];
            lastOffset = end;
            for (const auto i : c10::irange(start, end)) {
                const IndexT idx = newIndicesData[i];
                auto it = uniqueMap.find(idx);
                if (it == uniqueMap.end()) {
                    uniqueMap.emplace(idx, uniqueOffset);
                    newIndicesData[uniqueOffset] = idx;
                    if (ReturnCount) {
                        idsCountData[uniqueOffset] = 1;
                    }
                    uniqueOffset++;
                } else {
                    if (ReturnCount) {
                        idsCountData[it->second]++;
                    }
                    newLengthsData[linearIndex]--;
                }
            }
        }
    }

    for (const auto featureIdx : c10::irange(numFeatures)) {
        const auto blockSize = bucketSize;
        for (const auto batchIdx : c10::irange(batchSize)) {
            const auto linearIndex = featureIdx * batchSize + batchIdx;
            const OffsetT rowStart = offsetsData[linearIndex];
            const OffsetT rowEnd = offsetsData[linearIndex + 1];
            for (const auto i : c10::irange(rowStart, rowEnd)) {
                const IndexT idx = indicesData[i];
                const IndexT bucket = idx % blockSize;
                const auto hashMapIndex = bucket * numFeatures + featureIdx;
                // 为了性能考虑不做边界检查，因为一定在其中
                unbucketizePermuteData[i] = uniqueMaps[hashMapIndex].find(idx)->second;
            }
        }
    }
    return uniqueOffset;
}

// 核心分桶逻辑模板
template <bool Sequence,             // 是否序列模式
        bool HasWeight,            // 是否包含权重
        bool ReturnBucketMapping,  // 是否返回分桶映射
        typename OffsetT,          // 偏移量类型
        typename IndexT,           // 索引类型
        typename ScalarT,          // 标量类型
        bool DoUnique,             // 是否去重
        bool ReturnCount           // 是否返回key count
        >
void BlockBucketizeSparseFeaturesCpuKernel(const at::Tensor& lengths, const at::Tensor& indices,
                                           const std::optional<at::Tensor>& weights, const bool bucketizePos,
                                           const at::Tensor& blockSizes,
                                           const std::optional<at::Tensor>& totalNumBlocks, const int64_t bucketSize,
                                           at::Tensor newLengths, at::Tensor newIndices,
                                           std::optional<at::Tensor> newWeights, std::optional<at::Tensor> newPos,
                                           const std::optional<at::Tensor>& unbucketizePermute,
                                           const std::optional<at::Tensor>& batchSizePerFeature,
                                           const std::optional<std::vector<at::Tensor>>& blockBucketizePos,
                                           const std::optional<at::Tensor>& bucketMapping, const bool keepOrigIdx,
                                           at::Tensor idsCounts)
{
    // 基本参数校验
    const auto lengthsSize = lengths.numel();
    const auto newLengthsSize = lengthsSize * bucketSize;
    const int32_t numFeatures = blockSizes.numel();
    TORCH_CHECK(numFeatures > 0, "blockSizes must have at least one element");

    const int32_t batchSize = lengthsSize / numFeatures;

    // 预分配偏移量数组
    auto offsets = at::empty({lengthsSize + 1}, lengths.options());
    auto newOffsets = at::empty({newLengthsSize + 1}, lengths.options());

    // 数据指针获取
    OffsetT* lengthsData = GetSafeDataPtr<OffsetT>(lengths, "lengths");
    OffsetT* offsetsData = GetSafeDataPtr<OffsetT>(offsets, "offsets");
    IndexT* indicesData = GetSafeDataPtr<IndexT>(indices, "indices");

    // 计算原始偏移量
    PrefixSum(lengthsSize, lengthsData, offsetsData);
    TORCH_CHECK(offsetsData[lengthsSize] == indices.numel(),
                "Offset validation failed: offsets[lengthsSize] = ", offsetsData[lengthsSize],
                ", indices.numel() = ", indices.numel());

    // 新数据结构指针初始化
    OffsetT* newLengthsData = GetSafeDataPtr<OffsetT>(newLengths, "newLengths");
    OffsetT* newOffsetsData = GetSafeDataPtr<OffsetT>(newOffsets, "newOffsets");
    IndexT* newIndicesData = GetSafeDataPtr<IndexT>(newIndices, "newIndices");

    // 可选参数处理
    IndexT* unbucketizePermuteData = nullptr;
    if constexpr (Sequence) {
        unbucketizePermuteData = GetSafeDataPtr<IndexT>(unbucketizePermute.value(), "unbucketizePermute");
    }

    // 第一阶段: 计算新长度
    ComputeNewLengths<OffsetT, IndexT>(offsetsData, indicesData, newLengthsData, numFeatures,
                                       batchSize, bucketSize, lengthsSize);

    // 计算新偏移量
    PrefixSum(newLengthsSize, newLengthsData, newOffsetsData);

    // 第二阶段: 填充新索引
    FillNewIndices<Sequence, OffsetT, IndexT>(offsetsData, indicesData, newOffsetsData, newIndicesData,
                                              unbucketizePermuteData, numFeatures, batchSize, bucketSize,
                                              lengthsSize);

    // 去重逻辑 (需要时启用)
    if constexpr (DoUnique) {
        auto* idsCountData = GetSafeDataPtr<IndexT>(idsCounts, "idsCounts");
        int64_t uniqueSize = Deduplicate<OffsetT, IndexT, ReturnCount>(
            newLengthsData, newOffsetsData, offsetsData, indicesData, newIndicesData,
            unbucketizePermuteData, numFeatures, batchSize, bucketSize, idsCountData);
        newIndices.resize_(uniqueSize);
        if (ReturnCount) {
            idsCounts.resize_(uniqueSize);
        }
    }
}

// 对外接口函数
BucketResult BlockBucketizeSparseFeaturesCpu(
    const at::Tensor& lengths, const at::Tensor& indices,
    const bool bucketizePos, const bool sequence,
    const at::Tensor& blockSizes, const int64_t bucketSize,
    const std::optional<at::Tensor>& totalNumBlocks,
    const std::optional<at::Tensor>& weights,
    const std::optional<at::Tensor>& batchSizePerFeature, const int64_t maxBatchSize,
    const std::optional<std::vector<at::Tensor>>& blockBucketizePos,
    const bool returnBucketMapping, const bool keepOrigIdx, const bool doUnique,
    const bool returnCount)
{
    // 参数校验
    TORCH_CHECK(lengths.defined() && lengths.numel() > 0, "Lengths tensor is an empty tensor");

    TORCH_CHECK(indices.defined() && indices.numel() > 0, "Indices tensor is an empty tensor");

    TORCH_CHECK(lengths.scalar_type() == at::kLong, "Lengths tensor must be int64 type, got: ", lengths.scalar_type());

    TORCH_CHECK(indices.scalar_type() == at::kLong, "Indices tensor must be int64 type, got: ", indices.scalar_type());

    TORCH_CHECK(!weights.has_value(), "Weighted KJT is currently not supported");

    TORCH_CHECK(!bucketizePos, "Bucket position tracking is not implemented");

    TORCH_CHECK(!returnBucketMapping, "Bucket mapping return is not supported");

    TORCH_CHECK(!batchSizePerFeature, "batchSize PerFeature return is not supported");

    TORCH_CHECK(!blockBucketizePos, "block BucketizePos return is not supported");

    TORCH_CHECK(bucketSize > 0, "Bucket size must be greater than 0");

    // 初始化输出张量
    const auto lengthsSize = lengths.numel();
    const auto newLengthsSize = lengthsSize * bucketSize;

    auto newLengths = at::zeros({newLengthsSize}, lengths.options());
    auto newIndices = at::empty_like(indices);
    auto unbucketizePermute = at::empty(indices.sizes(), indices.options());
    auto idsCounts = sequence && doUnique && returnCount ?
        at::empty_like(indices) : torch::tensor({}, torch::dtype(torch::kInt64));

    // 根据序列模式选择不同内核
    if (sequence && doUnique) {
        if (returnCount) {
            BlockBucketizeSparseFeaturesCpuKernel<true, false, false, int64_t, int64_t, int64_t, true, true>(
                lengths, indices, weights, bucketizePos, blockSizes, totalNumBlocks, bucketSize, newLengths, newIndices,
                std::nullopt, std::nullopt, unbucketizePermute, batchSizePerFeature, blockBucketizePos, std::nullopt,
                keepOrigIdx, idsCounts);
        } else {
            BlockBucketizeSparseFeaturesCpuKernel<true, false, false, int64_t, int64_t, int64_t, true, false>(
                lengths, indices, weights, bucketizePos, blockSizes, totalNumBlocks, bucketSize, newLengths, newIndices,
                std::nullopt, std::nullopt, unbucketizePermute, batchSizePerFeature, blockBucketizePos, std::nullopt,
                keepOrigIdx, idsCounts);
        }
    } else if (sequence && !doUnique) {
        BlockBucketizeSparseFeaturesCpuKernel<true, false, false, int64_t, int64_t, int64_t, false, false>(
            lengths, indices, weights, bucketizePos, blockSizes, totalNumBlocks, bucketSize, newLengths, newIndices,
            std::nullopt, std::nullopt, unbucketizePermute, batchSizePerFeature, blockBucketizePos, std::nullopt,
            keepOrigIdx, idsCounts);
    } else {
        BlockBucketizeSparseFeaturesCpuKernel<false, false, false, int64_t, int64_t, int64_t, false, false>(
            lengths, indices, weights, bucketizePos, blockSizes, totalNumBlocks, bucketSize, newLengths, newIndices,
            std::nullopt, std::nullopt, unbucketizePermute, batchSizePerFeature, blockBucketizePos, std::nullopt,
            keepOrigIdx, idsCounts);
    }

    return {newLengths, newIndices, std::nullopt, std::nullopt, unbucketizePermute, std::nullopt, idsCounts};
}

}  // namespace hybrid