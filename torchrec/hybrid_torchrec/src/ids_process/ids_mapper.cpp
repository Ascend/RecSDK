#include "ids_mapper.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>

#include "unique.h"

namespace hybrid {
// 交换两个元素的值
std::tuple<at::Tensor, at::Tensor, at::Tensor> IdsMapper::UniqueAndLookup(const torch::Tensor& globalIds,
                                                                          bool highPrecison)
{
    TORCH_CHECK(globalIds.device() == torch::kCPU, "globalIds must be on CPU but on ", globalIds.device());
    TORCH_CHECK(globalIds.scalar_type() == at::kLong,
                "globalIds must be int64_t tensor expected but got a tensor with dtype: ", globalIds.scalar_type());
    at::ThreadLocalStateGuard tlsGrad(state);
    if (highPrecison)
        return FindOrInsertHighPrecison(globalIds);
    else
        return FindOrInsert(globalIds);
}

void IdsMapper::UniqueAndLookupOut(const torch::Tensor& globalIds, const torch::Tensor& hashIndices,
                                   const torch::Tensor& offset, const torch::Tensor& unique,
                                   const torch::Tensor& uniqueInverse, const torch::Tensor& uniqueOffset,
                                   int64_t tensorI)
{
    at::ThreadLocalStateGuard tlsGrad(state);
    RECORD_FUNCTION(c10::str("hybrid::UniqueAndLookupOut"), c10::ArrayRef<const c10::IValue>());

    int64_t* hashIndicesPtr = hashIndices.data_ptr<int64_t>();
    int64_t* globalIdsPtr = globalIds.data_ptr<int64_t>();
    int64_t* offsetPtr = offset.data_ptr<int64_t>();
    int64_t start = offsetPtr[tensorI];
    int64_t end = offsetPtr[tensorI + 1];
    int64_t* uniqueOffsetPtr = uniqueOffset.data_ptr<int64_t>();
    if (start == end) {
        uniqueOffsetPtr[tensorI + 1] = uniqueOffsetPtr[tensorI];
        return;
    }
    for (int64_t i = start; i < end; i++) {
        int64_t key = globalIdsPtr[i];
        auto findResult = ids2indicesMap.find(key);
        if (findResult == ids2indicesMap.end()) {
            std::lock_guard<std::mutex> lock(insertMute);
            auto findResult = ids2indicesMap.find(key);
            if (findResult == ids2indicesMap.end()) {
                int64_t r = maxIndex++;
                auto findResult = ids2indicesMap.find(key);
                ids2indicesMap.insert_or_assign(key, r);
                hashIndicesPtr[i] = r;
            } else {
                hashIndicesPtr[i] = findResult->second;
            }
        } else {
            hashIndicesPtr[i] = findResult->second;
        }
    }

    // Unique
    UniqueProcessing(hashIndices, offset, unique, uniqueInverse, uniqueOffset, tensorI);
}

void IdsMapper::UniqueProcessing(const torch::Tensor& hashIndices, const torch::Tensor& offset,
                                 const torch::Tensor& unique, const torch::Tensor& uniqueInverse,
                                 const torch::Tensor& uniqueOffset, int64_t tensorI)
{
    at::ThreadLocalStateGuard tlsGrad(state);
    RECORD_FUNCTION(c10::str("hybrid::UniqueProcessing"), c10::ArrayRef<const c10::IValue>());

    int64_t* hashIndicesPtr = hashIndices.data_ptr<int64_t>();
    int64_t* offsetPtr = offset.data_ptr<int64_t>();
    int64_t start = offsetPtr[tensorI];
    int64_t end = offsetPtr[tensorI + 1];
    int64_t* uniqueOffsetPtr = uniqueOffset.data_ptr<int64_t>();
    if (start == end) {
        uniqueOffsetPtr[tensorI + 1] = uniqueOffsetPtr[tensorI];
        return;
    }

    auto aHashMap = AllocFullHashMap();
    auto aHashMapPtr = aHashMap->data();
    int64_t* uniquePr = unique.data_ptr<int64_t>();
    int64_t* uniqueInversePtr = uniqueInverse.data_ptr<int64_t>();
    int64_t globalUniqueOffset = uniqueOffsetPtr[tensorI];
    int64_t thisUniqueOffset = 0;

    for (const auto i : c10::irange(start, end)) {
        int64_t key = hashIndicesPtr[i];
        if (aHashMapPtr[key] == -1) {
            aHashMapPtr[key] = thisUniqueOffset;
            uniquePr[globalUniqueOffset + thisUniqueOffset] = hashIndicesPtr[i];
            thisUniqueOffset++;
        }
    }

    if (tensorI == 0) {
        uniqueOffsetPtr[tensorI] = 0;
    }
    uniqueOffsetPtr[tensorI + 1] = uniqueOffsetPtr[tensorI] + thisUniqueOffset;

    for (const auto i : c10::irange(globalUniqueOffset, globalUniqueOffset + thisUniqueOffset)) {
        int64_t key = uniquePr[i];
        aHashMapPtr[key] = i - globalUniqueOffset;
    }

    for (const auto i : c10::irange(start, end)) {
        int64_t key = hashIndicesPtr[i];
        uniqueInversePtr[i] = aHashMapPtr[key];
    }

    for (const auto i : c10::irange(globalUniqueOffset, globalUniqueOffset + thisUniqueOffset)) {
        int64_t key = uniquePr[i];
        aHashMapPtr[key] = -1;
    }

    DeallocFullHashMap(std::move(aHashMap));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> IdsMapper::FindOrInsertHighPrecison(const torch::Tensor& globalIds)
{
    at::Tensor hashIndices = at::empty_like(globalIds);

    int64_t* hashIndicesPtr = hashIndices.data_ptr<int64_t>();
    int64_t* globalIdsPtr = globalIds.data_ptr<int64_t>();
    for (int64_t i = 0; i < globalIds.numel(); i++) {
        int64_t key = globalIdsPtr[i];
        auto findResult = ids2indicesMap.find(key);
        if (findResult == ids2indicesMap.end()) {
            int64_t r = maxIndex++;
            auto findResult = ids2indicesMap.find(key);
            ids2indicesMap.insert_or_assign(key, r);
            hashIndicesPtr[i] = r;
        } else {
            hashIndicesPtr[i] = findResult->second;
        }
    }

    at::Tensor unique;
    at::Tensor uniqueInverse;
    std::tie(unique, uniqueInverse) = UniqueParallel(hashIndices);
    return {hashIndices, unique, uniqueInverse};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> IdsMapper::FindOrInsert(const torch::Tensor& globalIds)
{
    at::Tensor hashIndices = at::empty_like(globalIds);

    int64_t* hashIndicesPtr = hashIndices.data_ptr<int64_t>();
    int64_t* globalIdsPtr = globalIds.data_ptr<int64_t>();
    at::parallel_for(0, globalIds.numel(), at::internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
            int64_t key = globalIdsPtr[i];
            auto findResult = ids2indicesMap.find(key);
            if (findResult == ids2indicesMap.end()) {
                std::lock_guard<std::mutex> lock(insertMute);
                auto findResult = ids2indicesMap.find(key);
                if (findResult == ids2indicesMap.end()) {
                    int64_t r = maxIndex++;
                    auto findResult = ids2indicesMap.find(key);
                    ids2indicesMap.insert_or_assign(key, r);
                    hashIndicesPtr[i] = r;
                } else {
                    hashIndicesPtr[i] = findResult->second;
                }
            } else {
                hashIndicesPtr[i] = findResult->second;
            }
        }
    });

    at::Tensor unique;
    at::Tensor uniqueInverse;
    std::tie(unique, uniqueInverse) = UniqueParallel(hashIndices);
    return {hashIndices, unique, uniqueInverse};
}
}  // namespace hybrid