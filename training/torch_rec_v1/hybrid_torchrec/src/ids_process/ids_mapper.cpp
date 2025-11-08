/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "ids_mapper.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>

#include "torch/torch.h"

#include "unique.h"
#include "common_utils.h"
namespace hybrid {

constexpr const int EXPAND_CAPACITY_RATE = 2;

std::tuple<at::Tensor, at::Tensor, at::Tensor> IdsMapper::UniqueAndLookup(const torch::Tensor& globalIds)
{
    TORCH_CHECK(globalIds.device() == torch::kCPU, "globalIds must be on CPU but on ", globalIds.device());
    TORCH_CHECK(globalIds.scalar_type() == at::kLong,
        "globalIds must be int64_t tensor expected but got a tensor with dtype: ", globalIds.scalar_type());
        at::ThreadLocalStateGuard tlsGrad(state);
    return FindOrInsertHighPrecison(globalIds);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> IdsMapper::FindOrInsertHighPrecison(const torch::Tensor& globalIds)
{
    at::Tensor hashIndices = at::empty_like(globalIds);

    int64_t* hashIndicesPtr = GetSafeDataPtr<int64_t>(hashIndices, "hashIndices");
    int64_t* globalIdsPtr = GetSafeDataPtr<int64_t>(globalIds, "globalIds");
    for (int64_t i = 0; i < globalIds.numel(); i++) {
        int64_t key = globalIdsPtr[i];
        TORCH_CHECK(key >= 0, " Invalid key value: ", key);
        auto findResult = ids2indicesMap.find(key);
        if (findResult == ids2indicesMap.end()) {
            int64_t r = maxIndex++;
            TORCH_CHECK(r < initMaxIndex, "Ids map reached maxIndex = ",
                initMaxIndex, " please reallocate a larger buffer.");
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

void IdsMapper::UniqueAndLookupOut(const torch::Tensor& globalIds, const torch::Tensor& hashIndices,
                                   const torch::Tensor& offset, const torch::Tensor& unique,
                                   const torch::Tensor& uniqueIds,
                                   const torch::Tensor& uniqueInverse, const torch::Tensor& uniqueOffset,
                                   int64_t tableId)
{
    at::ThreadLocalStateGuard tlsGrad(state);
    RECORD_FUNCTION(c10::str("hybrid::UniqueAndLookupOut"), c10::ArrayRef<const c10::IValue>());
    TORCH_CHECK(offset.numel() - 1 > tableId, "offset must be equal to table size + 1")
    // 数据指针获取
    int64_t* hashIndicesPtr = GetSafeDataPtr<int64_t>(hashIndices, "hashIndices");
    int64_t* globalIdsPtr = GetSafeDataPtr<int64_t>(globalIds, "globalIds");
    int64_t* offsetPtr = GetSafeDataPtr<int64_t>(offset, "offset");
    int64_t* uniqueOffsetPtr = GetSafeDataPtr<int64_t>(uniqueOffset, "uniqueOffset");

    int64_t start = offsetPtr[tableId];
    int64_t end = offsetPtr[tableId + 1];
    if (start == end) {
        uniqueOffsetPtr[tableId + 1] = uniqueOffsetPtr[tableId];
        return;
    }
    for (int64_t i = start; i < end; i++) {
        int64_t key = globalIdsPtr[i];
        TORCH_CHECK(key >= 0, " Invalid key value: ", key);
        auto findResult = ids2indicesMap.find(key);
        if (findResult == ids2indicesMap.end()) {
            std::lock_guard<std::mutex> lock(insertMute);
            // after lock, let's find(key) again to make sure that
            // the followings are executed sequentially by multi-threads
            auto findResult = ids2indicesMap.find(key);
            if (findResult == ids2indicesMap.end()) {
                int64_t r = maxIndex++;
                // When `onlyDeviceMem` is true, `initMaxIndex` indicates a complete table size, else only a cache size.
                if (onlyDeviceMem) {
                    TORCH_CHECK(r < initMaxIndex, "Ids map reached maxIndex = ",
                                initMaxIndex, " please reallocate a larger buffer.");
                }
                ids2indicesMap.insert_or_assign(key, r);
                indice2id.push_back(key);
                hashIndicesPtr[i] = r;
            } else {
                hashIndicesPtr[i] = findResult->second;
            }
        } else {
            hashIndicesPtr[i] = findResult->second;
        }
    }

    // Unique
    UniqueProcessing(hashIndices, offset, unique, uniqueIds, uniqueInverse, uniqueOffset, tableId);
}

void IdsMapper::UniqueProcessing(const torch::Tensor& hashIndices, const torch::Tensor& offset,
                                 const torch::Tensor& unique, const torch::Tensor& uniqueIds,
                                 const torch::Tensor& uniqueInverse,
                                 const torch::Tensor& uniqueOffset, int64_t tableId)
{
    at::ThreadLocalStateGuard tlsGrad(state);
    RECORD_FUNCTION(c10::str("hybrid::UniqueProcessing"), c10::ArrayRef<const c10::IValue>());

    int64_t* hashIndicesPtr = GetSafeDataPtr<int64_t>(hashIndices, "hashIndices");
    int64_t* offsetPtr = GetSafeDataPtr<int64_t>(offset, "offset");

    int64_t start = offsetPtr[tableId];
    int64_t end = offsetPtr[tableId + 1];

    auto aHashMap = AllocFullHashMap();
    auto aHashMapPtr = aHashMap->data();
    int64_t* uniquePtr = GetSafeDataPtr<int64_t>(unique, "unique");
    int64_t* uniqueIdsPtr = GetSafeDataPtr<int64_t>(uniqueIds, "uniqueIds");
    int64_t* uniqueInversePtr = GetSafeDataPtr<int64_t>(uniqueInverse, "uniqueInverse");
    int64_t* uniqueOffsetPtr = GetSafeDataPtr<int64_t>(uniqueOffset, "uniqueOffset");

    if (start == end) {
        uniqueOffsetPtr[tableId + 1] = uniqueOffsetPtr[tableId];
        return;
    }
    int64_t globalUniqueOffset = uniqueOffsetPtr[tableId];
    int64_t thisUniqueOffset = 0;

    for (const auto i : c10::irange(start, end)) {
        int64_t key = hashIndicesPtr[i];
        if (aHashMapPtr[key] == -1) {
            aHashMapPtr[key] = thisUniqueOffset;
            uniquePtr[globalUniqueOffset + thisUniqueOffset] = hashIndicesPtr[i];
            uniqueIdsPtr[globalUniqueOffset + thisUniqueOffset] = indice2id[hashIndicesPtr[i]];
            thisUniqueOffset++;
        }
    }

    if (tableId == 0) {
        uniqueOffsetPtr[tableId] = 0;
    }
    uniqueOffsetPtr[tableId + 1] = uniqueOffsetPtr[tableId] + thisUniqueOffset;

    for (const auto i : c10::irange(globalUniqueOffset, globalUniqueOffset + thisUniqueOffset)) {
        int64_t key = uniquePtr[i];
        aHashMapPtr[key] = i - globalUniqueOffset;
    }

    for (const auto i : c10::irange(start, end)) {
        int64_t key = hashIndicesPtr[i];
        uniqueInversePtr[i] = aHashMapPtr[key];
    }

    for (const auto i : c10::irange(globalUniqueOffset, globalUniqueOffset + thisUniqueOffset)) {
        int64_t key = uniquePtr[i];
        aHashMapPtr[key] = -1;
    }

    DeallocFullHashMap(std::move(aHashMap));
}

size_t IdsMapper::ProcessIds2Indices(IdsMapper& mapper, std::vector<int64_t>& uniqVec,
                                     const int64_t start, const int64_t end, const int64_t* gIdsPtr,
                                     int64_t* hashIdxPtr, int64_t* uniqueInvPtr)
{
    auto fullMap = mapper.AllocFullHashMap();
    int64_t* bitmap = fullMap->data();
    TORCH_CHECK(end - start >= 0 && end - start <= MAX_INDEX_LEN, "length is too large: ", end - start);
    uniqVec.reserve(end - start);

    for (int64_t i = start; i < end; ++i) {
        int64_t gid = gIdsPtr[i];

        auto it = mapper.ids2indicesMap.find(gid);
        if (it == mapper.ids2indicesMap.end()) {
            std::lock_guard<std::mutex> lk(mapper.insertMute);
            it = mapper.ids2indicesMap.find(gid);
            if (it == mapper.ids2indicesMap.end()) {
                int64_t newIdx = mapper.maxIndex++;
                mapper.ids2indicesMap.insert_or_assign(gid, newIdx);
                mapper.indice2id.push_back(gid);
                hashIdxPtr[i] = newIdx;
            } else {
                hashIdxPtr[i] = it->second;
            }
        } else {
            hashIdxPtr[i] = it->second;
        }

        int64_t hidx = hashIdxPtr[i];
        if (hidx >= static_cast<int64_t>(fullMap->size())) {
            TORCH_CHECK(hidx < (INT64_MAX - 1) / 2, "hidx is too large: ", hidx, ">=", (INT64_MAX - 1) / 2);
            fullMap->resize(hidx * EXPAND_CAPACITY_RATE + 1, -1);
            bitmap = fullMap->data();
        }

        if (bitmap[hidx] == -1) {
            bitmap[hidx] = static_cast<int64_t>(uniqVec.size());
            uniqVec.push_back(hidx);
        }
    }

    for (int64_t i = start; i < end; ++i) {
        uniqueInvPtr[i] = bitmap[hashIdxPtr[i]];
    }

    for (int64_t h : uniqVec) {
        bitmap[h] = -1;
    }
    mapper.DeallocFullHashMap(std::move(fullMap));
    return uniqVec.size();
}

torch::Tensor IdsMapper::ExportIdsAndIndices() const
{
    at::TensorOptions options = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
    torch::Tensor originalIds = at::empty({static_cast<int64_t>(indice2id.size())}, options);
    auto* originalIdsPtr = GetSafeDataPtr<int64_t>(originalIds, "originalIds");
    int64_t index = 0;
    for (auto& originalId : indice2id) {
        originalIdsPtr[index] = originalId;
        index++;
    }
    return originalIds;
}

void IdsMapper::LoadOriginalIds(const torch::Tensor originalIds)
{
    TORCH_CHECK(originalIds.dim() == 1, "originalIds dimension must be 1D.")
    auto dataSize = originalIds.numel();
    const auto* dataPtr = GetSafeDataPtr<int64_t>(originalIds, "originalIds");
    if (memStartIndex > 0) {
        for (auto i = 0; i < memStartIndex; ++i) {
            indice2id.emplace_back(0);
        }
    }
    for (int64_t i = 0; i < dataSize; ++i) {
        ids2indicesMap.emplace(dataPtr[i], maxIndex++);
        indice2id.emplace_back(dataPtr[i]);
    }
}

void IdsMapper::ParallelUniqueHashOut(
    const c10::List<c10::intrusive_ptr<IdsMapper>>& mappers,
    const torch::Tensor& globalIds,
    const torch::Tensor& hashIndices,
    const torch::Tensor& offsets,
    const torch::Tensor& unique,
    const torch::Tensor& uniqueIds,
    const torch::Tensor& uniqueInverse,
    const torch::Tensor& uniqueOffset)
{
    auto gIdsPtr = GetSafeDataPtr<int64_t>(globalIds, "globalIds");
    auto offPtr = GetSafeDataPtr<int64_t>(offsets, "offsets");
    auto hashIdxPtr = GetSafeDataPtr<int64_t>(hashIndices, "hashIndices");
    auto uniquePtr = GetSafeDataPtr<int64_t>(unique, "unique");
    auto uniqueIdsPtr = GetSafeDataPtr<int64_t>(uniqueIds, "uniqueIds");
    auto uniqueInvPtr = GetSafeDataPtr<int64_t>(uniqueInverse, "uniqueInverse");
    auto uniqOffPtr = GetSafeDataPtr<int64_t>(uniqueOffset, "uniqueOffset");

    const int64_t nTables = mappers.size();
    std::vector<int64_t> uniqueCnt(nTables);
    std::vector<std::vector<int64_t>> tableUniques(nTables);

    // 表间并行
    at::parallel_for(0, nTables, 1, [&](int64_t tbBegin, int64_t tbEnd) {
        for (int64_t t = tbBegin; t < tbEnd; ++t) {
            IdsMapper& mapper = *mappers[t];
            const int64_t start = offPtr[t];
            const int64_t end = offPtr[t + 1];

            if (start == end) {
                uniqueCnt[t] = 0;
                continue;
            }

            auto uniqueVecSize = ProcessIds2Indices(mapper, tableUniques[t], start, end, gIdsPtr,
                                                    hashIdxPtr, uniqueInvPtr);
            uniqueCnt[t] = static_cast<int64_t>(uniqueVecSize);
        }
    });

    // 最后再串行计算uniqueOffset
    uniqOffPtr[0] = 0;
    for (int64_t t = 0; t < nTables; ++t) {
        uniqOffPtr[t + 1] = uniqOffPtr[t] + uniqueCnt[t];
    }

    at::parallel_for(0, nTables, 1, [&](int64_t tbBegin, int64_t tbEnd) {
        for (int64_t t = tbBegin; t < tbEnd; ++t) {
            const int64_t base = uniqOffPtr[t];
            auto& mapper = *mappers[t];
            const auto& vec = tableUniques[t];
            for (size_t j = 0; j < vec.size(); ++j) {
                int64_t idx = vec[j];
                uniquePtr[base + j] = idx;
                uniqueIdsPtr[base + j] = mapper.indice2id[idx];
            }
        }
    });
}

}  // namespace hybrid