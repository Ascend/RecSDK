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

    int64_t* hashIndicesPtr = hashIndices.data_ptr<int64_t>();
    int64_t* globalIdsPtr = globalIds.data_ptr<int64_t>();
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
    int64_t* hashIndicesPtr = hashIndices.data_ptr<int64_t>();
    int64_t* globalIdsPtr = globalIds.data_ptr<int64_t>();
    int64_t* offsetPtr = offset.data_ptr<int64_t>();
    int64_t start = offsetPtr[tableId];
    int64_t end = offsetPtr[tableId + 1];
    int64_t* uniqueOffsetPtr = uniqueOffset.data_ptr<int64_t>();
    if (start == end) {
        uniqueOffsetPtr[tableId + 1] = uniqueOffsetPtr[tableId];
        return;
    }
    for (int64_t i = start; i < end; i++) {
        int64_t key = globalIdsPtr[i];
        TORCH_CHECK(key >= 0, " Invalid key value: ", key);
        auto it = ids2indicesMap.find(key);
        if (it != ids2indicesMap.end()) {
            hashIndicesPtr[i] = it->second;
            continue;
        }

        std::lock_guard<std::mutex> lock(insertMute);
        // after lock, let's find(key) again to make sure that
        // the followings are executed sequentially by multi-threads
        it = ids2indicesMap.find(key);
        if (it == ids2indicesMap.end()) {
            int64_t r = maxIndex++;
            // When `onlyDeviceMem` is true, `initMaxIndex` indicates a complete table size, else only a cache size.
            if (onlyDeviceMem) {
                TORCH_CHECK(r < initMaxIndex, "Ids map reached maxIndex = ",
                            initMaxIndex, " please reallocate a larger buffer.")
            }
            ids2indicesMap.insert_or_assign(key, r);
            indice2id.push_back(key);  // key's offset is r
            hashIndicesPtr[i] = r;
        } else {
            hashIndicesPtr[i] = it->second;
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

    int64_t* hashIndicesPtr = hashIndices.data_ptr<int64_t>();
    int64_t* offsetPtr = offset.data_ptr<int64_t>();
    int64_t start = offsetPtr[tableId];
    int64_t end = offsetPtr[tableId + 1];
    int64_t* uniqueOffsetPtr = uniqueOffset.data_ptr<int64_t>();
    if (start == end) {
        uniqueOffsetPtr[tableId + 1] = uniqueOffsetPtr[tableId];
        return;
    }

    auto aHashMap = AllocFullHashMap();
    auto aHashMapPtr = aHashMap->data();
    int64_t* uniquePtr = unique.data_ptr<int64_t>();
    int64_t* uniqueIdsPtr = uniqueIds.data_ptr<int64_t>();
    int64_t* uniqueInversePtr = uniqueInverse.data_ptr<int64_t>();
    if (uniqueOffsetPtr == nullptr || uniquePtr == nullptr || uniqueInversePtr == nullptr) {
        printf("[ERROR] unique, uniqueOffset and uniqueInverse cannot be None!");
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
    auto gIdsPtr = globalIds.data_ptr<int64_t>();
    auto offPtr = offsets.data_ptr<int64_t>();
    auto hashIdxPtr = hashIndices.data_ptr<int64_t>();
    auto uniquePtr = unique.data_ptr<int64_t>();
    auto uniqueIdsPtr = uniqueIds.data_ptr<int64_t>();
    auto uniqueInvPtr = uniqueInverse.data_ptr<int64_t>();
    auto uniqOffPtr = uniqueOffset.data_ptr<int64_t>();

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