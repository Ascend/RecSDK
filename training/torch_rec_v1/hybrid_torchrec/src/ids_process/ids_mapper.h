/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef IDS_MAPPER_H
#define IDS_MAPPER_H

#include <c10/util/flat_hash_map.h>
#include <omp.h>
#include <torch/torch.h>

#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <tuple>
#include <vector>

namespace hybrid {
constexpr int64_t MIN_IDS_LENGTH = 65536;
constexpr int64_t DOUBLE_INIT = 2;
constexpr int64_t PARTITION_LEN = 8192;
class IdsMapper : public torch::CustomClassHolder {
public:
    using Self = IdsMapper;
    explicit IdsMapper(int64_t initMaxIndex, bool onlyDeviceMem = true)
        : initMaxIndex(initMaxIndex), onlyDeviceMem(onlyDeviceMem){};
    IdsMapper(const IdsMapper& other) = delete;
    IdsMapper& operator=(const IdsMapper& other)
    {
        return *this;
    };
    std::tuple<at::Tensor, at::Tensor, at::Tensor> UniqueAndLookup(const torch::Tensor& globalIds);
    void UniqueAndLookupOut(const torch::Tensor& globalIds, const torch::Tensor& hashIndices,
                            const torch::Tensor& offset, const torch::Tensor& unique,
                            const torch::Tensor& uniqueIds,
                            const torch::Tensor& uniqueInverse, const torch::Tensor& uniqueOffset, int64_t tableId);

    std::unique_ptr<std::vector<int64_t>> AllocFullHashMap()
    {
        std::lock_guard<std::mutex> lock(allocMute);
        std::unique_ptr<std::vector<int64_t>> oneMap;
        if (fullHashMapQue.size() == 0) {
            oneMap = std::make_unique<std::vector<int64_t>>(maxIndex * DOUBLE_INIT, -1);
        } else {
            oneMap = std::move(fullHashMapQue.front());
            fullHashMapQue.pop();
        }
        if (oneMap->capacity() <= maxIndex) {
            oneMap->resize(maxIndex * DOUBLE_INIT, -1);
        }
        return oneMap;
    }

    void DeallocFullHashMap(std::unique_ptr<std::vector<int64_t>> oneMap)
    {
        std::lock_guard<std::mutex> lock(allocMute);
        fullHashMapQue.push(std::move(oneMap));
    }

    static size_t ProcessIds2Indices(IdsMapper& mapper, std::vector<int64_t>& uniqVec,
                                                const int64_t start, const int64_t end, const int64_t* gIdsPtr,
                                                int64_t* hashIdxPtr, int64_t* uniqueInvPtr);

    static void ParallelUniqueHashOut(
        const c10::List<c10::intrusive_ptr<IdsMapper>>& mappers,
        const torch::Tensor& globalIds,
        const torch::Tensor& hashIndices,
        const torch::Tensor& offsets,
        const torch::Tensor& unique,
        const torch::Tensor& uniqueIds,
        const torch::Tensor& uniqueInverse,
        const torch::Tensor& uniqueOffset);

private:

    void UniqueProcessing(const torch::Tensor& hashIndices, const torch::Tensor& offset,
        const torch::Tensor& unique, const torch::Tensor& uniqueIds, const torch::Tensor& uniqueInverse,
        const torch::Tensor& uniqueOffset, int64_t tableId);
    std::tuple<at::Tensor, at::Tensor, at::Tensor> FindOrInsertHighPrecison(const torch::Tensor& global_ids);
    ska::flat_hash_map<int64_t, int64_t> ids2indicesMap;
    std::vector<int64_t> indice2id;

    int numThread;
    std::queue<std::unique_ptr<std::vector<int64_t>>> fullHashMapQue;

    int64_t maxIndex = 0;
    int64_t initMaxIndex;
    bool onlyDeviceMem = true;  // 是否仅使用device memory

    std::mutex insertMute;
    std::mutex allocMute;
    // Profiling Record
    at::ThreadLocalState state;
};
}  // namespace hybrid
#endif