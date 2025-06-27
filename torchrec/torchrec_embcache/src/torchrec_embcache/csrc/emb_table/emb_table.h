/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_EMB_TABLE_H
#define EMBEDDING_CACHE_EMB_TABLE_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <unordered_map>

#include <glog/logging.h>
#include <torch/torch.h>

#include "common/common.h"
#include "common/constants.h"
#include "emb_memory_pool.h"
#include "hash_table/fast_hashmap.h"
#include "initializer.h"
#include "utils/string_tools.h"

constexpr int OPTIMIZER_SLOT_INDEX2 = 2;

namespace Embcache {

class EmbTable {
public:
    explicit EmbTable(const EmbConfig& embConfig)
        : embDim(embConfig.embDim),
          optimNum(embConfig.optimNum),
          weightInitMin(embConfig.weightInitMin),
          weightInitMax(embConfig.weightInitMax)
    {
        extEmbDim = (1 + optimNum) * embDim;
    }

    virtual ~EmbTable() = default;

    virtual void FindOrInsert(const std::vector<int64_t>& keys, float* outEmbs, std::vector<float*> outOptims) = 0;
    virtual void InsertOrAssign(const std::vector<int64_t>& keys, float* inEmbs, std::vector<float*> inOptims) = 0;
    virtual void RemoveEmbedding(const std::vector<int64_t>& keys) = 0;
    virtual void ForEachKey(const std::function<void(const int64_t, const float*)>& callback) = 0;

protected:
    int32_t embDim;
    int32_t extEmbDim;  // embDim + OptimNum * embDim
    int32_t optimNum;

    float weightInitMin;
    float weightInitMax;
};

class EmbTableUnorderedMap : public EmbTable {
public:
    explicit EmbTableUnorderedMap(const EmbConfig& embConfig) : EmbTable(embConfig) {}

    void FindOrInsert(const std::vector<int64_t>& keys, float* outEmbs, std::vector<float*> outOptims) override
    {
        std::lock_guard<std::mutex> lk(mtx);
        for (uint64_t i = 0; i < keys.size(); i++) {
            auto key = keys[i];
            auto it = table.find(key);
            if (it == table.end()) {
                auto res = table.emplace(key, extEmbDim);
                it = res.first;
                char* init_linear = getenv("INIT_LINEAR");
                if (init_linear) {
                    Initializer::GenLinear(it->second.data(), embDim, weightInitMin, weightInitMax);
                } else {
                    Initializer::GenUniform(it->second.data(), embDim, weightInitMin, weightInitMax);
                }
            }
            auto& emb = it->second;

            std::memcpy(outEmbs + i * embDim, emb.data(), embDim * sizeof(float));
            if (optimNum > 0) {
                std::memcpy(outOptims[0] + i * embDim, emb.data() + embDim, embDim * sizeof(float));
            }
            if (optimNum > 1) {
                std::memcpy(outOptims[1] + i * embDim, emb.data() + optimNum * embDim, embDim * sizeof(float));
            }
        }
    }

    void InsertOrAssign(const std::vector<int64_t>& keys, float* inEmbs, std::vector<float*> inOptims) override
    {
        std::lock_guard<std::mutex> lk(mtx);
        for (uint64_t i = 0; i < keys.size(); i++) {
            auto key = keys[i];
            auto it = table.find(key);
            if (it == table.end()) {
                auto res = table.emplace(key, extEmbDim);
                it = res.first;
            }
            auto& emb = it->second;

            std::memcpy(emb.data(), inEmbs + i * embDim, embDim * sizeof(float));
            if (optimNum > 0) {
                std::memcpy(emb.data() + embDim, inOptims[0] + i * embDim, embDim * sizeof(float));
            }

            if (optimNum > 1) {
                std::memcpy(emb.data() + optimNum * embDim, inOptims[1] + i * embDim, embDim * sizeof(float));
            }
        }
    }

    void RemoveEmbedding(const std::vector<int64_t>& keys) override
    {
        std::lock_guard<std::mutex> lk(mtx);
        for (auto key : keys) {
            table.erase(key);
        }
    }

    void ForEachKey(const std::function<void(const int64_t, const float*)>& callback) override
    {
        std::lock_guard<std::mutex> lk(mtx);

        for (const auto& [key, vec] : this->table) {
            callback(key, vec.data());
        }
    }

private:
    std::unordered_map<int64_t, std::vector<float>> table;
    std::mutex mtx;
};

class EmbTableFastHashMap : public EmbTable {
public:
    explicit EmbTableFastHashMap(const EmbConfig& embConfig) : EmbTable(embConfig)
    {
        uint64_t embMemoryPoolThreadNum = EmbMemPoolConfigConstants::refillThreadNum;
        char* threadNumStr = getenv("EMB_MEMORY_POOL_THREAD_NUM");
        if (threadNumStr) {
            embMemoryPoolThreadNum = atoi(threadNumStr);
        }
        memPoolPtr = std::make_shared<EmbMemoryPool>(embConfig, EmbMemPoolConfigConstants::bufferSize,
                                                     EmbMemPoolConfigConstants::hostVocabSize, embMemoryPoolThreadNum);
        hostVocabSize = EmbMemPoolConfigConstants::hostVocabSize;

        fastHashMapPtr = std::make_shared<FastHashMap>();

        uint64_t fastHashMapReserveBucketNum = FAST_HASHMAP_RESERVE_BUCKET_NUM;
        char* fastHashMapReserveStr = getenv("FAST_HASHMAP_RESERVE_BUCKET_NUM");
        if (fastHashMapReserveStr) {
            fastHashMapReserveBucketNum = atoi(fastHashMapReserveStr);
        }
        fastHashMapPtr->Init(fastHashMapReserveBucketNum);
        LOG(WARNING) << "FAST_HASHMAP_RESERVE_BUCKET_NUM:" << fastHashMapReserveBucketNum;
    }

    ~EmbTableFastHashMap() override
    {
        fastHashMapPtr->Destroy();
        memPoolPtr->Stop();
    }

    void FindOrInsert(const std::vector<int64_t>& keys, float* outEmbs, std::vector<float*> outOptims) override
    {
        at::parallel_for(
            0, keys.size(), std::ceil(keys.size() * 1.0 / at::get_num_threads()), [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    const auto key = keys[i];
                    uint64_t addrValue = 0;

                    FkvState ret = fastHashMapPtr->FindOrInsert(key, addrValue, [&]() {
                        const uint64_t currentSize = fastHashMapPtr->GetCurrentSize();
                        if (HM_UNLIKELY(currentSize >= hostVocabSize)) {
                            LOG(ERROR) << "No enough space at host, currentSize:" << currentSize
                                       << ", hostVocabSize:" << hostVocabSize;
                            return BeforePutFuncState::BEFORE_NO_SPACE;
                        }
                        return memPoolPtr->GetNewValueToBeInserted(addrValue);
                    });
                    if (ret == FkvState::FKV_FAIL) {
                        LOG(ERROR) << "fastHashMapPtr->FindOrInsert failed!";
                        throw std::runtime_error("fastHashMapPtr->FindOrInsert failed!");
                    }
                    if (ret == FkvState::FKV_BEFORE_PUT_FUNC_FAIL) {
                        LOG(ERROR) << "memory alloc failed!";
                        throw std::runtime_error("memory alloc failed!");
                    }

                    std::memcpy(outEmbs + i * embDim, (float*)addrValue, embDim * sizeof(float));
                    if (optimNum > 0) {
                        std::memcpy(outOptims[0] + i * embDim, (float*)addrValue + embDim, embDim * sizeof(float));
                    }
                    if (optimNum > 1) {
                        std::memcpy(outOptims[1] + i * embDim, (float*)addrValue + optimNum * embDim,
                                    embDim * sizeof(float));
                    }
                }
            });
    }

    void InsertOrAssign(const std::vector<int64_t>& keys, float* inEmbs, std::vector<float*> inOptims) override
    {
        at::parallel_for(
            0, keys.size(), std::ceil(keys.size() * 1.0 / at::get_num_threads()), [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    const auto key = keys[i];
                    uint64_t addrValue = 0;

                    FkvState ret = fastHashMapPtr->FindOrInsert(key, addrValue, [&]() {
                        const uint64_t currentSize = fastHashMapPtr->GetCurrentSize();
                        if (HM_UNLIKELY(currentSize >= hostVocabSize)) {
                            LOG(ERROR) << "No enough space at host, currentSize:" << currentSize
                                       << ", hostVocabSize:" << hostVocabSize;
                            return BeforePutFuncState::BEFORE_NO_SPACE;
                        }
                        return memPoolPtr->GetNewValueToBeInserted(addrValue);
                    });
                    if (ret == FkvState::FKV_FAIL) {
                        LOG(ERROR) << "fastHashMapPtr->InsertOrAssign failed!";
                        throw std::runtime_error("fastHashMapPtr->InsertOrAssign failed!");
                    }
                    if (ret == FkvState::FKV_BEFORE_PUT_FUNC_FAIL) {
                        LOG(ERROR) << "memory alloc failed!";
                        throw std::runtime_error("memory alloc failed!");
                    }

                    std::memcpy((float*)addrValue, inEmbs + i * embDim, embDim * sizeof(float));
                    if (optimNum > 0) {
                        std::memcpy((float*)addrValue + embDim, inOptims[0] + i * embDim, embDim * sizeof(float));
                    }
                    if (optimNum > 1) {
                        std::memcpy((float*)addrValue + OPTIMIZER_SLOT_INDEX2 * embDim, inOptims[1] + i * embDim,
                                    embDim * sizeof(float));
                    }
                }
            });
    }

    void RemoveEmbedding(const std::vector<int64_t>& keys) override
    {
        for (auto key : keys) {
            FkvState ret = fastHashMapPtr->Remove(key, [&](uint64_t value) {
                memPoolPtr->GetValueToBeRecycled(value);
                return BeforeRemoveFuncState::BEFORE_SUCCESS;
            });
            if (ret == FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL) {
                LOG(ERROR) << "remove embedding failed!";
                throw std::runtime_error("remove embedding failed!");
            }
        }
    }

    void ForEachKey(const std::function<void(const int64_t, const float*)>& callback) override
    {
        for (auto key : this->fastHashMapPtr->Export()) {
            callback(key.first, (float*)key.second);
        }
    }

private:
    std::shared_ptr<EmbMemoryPool> memPoolPtr;
    std::shared_ptr<FastHashMap> fastHashMapPtr;
    uint64_t hostVocabSize;
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_EMB_TABLE_H
