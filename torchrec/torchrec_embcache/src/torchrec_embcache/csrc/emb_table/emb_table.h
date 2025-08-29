/*
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

#include <torch/torch.h>
#include "securec.h"

#include "common/common.h"
#include "common/constants.h"
#include "emb_memory_pool.h"
#include "hash_table/fast_hashmap.h"
#include "initializer.h"
#include "utils/logger.h"

namespace Embcache {
constexpr long long EMB_SIZE_MAX = 1e9L;

class EmbTable {
public:
    explicit EmbTable(const EmbConfig& embConfig)
        : config_(embConfig),
          extEmbDim_((1 + embConfig.optimNum) * embConfig.embDim)
    {
    }

    virtual ~EmbTable() = default;

    virtual void FindOrInsert(const std::vector<int64_t>& keys, float* outEmbs, std::vector<float*> outOptims) = 0;
    virtual void InsertOrAssign(const std::vector<int64_t>& keys, float* inEmbs, std::vector<float*> inOptims) = 0;
    virtual void RemoveEmbedding(const std::vector<int64_t>& keys) = 0;
    virtual void ForEachKey(const std::function<void(const int64_t, const float*)>& callback) = 0;

protected:
    EmbConfig config_;
    int32_t extEmbDim_;  // embDim + OptimNum * embDim
};

class EmbTableUnorderedMap : public EmbTable {
public:
    explicit EmbTableUnorderedMap(const EmbConfig& embConfig) : EmbTable(embConfig) {}

    void FindOrInsert(const std::vector<int64_t>& keys, float* outEmbs, std::vector<float*> outOptims) override
    {
        std::lock_guard<std::mutex> lk(mtx_);
        auto embDim = config_.embDim;
        auto optimNum = config_.optimNum;
        if (outOptims.size() != optimNum) {
            LOG_ERROR("outOptims size {} is not equal to optimNum {}", outOptims.size(), optimNum);
            throw std::runtime_error("outOptims size is not equal to optimNum");
        }
        for (uint64_t i = 0; i < keys.size(); i++) {
            auto key = keys[i];
            auto it = table_.find(key);
            if (it == table_.end()) {
                auto res = table_.emplace(key, extEmbDim_);
                it = res.first;
                Initializer::InitEmbeddingWeights(it->second.data(), config_);
            }
            auto& emb = it->second;

            size_t size = embDim * sizeof(float);
            if (outEmbs == nullptr) {
                LOG_ERROR("outEmbs is nullptr");
                throw std::runtime_error("outEmbs is nullptr");
            }
            auto rc = memcpy_s(outEmbs + i * embDim, size, emb.data(), size);
            if (rc != 0) {
                LOG_ERROR("memcpy_s emb to outEmbs[{}] failed. ret: {}", i, rc);
                throw std::runtime_error("memcpy_s emb to outEmbs failed.");
            }

            if (optimNum > 0) {
                if (outOptims[0] == nullptr) {
                    LOG_ERROR("outOptims[0] is nullptr");
                    throw std::runtime_error("outOptims[0] is nullptr");
                }
                rc = memcpy_s(outOptims[0] + i * embDim, size, emb.data() + embDim, size);
                if (rc != 0) {
                    LOG_ERROR("memcpy_s optim1 to outOptims[{}][0] failed. ret: {}", i, rc);
                    throw std::runtime_error("memcpy_s optim1 to outOptims[0] failed.");
                }
            }

            if (optimNum > 1) {
                if (outOptims[1] == nullptr) {
                    LOG_ERROR("outOptims[1] is nullptr");
                    throw std::runtime_error("outOptims[1] is nullptr");
                }
                rc = memcpy_s(outOptims[1] + i * embDim, size, emb.data() + optimNum * embDim, size);
                if (rc != 0) {
                    LOG_ERROR("memcpy_s optim2 to outOptims[{}][1] failed. ret: {}", i, rc);
                    throw std::runtime_error("memcpy_s optim2 to outOptims[1] failed.");
                }
            }
        }
    }

    void InsertOrAssign(const std::vector<int64_t>& keys, float* inEmbs, std::vector<float*> inOptims) override
    {
        std::lock_guard<std::mutex> lk(mtx_);
        auto embDim = config_.embDim;
        auto optimNum = config_.optimNum;
        for (uint64_t i = 0; i < keys.size(); i++) {
            auto key = keys[i];
            auto it = table_.find(key);
            if (it == table_.end()) {
                auto res = table_.emplace(key, extEmbDim_);
                it = res.first;
            }
            auto& emb = it->second;

            size_t size = embDim * sizeof(float);
            if (inEmbs == nullptr) {
                LOG_ERROR("inEmbs is nullptr");
                throw std::runtime_error("inEmbs is nullptr");
            }
            auto rc = memcpy_s(emb.data(), size, inEmbs + i * embDim, size);
            if (rc != 0) {
                LOG_ERROR("memcpy_s emb[{}] to table failed. ret: {}", i, rc);
                throw std::runtime_error("memcpy_s emb to table failed.");
            }

            if (optimNum > 0) {
                if (inOptims[0] == nullptr) {
                    LOG_ERROR("inOptims[0] is nullptr");
                    throw std::runtime_error("inOptims[0] is nullptr");
                }
                rc = memcpy_s(emb.data() + embDim, size, inOptims[0] + i * embDim, size);
                if (rc != 0) {
                    LOG_ERROR("memcpy_s optim1[{}] to table failed. ret: {}", i, rc);
                    throw std::runtime_error("memcpy_s optim1 to table failed.");
                }
            }

            if (optimNum > 1) {
                if (inOptims[1] == nullptr) {
                    LOG_ERROR("inOptims[1] is nullptr");
                    throw std::runtime_error("inOptims[1] is nullptr");
                }
                rc = memcpy_s(emb.data() + optimNum * embDim, size, inOptims[1] + i * embDim, size);
                if (rc != 0) {
                    LOG_ERROR("memcpy_s optim2[{}] to table failed. ret: {}", i, rc);
                    throw std::runtime_error("memcpy_s optim2 to table failed.");
                }
            }
        }
    }

    void RemoveEmbedding(const std::vector<int64_t>& keys) override
    {
        std::lock_guard<std::mutex> lk(mtx_);
        for (auto key : keys) {
            table_.erase(key);
        }
    }

    void ForEachKey(const std::function<void(const int64_t, const float*)>& callback) override
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (this->table_.size() > EMB_SIZE_MAX) {
            auto errMsg = Logger::Format("Emb size exceed limit, table:{}, max:{}.", config_.tableName, EMB_SIZE_MAX);
            throw std::runtime_error(errMsg);
        }
        for (const auto& [key, vec] : this->table_) {
            callback(key, vec.data());
        }
    }

private:
    std::unordered_map<int64_t, std::vector<float>> table_;
    std::mutex mtx_;
};

class EmbTableFastHashMap : public EmbTable {
public:
    explicit EmbTableFastHashMap(const EmbConfig& embConfig) : EmbTable(embConfig)
    {
        memPoolPtr_ = std::make_shared<EmbMemoryPool>(embConfig, EmbMemPoolConfigConstants::bufferSize,
                                                     EmbMemPoolConfigConstants::hostVocabSize);
        hostVocabSize_ = EmbMemPoolConfigConstants::hostVocabSize;

        fastHashMapPtr_ = std::make_shared<FastHashMap>();

        uint64_t fastHashMapReserveBucketNum = FAST_HASHMAP_RESERVE_BUCKET_NUM;
        char* fastHashMapReserveStr = getenv("FAST_HASHMAP_RESERVE_BUCKET_NUM");
        if (fastHashMapReserveStr) {
            char* endptr = nullptr;
            fastHashMapReserveBucketNum = strtoul(fastHashMapReserveStr, &endptr, 10);
            if (endptr == fastHashMapReserveStr || *endptr != '\0') {
                LOG_ERROR("env FAST_HASHMAP_RESERVE_BUCKET_NUM is not a valid number");
                throw std::runtime_error("env FAST_HASHMAP_RESERVE_BUCKET_NUM is not a valid number");
            }
        }
        fastHashMapPtr_->Init(fastHashMapReserveBucketNum);
        LOG_INFO("FAST_HASHMAP_RESERVE_BUCKET_NUM: {}", fastHashMapReserveBucketNum);
    }

    EmbTableFastHashMap(const EmbTableFastHashMap&) = delete;
    EmbTableFastHashMap& operator=(const EmbTableFastHashMap&) = delete;
    EmbTableFastHashMap(EmbTableFastHashMap&&) = delete;
    EmbTableFastHashMap& operator=(EmbTableFastHashMap&&) = delete;

    ~EmbTableFastHashMap() override
    {
        fastHashMapPtr_->Destroy();
    }

    void FindOrInsert(const std::vector<int64_t>& keys, float* outEmbs, std::vector<float*> outOptims) override
    {
        auto embDim = config_.embDim;
        auto optimNum = config_.optimNum;
        at::parallel_for(
            0, keys.size(), std::ceil(keys.size() * 1.0 / at::get_num_threads()), [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    const auto key = keys[i];
                    uint64_t addrValue = 0;

                    FkvState ret = fastHashMapPtr_->FindOrInsert(key, addrValue, [&]() {
                        const uint64_t currentSize = fastHashMapPtr_->GetCurrentSize();
                        if (HM_UNLIKELY(currentSize >= hostVocabSize_)) {
                            LOG_ERROR("No enough space at host, currentSize: {}, hostVocabSize: {}", currentSize,
                                      hostVocabSize_);
                            return BeforePutFuncState::BEFORE_NO_SPACE;
                        }
                        return memPoolPtr_->GetNewValueToBeInserted(addrValue);
                    });
                    if (ret == FkvState::FKV_FAIL) {
                        LOG_ERROR("fastHashMapPtr->FindOrInsert failed!");
                        throw std::runtime_error("fastHashMapPtr->FindOrInsert failed!");
                    }
                    if (ret == FkvState::FKV_BEFORE_PUT_FUNC_FAIL) {
                        LOG_ERROR("memory alloc failed!");
                        throw std::runtime_error("memory alloc failed!");
                    }

                    size_t size = embDim * sizeof(float);
                    if (outEmbs == nullptr) {
                        LOG_ERROR("outEmbs is nullptr");
                        throw std::runtime_error("outEmbs is nullptr");
                    }
                    auto rc = memcpy_s(outEmbs + i * embDim, size, reinterpret_cast<float*>(addrValue), size);
                    if (rc != 0) {
                        LOG_ERROR("memcpy_s emb[{}] to outEmbs failed. ret: {}", i, rc);
                        throw std::runtime_error("memcpy_s emb to outEmbs failed.");
                    }

                    if (optimNum > 0) {
                        if (outOptims[0] == nullptr) {
                            LOG_ERROR("outOptims[0] is nullptr");
                            throw std::runtime_error("outOptims[0] is nullptr");
                        }
                        rc = memcpy_s(outOptims[0] + i * embDim, size,
                                      reinterpret_cast<float*>(addrValue) + embDim, size);
                        if (rc != 0) {
                            LOG_ERROR("memcpy_s optim1[{}] to outOptims[0] failed. ret: {}", i, rc);
                            throw std::runtime_error("memcpy_s optim1 to outOptims[0] failed.");
                        }
                    }

                    if (optimNum > 1) {
                        if (outOptims[1] == nullptr) {
                            LOG_ERROR("outOptims[1] is nullptr");
                            throw std::runtime_error("outOptims[1] is nullptr");
                        }
                        rc = memcpy_s(outOptims[1] + i * embDim, size,
                                      reinterpret_cast<float*>(addrValue) + optimNum * embDim, size);
                        if (rc != 0) {
                            LOG_ERROR("memcpy_s optim2[{}] to outOptims[1] failed. ret: {}", i, rc);
                            throw std::runtime_error("memcpy_s optim2 to outOptims[1] failed.");
                        }
                    }
                }
            });
    }

    void InsertOrAssign(const std::vector<int64_t>& keys, float* inEmbs, std::vector<float*> inOptims) override
    {
        auto embDim = config_.embDim;
        auto optimNum = config_.optimNum;
        at::parallel_for(
            0, keys.size(), std::ceil(keys.size() * 1.0 / at::get_num_threads()), [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    const auto key = keys[i];
                    uint64_t addrValue = 0;

                    FkvState ret = fastHashMapPtr_->FindOrInsert(key, addrValue, [&]() {
                        const uint64_t currentSize = fastHashMapPtr_->GetCurrentSize();
                        if (HM_UNLIKELY(currentSize >= hostVocabSize_)) {
                            LOG_ERROR("No enough space at host, currentSize: {}, hostVocabSize: {}", currentSize,
                                      hostVocabSize_);
                            return BeforePutFuncState::BEFORE_NO_SPACE;
                        }
                        return memPoolPtr_->GetNewValueToBeInserted(addrValue);
                    });
                    if (ret == FkvState::FKV_FAIL) {
                        LOG_ERROR("fastHashMapPtr->InsertOrAssign failed!");
                        throw std::runtime_error("fastHashMapPtr->InsertOrAssign failed!");
                    }
                    if (ret == FkvState::FKV_BEFORE_PUT_FUNC_FAIL) {
                        LOG_ERROR("memory alloc failed!");
                        throw std::runtime_error("memory alloc failed!");
                    }

                    size_t size = embDim * sizeof(float);
                    auto rc = memcpy_s((float*)addrValue, size, inEmbs + i * embDim, size);
                    if (rc != 0) {
                        LOG_ERROR("memcpy_s emb[{}] to addrValue failed. ret: {}", i, rc);
                        throw std::runtime_error("memcpy_s emb to addrValue failed.");
                    }
                    if (optimNum > 0) {
                        rc = memcpy_s((float*)addrValue + embDim, size, inOptims[0] + i * embDim, size);
                        if (rc != 0) {
                            LOG_ERROR("memcpy_s optim1[{}] to addrValue failed. ret: {}", i, rc);
                            throw std::runtime_error("memcpy_s optim1 to addrValue failed.");
                        }
                    }
                    if (optimNum > 1) {
                        rc = memcpy_s((float*)addrValue + optimNum * embDim, size, inOptims[1] + i * embDim, size);
                        if (rc != 0) {
                            LOG_ERROR("memcpy_s optim2[{}] to addrValue failed. ret: {}", i, rc);
                            throw std::runtime_error("memcpy_s optim2 to addrValue failed.");
                        }
                    }
                }
            });
    }

    void RemoveEmbedding(const std::vector<int64_t>& keys) override
    {
        for (auto key : keys) {
            FkvState ret = fastHashMapPtr_->Remove(key, [&](uint64_t value) {
                memPoolPtr_->GetValueToBeRecycled(value);
                return BeforeRemoveFuncState::BEFORE_SUCCESS;
            });
            if (ret == FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL) {
                LOG_ERROR("remove embedding failed!");
                throw std::runtime_error("remove embedding failed!");
            }
        }
    }

    void ForEachKey(const std::function<void(const int64_t, const float*)>& callback) override
    {
        auto keyEmbList = this->fastHashMapPtr_->Export();
        if (keyEmbList.size() > EMB_SIZE_MAX) {
            auto errMsg = Logger::Format("Emb size exceed limit, table:{}, max:{}.", config_.tableName, EMB_SIZE_MAX);
            throw std::runtime_error(errMsg);
        }
        for (auto key : keyEmbList) {
            callback(key.first, (float*)key.second);
        }
    }

private:
    std::shared_ptr<EmbMemoryPool> memPoolPtr_;
    std::shared_ptr<FastHashMap> fastHashMapPtr_;
    uint64_t hostVocabSize_;
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_EMB_TABLE_H