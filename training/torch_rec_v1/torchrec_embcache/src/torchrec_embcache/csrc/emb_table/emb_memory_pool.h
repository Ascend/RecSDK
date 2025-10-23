/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_EMB_MEMORY_POOL_H
#define EMBEDDING_CACHE_EMB_MEMORY_POOL_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "common/common.h"
#include "utils/thread_pool.h"
#include "utils/safe_queue.h"
#include "utils/logger.h"

namespace Embcache {

constexpr uint32_t MAX_EMB_MEMORY_POOL_SIZE = 200000;
using EmExpandMemUint = struct EmExpandMemoryUint {
    uint64_t address = 0;
    uint64_t capacity = 0;
    uint64_t leftCapacity = 0;

    EmExpandMemoryUint() = default;

    EmExpandMemoryUint(uint64_t a, uint64_t c) : address(a), capacity(c), leftCapacity(c) {}
};

class EmbMemoryPool {
public:
    EmbMemoryPool(const EmbConfig& embConfig, uint64_t bufferSize, uint64_t hostVocabSize)
        : embConfig_(embConfig),
          maxBufferSize_(bufferSize),
          totalLeftVocabSize_(hostVocabSize)
    {
        itemSize_ = (embConfig.optimNum + 1) * embConfig.embDim * sizeof(float);
        maxExpandSize_ = maxBufferSize_ * itemSize_;
        char* poolSizeStr = getenv("EMB_MEMORY_POOL_SIZE");
        if (poolSizeStr) {
            char* endptr = nullptr;
            embMemoryPoolSize_ = strtoul(poolSizeStr, &endptr, 10);
            if (endptr == poolSizeStr || *endptr != '\0') {
                LOG_ERROR("env EMB_MEMORY_POOL_SIZE is not a valid number");
                throw std::runtime_error("env EMB_MEMORY_POOL_SIZE is not a valid number");
            }
            if (embMemoryPoolSize_ == 0 || embMemoryPoolSize_ > MAX_EMB_MEMORY_POOL_SIZE) {
                LOG_ERROR("env EMB_MEMORY_POOL_SIZE {} should be in range (0, {}]",
                          embMemoryPoolSize_, MAX_EMB_MEMORY_POOL_SIZE);
                throw std::runtime_error("env EMB_MEMORY_POOL_SIZE is invalid");
            }
        }
        LOG_INFO("EmbMemoryPool embMemoryPoolSize: {}", embMemoryPoolSize_);
        for (uint64_t i = 0; i < embMemoryPoolSize_; i++) {
            Produce();
        }
    }

    EmbMemoryPool(const EmbMemoryPool& pool) = delete;

    EmbMemoryPool& operator=(const EmbMemoryPool& pool) = delete;

    EmbMemoryPool(EmbMemoryPool&& pool) = delete;

    EmbMemoryPool& operator=(EmbMemoryPool&& pool) = delete;

    ~EmbMemoryPool()
    {
        for (const auto& memUint : expandedMemory_) {
            free(reinterpret_cast<void*>(memUint.address));
        }
    }

    BeforePutFuncState GetNewValueToBeInserted(uint64_t& value, uint32_t maxRetry = 1000);
    void GetValueToBeRecycled(uint64_t value);

private:
    bool GetNewAddr(uint64_t& newAddr);
    void Produce();

private:
    std::vector<EmExpandMemUint> expandedMemory_;
    EmbConfig embConfig_;

private:
    uint64_t maxBufferSize_;
    uint64_t totalLeftVocabSize_;

    std::mutex getAddrMutex_;

    SafeQueue<uint64_t> bufferBin_;
    SafeQueue<uint64_t> recycleBin_;

    EmExpandMemUint currentMemoryUint_{};
    uint64_t dynamicExpandRatio_ = 2;

    uint64_t maxExpandSize_ = 0;
    uint64_t itemSize_;

    uint64_t embMemoryPoolSize_ = 102400;
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_EMB_MEMORY_POOL_H
