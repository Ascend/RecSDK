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

#include <glog/logging.h>

#include "common/common.h"
#include "utils/thread_pool.h"
#include "utils/safe_queue.h"

namespace Embcache {

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
        : embConfig(embConfig),
          maxBufferSize(bufferSize),
          totalLeftVocabSize(hostVocabSize)
    {
        itemSize = (embConfig.optimNum + 1) * embConfig.embDim * sizeof(float);
        maxExpandSize = maxBufferSize * itemSize;
        char* poolSizeStr = getenv("EMB_MEMORY_POOL_SIZE");
        if (poolSizeStr) {
            embMemoryPoolSize = atoi(poolSizeStr);
        }
        LOG(WARNING) << "EmbMemoryPool embMemoryPoolSize:" << embMemoryPoolSize;
        for (int i = 0; i < embMemoryPoolSize; i++) {
            Produce();
        }
    }

    EmbMemoryPool(const EmbMemoryPool& pool) = delete;

    EmbMemoryPool& operator=(const EmbMemoryPool& pool) = delete;

    ~EmbMemoryPool()
    {
        for (const auto& memUint : expandedMemory) {
            free(reinterpret_cast<void*>(memUint.address));
            }
    }

    BeforePutFuncState GetNewValueToBeInserted(uint64_t& value, uint32_t maxRetry = 1000);
    void GetValueToBeRecycled(uint64_t value);

private:
    bool GetNewAddr(uint64_t& newAddr);
    void Produce();

private:
    std::vector<EmExpandMemUint> expandedMemory;
    EmbConfig embConfig;

private:
    uint64_t maxBufferSize;
    uint64_t totalLeftVocabSize;

    std::mutex getAddrMutex;

    SafeQueue<uint64_t> BufferBin;
    SafeQueue<uint64_t> recycleBin;

    EmExpandMemUint currentMemoryUint{};
    uint64_t dynamicExpandRatio = 2;

    uint64_t maxExpandSize = 0;
    uint64_t itemSize;

    uint64_t embMemoryPoolSize = 102400;
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_EMB_MEMORY_POOL_H
