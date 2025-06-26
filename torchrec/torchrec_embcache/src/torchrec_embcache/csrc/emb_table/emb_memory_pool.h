/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
    EmbMemoryPool(const EmbConfig& embConfig, uint64_t bufferSize, uint64_t hostVocabSize, uint32_t refillThreadNum)
        : embConfig(embConfig),
          maxBufferSize(bufferSize),
          totalLeftVocabSize(hostVocabSize),
          numThreads(refillThreadNum)
    {
        itemSize = (embConfig.optimNum + 1) * embConfig.embDim * sizeof(float);
        maxExpandSize = maxBufferSize * itemSize;
        char* poolSizeStr = getenv("EMB_MEMORY_POOL_SIZE");
        if (poolSizeStr) {
            embMemoryPoolSize = atoi(poolSizeStr);
        }
        LOG(WARNING) << "EmbMemoryPool embMemoryPoolSize:" << embMemoryPoolSize << " and numThreads:" << numThreads;
        for (uint32_t i = 0; i < numThreads; i++) {
            producerThreads.emplace_back([this] { ProducerWorker(); });
        }
    }

    ~EmbMemoryPool()
    {
        {
            // Dont' remove brackets of this code block, otherwise may cause dead lock in ProducerWorker.
            // To let producerThreads quit, we need:
            //   1. stop is true;
            //   2. Make sure all ProducerWorker thread run at wait(lock),
            //      this condition will satisfy when we acquire lock below;
            //   3. Release lock below (lock only valid in this code block);
            //   4. Notify all thread, producerThread will get lock then return from wait, then meet stop flag, return.
            stop = true;
            std::lock_guard<std::mutex> lock(producerMutex);
        }
        producerCv.notify_all();
        fullCv.notify_all();
        for (auto& t : producerThreads) {
            t.join();
        }
    }

    void Stop();

    BeforePutFuncState GetNewValueToBeInserted(uint64_t& value, uint32_t maxRetry = 1000);
    void GetValueToBeRecycled(uint64_t value);

private:
    bool GetNewAddr(uint64_t& newAddr);
    void Produce();
    void ProducerWorker();

private:
    std::vector<EmExpandMemUint> expandedMemory;
    EmbConfig embConfig;

private:
    uint64_t maxBufferSize;
    uint64_t totalLeftVocabSize;
    uint32_t numThreads;

    std::atomic<uint64_t> currBufferSize{0};
    volatile std::atomic<bool> stop = false;
    volatile std::atomic<bool> full = false;
    std::mutex producerMutex;
    std::mutex getAddrMutex;
    std::condition_variable producerCv;
    std::condition_variable fullCv;

    SafeQueue<uint64_t> BufferBin;
    SafeQueue<uint64_t> recycleBin;

    std::vector<std::thread> producerThreads;
    EmExpandMemUint currentMemoryUint{};
    uint64_t dynamicExpandRatio = 2;

    uint64_t maxExpandSize = 0;
    uint64_t itemSize;

    uint64_t embMemoryPoolSize = 102400;
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_EMB_MEMORY_POOL_H
