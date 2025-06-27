/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "emb_memory_pool.h"

#include <stdexcept>

#include "securec.h"

#include "initializer.h"

namespace Embcache {

void EmbMemoryPool::Stop()
{
    stop = true;
    std::lock_guard<std::mutex> lock(producerMutex);
    producerCv.notify_all();
    fullCv.notify_all();
}

BeforePutFuncState EmbMemoryPool::GetNewValueToBeInserted(uint64_t& value, uint32_t maxRetry)
{
    for (uint32_t i = 0; i < maxRetry; i++) {
        if (BufferBin.pop(value)) {
            producerCv.notify_one();
            return BeforePutFuncState::BEFORE_SUCCESS;
        };
        producerCv.notify_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    LOG(ERROR) << "Failed to get new address for embedding, it is likely due to refill thread "
               << "memory allocation failure or max retry has been reached. "
               << "Please check for memory alloc error or increase refill thread num!";
    return BeforePutFuncState::BEFORE_FAIL;
}

void EmbMemoryPool::GetValueToBeRecycled(uint64_t value)
{
    std::lock_guard<std::mutex> lock(producerMutex);
    recycleBin.push(value);
    full = false;
    fullCv.notify_one();
}

bool EmbMemoryPool::GetNewAddr(uint64_t& newAddr)
{
    std::lock_guard<std::mutex> lg(getAddrMutex);
    if (HM_UNLIKELY(currentMemoryUint.leftCapacity <= 0)) {
        /* need to expand memory */
        uint64_t maxSize = std::min(maxExpandSize, totalLeftVocabSize * itemSize);
        uint64_t newSize =
            currentMemoryUint.capacity ? std::min(currentMemoryUint.capacity * dynamicExpandRatio, maxSize) : itemSize;
        if (newSize == 0) {
            if (recycleBin.GetLength() == 0) {
                full = true;
            }
            return false;
        }
        auto newAddress = reinterpret_cast<uint64_t>(malloc(newSize));
        if (newAddress == 0) {
            LOG(WARNING) << "Refill thread allocate memory failed!";
            return false;
        }
        expandedMemory.emplace_back(newAddress, newSize);
        currentMemoryUint.address = newAddress;
        currentMemoryUint.capacity = newSize;
        currentMemoryUint.leftCapacity = newSize;
        totalLeftVocabSize -= newSize / itemSize;
    }
    newAddr = currentMemoryUint.address + currentMemoryUint.capacity - currentMemoryUint.leftCapacity;
    currentMemoryUint.leftCapacity -= itemSize;
    return true;
}

void EmbMemoryPool::Produce()
{
    uint64_t newAddr;
    if (!recycleBin.pop(newAddr) && !GetNewAddr(newAddr)) {
        return;
    }

    // init embedding
    char* initLinear = getenv("INIT_LINEAR");
    if (initLinear) {
        Initializer::GenLinear((float*)newAddr, embConfig.embDim, embConfig.weightInitMin, embConfig.weightInitMax);
    } else {
        Initializer::GenUniform((float*)newAddr, embConfig.embDim, embConfig.weightInitMin, embConfig.weightInitMax);
    }

    // init optimizer
    auto ret = memset_s((float*)newAddr + embConfig.embDim, embConfig.optimNum * embConfig.embDim * sizeof(float), 0,
                        embConfig.optimNum * embConfig.embDim * sizeof(float));
    if (ret != EOK) {
        throw std::runtime_error("memset_s failed when init optimizer data.");
    }

    BufferBin.push(newAddr);
}

void EmbMemoryPool::ProducerWorker()
{
    std::unique_lock<std::mutex> lock(producerMutex);
    while (!stop) {
        if (full) {
            fullCv.wait(lock);
            continue;
        }
        if (BufferBin.GetLength() < embMemoryPoolSize) {
            Produce();
            continue;
        }
        producerCv.wait(lock);
    }
}

}  // namespace Embcache