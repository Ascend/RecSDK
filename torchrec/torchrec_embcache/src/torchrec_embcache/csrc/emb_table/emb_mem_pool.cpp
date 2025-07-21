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

BeforePutFuncState EmbMemoryPool::GetNewValueToBeInserted(uint64_t& value, uint32_t maxRetry)
{
    for (uint32_t i = 0; i < maxRetry; i++) {
        if (BufferBin.pop(value)) {
            GetEmbMemoryPool().enqueue([this] { Produce(); });
            return BeforePutFuncState::BEFORE_SUCCESS;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    LOG(ERROR) << "Failed to get new address for embedding, it is likely due to refill thread "
               << "memory allocation failure or max retry has been reached. "
               << "Please check for memory alloc error or increase refill thread num!";
    return BeforePutFuncState::BEFORE_FAIL;
}

void EmbMemoryPool::GetValueToBeRecycled(uint64_t value)
{
    recycleBin.push(value);
}

bool EmbMemoryPool::GetNewAddr(uint64_t& newAddr)
{
    std::lock_guard<std::mutex> lg(getAddrMutex);
    if (HM_UNLIKELY(currentMemoryUint.leftCapacity <= 0)) {
        /* need to expand memory */
        uint64_t maxSize = std::min(maxExpandSize, totalLeftVocabSize * itemSize);
        uint64_t newSize =
            currentMemoryUint.capacity ? std::min(currentMemoryUint.capacity * dynamicExpandRatio, maxSize) : itemSize;
        if (newSize == 0) { // 所有hostVocabSize均已分配
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
    Initializer::InitEmbeddingWeights(reinterpret_cast<float*>(newAddr), embConfig);

    // init optimizer
    auto ret = memset_s(reinterpret_cast<float*>(newAddr) + embConfig.embDim,
                        embConfig.optimNum * embConfig.embDim * sizeof(float),
                        0,
                        embConfig.optimNum * embConfig.embDim * sizeof(float));
    if (ret != EOK) {
        throw std::runtime_error("memset_s failed when init optimizer data.");
    }

    BufferBin.push(newAddr);
}

}  // namespace Embcache