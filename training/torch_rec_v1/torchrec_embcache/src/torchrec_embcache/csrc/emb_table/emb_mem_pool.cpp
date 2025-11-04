/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
        if (bufferBin_.pop(value)) {
            GetEmbMemoryPoolThreadPool().enqueue([this] { Produce(); });
            return BeforePutFuncState::BEFORE_SUCCESS;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    LOG_ERROR("Failed to get new address for embedding, it is likely due to refill thread "
              "memory allocation failure or max retry has been reached. "
              "Please check for memory alloc error or increase refill thread num!");
    return BeforePutFuncState::BEFORE_FAIL;
}

void EmbMemoryPool::GetValueToBeRecycled(uint64_t value)
{
    recycleBin_.push(value);
}

bool EmbMemoryPool::GetNewAddr(uint64_t& newAddr)
{
    std::lock_guard<std::mutex> lg(getAddrMutex_);
    if (HM_UNLIKELY(currentMemoryUint_.leftCapacity <= 0)) {
        /* need to expand memory */
        uint64_t maxSize = std::min(maxExpandSize_, totalLeftVocabSize_ * itemSize_);
        uint64_t newSize = currentMemoryUint_.capacity
                               ? std::min(currentMemoryUint_.capacity * dynamicExpandRatio_, maxSize)
                               : itemSize_;
        if (newSize == 0) {  // 所有hostVocabSize均已分配
            return false;
        }
        auto newAddress = reinterpret_cast<uint64_t>(malloc(newSize));
        if (newAddress == 0) {
            LOG_ERROR("Refill thread allocate memory failed!");
            return false;
        }
        expandedMemory_.emplace_back(newAddress, newSize);
        currentMemoryUint_.address = newAddress;
        currentMemoryUint_.capacity = newSize;
        currentMemoryUint_.leftCapacity = newSize;
        totalLeftVocabSize_ -= newSize / itemSize_;
    }
    newAddr = currentMemoryUint_.address + currentMemoryUint_.capacity - currentMemoryUint_.leftCapacity;
    currentMemoryUint_.leftCapacity -= itemSize_;
    return true;
}

void EmbMemoryPool::Produce()
{
    uint64_t newAddr;
    if (!recycleBin_.pop(newAddr) && !GetNewAddr(newAddr)) {
        return;
    }

    // init embedding
    if (embConfig_.initializerRandomPoolSize == -1) {
        Initializer::InitEmbeddingWeights(reinterpret_cast<float*>(newAddr), embConfig_);
    } else {
        Initializer::InitEmbeddingWeightsLimitPool(reinterpret_cast<float*>(newAddr), embConfig_);
    }

    // init optimizer
    auto ret = memset_s(reinterpret_cast<float*>(newAddr) + embConfig_.embDim,
                        embConfig_.optimNum * embConfig_.embDim * sizeof(float), 0,
                        embConfig_.optimNum * embConfig_.embDim * sizeof(float));
    if (ret != EOK) {
        throw std::runtime_error("memset_s failed when init optimizer data.");
    }

    bufferBin_.push(newAddr);
}

}  // namespace Embcache