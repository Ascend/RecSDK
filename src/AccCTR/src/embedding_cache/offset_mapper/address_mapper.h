/* Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 ==============================================================================*/

#ifndef MXREC_FASTER_QUERY_H
#define MXREC_FASTER_QUERY_H

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <queue>
#include <thread>
#include <utility>

#include "embedding_cache.h"
#include "offset_mapper/mapper_base.h"
#include "securec.h"

namespace EmbCache {
using EmExpandMemUint = struct em_expand_memory_uint_ {
    uint64_t address = 0;
    uint64_t capacity = 0;
    uint64_t leftCapacity = 0;

    em_expand_memory_uint_() = default;

    em_expand_memory_uint_(uint64_t a, uint64_t c) : address(a), capacity(c), leftCapacity(c) {}
};

template <typename T>
class QWithLock {
public:
    bool pop(T& ele)
    {
        std::lock_guard<std::mutex> lk(mut);
        if (dataQ.empty()) {
            return false;
        }
        ele = dataQ.front();
        dataQ.pop();
        return true;
    }

    void push(const T& ele)
    {
        std::lock_guard<std::mutex> lk(mut);
        dataQ.push(ele);
    }

    uint64_t GetLength()
    {
        std::lock_guard<std::mutex> lk(mut);
        return dataQ.size();
    }

private:
    std::mutex mut;
    std::queue<T> dataQ;
};

class AutoRefillEmbeddingMemoryPool {
public:
    std::vector<EmExpandMemUint> expandedMemory;
    uint32_t extEmbeddingSize;
    std::vector<InitializerInfo> initializerInfos;

    AutoRefillEmbeddingMemoryPool(uint64_t bufferSize, std::vector<InitializerInfo> initInfos, uint32_t extEmbSize,
                                  uint64_t hostVocabSize, uint32_t refillThreadNum = 1)
        : extEmbeddingSize(extEmbSize),
          initializerInfos(std::move(initInfos)),
          maxBufferSize(bufferSize),
          totalLeftVocabSize(hostVocabSize),
          numThreads(refillThreadNum)
    {
        itemSize = extEmbeddingSize * sizeof(float);
        maxExpandSize = maxBufferSize * itemSize;
        for (uint32_t i = 0; i < numThreads; i++) {
            producerThreads.emplace_back([this] { ProducerWorker(); });
        }
    }

    ~AutoRefillEmbeddingMemoryPool()
    {
        stop = true;
        std::lock_guard<std::mutex> lock(producerMutex);
        producerCv.notify_all();
        fullCv.notify_all();
        for (auto& t : producerThreads) {
            t.join();
        }
    }

    void Stop()
    {
        stop = true;
        std::lock_guard<std::mutex> lock(producerMutex);
        producerCv.notify_all();
        fullCv.notify_all();
    }

    BeforePutFuncState GetNewValueToBeInserted(uint64_t& value, uint32_t maxRetry = 10000)
    {
        for (uint32_t i = 0; i < maxRetry; i++) {
            if (BufferBin.pop(value)) {
                producerCv.notify_one();
                return BeforePutFuncState::BEFORE_SUCCESS;
            };
            producerCv.notify_one();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ock::ExternalLogger::PrintLog(
            ock::LogLevel::ERROR,
            "Failed to get new address for embedding, it is likely due to refill thread memory allocation failure "
            "or max retry has been reached. Please check for memory alloc error or increase refill thread num!");
        return BeforePutFuncState::BEFORE_FAIL;
    }

    void GetValueToBeRecycled(uint64_t value)
    {
        std::lock_guard<std::mutex> lock(producerMutex);
        recycleBin.push(value);
        full = false;
        fullCv.notify_one();
    }

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
    QWithLock<uint64_t> BufferBin;
    QWithLock<uint64_t> recycleBin;
    std::vector<std::thread> producerThreads;
    EmExpandMemUint currentMemoryUint{};
    uint64_t dynamicExpandRatio = 2;
    uint64_t maxExpandSize;
    uint64_t itemSize;

    bool GetNewAddr(uint64_t& newAddr)
    {
        std::lock_guard<std::mutex> lg(getAddrMutex);
        if (HM_UNLIKELY(currentMemoryUint.leftCapacity <= 0)) {
            /* need to expand memory */
            uint64_t maxSize = std::min(maxExpandSize, totalLeftVocabSize * itemSize);
            uint64_t newSize = currentMemoryUint.capacity
                                   ? std::min(currentMemoryUint.capacity * dynamicExpandRatio, maxSize)
                                   : itemSize;
            if (newSize == 0) {
                if (recycleBin.GetLength() == 0) {
                    full = true;
                }
                return false;
            }
            auto newAddress = (uint64_t)malloc(newSize);
            if (newAddress == 0) {
                ock::ExternalLogger::PrintLog(ock::LogLevel::WARN, "Refill thread allocate memory failed!");
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

    void Produce()
    {
        uint64_t newAddr;
        if (!recycleBin.pop(newAddr)) {
            if (!GetNewAddr(newAddr)) {
                return;
            }
        }
        GenerateData(newAddr);
        BufferBin.push(newAddr);
    }

    void GenerateData(const uint64_t& addr)
    {
        auto* embAddr = reinterpret_cast<float*>(addr);
        for (const auto& initializerInfo : initializerInfos) {
            initializerInfo.initializer->GenerateData(embAddr, INVALID_EMB_SIZE);
        }
    }

    void ProducerWorker()
    {
        std::unique_lock<std::mutex> lock(producerMutex);
        while (!stop) {
            if (full) {
                fullCv.wait(lock);
                continue;
            }
            if (BufferBin.GetLength() < maxBufferSize) {
                Produce();
                continue;
            }
            producerCv.wait(lock);
        }
    }
};

class AddressMapper : public MapperBase {
public:
    AddressMapper() = default;

    ~AddressMapper() = default;

    bool Initialize(uint32_t reserve, uint32_t vocabSize, std::shared_ptr<AutoRefillEmbeddingMemoryPool> expendInfoPtr)
    {
        hostVocabSize = vocabSize;
        emExpendMemInfoPtr = expendInfoPtr;
        return MapperBase::Initialize(reserve);
    }

    void UnInitialize() override
    {
        emExpendMemInfoPtr->Stop();
        FreeExpandedMemory();
        MapperBase::UnInitialize();
    }

    FkvState Remove(uint64_t key)
    {
        return MapperBase::Remove(key, [&](uint64_t value) {
            emExpendMemInfoPtr->GetValueToBeRecycled(value);
            return BeforeRemoveFuncState::BEFORE_SUCCESS;
        });
    }

    FkvState FindAndPutIfNotFound(uint64_t key, uint64_t& value)
    {
        FkvState ret = MapperBase::FindAndPutIfNotFound(key, value, [&]() {
            if (HM_UNLIKELY(current_size.load() > hostVocabSize)) {
                ock::ExternalLogger::PrintLog(ock::LogLevel::ERROR, "host does not have enough space");
                return BeforePutFuncState::BEFORE_NO_SPACE;
            }
            return emExpendMemInfoPtr->GetNewValueToBeInserted(value);
        });
        if (ret == FkvState::FKV_FAIL) {
            ock::ExternalLogger::PrintLog(ock::LogLevel::ERROR, "FindAndPutIfNotFound failed!");
            return ret;
        }
        if (ret == FkvState::FKV_BEFORE_PUT_FUNC_FAIL) {
            ock::ExternalLogger::PrintLog(ock::LogLevel::ERROR, "malloc failed");
            return ret;
        }
        return ret;
    }

    // 如果多线程使用，严格保证传入的key线程间不会重复(unique key)，否则可能出现未定义结果
    FkvState FindAndRemoveIfFound(uint64_t key, const uint64_t startAddr)
    {
        return MapperBase::Remove(key, [&](uint64_t value) {
            uint64_t memSize = emExpendMemInfoPtr->extEmbeddingSize * sizeof(float);
            auto rc = memcpy_s(reinterpret_cast<void*>(startAddr), memSize, reinterpret_cast<void*>(value), memSize);
            if (rc != 0) {
                ock::ExternalLogger::PrintLog(ock::LogLevel::ERROR,
                                              "memcpy_s failed... dstSize: " + std::to_string(memSize));
                return BeforeRemoveFuncState::BEFORE_FAIL;
            }
            emExpendMemInfoPtr->GetValueToBeRecycled(value);
            return BeforeRemoveFuncState::BEFORE_SUCCESS;
        });
    }

    uint32_t GetUsage()
    {
        return MapperBase::current_size;
    }

private:
    void FreeExpandedMemory()
    {
        for (auto& memUint : emExpendMemInfoPtr->expandedMemory) {
            free(reinterpret_cast<float*>(memUint.address));
        }
    }

private:
    uint32_t hostVocabSize;
    std::shared_ptr<AutoRefillEmbeddingMemoryPool> emExpendMemInfoPtr;
};
}  // namespace EmbCache
#endif  // MXREC_FASTER_QUERY_H
