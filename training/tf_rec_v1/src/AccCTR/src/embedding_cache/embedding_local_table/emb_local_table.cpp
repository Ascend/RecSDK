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

#include "emb_local_table.h"

#include <algorithm>
#include <thread>

#include "error_code.h"
#include "securec.h"

using namespace std;
using namespace EmbCache;
using namespace ock;
using namespace ock::ctr;

bool EmbLocalTable::Initialize(const EmbCacheInfo& embCacheInfo, uint64_t reserve,
                               const std::vector<InitializerInfo>& initializerInfos, const EmbPoolParam& embPoolParam)
{
    emExpendMemInfo = make_shared<AutoRefillEmbeddingMemoryPool>(embPoolParam.prefillBufferSize, initializerInfos,
                                                                 embCacheInfo.extEmbeddingSize, embCacheInfo.vocabSize,
                                                                 embPoolParam.refillThreadNum);
    embeddingSize = embCacheInfo.embeddingSize;
    extEmbeddingSize = embCacheInfo.extEmbeddingSize;
    return embMap.Initialize(reserve, embCacheInfo.vocabSize, emExpendMemInfo);
}

void EmbLocalTable::UnInitialize()
{
    embMap.UnInitialize();
}

int EmbLocalTable::FindAndPutIfNotFound(uint64_t key, uint64_t& value)
{
    FkvState ret = embMap.FindAndPutIfNotFound(key, value);
    if (ret == FkvState::FKV_FAIL) {
        return H_ERROR;
    }
    if (ret == FkvState::FKV_BEFORE_PUT_FUNC_FAIL) {
        return H_MEMORY_ALLOC_ERROR;
    }
    if (ret == FkvState::FKV_NO_SPACE) {
        return H_HOST_VOCAB_SIZE_TOO_SMALL;
    }
    return H_OK;
}

bool EmbLocalTable::Remove(uint64_t key)
{
    return embMap.Remove(key) != FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL;
}

int EmbLocalTable::RemoveByKeys(const std::vector<uint64_t>& keys, uint32_t threadNum)
{
    if (threadNum == 1) {
        for (uint64_t key : keys) {
            if (!Remove(key)) {
                return H_ERROR;
            }
        }
        return H_OK;
    }
    // 每个线程处理[start[threadId],start[threadId+1])这个区间的key
    uint32_t m = keys.size() % threadNum;
    vector<uint64_t> start(threadNum + 1);
    // 前keys.size()%threadNum个线程向上取整
    for (uint32_t threadId = 0; threadId < m; threadId++) {
        start[threadId] = ((keys.size() + threadNum - 1) / threadNum) * threadId;
    }
    // 后面的向下取整
    for (uint32_t threadId = m; threadId <= threadNum; threadId++) {
        start[threadId] = (keys.size() / threadNum) * threadId + m;
    }

    vector<future<ock::ctr::CTRCode>> threads(threadNum);
    for (uint32_t threadId = 0; threadId < threadNum; threadId++) {
        threads[threadId] = std::async(std::launch::async, [&, threadId]() {
            for (uint64_t i = start[threadId]; i < start[threadId + 1]; i++) {
                if (!Remove(keys[i])) {
                    return H_ERROR;
                }
            }
            return H_OK;
        });
    }
    for (auto& t : threads) {
        auto res = t.get();
        if (res != H_OK) {
            return res;
        }
    }
    return H_OK;
}

int EmbLocalTable::OneThreadHandle(uint64_t startAddr, const std::vector<uint64_t>& keys, bool isGather)
{
    for (uint64_t i = 0; i < keys.size(); i++) {
        uint64_t embAddr;
        int ret = FindAndPutIfNotFound(keys[i], embAddr);
        if (ret != H_OK) {
            return ret;
        }
        uint64_t memSize = emExpendMemInfo->extEmbeddingSize * sizeof(float);
        auto addr = startAddr + i * memSize;
        if (isGather) {
            auto rc = memcpy_s(reinterpret_cast<void*>(addr), memSize, reinterpret_cast<void*>(embAddr), memSize);
            if (rc != 0) {
                ExternalLogger::PrintLog(LogLevel::ERROR,
                                         "gather memcpy_s failed... dstSize: " + std::to_string(memSize));
                return H_COPY_ERROR;
            }
        } else {
            auto rc = memcpy_s(reinterpret_cast<void*>(embAddr), memSize,  // 按顺序把新的embedding拷贝到对应地址中
                               reinterpret_cast<void*>(addr), memSize);
            if (rc != 0) {
                ExternalLogger::PrintLog(LogLevel::ERROR,
                                         "scatter memcpy_s failed... dstSize: " + std::to_string(memSize));
                return H_COPY_ERROR;
            }
        }
    }

    return H_OK;
}

int EmbLocalTable::Gather(uint64_t startAddr, const vector<uint64_t>& keys, uint32_t threadNum)
{
    if (threadNum == 1) {
        return OneThreadHandle(startAddr, keys, true);
    }

    // 每个线程处理[start[threadId],start[threadId+1])这个区间的key
    uint32_t m = keys.size() % threadNum;
    vector<uint64_t> start(threadNum + 1);
    // 前keys.size()%threadNum个线程向上取整
    for (uint32_t threadId = 0; threadId < m; threadId++) {
        start[threadId] = ((keys.size() + threadNum - 1) / threadNum) * threadId;
    }
    // 后面的向下取整
    for (uint32_t threadId = m; threadId <= threadNum; threadId++) {
        start[threadId] = (keys.size() / threadNum) * threadId + m;
    }

    vector<thread> threads(threadNum);
    int ret = H_OK;
    for (uint32_t threadId = 0; threadId < threadNum; threadId++) {
        threads[threadId] = thread([&, threadId] {
            for (uint64_t i = start[threadId]; i < start[threadId + 1]; i++) {
                uint64_t embAddr;
                int temp_ret = FindAndPutIfNotFound(keys[i], embAddr);
                if (temp_ret != H_OK) {
                    ret = temp_ret;
                    return;
                }
                uint64_t memSize = emExpendMemInfo->extEmbeddingSize * sizeof(float);
                auto addr = startAddr + i * memSize;
                auto rc = memcpy_s(reinterpret_cast<void*>(addr), memSize, reinterpret_cast<void*>(embAddr), memSize);
                if (rc != 0) {
                    ExternalLogger::PrintLog(LogLevel::ERROR, "memcpy_s failed... dstSize: " + std::to_string(memSize));
                    ret = H_COPY_ERROR;
                    return;
                }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    return ret;
}

int EmbLocalTable::GatherAddrs(const std::vector<uint64_t>& keys, std::vector<float*>& addrs, uint32_t threadNum)
{
    if (threadNum == 1) {
        addrs.resize(keys.size());
        for (uint64_t i = 0; i < keys.size(); i++) {
            int temp_ret = FindAndPutIfNotFound(keys[i], reinterpret_cast<uint64_t&>(addrs[i]));
            if (temp_ret != H_OK) {
                return temp_ret;
            }
        }
        return H_OK;
    }
    // 每个线程处理[start[threadId],start[threadId+1])这个区间的key
    uint32_t m = keys.size() % threadNum;
    vector<uint64_t> start(threadNum + 1);
    // 前keys.size()%threadNum个线程向上取整
    for (uint32_t threadId = 0; threadId < m; threadId++) {
        start[threadId] = ((keys.size() + threadNum - 1) / threadNum) * threadId;
    }
    // 后面的向下取整
    for (uint32_t threadId = m; threadId <= threadNum; threadId++) {
        start[threadId] = (keys.size() / threadNum) * threadId + m;
    }
    addrs.resize(keys.size());

    vector<thread> threads(threadNum);
    int ret = H_OK;
    for (uint32_t threadId = 0; threadId < threadNum; threadId++) {
        threads[threadId] = thread([&, threadId] {
            for (uint64_t i = start[threadId]; i < start[threadId + 1]; i++) {
                int temp_ret = FindAndPutIfNotFound(keys[i], reinterpret_cast<uint64_t&>(addrs[i]));
                if (temp_ret != H_OK) {
                    ret = temp_ret;
                    return;
                }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    return ret;
}

// 如果多线程使用，严格保证传入的key线程间不会重复(unique key)，否则可能出现未定义结果
int EmbLocalTable::GatherAndRemove(uint64_t startAddr, const vector<uint64_t>& keys, uint32_t threadNum)
{
    if (threadNum == 1) {
        for (uint64_t i = 0; i < keys.size(); i++) {
            uint64_t memSize = emExpendMemInfo->extEmbeddingSize * sizeof(float);
            auto addr = startAddr + i * memSize;
            auto ret = embMap.FindAndRemoveIfFound(keys[i], addr);  // 如果找到了就拷贝出来然后把key删了
            if (ret == FkvState::FKV_NOT_EXIST) {  // 没找到key，给一个新的初始化值并且不需要存入key
                auto* embAddr = reinterpret_cast<float*>(addr);
                for (const auto& initializerInfo : emExpendMemInfo->initializerInfos) {
                    initializerInfo.initializer->GenerateData(embAddr, INVALID_EMB_SIZE);
                }
            } else if (ret == FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL) {
                ExternalLogger::PrintLog(LogLevel::ERROR, "memcpy_s failed... dstSize: " + std::to_string(memSize));
                return H_COPY_ERROR;
            }
        }
        return H_OK;
    }

    // 每个线程处理[start[threadId],start[threadId+1])这个区间的key
    uint32_t m = keys.size() % threadNum;
    vector<uint64_t> start(threadNum + 1);
    // 前keys.size()%threadNum个线程向上取整
    for (uint32_t threadId = 0; threadId < m; threadId++) {
        start[threadId] = ((keys.size() + threadNum - 1) / threadNum) * threadId;
    }
    // 后面的向下取整
    for (uint32_t threadId = m; threadId <= threadNum; threadId++) {
        start[threadId] = (keys.size() / threadNum) * threadId + m;
    }

    vector<thread> threads(threadNum);
    int retVal = H_OK;
    for (uint32_t threadId = 0; threadId < threadNum; threadId++) {
        threads[threadId] = thread([&, threadId] {
            for (uint64_t i = start[threadId]; i < start[threadId + 1]; i++) {
                uint64_t memSize = emExpendMemInfo->extEmbeddingSize * sizeof(float);
                auto addr = startAddr + i * memSize;
                auto ret = embMap.FindAndRemoveIfFound(keys[i], addr);  // 如果找到了就拷贝出来然后把key删了
                if (ret == FkvState::FKV_NOT_EXIST) {  // 没找到key，给一个新的初始化值并且不需要存入key
                    auto* embAddr = reinterpret_cast<float*>(addr);
                    for (const auto& initializerInfo : emExpendMemInfo->initializerInfos) {
                        initializerInfo.initializer->GenerateData(embAddr, INVALID_EMB_SIZE);
                    }
                } else if (ret == FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL) {
                    ExternalLogger::PrintLog(LogLevel::ERROR, "memcpy_s failed... dstSize: " + std::to_string(memSize));
                    retVal = H_COPY_ERROR;
                    return;
                }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    return retVal;
}

int EmbLocalTable::Scatter(const uint64_t startAddr, const vector<uint64_t>& keys, uint32_t threadNum)
{
    if (threadNum == 1) {  // 单线程版本
        return OneThreadHandle(startAddr, keys, false);
    }

    // 多线程版本
    // 每个线程处理[start[threadId],start[threadId+1])这个区间的key
    uint32_t m = keys.size() % threadNum;
    vector<uint64_t> start(threadNum + 1);
    // 前keys.size()%threadNum个线程向上取整
    for (uint32_t threadId = 0; threadId < m; threadId++) {
        start[threadId] = ((keys.size() + threadNum - 1) / threadNum) * threadId;
    }
    // 后面的向下取整
    for (uint32_t threadId = m; threadId <= threadNum; threadId++) {
        start[threadId] = (keys.size() / threadNum) * threadId + m;
    }

    vector<thread> threads(threadNum);
    int ret = H_OK;
    for (uint32_t threadId = 0; threadId < threadNum; threadId++) {
        threads[threadId] = thread([&, threadId] {
            for (uint64_t i = start[threadId]; i < start[threadId + 1]; i++) {
                uint64_t embAddr;
                int temp_ret = FindAndPutIfNotFound(keys[i], embAddr);  // 获取每个key的embedding对应首地址
                if (temp_ret != H_OK) {
                    ret = temp_ret;
                    return;
                }
                uint64_t memSize = emExpendMemInfo->extEmbeddingSize * sizeof(float);
                auto addr = startAddr + i * memSize;
                auto rc = memcpy_s(reinterpret_cast<void*>(embAddr), memSize,  // 按顺序把新的embedding拷贝到对应地址中
                                   reinterpret_cast<void*>(addr), memSize);
                if (rc != 0) {
                    ExternalLogger::PrintLog(LogLevel::ERROR, "memcpy_s failed... dstSize: " + std::to_string(memSize)
                                                                  + ", embAddr:" + std::to_string(embAddr)
                                                                  + ", rc:" + std::to_string(rc));
                    ret = H_COPY_ERROR;
                    return;
                }
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    return ret;
}

// 导出存储的所有kv对
vector<pair<uint64_t, uint64_t>> EmbLocalTable::ExportVec()
{
    return embMap.ExportVec();
}

template <class T>
void EmbLocalTable::insertData(vector<char>& buffer, T& data) const
{
    buffer.insert(buffer.end(), reinterpret_cast<char*>(&data), reinterpret_cast<char*>(&data) + sizeof(data));
}

template <class T>
bool EmbLocalTable::getData(const vector<char>& buffer, T& data, uint64_t& i) const
{
    if (i + sizeof(T) > buffer.size()) {
        return false;
    }
    data = *reinterpret_cast<const T*>(&buffer[i]);
    i += sizeof(T);
    return true;
}

// 把所存储的key-embedding信息序列化
vector<char> EmbLocalTable::Serialize()
{
    vector<char> buffer;
    vector<pair<uint64_t, uint64_t>> kvVec = ExportVec();

    for (auto& p : kvVec) {
        uint64_t key = p.first;
        uint64_t value = p.second;
        insertData(buffer, key);
        auto* addr = reinterpret_cast<float*>(value);
        buffer.insert(buffer.end(), reinterpret_cast<char*>(addr),
                      reinterpret_cast<char*>((addr + emExpendMemInfo->extEmbeddingSize)));
    }
    return buffer;
}

// 反序列化key-embedding，存进map
bool EmbLocalTable::Deserialize(const vector<char>& buffer)
{
    uint64_t i = 0;
    while (i < buffer.size()) {
        uint64_t key;
        if (!getData(buffer, key, i)) {
            ExternalLogger::PrintLog(LogLevel::ERROR, "get data failed!");
            return false;
        }
        uint64_t value = 0;
        if (FindAndPutIfNotFound(key, value) != H_OK) {
            ExternalLogger::PrintLog(LogLevel::ERROR, "FindAndPutIfNotFound failed!");
            return false;
        }

        auto* addr = reinterpret_cast<float*>(value);
        for (uint32_t j = 0; j < emExpendMemInfo->extEmbeddingSize; j++) {
            if (!getData(buffer, addr[j], i)) {
                ExternalLogger::PrintLog(LogLevel::ERROR, "get data failed!");
                return false;
            }
        }
    }
    return true;
}

uint32_t EmbLocalTable::GetUsage()
{
    return embMap.currentSize;
}

void EmbLocalTable::GetEmbTableInfos(std::vector<uint64_t>& keys, std::vector<std::vector<float>>& embeddings,
                                     std::vector<std::vector<float>>& optimizerSlots)
{
    vector<pair<uint64_t, uint64_t>> kvVec = ExportVec();

    for (auto& p : kvVec) {
        std::vector<float> curEmbedding;
        auto* addr = reinterpret_cast<float*>(p.second);
        if (addr == nullptr) {
            ExternalLogger::PrintLog(
                LogLevel::ERROR,
                "The new key:" + std::to_string(p.first) + " is being inserted in parallel. Addr is null!");
            throw std::runtime_error("GetEmbTableInfos fail. Addr is null");
        }
        keys.emplace_back(p.first);
        curEmbedding.insert(curEmbedding.end(), addr, reinterpret_cast<float*>((addr + embeddingSize)));
        embeddings.emplace_back(curEmbedding);
        if (extEmbeddingSize > embeddingSize) {
            std::vector<float> curOptimizerSlot;
            curOptimizerSlot.insert(curOptimizerSlot.end(), reinterpret_cast<float*>(addr + embeddingSize),
                                    reinterpret_cast<float*>((addr + extEmbeddingSize)));
            optimizerSlots.emplace_back(curOptimizerSlot);
        }
    }
}

bool EmbLocalTable::LoadEmbTableInfos(const std::vector<uint64_t>& keys,
                                      const std::vector<std::vector<float>>& embeddings,
                                      const std::vector<std::vector<float>>& optimizerSlots)
{
    if (keys.size() != embeddings.size()) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "the size of keys and embeddings should be same!");
        return false;
    }
    uint32_t optimizerSlotSize = extEmbeddingSize - embeddingSize;
    if (optimizerSlotSize > 0) {
        if (keys.size() != optimizerSlots.size()) {
            ExternalLogger::PrintLog(LogLevel::ERROR, "the size of keys and optimizerSlots should be same!");
            return false;
        }
    }
    for (uint64_t i = 0; i < keys.size(); i++) {
        uint64_t value = 0;
        if (FindAndPutIfNotFound(keys[i], value) != H_OK) {
            ExternalLogger::PrintLog(
                LogLevel::ERROR,
                "FindAndPutIfNotFound failed! Table:" + name + ", Try put keys' quantity:" +
                std::to_string(keys.size()) + ", already put quantity:" + std::to_string(i) + ".");
            return false;
        }
        if (embeddings[i].size() != embeddingSize) {
            ExternalLogger::PrintLog(LogLevel::ERROR,
                                     "The size of entering Embedding does not equals to embeddingSize");
            return false;
        }
        auto* addr = reinterpret_cast<float*>(value);
        if (addr == nullptr) {
            ExternalLogger::PrintLog(LogLevel::ERROR, "addr is nullptr.");
            return false;
        }
        auto rc = memcpy_s(addr, embeddingSize * sizeof(float), embeddings[i].data(), embeddingSize * sizeof(float));
        if (rc != 0) {
            ExternalLogger::PrintLog(LogLevel::ERROR, "embedding memcpy_s failed... ");
            return false;
        }
        if (optimizerSlotSize > 0) {
            if (optimizerSlots[i].size() != optimizerSlotSize) {
                ExternalLogger::PrintLog(
                    LogLevel::ERROR,
                    "The size of entering optimizerSlot does not equals to extEmbeddingSize - embeddingSize");
                return false;
            }
            auto rc2 = memcpy_s(reinterpret_cast<float*>(addr + embeddingSize), optimizerSlotSize * sizeof(float),
                                optimizerSlots[i].data(), optimizerSlotSize * sizeof(float));
            if (rc2 != 0) {
                ExternalLogger::PrintLog(LogLevel::ERROR, "optimizerSlot memcpy_s failed... ");
                return false;
            }
        }
    }
    return true;
}