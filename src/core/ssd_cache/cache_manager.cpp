/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "cache_manager.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

#include "utils/common.h"
#include "utils/time_cost.h"

using namespace MxRec;

void CacheManager::Init(ock::ctr::EmbCacheManagerPtr embCachePtr, vector<EmbInfo>& mgmtEmbInfo)
{
    LOG_INFO("CacheManager Init method begin");
    this->embCache = std::move(embCachePtr);
    for (auto& emb : mgmtEmbInfo) {
        EmbBaseInfo baseInfo {emb.ssdVocabSize, emb.ssdDataPath, false};
        embBaseInfos.emplace(emb.name, baseInfo);
        preProcessMapper[emb.name].Initialize(emb.name, emb.hostVocabSize, emb.ssdVocabSize);
    }
    ssdEngine->Start();
    LOG_INFO("CacheManager Init method end");
}

bool CacheManager::IsKeyInSSD(const string& embTableName, emb_cache_key_t key)
{
    return ssdEngine->IsKeyExist(embTableName, key);
}

/// 淘汰SSD中Emb信息
/// \param embTableName emb表名
/// \param keys 淘汰key列表
void CacheManager::EvictSSDEmbedding(const string& embTableName, const vector<emb_cache_key_t>& keys)
{
    if (keys.empty()) {
        return;
    }

    int keyStep = preProcessStep;
    unordered_map<emb_cache_key_t, freq_num_t>& ssdMap = preProcessMapper[embTableName].excludeDDRKeyCountMap;
    LFUCache& ddrLfu = preProcessMapper[embTableName].lfuCache;
    std::vector<emb_cache_key_t> ssdKeysToBeDeleted;
    // 1 删除缓存中记录的key的次数
    for (auto &key: keys) {
        auto it = ssdMap.find(key);
        if (it != ssdMap.end()) {
            ssdMap.erase(it);
            ssdKeysToBeDeleted.emplace_back(key);
        } else {
            ddrLfu.Pop(key);
        }
    }

    ssdEvictThreads.emplace_back([=]() mutable {
        // 2 删除SSD中保存的Emb数据
        std::unique_lock<std::mutex> lk(evictWaitMut);
        evictWaitCond.wait(lk, [keyStep, this] {
            return embeddingTaskStep == keyStep;
        });
        ssdEngine->DeleteEmbeddings(embTableName, ssdKeysToBeDeleted);
    });
}

/// 放入key，新增/更新(次数+1)次数
/// \param embTableName emb表名
/// \param key key
/// \param type 记录类型
void CacheManager::PutKey(const string& embTableName, const emb_key_t& key, RecordType type)
{
    if (type == RecordType::DDR) {
        ddrKeyFreqMap[embTableName].Put(key);
        return;
    }
    auto& hashMap = excludeDDRKeyCountMap[embTableName];
    const auto& it = hashMap.find(key);
    freq_num_t count = it == hashMap.end() ? 1 : it->second + 1;
    hashMap[key] = count;
}

void CacheManager::CreateSSDTableIfNotExist(const std::string& embTableName)
{
    if (embBaseInfos[embTableName].isExist) {
        return;
    }
    if (!ssdEngine->IsTableExist(embTableName)) {
        ssdEngine->CreateTable(embTableName, embBaseInfos[embTableName].savePath,
                               embBaseInfos[embTableName].maxTableSize);
        embBaseInfos[embTableName].isExist = true;
        LOG_INFO("create ssd table end, embTableName:" + embTableName);
        return;
    }
    // 续训场景：embBaseInfos 没有保存，不会初始化；SSD表会初始化，此时表已存在
    embBaseInfos[embTableName].isExist = true;
    LOG_INFO("ssd table is exist, embTableName:" + embTableName);
}

CacheManager::~CacheManager()
{
    for (auto &t : ssdEvictThreads) {
        t.join();
    }
    ssdEngine->Stop();
    ddrKeyFreqMap.clear();
    excludeDDRKeyCountMap.clear();
}

/// 加载数据到CacheManager
/// \param ddrFreqInitMap ddr内key频次数据
/// \param excludeDdrFreqInitMap 非DDR key频次数据
/// \param step 加载SSDEngine传入步数
void CacheManager::Load(const std::vector<EmbInfo> &mgmtEmbInfo, int step,
                        map<string, unordered_set<emb_cache_key_t>>& trainKeySet)
{
    // 加载SSDEngine数据
#ifndef GTEST
    for (auto& it : embBaseInfos) {
        string embTableName = it.first;
        EmbBaseInfo& embBase = it.second;
        ssdEngine->Load(embTableName, embBase.savePath, embBase.maxTableSize, step);
    }
    auto tableKeysVec = ssdEngine->ExportTableKey();
    for (auto &it: tableKeysVec) {
        auto &embTableName = it.first;
        auto &keys = it.second;
        for (auto key: keys) {
            preProcessMapper[embTableName].excludeDDRKeyCountMap[key] = 1;
            trainKeySet[embTableName].insert(key);
        }
    }
    for (const auto &embInfo: mgmtEmbInfo) {
        const std::string &tableName = embInfo.name;
        std::vector<char> buffer;
        int rc = embCache->Serialize(tableName, buffer);
        if (rc != 0) {
            throw std::runtime_error("Serialize failed!");
        }
        uint64_t memSize = sizeof(uint64_t) + embInfo.extEmbeddingSize * sizeof(float);
        for (uint64_t i = 0; i < buffer.size(); i += memSize) {
            uint64_t key = *reinterpret_cast<uint64_t *>(&buffer[i]);
            preProcessMapper[tableName].lfuCache.Put(key);
        }
    }
#endif
}

void CacheManager::SaveSSDEngine(int step)
{
#ifndef GTEST
    ssdEngine->Save(step);
#endif
}

int64_t CacheManager::GetTableEmbeddingSize(const string& tableName)
{
    if (ssdEngine == nullptr) {
        throw runtime_error("SSDEngine not init");
    }
    return ssdEngine->GetTableEmbeddingSize(tableName);
}

void CacheManager::ProcessSwapOutKeys(const string& tableName, const vector<emb_cache_key_t>& swapOutKeys,
                                      SwapOutInfo& info)
{
    auto& swapOutDDRKeys = info.swapOutDDRKeys;
    auto& swapOutDDRAddrOffs = info.swapOutDDRAddrOffs;
    auto& swapOutSSDKeys = info.swapOutSSDKeys;
    auto& swapOutSSDAddrOffs = info.swapOutSSDAddrOffs;

    // 处理一下没见过的key，看是更新到DDR还是SSD中
    auto& keyMapper = preProcessMapper[tableName];
    size_t availableDDRSize = keyMapper.DDRAvailableSize();
    for (size_t i = 0; i < swapOutKeys.size(); ++i) {
        emb_cache_key_t key = swapOutKeys[i];
        if (keyMapper.IsDDRKeyExist(key)) {
            keyMapper.lfuCache.Put(key);
            swapOutDDRKeys.push_back(key);
            swapOutDDRAddrOffs.push_back(i);
        } else if (keyMapper.IsSSDKeyExist(key)) {
            keyMapper.excludeDDRKeyCountMap[key]++;
            swapOutSSDKeys.push_back(key);
            swapOutSSDAddrOffs.push_back(i);
        } else if (availableDDRSize > 0) {
            keyMapper.InsertDDRKey(key);
            swapOutDDRKeys.push_back(key);
            swapOutDDRAddrOffs.push_back(i);
            availableDDRSize--;
        } else {
            keyMapper.InsertSSDKey(key);
            swapOutSSDKeys.push_back(key);
            swapOutSSDAddrOffs.push_back(i);
        }
    }
}

void CacheManager::ProcessSwapInKeys(const string& tableName, const vector<emb_cache_key_t>& swapInKeys,
                                     vector<emb_cache_key_t>& DDRToSSDKeys, vector<emb_cache_key_t>& SSDToDDRKeys)
{
    auto& keyMapper = preProcessMapper[tableName];
    size_t externalDDRSize = 0;
    std::vector<emb_cache_key_t> firstSeenKeys;
    for (emb_cache_key_t key : swapInKeys) {
        if (keyMapper.IsDDRKeyExist(key)) {
            continue;
        }
        externalDDRSize++;
        if (keyMapper.IsSSDKeyExist(key)) {
            SSDToDDRKeys.push_back(key);
        } else {
            firstSeenKeys.push_back(key);
        }
    }

    auto ddrAvailableSize = keyMapper.DDRAvailableSize();
    if (externalDDRSize > ddrAvailableSize) {  // 需要DDR--->SSD
        size_t transNum = externalDDRSize - ddrAvailableSize;

        if (transNum > keyMapper.SSDAvailableSize()) {
            throw invalid_argument("SSD table size too small, key quantity exceed while transferring DDR data to SSD");
        }
        // DDR--->SSD
        keyMapper.GetAndDeleteLeastFreqDDRKey2SSD(transNum, swapInKeys, DDRToSSDKeys);
    }

    // SSD--->DDR
    for (uint64_t key : SSDToDDRKeys) {
        keyMapper.InsertDDRKey(key);
        keyMapper.RemoveSSDKey(key);
    }
    for (uint64_t key : firstSeenKeys) {
        keyMapper.InsertDDRKey(key);
    }
    preProcessStep++;
}

void CacheManager::UpdateSSDEmb(string tableName, float* embPtr, uint32_t extEmbeddingSize,
                                vector<emb_cache_key_t>& keys, const vector<uint64_t>& swapOutSSDddrOffs)
{
    vector<float*> embeddingsAddr(keys.size());
    for (uint64_t i = 0; i < swapOutSSDddrOffs.size(); i++) {
        embeddingsAddr[i] = embPtr + swapOutSSDddrOffs[i] * extEmbeddingSize;
    }
    ssdEngine->InsertEmbeddingsByAddr(tableName, keys, embeddingsAddr, extEmbeddingSize);
}

void CacheManager::TransferDDR2SSD(string tableName, uint32_t extEmbeddingSize, vector<emb_cache_key_t>& keys,
                                   vector<float*>& addrs)
{
    CreateSSDTableIfNotExist(tableName);
    ssdEngine->InsertEmbeddingsByAddr(tableName, keys, addrs, extEmbeddingSize);
    for (auto addr : addrs) {
        free(addr);
        addr = nullptr;
    }
}

void CacheManager::FetchSSDEmb2DDR(string tableName, uint32_t extEmbeddingSize, vector<emb_cache_key_t>& keys,
                                   const vector<float*>& addrs)
{
    auto embeddings = ssdEngine->FetchEmbeddings(tableName, keys);
    for (uint64_t i = 0; i < embeddings.size(); i++) {
        int rc = memcpy_s(addrs[i], extEmbeddingSize * sizeof(float), embeddings[i].data(),
                          extEmbeddingSize * sizeof(float));
        if (rc != 0) {
            throw runtime_error("memcpy_s failed, rc: " + to_string(rc));
        }
    }
    ssdEngine->DeleteEmbeddings(tableName, keys);

    embeddingTaskStep++;
    evictWaitCond.notify_all();
}
