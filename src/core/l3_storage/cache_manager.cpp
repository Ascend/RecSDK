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

void CacheManager::Init(ock::ctr::EmbCacheManagerPtr embCachePtr, vector<EmbInfo>& mgmtEmbInfo,
                        shared_ptr<L3Storage> level3Storage)
{
    LOG_INFO("CacheManager Init method begin");
    if (level3Storage == nullptr) {
        auto error = Error(ModuleName::M_L3_STORAGE, ErrorType::LOGIC_ERROR,
                           "Attribute:level3Storage is nullptr.");
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }

    this->embCache = std::move(embCachePtr);
    for (auto& emb : mgmtEmbInfo) {
        EmbBaseInfo baseInfo {emb.ssdVocabSize, emb.ssdDataPath, false, emb.extEmbeddingSize};
        embBaseInfos.emplace(emb.name, baseInfo);
        preProcessMapper[emb.name].Initialize(emb.name, emb.hostVocabSize, emb.ssdVocabSize);
    }
    this->l3Storage = level3Storage;
    this->l3Storage->Start();
    LOG_INFO("CacheManager Init method end");
}

bool CacheManager::IsKeyInL3Storage(const string& embTableName, emb_cache_key_t key)
{
    return l3Storage->IsKeyExist(embTableName, key);
}

/// 淘汰三级存储中Emb信息
/// \param embTableName emb表名
/// \param keys 淘汰key列表
void CacheManager::EvictL3StorageEmbedding(const string& embTableName, const vector<emb_cache_key_t>& keys)
{
    if (keys.empty()) {
        return;
    }

    int keyStep = preProcessStep;
    unordered_map<emb_cache_key_t, freq_num_t>& l3StorageMap = preProcessMapper[embTableName].excludeDDRKeyCountMap;
    LFUCache& ddrLfu = preProcessMapper[embTableName].lfuCache;
    std::vector<emb_cache_key_t> l3StorageKeysToBeDeleted;
    // 1 删除缓存中记录的key的次数
    for (auto &key: keys) {
        auto it = l3StorageMap.find(key);
        if (it != l3StorageMap.end()) {
            l3StorageMap.erase(it);
            l3StorageKeysToBeDeleted.emplace_back(key);
        } else {
            ddrLfu.Pop(key);
        }
    }

    l3StorageEvictThreads.emplace_back([=]() mutable {
        // 2 删除L3Storage中保存的Emb数据
        std::unique_lock<std::mutex> lk(evictWaitMut);
        evictWaitCond.wait(lk, [keyStep, this] {
            return embeddingTaskStep == keyStep;
        });
        l3Storage->DeleteEmbeddings(embTableName, l3StorageKeysToBeDeleted);
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

void CacheManager::CreateL3StorageTableIfNotExist(const std::string& embTableName)
{
    if (embBaseInfos[embTableName].isExist) {
        return;
    }
    if (!l3Storage->IsTableExist(embTableName)) {
        l3Storage->CreateTable(embTableName, embBaseInfos[embTableName].savePath,
                               embBaseInfos[embTableName].maxTableSize);
        embBaseInfos[embTableName].isExist = true;
        LOG_INFO("create l3Storage table end, embTableName:{}", embTableName);
        return;
    }
    // 续训场景：embBaseInfos 没有保存，不会初始化；L3Storage表会初始化，此时表已存在
    embBaseInfos[embTableName].isExist = true;
    LOG_INFO("l3Storage table is exist, embTableName:{}", embTableName);
}

CacheManager::~CacheManager()
{
    for (auto& t : l3StorageEvictThreads) {
        t.join();
    }
    l3Storage->Stop();
    ddrKeyFreqMap.clear();
    excludeDDRKeyCountMap.clear();
}

/// 加载数据到CacheManager
/// \param ddrFreqInitMap ddr内key频次数据
/// \param excludeDdrFreqInitMap 非DDR key频次数据
/// \param step 加载L3Storage传入步数
void CacheManager::Load(const std::vector<EmbInfo> &mgmtEmbInfo, int step,
                        map<string, unordered_set<emb_cache_key_t>>& trainKeySet)
{
    // 加载L3Storage数据
#ifndef GTEST
    for (auto& it : embBaseInfos) {
        string embTableName = it.first;
        EmbBaseInfo& embBase = it.second;
        l3Storage->Load(embTableName, embBase.savePath, embBase.maxTableSize, step);
    }
    auto tableKeysVec = l3Storage->ExportTableKey();
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
            auto error = Error(ModuleName::M_L3_STORAGE, ErrorType::LOGIC_ERROR,
                               "Serialize failed!");
            LOG_ERROR(error.ToString());
            throw std::runtime_error(error.ToString());
        }
        uint64_t memSize = sizeof(uint64_t) + embInfo.extEmbeddingSize * sizeof(float);
        for (uint64_t i = 0; i < buffer.size(); i += memSize) {
            uint64_t key = *reinterpret_cast<uint64_t *>(&buffer[i]);
            preProcessMapper[tableName].lfuCache.Put(key);
        }
    }
#endif
}

void CacheManager::Save(int step)
{
#ifndef GTEST
    l3Storage->Save(step);
#endif
}

int64_t CacheManager::GetTableUsage(const string& tableName)
{
    if (l3Storage == nullptr) {
        auto error = Error(ModuleName::M_L3_STORAGE, ErrorType::LOGIC_ERROR, "L3Storage not init.");
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }
    return l3Storage->GetTableUsage(tableName);
}

void CacheManager::ProcessSwapOutKeys(const string& tableName, const vector<emb_cache_key_t>& swapOutKeys,
                                      HBMSwapOutInfo& info)
{
    auto& swapOutDDRKeys = info.swapOutDDRKeys;
    auto& swapOutDDRAddrOffs = info.swapOutDDRAddrOffs;
    auto& swapOutL3StorageKeys = info.swapOutL3StorageKeys;
    auto& swapOutL3StorageAddrOffs = info.swapOutL3StorageAddrOffs;

    // 处理一下没见过的key，看是更新到DDR还是L3Storage中
    auto& keyMapper = preProcessMapper[tableName];
    size_t availableDDRSize = keyMapper.DDRAvailableSize();
    for (size_t i = 0; i < swapOutKeys.size(); ++i) {
        emb_cache_key_t key = swapOutKeys[i];
        if (keyMapper.IsDDRKeyExist(key)) {
            keyMapper.lfuCache.Put(key);
            swapOutDDRKeys.push_back(key);
            swapOutDDRAddrOffs.push_back(i);
        } else if (keyMapper.IsL3StorageKeyExist(key)) {
            keyMapper.excludeDDRKeyCountMap[key]++;
            swapOutL3StorageKeys.push_back(key);
            swapOutL3StorageAddrOffs.push_back(i);
        } else if (availableDDRSize > 0) {
            keyMapper.InsertDDRKey(key);
            swapOutDDRKeys.push_back(key);
            swapOutDDRAddrOffs.push_back(i);
            availableDDRSize--;
        } else {
            keyMapper.InsertL3StorageKey(key);
            swapOutL3StorageKeys.push_back(key);
            swapOutL3StorageAddrOffs.push_back(i);
        }
    }
}

void CacheManager::ProcessSwapInKeys(const string& tableName, const vector<emb_cache_key_t>& swapInKeys,
                                     vector<emb_cache_key_t>& DDRToL3StorageKeys,
                                     vector<emb_cache_key_t>& L3StorageToDDRKeys)
{
    auto& keyMapper = preProcessMapper[tableName];
    size_t externalDDRSize = 0;
    std::vector<emb_cache_key_t> firstSeenKeys;
    for (emb_cache_key_t key : swapInKeys) {
        if (keyMapper.IsDDRKeyExist(key)) {
            continue;
        }
        externalDDRSize++;
        if (keyMapper.IsL3StorageKeyExist(key)) {
            L3StorageToDDRKeys.push_back(key);
        } else {
            firstSeenKeys.push_back(key);
        }
    }

    auto ddrAvailableSize = keyMapper.DDRAvailableSize();
    if (externalDDRSize > ddrAvailableSize) {  // 需要DDR--->L3Storage
        size_t transNum = externalDDRSize - ddrAvailableSize;

        if (transNum > keyMapper.L3StorageAvailableSize()) {
            auto error = Error(ModuleName::M_L3_STORAGE, ErrorType::LOGIC_ERROR,
                               "L3Storage table size too small, key quantity exceed while transferring DDR data"
                               " to L3Storage.");
            LOG_ERROR(error.ToString());
            throw std::invalid_argument(error.ToString());
        }
        // DDR--->L3Storage
        keyMapper.GetAndDeleteLeastFreqDDRKey2L3Storage(transNum, swapInKeys, DDRToL3StorageKeys);
    }

    // L3Storage--->DDR
    for (uint64_t key : L3StorageToDDRKeys) {
        keyMapper.InsertDDRKey(key);
        keyMapper.RemoveL3StorageKey(key);
    }
    for (uint64_t key : firstSeenKeys) {
        keyMapper.InsertDDRKey(key);
    }
    preProcessStep++;
}

void CacheManager::UpdateL3StorageEmb(string tableName, float* embPtr, uint32_t extEmbeddingSize,
                                      vector<emb_cache_key_t>& keys, const vector<uint64_t>& swapOutL3StorageOffs)
{
    vector<float*> embeddingsAddr(keys.size());
    for (uint64_t i = 0; i < swapOutL3StorageOffs.size(); i++) {
        embeddingsAddr[i] = embPtr + swapOutL3StorageOffs[i] * extEmbeddingSize;
    }
    l3Storage->InsertEmbeddingsByAddr(tableName, keys, embeddingsAddr, extEmbeddingSize);
}

void CacheManager::TransferDDR2L3Storage(string tableName, uint32_t extEmbeddingSize, vector<emb_cache_key_t>& keys,
                                         vector<float*>& addrs)
{
    CreateL3StorageTableIfNotExist(tableName);
    l3Storage->InsertEmbeddingsByAddr(tableName, keys, addrs, extEmbeddingSize);
    for (auto addr : addrs) {
        free(addr);
        addr = nullptr;
    }
}

void CacheManager::FetchL3StorageEmb2DDR(string tableName, uint32_t extEmbeddingSize, vector<emb_cache_key_t>& keys,
                                         const vector<float*>& addrs)
{
    auto embeddings = l3Storage->FetchEmbeddings(tableName, keys);
    for (uint64_t i = 0; i < embeddings.size(); i++) {
        int rc = memcpy_s(addrs[i], extEmbeddingSize * sizeof(float), embeddings[i].data(),
                          extEmbeddingSize * sizeof(float));
        if (rc != 0) {
            auto error = Error(ModuleName::M_L3_STORAGE, ErrorType::IO_ERROR,
                               Logger::Format("Invoke memcpy_s failed, rc:{}. You can query the meaning of "
                                   "security function error code.", to_string(rc)));
            LOG_ERROR(error.ToString());
            throw std::runtime_error(error.ToString());
        }
    }
    l3Storage->DeleteEmbeddings(tableName, keys);

    embeddingTaskStep++;
    evictWaitCond.notify_all();
}

void CacheManager::BackUpTrainStatus()
{
    ddrKeyFreqMapBackUp = ddrKeyFreqMap;
    excludeDDRKeyCountMapBackUp = excludeDDRKeyCountMap;
}

void CacheManager::RecoverTrainStatus()
{
    for (const auto& pair: excludeDDRKeyCountMapBackUp) {
        auto tableName = pair.first;

        std::vector<emb_cache_key_t> ssdKeysBeforeEval;
        std::vector<emb_cache_key_t> ssdKeysAfterEval;
        std::vector<emb_cache_key_t> swapInKeys;
        std::vector<emb_cache_key_t> swapOutKeys;

        for (const auto& keyMap : pair.second) {
            ssdKeysBeforeEval.push_back(keyMap.first);
        }
        for (const auto& keyMap : excludeDDRKeyCountMap[tableName]) {
            ssdKeysAfterEval.push_back(keyMap.first);
        }

        GetSwapInAndSwapOutKeys(ssdKeysBeforeEval, ssdKeysAfterEval, swapInKeys, swapOutKeys);

        // ddr <-> ssd
        // ddr-> lookup address, ssd->insert embedding , ddr->remove embedding
        vector<float*> swapInKeysAddr;
        int rc = embCache->EmbeddingLookupAddrs(tableName, swapInKeys, swapInKeysAddr);
        CheckEmbCacheReturnCode("EmbeddingLookUpAddrs", rc);

        auto extEmbeddingSize = embBaseInfos[tableName].extEmbeddingSize;
        l3Storage->InsertEmbeddingsByAddr(tableName, swapInKeys, swapInKeysAddr, extEmbeddingSize);
        rc = embCache->EmbeddingRemove(tableName, swapInKeys);
        CheckEmbCacheReturnCode("EmbeddingRemove", rc);

        // ssd->fetch embedding, ddr->EmbeddingUpdate, ssd->delete embedding
        auto swapOutEmbeddings = l3Storage->FetchEmbeddings(tableName, swapOutKeys);
        vector<float> swapOutFlattenEmbeddings;
        for (auto& emb : swapOutEmbeddings) {
            swapOutFlattenEmbeddings.insert(swapOutFlattenEmbeddings.cend(), emb.cbegin(), emb.cend());
        }
        rc = embCache->EmbeddingUpdate(tableName, swapOutKeys, swapOutFlattenEmbeddings.data());
        l3Storage->DeleteEmbeddings(tableName, swapOutKeys);
    }

    ddrKeyFreqMap = ddrKeyFreqMapBackUp;
    excludeDDRKeyCountMap = excludeDDRKeyCountMapBackUp;
}

void CacheManager::GetSwapInAndSwapOutKeys(vector<emb_cache_key_t>& ssdKeysBeforeEval,
                                           vector<emb_cache_key_t>& ssdKeysAfterEval,
                                           vector<emb_cache_key_t>& swapInKeys, vector<emb_cache_key_t>& swapOutKeys)
{
    std::sort(ssdKeysBeforeEval.begin(), ssdKeysBeforeEval.end());
    std::sort(ssdKeysAfterEval.begin(), ssdKeysAfterEval.end());
    vector<emb_cache_key_t> intersectionKeys;
    std::set_intersection(ssdKeysBeforeEval.begin(), ssdKeysBeforeEval.end(), ssdKeysAfterEval.begin(),
                          ssdKeysAfterEval.end(), std::back_inserter(intersectionKeys));

    std::set_difference(ssdKeysBeforeEval.begin(), ssdKeysBeforeEval.end(), intersectionKeys.begin(),
                        intersectionKeys.end(), std::back_inserter(swapInKeys));
    std::set_difference(ssdKeysAfterEval.begin(), ssdKeysAfterEval.end(), intersectionKeys.begin(),
                        intersectionKeys.end(), std::back_inserter(swapOutKeys));
}

void CacheManager::CheckEmbCacheReturnCode(const string& funcName, int retCode)
{
    if (retCode == 0) {
        return;
    }
    auto error = Error(ModuleName::M_L3_STORAGE, ErrorType::LOGIC_ERROR,
                       Logger::Format("Invoke embCache func:{} failed, error code:{}.",
                           funcName, std::to_string(retCode)));
    LOG_ERROR(error.ToString());
    throw std::runtime_error(error.ToString());
}
