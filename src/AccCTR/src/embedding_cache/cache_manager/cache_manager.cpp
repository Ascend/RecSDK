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

#include "cache_manager.h"

#include <unordered_set>

#include "external_logger.h"

using namespace EmbCache;
using namespace ock;
using namespace ock::ctr;

int64_t EmbCache::INVALID_KEY = -1;

int EmbCacheManagerImpl::CreateCacheForTable(const EmbCacheInfo& embCacheInfo,
                                             const std::vector<InitializerInfo>& initializerInfos, int64_t invalidKey,
                                             uint64_t prefillBufferSize, uint32_t refillThreadNum)
{
    int checkTableNameRet = CheckCreateTableName(embCacheInfo.tableName);
    if (checkTableNameRet != H_OK) {
        return checkTableNameRet;
    }

    if (embCacheInfo.extEmbeddingSize == 0 || embCacheInfo.embeddingSize == 0 || embCacheInfo.vocabSize == 0 ||
        embCacheInfo.maxCacheSize == 0) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "size must be positive");
        return H_SIZE_ZERO;
    }

    if (embCacheInfo.vocabSize < embCacheInfo.maxCacheSize) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "host vocabSize:" + std::to_string(embCacheInfo.vocabSize) +
        " must be greater than or equal to device vocabSize:" + std::to_string(embCacheInfo.maxCacheSize) +
        ", please increase [host vocabSize] in [create_table] interface");
        return H_HOST_VOCAB_SIZE_TOO_SMALL;
    }

    auto om = offsetMappers.find(embCacheInfo.tableName);
    auto embTable = embTables.find(embCacheInfo.tableName);
    if (om != offsetMappers.end() || embTable != embTables.end()) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "This table has already been created");
        return H_TABLE_CREATE_DUPLICATE;
    }

    if (embCacheInfo.extEmbeddingSize % embCacheInfo.embeddingSize != 0) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "extEmbeddingSize = embeddingSize + optimizerSize, "
                                                  "which is divisible by embeddingSize");
        return H_EXT_EMBEDDING_SIZE_INVALID;
    }

    if (!CheckInitializer(embCacheInfo.extEmbeddingSize, initializerInfos)) {
        return H_INITIALIZER_INVALID;
    }

    if ((prefillBufferSize < 1) || (prefillBufferSize > embCacheInfo.vocabSize)) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "prefillBufferSize has to be between [1, hostVocabSize]");
        return H_PREFILL_BUFFER_SIZE_INVALID;
    }

    if (!CheckValidThreadNum(refillThreadNum)) {
        return H_THREAD_NUM_ERROR;
    }

    uint32_t reserve = embCacheInfo.vocabSize / VOCAB_CACHE_RATIO;
    if (!offsetMappers[embCacheInfo.tableName].Initialize(reserve, embCacheInfo.maxCacheSize)) {
        offsetMappers[embCacheInfo.tableName].UnInitialize();
        offsetMappers.erase(embCacheInfo.tableName);
        return H_MEMORY_ALLOC_ERROR;
    }

    EmbPoolParam embPoolParam{prefillBufferSize, refillThreadNum};

    if (!embTables[embCacheInfo.tableName].Initialize(embCacheInfo, reserve, initializerInfos, embPoolParam)) {
        offsetMappers.erase(embCacheInfo.tableName);
        embTables.erase(embCacheInfo.tableName);
        return H_MEMORY_ALLOC_ERROR;
    }

    embCacheInfos.insert({embCacheInfo.tableName, embCacheInfo});
    INVALID_KEY = invalidKey;
    return H_OK;
}

int EmbCacheManagerImpl::GetSwapPairsAndKey2Offset(const std::string& tableName, std::vector<uint64_t>& keys,
                                                   KeyOffsetPair& swapInKoPair, KeyOffsetPair& swapOutKoPair)
{
    int checkRet = CheckGetSwapPairsAndKey2Offset(tableName, swapInKoPair, swapOutKoPair);
    if (checkRet != H_OK) {
        return checkRet;
    }
    return offsetMappers[tableName].GetSwapPairsAndKey2Offset(keys, swapInKoPair, swapOutKoPair);
}

int EmbCacheManagerImpl::EmbeddingLookup(const std::string& tableName, const std::vector<uint64_t>& keys,
                                         float* embAddr, uint32_t threadNum)
{
    int checkTableNameRet = CheckValidTableName(tableName);
    if (checkTableNameRet != H_OK) {
        return checkTableNameRet;
    }

    if (!CheckValidThreadNum(threadNum)) {
        return H_THREAD_NUM_ERROR;
    }

    if (keys.empty()) {
        return H_OK;
    }

    if (embAddr == nullptr) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "embAddr is nullptr");
        return H_ADDRESS_NULL;
    }

    return embTables[tableName].Gather(reinterpret_cast<uint64_t>(embAddr), keys, threadNum);
}

int EmbCacheManagerImpl::EmbeddingLookupAddrs(const std::string& tableName, const std::vector<uint64_t>& keys,
                                              std::vector<float*>& addrs, uint32_t threadNum)
{
    int checkTableNameRet = CheckValidTableName(tableName);
    if (checkTableNameRet != H_OK) {
        return checkTableNameRet;
    }

    if (!CheckValidThreadNum(threadNum)) {
        return H_THREAD_NUM_ERROR;
    }

    if (keys.empty()) {
        return H_OK;
    }

    return embTables[tableName].GatherAddrs(keys, addrs, threadNum);
}

// 如果多线程使用，严格保证传入的key线程间不会重复(unique key)，否则可能出现未定义结果
int EmbCacheManagerImpl::EmbeddingLookupAndRemove(const std::string& tableName, const std::vector<uint64_t>& keys,
                                                  float* embAddr, uint32_t threadNum)
{
    int checkTableNameRet = CheckValidTableName(tableName);
    if (checkTableNameRet != H_OK) {
        return checkTableNameRet;
    }

    if (!CheckValidThreadNum(threadNum)) {
        return H_THREAD_NUM_ERROR;
    }

    if (keys.empty()) {
        return H_OK;
    }

    if (embAddr == nullptr) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "embAddr is nullptr");
        return H_ADDRESS_NULL;
    }

    return embTables[tableName].GatherAndRemove(reinterpret_cast<uint64_t>(embAddr), keys, threadNum);
}

int EmbCacheManagerImpl::EmbeddingUpdate(const std::string& tableName, const std::vector<uint64_t>& keys,
                                         float* embAddr, uint32_t threadNum)
{
    int checkTableNameRet = CheckValidTableName(tableName);
    if (checkTableNameRet != H_OK) {
        return checkTableNameRet;
    }

    if (!CheckValidThreadNum(threadNum)) {  // 检查thread是否小于核数
        return H_THREAD_NUM_ERROR;
    }

    if (keys.empty()) {
        return H_OK;
    }

    if (embAddr == nullptr) {  // 检查embAddr是不是空指针
        ExternalLogger::PrintLog(LogLevel::ERROR, "embAddr is nullptr");
        return H_ADDRESS_NULL;
    }

    return embTables[tableName].Scatter(reinterpret_cast<uint64_t>(embAddr), keys, threadNum);
}

int EmbCacheManagerImpl::EmbeddingRemove(const std::string& tableName, const std::vector<uint64_t>& keys,
                                         uint32_t threadNum)
{
    int checkTableNameRet = CheckValidTableName(tableName);
    if (checkTableNameRet != H_OK) {
        return checkTableNameRet;
    }

    if (!CheckValidThreadNum(threadNum)) {  // 检查thread是否小于核数
        return H_THREAD_NUM_ERROR;
    }

    if (keys.empty()) {
        return H_OK;
    }

    return embTables[tableName].RemoveByKeys(keys, threadNum);
}

int EmbCacheManagerImpl::RemoveEmbsByKeys(const std::string& tableName, const std::vector<uint64_t>& keys)
{
    int checkTableNameRet = CheckValidTableName(tableName);
    if (checkTableNameRet != H_OK) {
        return checkTableNameRet;
    }
    const auto& om = offsetMappers.find(tableName);
    const auto& embTable = embTables.find(tableName);
    for (auto key : keys) {
        if (key == static_cast<uint64_t>(INVALID_KEY)) {
            ExternalLogger::PrintLog(LogLevel::WARN, "Try to evict invalid key");
            continue;
        }
        om->second.Remove(key);
        embTable->second.Remove(key);
    }
    return H_OK;
}

int EmbCacheManagerImpl::GetEmbTableNames(std::vector<std::string>& allTableNames)
{
    if (!allTableNames.empty()) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "allTableNames should be empty");
        return H_ARG_NOT_EMPTY;
    }
    allTableNames.reserve(embTables.size());
    for (auto& embTable : embTables) {
        allTableNames.emplace_back(embTable.first);
    }
    return H_OK;
}

int EmbCacheManagerImpl::ExportDeviceKeyOffsetPairs(const std::string& tableName,
                                                    std::vector<std::pair<uint64_t, uint64_t>>& koVec)
{
    int checkTableNameRet = CheckValidTableName(tableName);
    if (checkTableNameRet != H_OK) {
        return checkTableNameRet;
    }
    OffsetMapper& om = offsetMappers[tableName];
    koVec = om.ExportSortedKVPairs();
    return H_OK;
}

int EmbCacheManagerImpl::Serialize(const std::string& tableName, std::vector<char>& buffer)
{
    int checkTableNameRet = CheckValidTableName(tableName);
    if (checkTableNameRet != H_OK) {
        return checkTableNameRet;
    }
    buffer = embTables[tableName].Serialize();
    return H_OK;
}

int EmbCacheManagerImpl::Deserialize(const std::string& tableName, const std::vector<char>& buffer)
{
    int checkTableNameRet = CheckValidTableName(tableName);
    if (checkTableNameRet != H_OK) {
        return checkTableNameRet;
    }
    if (!embTables[tableName].Deserialize(buffer)) {
        return H_LOAD_ERROR;
    }
    return H_OK;
}

int EmbCacheManagerImpl::GetEmbTableInfos(std::string tableName, std::vector<uint64_t>& keys,
                                          std::vector<std::vector<float>>& embeddings,
                                          std::vector<std::vector<float>>& optimizerSlots)
{
    int checkTableNameRet = CheckValidTableName(tableName);
    if (checkTableNameRet != H_OK) {
        return checkTableNameRet;
    }
    if (!keys.empty()) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "keys should be empty");
        return H_ARG_NOT_EMPTY;
    }
    if (!embeddings.empty()) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "embeddings should be empty");
        return H_ARG_NOT_EMPTY;
    }
    if (!optimizerSlots.empty()) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "optimizerSlots should be empty");
        return H_ARG_NOT_EMPTY;
    }
    embTables[tableName].GetEmbTableInfos(keys, embeddings, optimizerSlots);
    return H_OK;
}

int EmbCacheManagerImpl::LoadEmbTableInfos(std::string tableName, const std::vector<uint64_t>& keys,
                                           const std::vector<std::vector<float>>& embeddings,
                                           const std::vector<std::vector<float>>& optimizerSlots)
{
    int checkTableNameRet = CheckValidTableName(tableName);
    if (checkTableNameRet != H_OK) {
        return checkTableNameRet;
    }
    if (!embTables[tableName].LoadEmbTableInfos(keys, embeddings, optimizerSlots)) {
        return H_LOAD_ERROR;
    }
    return H_OK;
}

void EmbCacheManagerImpl::Destroy()
{
    for (auto it = offsetMappers.begin(); it != offsetMappers.end(); it++) {
        it->second.UnInitialize();
    }
    for (auto it = embTables.begin(); it != embTables.end(); it++) {
        it->second.UnInitialize();
    }
    embCacheInfos.clear();
    offsetMappers.clear();
    embTables.clear();
}

int EmbCacheManagerImpl::CheckValidTableName(const std::string& tableName)
{
    if (tableName.size() > TABLE_NAME_MAX_SIZE) {
        ExternalLogger::PrintLog(LogLevel::ERROR,
                                 "tableName size can not larger than " + std::to_string(TABLE_NAME_MAX_SIZE));
        return H_TABLE_NAME_TOO_LONG;
    }
    auto om = offsetMappers.find(tableName);
    auto embTable = embTables.find(tableName);
    if (om == offsetMappers.end() || embTable == embTables.end()) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "can not find table");
        return H_TABLE_NOT_EXIST;
    }
    return H_OK;
}

bool EmbCacheManagerImpl::CheckInitializer(uint32_t extEmbSize, std::vector<InitializerInfo> initializerInfos)
{
    std::sort(initializerInfos.begin(), initializerInfos.end(),
              [](const auto& u, const auto& v) { return u.start < v.start; });
    uint32_t cur_pos = 0;
    for (const auto& info : initializerInfos) {
        if (info.initializer == nullptr) {
            ExternalLogger::PrintLog(LogLevel::ERROR, "initializer is nullptr");
            return false;
        }
        if (info.start != cur_pos) {
            ExternalLogger::PrintLog(LogLevel::ERROR, "Initializers got coverage problems");
            return false;
        }
        cur_pos += info.len;
    }
    // 最后判断
    if (cur_pos != extEmbSize) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "Initializers got coverage problems");
        return false;
    }
    return true;
}

bool EmbCacheManagerImpl::CheckValidThreadNum(uint32_t threadNum)
{
    uint32_t processCoreNum = std::thread::hardware_concurrency();
    if (threadNum > processCoreNum) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "ThreadNum can not larger than cpu core num");
        return false;
    }

    if (threadNum == 0) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "ThreadNum can not be zero");
        return false;
    }
    return true;
}

int EmbCacheManagerImpl::CheckGetSwapPairsAndKey2Offset(const std::string& tableName, const KeyOffsetPair& swapInKoPair,
                                                        const KeyOffsetPair& swapOutKoPair)
{
    if (!swapInKoPair.first.empty() || !swapInKoPair.second.empty() || !swapOutKoPair.first.empty() ||
        !swapOutKoPair.second.empty()) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "koPair should be empty");
        return H_ARG_NOT_EMPTY;
    }

    int checkTableNameRet = CheckValidTableName(tableName);
    if (checkTableNameRet != H_OK) {
        return checkTableNameRet;
    }

    return H_OK;
}

int EmbCacheManagerImpl::CheckCreateTableName(const std::string& tableName)
{
    if (tableName.empty()) {
        ExternalLogger::PrintLog(LogLevel::ERROR, "tableName can not be empty");
        return H_TABLE_NAME_EMPTY;
    }

    if (tableName.size() > TABLE_NAME_MAX_SIZE) {
        ExternalLogger::PrintLog(LogLevel::ERROR,
                                 "tableName size can not larger than " + std::to_string(TABLE_NAME_MAX_SIZE));
        return H_TABLE_NAME_TOO_LONG;
    }
    return H_OK;
}

uint32_t EmbCacheManagerImpl::GetUsage(const std::string& tableName)
{
    return embTables[tableName].GetUsage();
}
