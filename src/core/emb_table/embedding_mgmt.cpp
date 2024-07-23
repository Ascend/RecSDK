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
#include "emb_table/embedding_mgmt.h"

#include <future>

#include "emb_table/embedding_static.h"
#include "emb_table/embedding_dynamic.h"
#include "emb_table/embedding_ddr.h"
#include "file_system/file_system_handler.h"
#include "utils/logger.h"

using namespace MxRec;

EmbeddingMgmt::EmbeddingMgmt()
{
}

void EmbeddingMgmt::Init(const RankInfo& rInfo, const vector<EmbInfo>& eInfos, int seed)
{
    for (size_t i = 0; i < eInfos.size(); ++i) {
        if (rInfo.isDDR) {
            embeddings[eInfos[i].name] = std::make_shared<EmbeddingDDR>(eInfos[i], rInfo, seed);
            continue;
        }
        if (rInfo.useDynamicExpansion) {
            embeddings[eInfos[i].name] = std::make_shared<EmbeddingDynamic>(eInfos[i], rInfo, seed);
            continue;
        }
        embeddings[eInfos[i].name] = std::make_shared<EmbeddingStatic>(eInfos[i], rInfo, seed);
    }
}

EmbeddingMgmt* EmbeddingMgmt::Instance()
{
    static EmbeddingMgmt mgmt;
    return &mgmt;
}

void EmbeddingMgmt::Key2Offset(const std::string& name, std::vector<emb_key_t>& keys, int channel)
{
    embeddings[name]->Key2Offset(keys, channel);
}

void EmbeddingMgmt::Key2OffsetForDp(const std::string& name, std::vector<emb_key_t>& keys, int channel)
{
    embeddings[name]->Key2OffsetForDp(keys, channel);
}

size_t EmbeddingMgmt::GetMaxOffset(const std::string& name)
{
    return embeddings[name]->GetMaxOffset();
}

std::map<EmbNameT, size_t> EmbeddingMgmt::GetMaxOffset()
{
    std::map<EmbNameT, size_t> maxoffset;
    for (auto &it: embeddings) {
        maxoffset[it.first] = it.second->GetMaxOffset();
    }
    return maxoffset;
}

KeyOffsetMemT EmbeddingMgmt::GetKeyOffsetMap()
{
    KeyOffsetMemT keyOffsetMap;
    for (auto &it: embeddings) {
        keyOffsetMap[it.first] = it.second->GetKeyOffsetMap();
    }
    return keyOffsetMap;
}

void EmbeddingMgmt::EvictKeys(const string& name, const vector<emb_cache_key_t>& keys)
{
    LOG_ERROR("evict keys for {}", name);
    if (keys.size() != 0) {
        embeddings[name]->EvictKeys(keys);
    }
    embeddings[name]->EvictInitDeviceEmb();
}

void EmbeddingMgmt::EvictKeysCombine(const vector<emb_cache_key_t>& keys)
{
    if (keys.size() != 0) {
        for (auto& table: embeddings) {
            table.second->EvictKeys(keys);
        }
    }
    for (auto& table: embeddings) {
        // 初始化 dev
        table.second->EvictInitDeviceEmb();
    }
}

int64_t EmbeddingMgmt::GetSize(const std::string &name)
{
    return embeddings[name]->size();
}

int64_t EmbeddingMgmt::GetCapacity(const std::string &name)
{
    return embeddings[name]->capacity();
}

void EmbeddingMgmt::Load(const string& name, const string& filePath,
                         map<string, unordered_set<emb_cache_key_t>>& trainKeySet)
{
    embeddings[name]->SetFileSystemPtr(filePath);
    embeddings[name]->Load(filePath, trainKeySet);
    embeddings[name]->UnsetFileSystemPtr();
}

void EmbeddingMgmt::Load(const string& filePath, map<string, unordered_set<emb_cache_key_t>>& trainKeySet)
{
    for (auto& tablePair: embeddings) {
        tablePair.second->SetFileSystemPtr(filePath);
        tablePair.second->Load(filePath, trainKeySet);
        tablePair.second->UnsetFileSystemPtr();
    }
}

void EmbeddingMgmt::Save(const string& name, const string& filePath)
{
    embeddings[name]->SetFileSystemPtr(filePath);
    embeddings[name]->Save(filePath);
    embeddings[name]->UnsetFileSystemPtr();
}

void EmbeddingMgmt::Save(const string& filePath)
{
    for (auto& tablePair: embeddings) {
        tablePair.second->SetFileSystemPtr(filePath);
    }
    // use multi-thread to prevent receiving save_d2h blocked when table order different between cpp and python
    vector<future<void>> futures;
    for (auto& tablePair: embeddings) {
        futures.emplace_back(
            std::async(std::launch::async, [table = tablePair.second, filePath] { table->Save(filePath); }));
    }
    for (auto& f: futures) {
        f.get();  // get() will repost exception if happened
    }

    for (auto& tablePair: embeddings) {
        tablePair.second->UnsetFileSystemPtr();
    }
}

OffsetMapT EmbeddingMgmt::GetDeviceOffsets()
{
    OffsetMapT AllDeviceOffsets;
    for (auto& tablePair: embeddings) {
        AllDeviceOffsets[tablePair.first] = tablePair.second ->GetDeviceOffset();
    }
    return AllDeviceOffsets;
}

void EmbeddingMgmt::SetOptimizerInfo(const string& name, OptimizerInfo& optimizerInfo)
{
    embeddings[name]->SetOptimizerInfo(optimizerInfo);
}

OffsetMapT EmbeddingMgmt::GetLoadOffsets()
{
    OffsetMapT AllLoadOffsets;
    for (auto& tablePair: embeddings) {
        AllLoadOffsets[tablePair.first] = tablePair.second ->GetLoadOffset();
    }
    return AllLoadOffsets;
}

void EmbeddingMgmt::SetCacheManagerForEmbTable(CacheManager* cacheManager)
{
    for (auto& table: embeddings) {
        table.second->SetCacheManager(cacheManager);
    }
}

void EmbeddingMgmt::SetHDTransferForEmbTable(HDTransfer* hdTransfer)
{
    for (auto& table: embeddings) {
        table.second->SetHDTransfer(hdTransfer);
    }
}

void EmbeddingMgmt::SetEmbCacheForEmbTable(const ock::ctr::EmbCacheManagerPtr& embCache)
{
    for (auto& table: embeddings) {
        table.second->SetEmbCache(embCache);
    }
}
