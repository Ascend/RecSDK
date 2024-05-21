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
#include "emb_table/embedding_static.h"
#include "emb_table/embedding_dynamic.h"
#include "emb_table/embedding_ddr.h"
#include "utils/logger.h"

using namespace MxRec;

EmbeddingMgmt::EmbeddingMgmt()
{
}

void EmbeddingMgmt::Init(const RankInfo& rInfo, const vector<EmbInfo>& eInfos,
    const vector<ThresholdValue>& thresholdValues, int seed)
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

size_t EmbeddingMgmt::GetMaxOffset(const std::string& name)
{
    embeddings[name]->GetMaxOffset();
}

void EmbeddingMgmt::LoadMaxOffset(OffsetMemT& loadData)
{
    LOG_ERROR("load max offset");
}

void EmbeddingMgmt::LoadKeyOffsetMap(KeyOffsetMemT& loadData)
{
    LOG_ERROR("load key offset");
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

void EmbeddingMgmt::EvictKeys(const string& name, const vector<emb_key_t>& keys)
{
    LOG_ERROR("evict keys for {}", name);
    if (keys.size() != 0) {
        embeddings[name]->EvictKeys(keys);
    }
    embeddings[name]->EvictInitDeviceEmb();
}

void EmbeddingMgmt::EvictKeysCombine(const vector<emb_key_t>& keys)
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

void EmbeddingMgmt::FindOffset(const std::string& name, const vector<emb_key_t>& keys,
                               size_t currentBatchId, size_t keepBatchId, int channel)
{
    return embeddings[name]->FindOffset(keys, currentBatchId, keepBatchId, channel);
}

const std::vector<size_t>& EmbeddingMgmt::GetMissingKeys(const std::string& name)
{
    return embeddings[name]->GetMissingKeys();
}

void EmbeddingMgmt::ClearMissingKeys(const std::string& name)
{
    return embeddings[name]->ClearMissingKeys();
}

std::shared_ptr<EmbeddingTable> EmbeddingMgmt::GetTable(const string& name)
{
    auto it = embeddings.find(name);
    if (it == embeddings.end()) {
        LOG_ERROR("table not found");
    }
    return std::dynamic_pointer_cast<EmbeddingTable>(it->second);
}

void EmbeddingMgmt::Load(const string& name, const string& filePath)
{
    return embeddings[name]->Load(filePath);
}


void EmbeddingMgmt::Load(const string& filePath)
{
    for (auto& tablePair: embeddings) {
        tablePair.second->Load(filePath);
    }
}

void EmbeddingMgmt::Save(const string& name, const string& filePath)
{
    return embeddings[name]->Save(filePath);
}

void EmbeddingMgmt::Save(const string& filePath)
{
    for (auto& tablePair: embeddings) {
        tablePair.second->Save(filePath);
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

EmbHashMemT EmbeddingMgmt::GetEmbHashMaps()
{
    EmbHashMemT EmbHashMaps;
    for (auto& tablePair: embeddings) {
        EmbHashMaps[tablePair.first].hostHashMap = tablePair.second ->GetKeyOffsetMap();
        EmbHashMaps[tablePair.first].devVocabSize = tablePair.second ->GetDevVocabSize();
        EmbHashMaps[tablePair.first].hostVocabSize = tablePair.second ->GetHostVocabSize();
        EmbHashMaps[tablePair.first].maxOffset = tablePair.second ->GetMaxOffset();
    }
    return EmbHashMaps;
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

void EmbeddingMgmt::EnableSSD()
{
    for (auto& table: embeddings) {
        table.second->EnableSSD();
    }
}

void EmbeddingMgmt::LockSave()
{
    for (auto& table: embeddings) {
        table.second->mutSave_.lock();
    }
    LOG_DEBUG("LockSave");
}

void EmbeddingMgmt::UnLockSave()
{
    for (auto& table: embeddings) {
        table.second->mutSave_.unlock();
    }
    LOG_DEBUG("UnLockSave");
}