/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: EmbeddingMgmt管理类
 * Author: MindX SDK
 * Date: 2023/12/11
 */

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

int EmbeddingMgmt::Load(const string& name, const string& filePath)
{
    return embeddings[name]->Load(filePath);
}

int EmbeddingMgmt::Save(const string& name, const string& filePath)
{
    return embeddings[name]->Save(filePath);
}

int EmbeddingMgmt::Save(const string& filePath)
{
    for (auto& tablePair: embeddings) {
        tablePair.second->Save(filePath);
    }
}
