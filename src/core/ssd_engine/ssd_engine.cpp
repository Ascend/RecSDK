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

#include "ssd_engine.h"

using namespace MxRec;
using namespace std;

bool SSDEngine::IsTableExist(const string &tableName)
{
    CheckSSDEngineIsRunning();
    auto it = as_const(tableMap).find(tableName);
    return !(it == tableMap.end());
}

bool SSDEngine::IsKeyExist(const string &tableName, emb_cache_key_t key)
{
    CheckSSDEngineIsRunning();
    auto it = as_const(tableMap).find(tableName);
    CheckTableExist(it == tableMap.end(), tableName);
    return it->second->IsKeyExist(key);
}

void SSDEngine::CreateTable(const string &tableName, vector<string> savePaths, uint64_t maxTableSize)
{
    CheckSSDEngineIsRunning();
    if (savePaths.empty()) {
        Table::ThrowInvalidArgError("SSDEngine input savePaths is empty.");
    }
    auto it = as_const(tableMap).find(tableName);
    if (it != tableMap.end()) {
        Table::ThrowInvalidArgError(Logger::Format("Table:{} already exist.", tableName));
    }
    tableMap[tableName] = make_shared<Table>(tableName, savePaths, maxTableSize, compactThreshold);
}

void SSDEngine::InsertEmbeddings(const string& tableName, vector<emb_cache_key_t>& keys,
                                 vector<vector<float>>& embeddings)
{
    CheckSSDEngineIsRunning();
    auto it = as_const(tableMap).find(tableName);
    CheckTableExist(it == tableMap.end(), tableName);

    if (keys.size() != embeddings.size()) {
        Table::ThrowInvalidArgError("Param keys' length not equal to embeddings' length.");
    }

    it->second->InsertEmbeddings(keys, embeddings);
}

void SSDEngine::DeleteEmbeddings(const string &tableName, vector<emb_cache_key_t> &keys)
{
    CheckSSDEngineIsRunning();
    auto it = as_const(tableMap).find(tableName);
    CheckTableExist(it == tableMap.end(), tableName);

    it->second->DeleteEmbeddings(keys);
}

int64_t SSDEngine::GetTableAvailableSpace(const string &tableName)
{
    CheckSSDEngineIsRunning();
    auto it = as_const(tableMap).find(tableName);
    CheckTableExist(it == tableMap.end(), tableName);

    return it->second->GetTableAvailableSpace();
}

void SSDEngine::Save(int step, const map<string, map<emb_key_t, KeyInfo>>& keyInfoMap)
{
    CheckSSDEngineIsRunning();

    if (step == loadStep) {
        LOG_INFO("save step equal to load step, skip saving, step:{}", step);
        return;
    }

    for (auto item: as_const(tableMap)) {
        item.second->Save(step, keyInfoMap.at(item.first));
    }
    saveStep = step;
}

void SSDEngine::Save(int step)
{
    CheckSSDEngineIsRunning();

    if (step == loadStep) {
        LOG_INFO("save step equal to load step, skip saving, step:{}", step);
        return;
    }

    for (auto item: as_const(tableMap)) {
        item.second->Save(step);
    }
    saveStep = step;
}

void SSDEngine::Load(const string &tableName, vector<string> savePaths, uint64_t maxTableSize, int step)
{
    CheckSSDEngineIsRunning();

    if (step == saveStep) {
        LOG_INFO("load step equal to save step, skip loading, step:{}", step);
        return;
    }

    auto it = as_const(tableMap).find(tableName);
    if (it != tableMap.end()) {
        Table::ThrowInvalidArgError("Table already exist.");
    }

    tableMap[tableName] = make_shared<Table>(tableName, savePaths, maxTableSize, compactThreshold, step);
    loadStep = step;
}

void SSDEngine::Start()
{
    if (isRunning) {
        return;
    }
    isRunning = true;
    compactThread = make_shared<thread>([this] { CompactMonitor(); });
    LOG_INFO("SSDEngine start");
}

/// 压缩监控方法，达到检查周期时调用表的压缩接口
void SSDEngine::CompactMonitor()
{
    LOG_DEBUG("SSDEngine start CompactMonitor");
    auto start = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
    chrono::microseconds loopDuration = 100ms;
    chrono::seconds duration;
    while (isRunning) {
        duration = chrono::duration_cast<std::chrono::seconds>(end - start);
        if (duration >= compactPeriod) {
            LOG_DEBUG("SSDEngine CompactMonitor start compact");
            for (const auto &item: as_const(tableMap)) {
                item.second->Compact(false);
            }
            LOG_DEBUG("SSDEngine CompactMonitor end compact");
            start = chrono::high_resolution_clock::now();
        }
        this_thread::sleep_for(loopDuration);
        end = chrono::high_resolution_clock::now();
    }
    LOG_DEBUG("SSDEngine end CompactMonitor");
}

vector<vector<float>> SSDEngine::FetchEmbeddings(const string &tableName, vector<emb_cache_key_t> &keys)
{
    CheckSSDEngineIsRunning();
    auto it = as_const(tableMap).find(tableName);
    CheckTableExist(it == tableMap.end(), tableName);

    return it->second->FetchEmbeddings(keys);
}

void SSDEngine::Stop()
{
    CheckSSDEngineIsRunning();
    isRunning = false;
    compactThread->join();
    tableMap.clear();
    compactThread = nullptr;

    LOG_INFO("SSDEngine stop");
}

/// 设置文件压缩的周期
/// \param seconds 文件压缩的周期
void SSDEngine::SetCompactPeriod(chrono::seconds seconds)
{
    compactPeriod = seconds;
}

/// 设置文件压缩的阈值
/// \param threshold 无效数据占比阈值
void SSDEngine::SetCompactThreshold(double threshold)
{
    if (threshold >= 0 && threshold <= 1) {
        compactThreshold = threshold;
        return;
    }
    Table::ThrowInvalidArgError("Compact threshold should in range [0, 1].");
}

int64_t SSDEngine::GetTableUsage(const string &tableName)
{
    CheckSSDEngineIsRunning();
    auto it = as_const(tableMap).find(tableName);
    if (it == tableMap.end()) {
        return -1;
    }
    return static_cast<int64_t>(it->second->GetTableUsage());
}

void SSDEngine::InsertEmbeddingsByAddr(const string& tableName, vector<emb_cache_key_t>& keys,
                                       vector<float*>& embeddingsAddr, uint64_t extEmbeddingSize)
{
    CheckSSDEngineIsRunning();
    auto it = as_const(tableMap).find(tableName);
    CheckTableExist(it == tableMap.end(), tableName);

    if (keys.size() != embeddingsAddr.size()) {
        Table::ThrowInvalidArgError("Param keys' length not equal to embeddings' length.");
    }

    it->second->InsertEmbeddingsByAddr(keys, embeddingsAddr, extEmbeddingSize);
}

vector<pair<string, vector<emb_cache_key_t>>> SSDEngine::ExportTableKey()
{
    vector<pair<string, vector<emb_cache_key_t>>> tableKeysVec;
    for (const auto& p : tableMap) {
        tableKeysVec.emplace_back(p.first, p.second->ExportKeys());
    }
    return tableKeysVec;
}

void SSDEngine::CheckSSDEngineIsRunning() const
{
    if (isRunning) {
        return;
    }
    auto error = Error(ModuleName::M_SSD_ENGINE, ErrorType::LOGIC_ERROR, "SSDEngine not running.");
    LOG_ERROR(error.ToString());
    throw std::invalid_argument(error.ToString());
}

void SSDEngine::CheckTableExist(bool isThrowError, const string& tableName)
{
    if (isThrowError) {
        auto error = Error(ModuleName::M_SSD_ENGINE, ErrorType::LOGIC_ERROR,
                           Logger::Format("Table:{} not found in 'tableMap'.", tableName));
        LOG_ERROR(error.ToString());
        throw std::invalid_argument(error.ToString());
    }
}
