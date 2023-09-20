/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 */
#include "ssd_engine.h"

using namespace MxRec;
using namespace std;

bool SSDEngine::IsTableExist(const string &tableName)
{
    if (!isRunning) {
        throw invalid_argument("SSDEngine not running");
    }
    auto it = as_const(tableMap).find(tableName);
    return !(it == tableMap.end());
}

bool SSDEngine::IsKeyExist(const string &tableName, emb_key_t key)
{
    if (!isRunning) {
        throw invalid_argument("SSDEngine not running");
    }
    auto it = as_const(tableMap).find(tableName);
    if (it == tableMap.end()) {
        throw invalid_argument("table not found");
    }
    return it->second->IsKeyExist(key);
}

void SSDEngine::CreateTable(const string &tableName, vector<string> savePaths, uint64_t maxTableSize)
{
    if (!isRunning) {
        throw invalid_argument("SSDEngine not running");
    }
    if (savePaths.empty()) {
        throw invalid_argument("SSDEngine input savePaths is empty");
    }
    auto it = as_const(tableMap).find(tableName);
    if (it != tableMap.end()) {
        throw invalid_argument("table already exist");
    }
    tableMap[tableName] = make_shared<Table>(tableName, savePaths, maxTableSize, compactThreshold);
}

void SSDEngine::InsertEmbeddings(const string &tableName, vector<emb_key_t> &keys, vector<vector<float>> &embeddings)
{
    if (!isRunning) {
        throw invalid_argument("SSDEngine not running");
    }
    auto it = as_const(tableMap).find(tableName);
    if (it == tableMap.end()) {
        throw invalid_argument("table not found");
    }

    if (keys.size() != embeddings.size()) {
        throw invalid_argument("keys' length not equal to embeddings' length");
    }

    it->second->InsertEmbeddings(keys, embeddings);
}

void SSDEngine::DeleteEmbeddings(const string &tableName, vector<emb_key_t> &keys)
{
    if (!isRunning) {
        throw invalid_argument("SSDEngine not running");
    }
    auto it = as_const(tableMap).find(tableName);
    if (it == tableMap.end()) {
        throw invalid_argument("table not found");
    }

    it->second->DeleteEmbeddings(keys);
}

int64_t SSDEngine::GetTableAvailableSpace(const string &tableName)
{
    if (!isRunning) {
        throw invalid_argument("SSDEngine not running");
    }
    auto it = as_const(tableMap).find(tableName);
    if (it == tableMap.end()) {
        throw invalid_argument("table not found");
    }

    return it->second->GetTableAvailableSpace();
}

void SSDEngine::Save(int step)
{
    if (!isRunning) {
        throw invalid_argument("SSDEngine not running");
    }
    for (auto item: as_const(tableMap)) {
        item.second->Save(step);
    }
}

void SSDEngine::Load(const string &tableName, vector<string> savePaths, uint64_t maxTableSize, int step)
{
    if (!isRunning) {
        throw invalid_argument("SSDEngine not running");
    }
    auto it = as_const(tableMap).find(tableName);
    if (it != tableMap.end()) {
        throw invalid_argument("table already exist");
    }

    tableMap[tableName] = make_shared<Table>(tableName, savePaths, maxTableSize, compactThreshold, step);
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

vector<vector<float>> SSDEngine::FetchEmbeddings(const string &tableName, vector<emb_key_t> &keys)
{
    if (!isRunning) {
        throw invalid_argument("SSDEngine not running");
    }
    auto it = as_const(tableMap).find(tableName);
    if (it == tableMap.end()) {
        throw invalid_argument("table not found");
    }

    return it->second->FetchEmbeddings(keys);
}

void SSDEngine::Stop()
{
    if (!isRunning) {
        throw invalid_argument("SSDEngine not running");
    }
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
    throw invalid_argument("compact threshold should in range [0, 1]");
}