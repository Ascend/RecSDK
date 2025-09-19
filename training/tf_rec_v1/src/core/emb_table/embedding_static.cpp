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

#include "emb_table/embedding_static.h"

#include "log/logger.h"
#include "error/error.h"
#include "file_system/file_system_handler.h"
#include "hybrid_mgmt/hybrid_mgmt.h"

using namespace MxRec;

EmbeddingStatic::EmbeddingStatic()
{
}

EmbeddingStatic::EmbeddingStatic(const EmbInfo& info, const RankInfo& rankInfo, int inSeed)
    : EmbeddingTable(info, rankInfo, inSeed)
{
}

EmbeddingStatic::~EmbeddingStatic()
{
}

void EmbeddingStatic::Key2Offset(std::vector<emb_key_t>& keys, int channel)
{
    auto ret = FindKeyOffset(keys);
    if (!ret.empty()) {
        EmplaceKeyOffset(keys, ret, channel);
    }
}

std::vector<size_t> EmbeddingStatic::FindKeyOffset(std::vector<emb_key_t>& keys)
{
    std::shared_lock<std::shared_mutex> lock(keyOffsetMutex_);
    std::vector<size_t> newKeysIdx;
    for (size_t i = 0; i < keys.size(); ++i) {
        auto& key = keys[i];
        if (key == INVALID_KEY_VALUE) {
            continue;
        }
        const auto& iter = keyOffsetMap.find(key);
        if (iter != keyOffsetMap.end()) {
            key = iter->second;
            continue;
        }
        newKeysIdx.emplace_back(i);
    }
    return newKeysIdx;
}

void EmbeddingStatic::EmplaceKeyOffset(std::vector<emb_key_t>& keys, const std::vector<size_t>& newKeysIdx, int channel)
{
    std::unique_lock<std::shared_mutex> lock(keyOffsetMutex_);
    for (const auto& it : newKeysIdx) {
        auto& key = keys[it];
        if (!evictDevPos.empty() && channel == TRAIN_CHANNEL_ID) {
            // 新值, emb有pos可复用
            auto offset = evictDevPos.back();
            auto ret = keyOffsetMap.try_emplace(key, offset);
            if (ret.second) {
                evictDevPos.pop_back();
            }
            key = ret.first->second;
            continue;
        } else if (channel != TRAIN_CHANNEL_ID) {
            key = INVALID_KEY_VALUE;
            continue;
        }
        auto ret = keyOffsetMap.try_emplace(key, maxOffset);
        if (ret.second) {
            RecordPaddingKeysOffset(channel, key, maxOffset);
            maxOffset++;
        }
        key = ret.first->second;
    }
    if (maxOffset > devVocabSize) {
        string errMsg = Logger::Format("Device cache overflow! Please set a grater value for `device_vocabulary_size` "
                                       "parameter when invoke `create_table` function. "
                                       "Current offset:{}, device_vocabulary_size:{}, table:{}.",
                                       maxOffset, devVocabSize, name);
        auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::INVALID_ARGUMENT, errMsg);
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }
}

void EmbeddingStatic::Key2OffsetForDp(std::vector<emb_key_t>& keys, int channel)
{
    std::unique_lock<std::shared_mutex> lock(keyOffsetMutex_); // lock for PROCESS_THREAD
    for (emb_key_t& key : keys) {
        if (key == INVALID_KEY_VALUE) {
            continue;
        }
        const auto& iter = keyOffsetMap.find(key);
        if (iter != keyOffsetMap.end()) {
            key = iter->second;
            continue;
        }
        // New key.
        if (channel == TRAIN_CHANNEL_ID) {
            auto error =
                Error(ModuleName::M_EMB_TABLE, ErrorType::NOT_FOUND,
                      StringFormat("LookupKeys contains invalid key:%d,"
                                   " key must exist in the offset map, table:%s.", key, name.c_str()));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString());
        }
        key = INVALID_KEY_VALUE;
    }
}

int64_t EmbeddingStatic::capacity() const
{
    return this->devVocabSize;
}

void EmbeddingStatic::Save(const string& savePath, const int pythonBatchId, bool saveDelta,
                           const map<emb_key_t, KeyInfo>& keyInfo)
{
    // Param pythonBatchId not use in this method, and only use in embedding_ddr.
    SaveKey(savePath, saveDelta, keyInfo);
}

void EmbeddingStatic::SaveKey(const string& savePath, bool saveDelta, const map<emb_key_t, KeyInfo>& keyInfo)
{
    stringstream ss;
    ss << savePath << "/" << name << "/key/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    deviceKey.clear();
    deviceOffset.clear();

    if (saveDelta) {
        for (const auto& it : keyInfo) {
            auto result = keyOffsetMap.find(it.first);
            if (result == keyOffsetMap.end()) {
                auto error = MxRec::Error(ModuleName::M_EMB_TABLE, ErrorType::NOT_FOUND,
                                          StringFormat("Key: %s not in keyOffsetMap, please check if deltaMap "
                                                       "update correctly or get keyInfo from deltaMap is correct,"
                                                       " table:%s.", it.first, name.c_str()));
                LOG_ERROR(error.ToString());
                throw runtime_error(StringFormat("Key: %s not in keyOffsetMap.", it.first));
            }
            deviceKey.push_back(result->first);
            deviceOffset.push_back(result->second);
        }
    } else {
        for (const auto& it: keyOffsetMap) {
            deviceKey.push_back(it.first);
            deviceOffset.push_back(it.second);
        }
    }

    LOG_INFO("Get device keys and offsets, table: {}, save path: {}, rank id: {}, device key size: {}, device offset "
             "size: {}.", name, savePath, rankId_, deviceKey.size(), deviceOffset.size());

    CheckFileSystemPtr();

    size_t writeSize = static_cast<size_t>(deviceKey.size() * sizeof(int64_t));
    ssize_t res = fileSystemPtr_->Write(ss.str(), reinterpret_cast<const char *>(deviceKey.data()), writeSize);
    if (res == -1) {
        auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::LOGIC_ERROR,
                           StringFormat("Save keys failed. An error occurred while writing file %s, table:%s",
                                        ss.str().c_str(), name.c_str()));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }
    if (res != writeSize) {
        auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::LOGIC_ERROR,
                           StringFormat("Save keys failed. Expected to write %d bytes, "
                                        "but actually write %d bytes to file %s, table:%s",
                                        writeSize, res, ss.str().c_str(), name.c_str()));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }
}

void EmbeddingStatic::Load(const string& savePath, map<string, unordered_set<emb_cache_key_t>>& trainKeySet,
                           const vector<string>& warmStartTables)
{
    LoadKey(savePath);
}

void EmbeddingStatic::LoadKey(const string& savePath)
{
    CheckFileSystemPtr();

    stringstream ss;
    ss << savePath << "/" << name << "/key/slice.data";
    size_t fileSize = fileSystemPtr_->GetFileSize(ss.str());

    CheckReadKeyFileSize(ss.str(), fileSize);
    int64_t* buf = static_cast<int64_t*>(malloc(fileSize));
    CheckLoadKeyMallocPtr(buf, fileSize);

    try {
        ssize_t res = fileSystemPtr_->Read(ss.str(), reinterpret_cast<char*>(buf), fileSize);
        CheckReadKeyFileBytes(res, ss.str(), fileSize);
    } catch (std::runtime_error& e) {
        free(static_cast<void*>(buf));
        auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::IO_ERROR,
                           StringFormat("Failed to read file, table:%s, error: %s.", name.c_str(), e.what()));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }

    size_t loadKeySize = fileSize / sizeof(int64_t);
    loadOffset.clear();
    int keyCount = 0;
    for (int i = 0; i < loadKeySize; i = i + 1) {
        if (!embInfo_.isDp && buf[i] % rankSize_ != rankId_) {
            continue;
        }
        keyOffsetMap[buf[i]] = keyCount;
        loadOffset.push_back(i);
        keyCount++;
    }

    if (loadOffset.size() > devVocabSize) {
        free(static_cast<void*>(buf));
        auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::LOGIC_ERROR,
                           StringFormat("Load keys failed, table:%s, load key size:%d exceeds device vocab size:%d.",
                                        name.c_str(), loadOffset.size(), devVocabSize));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }

    maxOffset = keyOffsetMap.size();
    free(static_cast<void*>(buf));
}

vector<int64_t> EmbeddingStatic::GetDeviceOffset()
{
    return deviceOffset;
}

void EmbeddingStatic::BackUpTrainStatus()
{
    keyOffsetMapBackUp = keyOffsetMap;
}

void EmbeddingStatic::RecoverTrainStatus()
{
    if (keyOffsetMapBackUp.size()!=0) {
        keyOffsetMap = keyOffsetMapBackUp;
        keyOffsetMapBackUp.clear();
    }
}
