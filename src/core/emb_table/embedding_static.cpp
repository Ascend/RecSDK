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

#include "utils/logger.h"
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
    std::lock_guard<std::mutex> lk(mut_); // lock for PROCESS_THREAD
    for (emb_key_t& key : keys) {
        if (key == INVALID_KEY_VALUE) {
            continue;
        }
        const auto& iter = keyOffsetMap.find(key);
        if (iter != keyOffsetMap.end()) {
            key = iter->second;
            continue;
        }
        if (evictDevPos.size() != 0 && channel == TRAIN_CHANNEL_ID) {
            // 新值, emb有pos可复用
            size_t offset = evictDevPos.back();
            keyOffsetMap[key] = offset;
            key = offset;
            evictDevPos.pop_back();
            continue;
        }
        // 新值
        if (channel != TRAIN_CHANNEL_ID) {
            key = INVALID_KEY_VALUE;
            continue;
        }
        keyOffsetMap[key] = maxOffset;
        key = maxOffset++;
    }
    if (maxOffset > devVocabSize) {
        LOG_ERROR("dev cache overflow {} > {}", maxOffset, devVocabSize);
        throw std::runtime_error("dev cache overflow!");
    }
}

void EmbeddingStatic::Key2OffsetForDp(std::vector<emb_key_t>& keys, int channel)
{
    std::lock_guard<std::mutex> lk(mut_); // lock for PROCESS_THREAD
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
            throw runtime_error(StringFormat("Error: LookupKeys contains invalid key %d, "
                                             "the key must exist in the offset map.", key));
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

    for (const auto& it: keyOffsetMap) {
        // When saving a delta model, you need to first extract the keys from deltaMap[name] where isChanged is true
        // from the keyOffsetMap.
        if (saveDelta) {
            auto result = keyInfo.find(it.first);
            if (result == keyInfo.end() || !result->second.isChanged) {
                continue;
            }
        }
        deviceKey.push_back(it.first);
        deviceOffset.push_back(it.second);
    }
    LOG_INFO("Device key size: {}, device offset size: {}.", deviceKey.size(), deviceOffset.size());

    if (fileSystemPtr_ == nullptr) {
        throw runtime_error("failed to obtain the file system pointer, the file system pointer is null.");
    }

    size_t writeSize = static_cast<size_t>(deviceKey.size() * sizeof(int64_t));
    ssize_t res = fileSystemPtr_->Write(ss.str(), reinterpret_cast<const char *>(deviceKey.data()), writeSize);
    if (res == -1) {
        throw runtime_error(StringFormat("Error: Save keys failed. "
                                         "An error occurred while writing file: %s.", ss.str().c_str()));
    }
    if (res != writeSize) {
        throw runtime_error(StringFormat("Error: Save keys failed. Expected to write %d bytes, "
                                         "but actually write %d bytes to file %s.", writeSize, res, ss.str().c_str()));
    }
}

void EmbeddingStatic::Load(const string& savePath, map<string, unordered_set<emb_cache_key_t>>& trainKeySet)
{
    LoadKey(savePath);
}

void EmbeddingStatic::LoadKey(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/" << name << "/key/slice.data";

    if (fileSystemPtr_ == nullptr) {
        throw runtime_error("failed to obtain the file system pointer, the file system pointer is null.");
    }
    size_t fileSize = fileSystemPtr_->GetFileSize(ss.str());
    if (fileSize >= FILE_MAX_SIZE) {
        throw runtime_error(StringFormat("Error: Load keys failed. "
                                         "file %s size %d is too big.", ss.str().c_str(), fileSize));
    }

    int64_t* buf = static_cast<int64_t*>(malloc(fileSize));
    if (buf == nullptr) {
        throw runtime_error(StringFormat("Error: Load keys failed. "
                                         "failed to allocate %d bytes using malloc.", fileSize));
    }

    ssize_t res = fileSystemPtr_->Read(ss.str(), reinterpret_cast<char *>(buf), fileSize);
    if (res == -1) {
        throw runtime_error(StringFormat("Error: Load keys failed. "
                                         "An error occurred while reading file: %s.", ss.str().c_str()));
    }
    if (res != fileSize) {
        throw runtime_error(StringFormat("Error: Load keys failed. Expected to read %d bytes, "
                                         "but actually read %d bytes to file %s.", fileSize, res, ss.str().c_str()));
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
        throw runtime_error(StringFormat("Error: Load keys failed. Load key size :%d exceeds device vocab size: %d.",
                                         loadOffset.size(), devVocabSize));
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
