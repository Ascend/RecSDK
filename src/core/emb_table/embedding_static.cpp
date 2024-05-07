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

int64_t EmbeddingStatic::capacity() const
{
    return this->devVocabSize;
}

void EmbeddingStatic::Save(const string& savePath)
{
    SaveKey(savePath);
}

int EmbeddingStatic::SaveKey(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/" << name << "/key/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    deviceKey.clear();
    deviceOffset.clear();

    for (const auto& it: keyOffsetMap) {
        deviceKey.push_back(it.first);
        deviceOffset.push_back(it.second);
    }

    size_t writeSize = static_cast<size_t>(deviceKey.size() * sizeof(int64_t));
    ssize_t res = fileSystemPtr->Write(ss.str(), reinterpret_cast<const char *>(deviceKey.data()), writeSize);
    if (res == -1) {
        throw runtime_error("Error: Save keys failed. An error occurred while writing file: {}.", ss.str());
    }
    if (res != writeSize) {
        throw runtime_error(
                "Error: Save keys failed. Expected to write {} bytes, but actually write {} bytes to file {}.",
                writeSize, res, ss.str());
    }
}

void EmbeddingStatic::Load(const string& savePath)
{
    LoadKey(savePath);
}

void EmbeddingStatic::LoadKey(const string &savePath)
{
    stringstream ss;
    ss << savePath << "/" << name << "/key/slice.data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    size_t fileSize = fileSystemPtr->GetFileSize(ss.str());
    if (fileSize >= FILE_MAX_SIZE) {
        throw runtime_error("Error: Load keys failed. file {} size {}  is too big.", ss.str(), fileSize);
    }

    int64_t* buf = static_cast<int64_t *>(malloc(fileSize));
    if (buf == nullptr) {
        throw runtime_error("Error: Load keys failed. failed to allocate {} bytes using malloc.", fileSize);
    }

    ssize_t res = fileSystemPtr->Read(ss.str(), reinterpret_cast<char *>(buf), fileSize);
    if (res == -1) {
        throw runtime_error("Error: Load keys failed. An error occurred while reading file: {}.", ss.str());
    }
    if (res != fileSize) {
        throw runtime_error(
                "Error: Load keys failed. Expected to read {} bytes, but actually read {} bytes to file {}.",
                fileSize, res, ss.str());
    }

    size_t loadKeySize = fileSize / sizeof(int64_t);
    loadOffset.clear();
    int keyCount = 0;
    for (int i = 0; i < loadKeySize; i = i + 1) {
        if (buf[i] % rankSize_ == rankId_) {
            keyOffsetMap[buf[i]] = keyCount;
            loadOffset.push_back(i);
            keyCount++;
        }
    }

    if (loadOffset.size() > devVocabSize) {
        throw runtime_error("Error: Load keys failed. Load key size :{} exceeds device vocab size: {}.",
                            loadOffset.size(), devVocabSize);
    }

    maxOffset = keyOffsetMap.size();

    free(static_cast<void*>(buf));
}

vector<int64_t> EmbeddingStatic::GetDeviceOffset()
{
    return deviceOffset;
}