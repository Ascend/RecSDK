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

#include "emb_table/embedding_table.h"
#include "utils/logger.h"
#include "utils/singleton.h"
#include "hd_transfer/hd_transfer.h"
#include "file_system/file_system_handler.h"

using namespace MxRec;

EmbeddingTable::EmbeddingTable()
{
}

EmbeddingTable::EmbeddingTable(const EmbInfo& info, const RankInfo& rankInfo, int inSeed)
    : name(info.name), hostVocabSize(info.hostVocabSize), devVocabSize(info.devVocabSize),
      ssdVocabSize(info.ssdVocabSize), freeSize_(0), maxOffset(0), isDynamic_(rankInfo.useDynamicExpansion),
      embSize_(info.embeddingSize), extEmbSize_(info.extEmbeddingSize),
      embInfo_(info), seed_(inSeed), rankId_(rankInfo.rankId), rankSize_(rankInfo.rankSize)
{
    LOG_INFO("table {} isDynamic = {} embeddingSize {} extSize {}", name, isDynamic_, embSize_, extEmbSize_);
}

EmbeddingTable::~EmbeddingTable()
{
}

void EmbeddingTable::Key2Offset(std::vector<emb_key_t>& keys, int channel)
{
    return;
}

size_t EmbeddingTable::GetMaxOffset()
{
    return maxOffset;
}

int64_t EmbeddingTable::capacity() const
{
    return static_cast<int64_t>(devVocabSize);
}

size_t EmbeddingTable::size() const
{
    return maxOffset;
}

void EmbeddingTable::EvictKeys(const std::vector<emb_cache_key_t>& keys)
{
    std::lock_guard<std::mutex> lk(mut_); // lock for PROCESS_THREAD
    size_t keySize = keys.size();
    for (size_t i = 0; i < keySize; i++) {
        emb_key_t key = keys[i];
        if (key == INVALID_KEY_VALUE) {
            LOG_WARN("evict key is INVALID_KEY_VALUE!");
            continue;
        }
        const auto& iter = keyOffsetMap.find(key);
        if (iter == keyOffsetMap.end()) { // not found
            continue;
        }
        keyOffsetMap.erase(iter);
        evictDevPos.emplace_back(iter->second);
        LOG_TRACE("evict embName:{}, offset:{}", name, iter->second);
    }
    LOG_INFO("EvictKeys: table [{}] evict size on dev:{}", name, evictDevPos.size());
}

const std::vector<int64_t>& EmbeddingTable::GetEvictedKeys()
{
    return evictDevPos;
}

const std::vector<int64_t>& EmbeddingTable::GetHostEvictedKeys()
{
    return evictHostPos;
}

void EmbeddingTable::EvictInitDeviceEmb()
{
    if (evictDevPos.size() > devVocabSize) {
        LOG_ERROR("{} overflow! init evict dev, evictOffset size {} bigger than dev vocabSize {}",
            name, evictDevPos.size(), devVocabSize);
        throw runtime_error(
            Logger::Format("{} overflow! init evict dev, evictOffset size {} bigger than dev vocabSize {}",
                name, evictDevPos.size(), devVocabSize).c_str());
    }

    vector<Tensor> tmpDataOut;
    Tensor tmpData = Vec2TensorI32(evictDevPos);
    tmpDataOut.emplace_back(tmpData);
    tmpDataOut.emplace_back(Tensor(tensorflow::DT_INT32, { 1 }));

    auto evictLen = tmpDataOut.back().flat<int32>();
    evictLen(0) = static_cast<int>(evictDevPos.size());

    // evict key发送给dev侧，dev侧初始化emb
    auto trans = Singleton<HDTransfer>::GetInstance();
    trans->Send(TransferChannel::EVICT, tmpDataOut, TRAIN_CHANNEL_ID, name);

    LOG_INFO(KEY_PROCESS "hbm EvictInitDeviceEmb: [{}]! send offsetSize:{}", name, evictDevPos.size());
}

absl::flat_hash_map<emb_key_t, int64_t> EmbeddingTable::GetKeyOffsetMap()
{
    return keyOffsetMap;
}

void EmbeddingTable::SetFileSystemPtr(const string& savePath)
{
    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    fileSystemPtr_ = fileSystemHandler->Create(savePath);
}

void EmbeddingTable::UnsetFileSystemPtr()
{
    fileSystemPtr_ = nullptr;
}

vector<int64_t> EmbeddingTable::GetLoadOffset()
{
    return loadOffset;
}

void EmbeddingTable::Load(const string& filePath, map<string, unordered_set<emb_cache_key_t>>& trainKeySet)
{
}

void EmbeddingTable::Save(const string& filePath, bool saveDelta)
{
}

void EmbeddingTable::BackUpTrainStatus()
{
}

void EmbeddingTable::RecoverTrainStatus()
{
}

void EmbeddingTable::MakeDir(const string& dirName)
{
    if (fileSystemPtr_ == nullptr) {
        throw runtime_error("failed to obtain the file system pointer, the file system pointer is null.");
    }
    fileSystemPtr_->CreateDir(dirName);
}

void EmbeddingTable::SetCacheManager(CacheManager *cm)
{
}

TableInfo EmbeddingTable::GetTableInfo()
{
    TableInfo ti = {
        .name=name,
        .hostVocabSize=hostVocabSize,
        .devVocabSize=devVocabSize,
        .maxOffset=maxOffset,
        .keyOffsetMap=keyOffsetMap,
    };
    return ti;
}

vector<int64_t> EmbeddingTable::GetDeviceOffset()
{
    return vector<int64_t>{};
}

void EmbeddingTable::SetOptimizerInfo(OptimizerInfo& optimizerInfo)
{
}

void EmbeddingTable::SetHDTransfer(HDTransfer *hdTransfer)
{
}

void EmbeddingTable::SetEmbCache(ock::ctr::EmbCacheManagerPtr embCache)
{
}
