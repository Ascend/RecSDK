/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: emb table
 * Author: MindX SDK
 * Date: 2023/12/11
 */

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
    : name_(info.name), hostVocabSize_(info.hostVocabSize), devVocabSize_(info.devVocabSize),
      freeSize_(0), maxOffset_(0), isDynamic_(rankInfo.useDynamicExpansion),
      embSize_(info.embeddingSize), extEmbSize_(info.extEmbeddingSize),
      embInfo_(info), seed_(inSeed), rankId_(rankInfo.rankId)
{
    LOG_TRACE("table {} isDynamic = {} embeddingSize {} extSize {}",
              name_, isDynamic_, embSize_, extEmbSize_);
}

EmbeddingTable::~EmbeddingTable()
{
}

void EmbeddingTable::Key2Offset(std::vector<emb_key_t>& keys, int channel)
{
    return;
}

void EmbeddingTable::FindOffset(const vector<emb_key_t>& keys,
                                size_t currentBatchId, size_t keepBatchId, int channelId)
{
    return;
}

std::vector<int32_t> EmbeddingTable::FindOffset(const vector<emb_key_t>& keys,
                                                size_t batchId, int channelId,
                                                std::vector<size_t>& swapPos)
{
    return {};
}

size_t EmbeddingTable::GetMaxOffset()
{
    return maxOffset_;
}

int64_t EmbeddingTable::capacity() const
{
    return static_cast<int64_t>(devVocabSize_);
}

size_t EmbeddingTable::size() const
{
    return maxOffset_;
}

void EmbeddingTable::EvictKeys(const std::vector<emb_key_t>& keys)
{
    std::lock_guard<std::mutex> lk(mut_); // lock for PROCESS_THREAD
    size_t keySize = keys.size();
    for (size_t i = 0; i < keySize; i++) {
        emb_key_t key = keys[i];
        if (key == INVALID_KEY_VALUE) {
            LOG_WARN("evict key is INVALID_KEY_VALUE!");
            continue;
        }
        const auto& iter = keyOffsetMap_.find(key);
        if (iter == keyOffsetMap_.end()) { // not found
            continue;
        }
        keyOffsetMap_.erase(iter);
        evictPos_.emplace_back(iter->second);
        LOG_TRACE("evict embName:{}, offset:{}", name_, iter->second);
    }
    LOG_INFO("EvictKeys: table [{}] evict size on dev:{}", name_, evictPos_.size());
}

const std::vector<int64_t>& EmbeddingTable::GetEvictedKeys()
{
    return evictPos_;
}

const std::vector<int64_t>& EmbeddingTable::GetHostEvictedKeys()
{
    return evictHostPos_;
}

void EmbeddingTable::EvictInitDeviceEmb()
{
    if (evictPos_.size() > devVocabSize_) {
        LOG_ERROR("{} overflow! init evict dev, evictOffset size {} bigger than dev vocabSize {}",
            name_, evictPos_.size(), devVocabSize_);
        throw runtime_error(
            Logger::Format("{} overflow! init evict dev, evictOffset size {} bigger than dev vocabSize {}",
                name_, evictPos_.size(), devVocabSize_).c_str());
    }

    vector<Tensor> tmpDataOut;
    Tensor tmpData = Vec2TensorI32(evictPos_);
    tmpDataOut.emplace_back(tmpData);
    tmpDataOut.emplace_back(Tensor(tensorflow::DT_INT32, { 1 }));

    auto evictLen = tmpDataOut.back().flat<int32>();
    evictLen(0) = static_cast<int>(evictPos_.size());

    // evict key发送给dev侧，dev侧初始化emb
    auto trans = Singleton<HDTransfer>::GetInstance();
    trans->Send(TransferChannel::EVICT, tmpDataOut, TRAIN_CHANNEL_ID, name_);

    LOG_INFO(KEY_PROCESS "hbm EvictInitDeviceEmb: [{}]! send offsetSize:{}", name_, evictPos_.size());
}

absl::flat_hash_map<emb_key_t, int64_t> EmbeddingTable::GetKeyOffsetMap()
{
    return keyOffsetMap_;
}

void EmbeddingTable::ClearMissingKeys()
{
    missingKeysHostPos_.clear();
}

const std::vector<size_t>& EmbeddingTable::GetMissingKeys()
{
    return missingKeysHostPos_;
}

void EmbeddingTable::SetStartCount()
{
}

void EmbeddingTable::ClearLookupAndSwapOffset()
{
}

size_t EmbeddingTable::GetDevVocabSize()
{
    return devVocabSize_;
}

size_t EmbeddingTable::GetHostVocabSize()
{
    return hostVocabSize_;
}

int EmbeddingTable::Load(const string& filePath)
{
    return 0;
}

int EmbeddingTable::Save(const string& filePath)
{
    return 0;
}

void EmbeddingTable::MakeDir(const string& dirName)
{
    auto fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(dirName);
    fileSystemPtr->CreateDir(dirName);
}
