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
    : name(info.name), hostVocabSize(info.hostVocabSize), devVocabSize(info.devVocabSize),
      freeSize_(0), maxOffset(0), isDynamic_(rankInfo.useDynamicExpansion),
      embSize_(info.embeddingSize), extEmbSize_(info.extEmbeddingSize),
      embInfo_(info), seed_(inSeed), rankId_(rankInfo.rankId)
{
    LOG_TRACE("table {} isDynamic = {} embeddingSize {} extSize {}",
              name, isDynamic_, embSize_, extEmbSize_);
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
    return devVocabSize;
}

size_t EmbeddingTable::GetHostVocabSize()
{
    return hostVocabSize;
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

void EmbeddingTable::SetCacheManager(CacheManager *cm)
{
}

void EmbeddingTable::EnableSSD()
{
    isSSDEnabled_ = true;
}

void EmbeddingTable::RefreshFreqInfoWithSwap()
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
        .evictDevPos=evictDevPos,
        .evictHostPos=evictHostPos,
    };
    return ti;
}
