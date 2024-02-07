/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: EmbeddingDDR DDR 模式embedding表实现
 * Author: MindX SDK
 * Date: 2023/12/11
 */

#include "emb_table/embedding_ddr.h"
#include "utils/logger.h"
#include "utils/singleton.h"
#include "host_emb/host_emb.h"
#include "hd_transfer/hd_transfer.h"
#include "file_system/file_system_handler.h"

using namespace MxRec;

constexpr int ELEMENT_NUM = 4;
constexpr int CURRENT_UPDATE_IDX = 0;
constexpr int HOST_VOCAB_SIZE_IDX = 1;
constexpr int DEV_VOCAB_SIZE_IDX = 2;
constexpr int MAX_OFFSET_IDX = 3;

constexpr int EMB_INFO_ELEMENT_NUM = 3;
constexpr int EMB_INFO_EXT_SIZE_IDX = 0;
constexpr int EMB_INFO_DEV_VOCAB_SIZE_IDX = 1;
constexpr int EMB_INFO_HOST_VOCAB_SIZE_IDX = 2;

EmbeddingDDR::EmbeddingDDR()
{
}

EmbeddingDDR::EmbeddingDDR(const EmbInfo& info, const RankInfo& rankInfo, int inSeed)
    : EmbeddingTable(info, rankInfo, inSeed)
{
    LOG_INFO("Init DDR table [{}] devVocabSize = {} hostVocabSize = {}", name_, devVocabSize_, hostVocabSize_);
    currentUpdatePos = 0;
    devOffset2Key.resize(devVocabSize_);
    devOffset2Batch.resize(devVocabSize_);
    std::fill(devOffset2Batch.begin(), devOffset2Batch.end(), -1);
    std::fill(devOffset2Key.begin(), devOffset2Key.end(), -1);
}

EmbeddingDDR::~EmbeddingDDR()
{
}

void EmbeddingDDR::Key2Offset(std::vector<emb_key_t>& splitKey, int channel)
{
}

int64_t EmbeddingDDR::capacity() const
{
    return capacity_;
}

std::vector<int32_t> EmbeddingDDR::FindOffset(const vector<emb_key_t>& keys,
                                              size_t batchId, int channelId,
                                              std::vector<size_t>& swapPos)
{
    devOffset2KeyOld.clear();
    oldSwap.clear();
    maxOffsetOld = maxOffset_;

    UpdateBatchId(keys, batchId);
    std::vector<int32_t> lookUpVec;
    for (size_t i = 0; i < keys.size(); i++) {
        emb_key_t key = keys[i];
        if (key == INVALID_KEY_VALUE) {
            lookUpVec.emplace_back(INVALID_KEY_VALUE);
            continue;
        }
        emb_key_t offset = FindOffsetHelper(key, channelId);
        if (offset == INVALID_KEY_VALUE) {
            lookUpVec.emplace_back(INVALID_KEY_VALUE);
            continue;
        }
        if (offset < devVocabSize_) {
            // 偏移小于等于HBM容量：直接放入查询向量；更新偏移之前关联的key和当前关联的key
            lookUpVec.push_back(offset);
            devOffset2KeyOld.emplace_back(offset, static_cast<int>(devOffset2Key[offset]));
            devOffset2Key[offset] = key;
        } else {
            // 偏移大于HBM容量：记录在host emb上的偏移；找到需要交换的HBM偏移
            missingKeysHostPos_.emplace_back(offset - devVocabSize_);
            offset = FindSwapPosOld(key, offset, batchId, swapPos);
            lookUpVec.emplace_back(offset);
        }
    }
    if (batchId == 0) {
        LOG_INFO("max offset {}", maxOffset_);
    }
    LOG_TRACE("keyOffsetMap_, {}", MapToString(keyOffsetMap_));
    return lookUpVec;
}

emb_key_t EmbeddingDDR::FindOffsetHelper(const emb_key_t& key, int channelId)
{
    const auto& iter = keyOffsetMap_.find(key);
    emb_key_t offset = INVALID_KEY_VALUE;
    if (iter != keyOffsetMap_.end()) {
        offset = iter->second;
        LOG_TRACE("devVocabSize, {} , offset , {}", devVocabSize_, offset);
        if (offset >= devVocabSize_) {
            ddr2HbmKeys.emplace_back(key);
        }
        return offset;
    }
    if (channelId != TRAIN_CHANNEL_ID) {
        return offset;
    }
    if (evictPos_.size() != 0) { // 优先复用hbm表
        offset = evictPos_.back();
        keyOffsetMap_[key] = offset;
        LOG_TRACE("ddr mode, dev evictPos is not null, key [{}] reuse offset [{}], evictSize [{}]",
            key, offset, evictPos_.size());
        evictPos_.pop_back();
        LOG_ERROR("dev evicted offset = {}", offset);
        return offset;
    }

    if (evictHostPos_.size() != 0) { // hbm不足，再复用host/ddr表
        offset = evictHostPos_.back();
        keyOffsetMap_[key] = offset;
        LOG_TRACE("ddr mode, host evictPos is not null, key [{}] reuse offset [{}], evictSize [{}]",
            key, offset, evictHostPos_.size());
        evictHostPos_.pop_back();
        LOG_ERROR("host evicted offset = {}", offset);
        return offset;
    }
    keyOffsetMap_[key] = maxOffset_;
    offset = maxOffset_;
    maxOffset_++;
    if (maxOffset_ == devVocabSize_) {
        LOG_INFO("start using host vocab!");
    }
    if (maxOffset_ > (hostVocabSize_ + devVocabSize_)) {
        LOG_ERROR("hostVocabSize too small! dev:{} host:{}", devVocabSize_, hostVocabSize_);
        throw runtime_error("hostVocabSize too small");
    }
    return offset;
}

void EmbeddingDDR::UpdateBatchId(const vector<emb_key_t>& keys, size_t currentBatchId)
{
    for (size_t i = 0; i < keys.size(); i++) {
        size_t offset;
        emb_key_t key = keys[i];
        if (key == -1) {
            continue;
        }
        const auto& iter = keyOffsetMap_.find(key);
        if (iter != keyOffsetMap_.end()) {
            offset = iter->second;

            LOG_TRACE("key will be used, {} , offset , {}", key, offset);
            if (offset < devVocabSize_) {
                // devOffset2Batch size equal to devVocabSize, unnecessary to check index boundary
                devOffset2Batch[offset] = static_cast<int>(currentBatchId);
            }
        }
    }
}

/// 利用devOffset2Batch上key最近使用的batchId，来选择需要淘汰的key，记录淘汰位置和device侧所需的keys
/// \param embName 表名
/// \param key 输入特征
/// \param hostOffset 全局偏移
/// \param currentBatchId 已处理的batch数
/// \param keepBatchId 处理batch的次数，多个预取一起处理算一次
/// \return 是否找到需要交换的位置
emb_key_t EmbeddingDDR::FindSwapPosOld(emb_key_t key, size_t hostOffset, size_t batchId,
                                       std::vector<size_t>& swapPos)
{
    bool notFind = true;
    emb_key_t offset = INVALID_KEY_VALUE;
    while (notFind) {
        // 找到本次预取之前的偏移（保证所有预取batch的key都在HBM中）
        if (currentUpdatePos >= devOffset2Batch.size()) {
            LOG_ERROR("outofrange {} >= {}", currentUpdatePos, devOffset2Batch.size());
            throw runtime_error("currentUpdatePos out of range");
        }

        if (devOffset2Batch[currentUpdatePos] < static_cast<int>(batchId)) {
            devOffset2Batch[currentUpdatePos] = static_cast<int>(batchId);
            swapPos.emplace_back(currentUpdatePos); // 记录需要被换出的HBM偏移
            offset = currentUpdatePos;
            keyOffsetMap_[key] = currentUpdatePos;  // 更新key对应的HBM偏移
            // 记录HBM偏移之前的key
            devOffset2KeyOld.emplace_back(currentUpdatePos, devOffset2Key[currentUpdatePos]);
            auto& oldKey = devOffset2Key[currentUpdatePos];
            oldSwap.emplace_back(oldKey, key); // 记录交换的两个key oldKey:HBM->DDR key:DDR->HBM
            keyOffsetMap_[oldKey] = hostOffset; // 更新被替换的key的偏移
            oldKey = key;
            notFind = false;
        }
        currentUpdatePos++; // 查找位置+1
        freeSize_--;        // HBM可用空间-1

        // 遍历完一遍整个HBM表后，从头开始遍历
        if (currentUpdatePos == devVocabSize_) {
            currentUpdatePos = 0;
        }

        /**
         * currentUpdatePos已经绕了HBM一圈
         * 已经找完整个HBM空间，且没找到可用位置，表示HBM空间不足以放下整个batch（预取batch数）的key，
         * 无法正常执行训练，故运行时错误退出
         */
        if (currentUpdatePos == currentUpdatePosStart && notFind) {
            LOG_ERROR("devVocabSize is too small");
            throw runtime_error("devVocabSize is too small");
        }
    }
    return offset;
}

/*
* 删除淘汰key的映射关系，并将其offset更新到evictPos，待后续复用
*/
void EmbeddingDDR::EvictDeleteEmb(const vector<emb_key_t>& keys)
{
    EASY_FUNCTION()
    size_t keySize = keys.size();
    vector<emb_key_t> evictHBMKeys;
    vector<emb_key_t> evictDDRKeys;
    for (size_t i = 0; i < keySize; ++i) {
        size_t offset;
        emb_key_t key = keys[i];
        if (key == INVALID_KEY_VALUE) {
            LOG_WARN("evict key equal -1!");
            continue;
        }
        const auto& iter = keyOffsetMap_.find(key);
        if (iter == keyOffsetMap_.end()) {
            // 淘汰依据keyProcess中的history，hashmap映射关系创建于ParseKey；两者异步，造成淘汰的值在hashmap里可能未创建
            continue;
        }
        offset = iter->second;
        keyOffsetMap_.erase(iter);
        LOG_TRACE("evict embName {}, offset {}", name_, offset);

        if (offset < devVocabSize_) {
            // offset 在device中
            devOffset2Batch[offset] = -1;
            devOffset2KeyOld.emplace_back(offset, devOffset2Key[offset]);
            devOffset2Key[offset] = -1;
            evictPos_.emplace_back(offset);
            evictHBMKeys.emplace_back(key);
        } else {
            // offset 在Host
            evictHostPos_.emplace_back(offset);
            evictDDRKeys.emplace_back(key); // 删除映射表、初始化host表、发送dev淘汰位置
        }
    }

    LOG_INFO("ddr EvictDeleteEmb, emb: [{}], hostEvictSize: {}, devEvictSize: {}",
        name_, evictPos_.size(), evictHostPos_.size());
    LOG_TRACE("keyOffsetMap_, {}", MapToString(keyOffsetMap_));
}

/// DDR模式下的淘汰：删除映射表、初始化host表、发送dev淘汰位置
/// \param embName
/// \param keys
void EmbeddingDDR::EvictKeys(const vector<emb_key_t>& keys)
{
    EASY_FUNCTION()
    for (const emb_key_t& key : keys) {
        size_t offset;
        if (key == INVALID_KEY_VALUE) {
            LOG_WARN("evict key equal -1!");
            continue;
        }
        const auto& iter = keyOffsetMap_.find(key);
        if (iter == keyOffsetMap_.end()) {
            continue;
        }
        // 淘汰依据keyProcess中的history，hashmap映射关系创建于ParseKey；两者异步，造成淘汰的值在hashmap里可能未创建
        offset = iter->second;
        keyOffsetMap_.erase(iter);
        LOG_TRACE("evict embName {}, offset {}", name_, offset);

        if (offset < devVocabSize_) {
            devOffset2Batch[offset] = INVALID_KEY_VALUE;
            devOffset2KeyOld.emplace_back(offset, devOffset2Key[offset]);
            devOffset2Key[offset] = INVALID_KEY_VALUE;
            evictPos_.emplace_back(offset);
        } else {
            evictHostPos_.emplace_back(offset);
        }
    }
}

void EmbeddingDDR::ClearLookupAndSwapOffset()
{
    ddr2HbmKeys.clear();
}

void EmbeddingDDR::SetStartCount()
{
    currentUpdatePosStart = currentUpdatePos;
    freeSize_ = devVocabSize_;
}

int EmbeddingDDR::Load(const string& savePath)
{
    LoadHashMap(savePath);
    LoadDevOffset(savePath);
    LoadCurrStat(savePath);
    LoadEvictPos(savePath);
    LoadEmbInfo(savePath);
    LoadEmbData(savePath);
}

int EmbeddingDDR::Save(const string& savePath)
{
    SaveHashMap(savePath);
    SaveDevOffset(savePath);
    SaveCurrStat(savePath);
    SaveEvictPos(savePath);
    SaveEmbInfo(savePath);
    SaveEmbData(savePath);
}

int EmbeddingDDR::LoadHashMap(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/HashTable/DDR/" << name_ <<"/embedding_hashmap/slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    size_t fileSize = 0;
    try {
        fileSize = fileSystemPtr->GetFileSize(ss.str());
    } catch (exception& e) {
        LOG_ERROR("open file {} failed:{}", ss.str(), strerror(errno));
        return -1;
    }
    if (fileSize >= FILE_MAX_SIZE) {
        LOG_ERROR("file {} size = {} is too big", ss.str(), fileSize);
        return -1;
    }

    int64_t* buf = static_cast<int64_t*>(malloc(fileSize));
    if (buf == nullptr) {
        LOG_ERROR("malloc failed: {}", strerror(errno));
        return -1;
    }
    fileSystemPtr->Read(ss.str(), reinterpret_cast<char*>(buf), fileSize);
    for (int i = 0; i < fileSize / sizeof(int64_t); i = i + 2) { // key, offset进行pair对存储
        keyOffsetMap_[buf[i]] = buf[i + 1];
    }
    free(static_cast<void*>(buf));
    return 0;
}

int EmbeddingDDR::LoadDevOffset(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/HashTable/DDR/" << name_ <<"/dev_offset_2_Batch_n_Key/slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());
    size_t fileSize = 0;
    try {
        fileSize = fileSystemPtr->GetFileSize(ss.str());
    } catch (exception& e) {
        LOG_ERROR("open file {} failed:{}", ss.str(), strerror(errno));
        return -1;
    }
    if (fileSize >= FILE_MAX_SIZE) {
        LOG_ERROR("file {} size = {} is too big", ss.str(), fileSize);
        return -1;
    }

    devOffset2Key.resize(fileSize / sizeof(emb_key_t));
    fileSystemPtr->Read(ss.str(), reinterpret_cast<char*>(devOffset2Key.data()), fileSize);
    return 0;
}

int EmbeddingDDR::LoadCurrStat(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/HashTable/DDR/" << name_ <<"/embedding_current_status/slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    size_t raw[ELEMENT_NUM] = {0};
    fileSystemPtr->Read(ss.str(), reinterpret_cast<char*>(raw), sizeof(raw));
    currentUpdatePos = raw[CURRENT_UPDATE_IDX];
    hostVocabSize_ = raw[HOST_VOCAB_SIZE_IDX];
    devVocabSize_ = raw[MAX_OFFSET_IDX];
    maxOffset_ = raw[MAX_OFFSET_IDX];
    return 0;
}

int EmbeddingDDR::LoadEvictPos(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/HashTable/DDR/" << name_ <<"/evict_pos/slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    size_t fileSize = 0;
    try {
        fileSize = fileSystemPtr->GetFileSize(ss.str());
    } catch (exception& e) {
        LOG_ERROR("open file {} failed:{}", ss.str(), strerror(errno));
        return -1;
    }
    if (fileSize >= FILE_MAX_SIZE) {
        LOG_ERROR("File {} size = {} is too big", ss.str(), fileSize);
        return -1;
    }
    evictPos_.resize(fileSize / sizeof(int64_t));

    fileSystemPtr->Read(ss.str(), reinterpret_cast<char*>(evictPos_.data()), fileSize);
    return 0;
}

int EmbeddingDDR::LoadEmbInfo(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/HashTable/DDR/" << name_ <<"/embedding_info/slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    size_t raw[EMB_INFO_ELEMENT_NUM] = {0};
    fileSystemPtr->Read(ss.str(), reinterpret_cast<char*>(raw), sizeof(raw));
    extEmbSize_ = raw[EMB_INFO_EXT_SIZE_IDX];
    devVocabSize_ = raw[EMB_INFO_DEV_VOCAB_SIZE_IDX];
    hostVocabSize_ = raw[EMB_INFO_HOST_VOCAB_SIZE_IDX];
    return 0;
}

int EmbeddingDDR::LoadEmbData(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/HashTable/DDR/" << name_ <<"/embedding_data/slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    HostEmb* hostEmbs = Singleton<MxRec::HostEmb>::GetInstance();
    HostEmbTable& table = hostEmbs->GetEmb(name_);
    if (table.embData.empty()) {
        LOG_ERROR("hostEmb data is empty");
        return -1;
    }
    fileSystemPtr->Read(ss.str(), table.embData);
    return 0;
}

int EmbeddingDDR::SaveHashMap(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/HashTable/DDR/" << name_ <<"/embedding_hashmap/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    vector<int64_t> raw;
    for (const auto& it : keyOffsetMap_) {
        raw.push_back(it.first);
        raw.push_back(static_cast<int64_t>(it.second));
    }
    fileSystemPtr->Write(ss.str(), reinterpret_cast<const char*>(raw.data()),
                         static_cast<size_t>(raw.size() * sizeof(int64_t)));
    return 0;
}

int EmbeddingDDR::SaveDevOffset(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/HashTable/DDR/" << name_ <<"/dev_offset_2_Batch_n_Key/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    fileSystemPtr->Write(ss.str(), reinterpret_cast<const char*>(devOffset2Key.data()),
                         static_cast<size_t>(devOffset2Key.size() * sizeof(emb_key_t)));
    return 0;
}

int EmbeddingDDR::SaveCurrStat(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/HashTable/DDR/"<< name_ <<"/embedding_current_status/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    size_t raw[ELEMENT_NUM] = {0};
    raw[CURRENT_UPDATE_IDX] = currentUpdatePos;
    raw[HOST_VOCAB_SIZE_IDX] = hostVocabSize_;
    raw[DEV_VOCAB_SIZE_IDX] = devVocabSize_;
    raw[MAX_OFFSET_IDX] = maxOffset_;
    fileSystemPtr->Write(ss.str(), reinterpret_cast<const char*>(raw), sizeof(raw));
    return 0;
}

int EmbeddingDDR::SaveEvictPos(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/HashTable/DDR/" << name_ << "/evict_pos/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    fileSystemPtr->Write(ss.str(), reinterpret_cast<const char*>(evictPos_.data()),
                         static_cast<size_t>(evictPos_.size() * sizeof(int64_t)));
    return 0;
}

int EmbeddingDDR::SaveEmbInfo(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/HashTable/DDR/"<< name_ <<"/embedding_info/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    size_t raw[EMB_INFO_ELEMENT_NUM] = {};
    raw[EMB_INFO_EXT_SIZE_IDX] = extEmbSize_;
    raw[EMB_INFO_DEV_VOCAB_SIZE_IDX] = devVocabSize_;
    raw[EMB_INFO_HOST_VOCAB_SIZE_IDX] = hostVocabSize_;
    fileSystemPtr->Write(ss.str(), reinterpret_cast<const char*>(raw), sizeof(raw));
    return 0;
}

int EmbeddingDDR::SaveEmbData(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/HashTable/DDR/"<< name_ <<"/embedding_data/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    HostEmb* hostEmbs = Singleton<MxRec::HostEmb>::GetInstance();
    HostEmbTable& table = hostEmbs->GetEmb(name_);
    if (table.embData.empty()) {
        LOG_ERROR("host embedding data is empty");
        return 0;
    }
    vector<float*> content;
    for (vector<float>& emb : table.embData) {
        content.push_back(emb.data());
    }
    size_t dataSize = table.embData[0].size();
    fileSystemPtr->Write(ss.str(), content, dataSize * sizeof(float));
    return 0;
}
