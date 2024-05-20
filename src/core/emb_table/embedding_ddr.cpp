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

#include "emb_table/embedding_ddr.h"
#include <utility>
#include "utils/logger.h"
#include "utils/singleton.h"
#include "host_emb/host_emb.h"
#include "file_system/file_system_handler.h"
#include "ssd_cache/cache_manager.h"
#include "emb_table/embedding_mgmt.h"

using namespace MxRec;

EmbeddingDDR::EmbeddingDDR()
{
}

EmbeddingDDR::EmbeddingDDR(const EmbInfo& info, const RankInfo& rankInfo, int inSeed)
    : EmbeddingTable(info, rankInfo, inSeed)
{
    LOG_INFO("Init DDR table [{}] devVocabSize = {} hostVocabSize = {}", name, devVocabSize, hostVocabSize);
    currentUpdatePos = 0;
    devOffset2Key.resize(devVocabSize);
    devOffset2Batch.resize(devVocabSize);
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
    maxOffsetOld = maxOffset;

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
        AddKeyFreqInfo(key, RecordType::NOT_DDR);
        if (offset < devVocabSize) {
            // 偏移小于等于HBM容量：直接放入查询向量；更新偏移之前关联的key和当前关联的key
            lookUpVec.push_back(offset);
            devOffset2KeyOld.emplace_back(offset, static_cast<int>(devOffset2Key[offset]));
            devOffset2Key[offset] = key;
        } else {
            // 偏移大于HBM容量：记录在host emb上的偏移；找到需要交换的HBM偏移
            missingKeysHostPos_.emplace_back(offset - devVocabSize);
            offset = FindSwapPosOld(key, offset, batchId, swapPos);
            lookUpVec.emplace_back(offset);
        }
    }
    if (batchId == 0) {
        LOG_INFO("max offset {}", maxOffset);
    }
    LOG_TRACE("keyOffsetMap, {}", MapToString(keyOffsetMap));
    return lookUpVec;
}

emb_key_t EmbeddingDDR::FindOffsetHelper(const emb_key_t& key, int channelId)
{
    const auto& iter = keyOffsetMap.find(key);
    emb_key_t offset = INVALID_KEY_VALUE;
    if (iter != keyOffsetMap.end()) {
        offset = iter->second;
        LOG_TRACE("devVocabSize, {} , offset , {}", devVocabSize, offset);
        if (isSSDEnabled_ && offset >= devVocabSize) {
            ddr2HbmKeys.emplace_back(key);
        }
        return offset;
    }
    if (channelId != TRAIN_CHANNEL_ID) {
        return offset;
    }
    if (evictDevPos.size() != 0) { // 优先复用hbm表
        offset = evictDevPos.back();
        keyOffsetMap[key] = offset;
        LOG_TRACE("ddr mode, dev evictDevPos is not null, key [{}] reuse offset [{}], evictSize [{}]",
            key, offset, evictDevPos.size());
        evictDevPos.pop_back();
        LOG_ERROR("dev evicted offset = {}", offset);
        return offset;
    }

    if (evictHostPos.size() != 0) { // hbm不足，再复用host/ddr表
        offset = evictHostPos.back();
        keyOffsetMap[key] = offset;
        LOG_TRACE("ddr mode, host evictPos is not null, key [{}] reuse offset [{}], evictSize [{}]",
            key, offset, evictHostPos.size());
        evictHostPos.pop_back();
        LOG_TRACE("host evicted offset = {}", offset);
        return offset;
    }
    keyOffsetMap[key] = maxOffset;
    offset = maxOffset;
    maxOffset++;
    if (maxOffset == devVocabSize) {
        LOG_INFO("start using host vocab!");
    }
    if (maxOffset > (hostVocabSize + devVocabSize)) {
        LOG_ERROR("hostVocabSize too small! dev:{} host:{}", devVocabSize, hostVocabSize);
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
        const auto& iter = keyOffsetMap.find(key);
        if (iter != keyOffsetMap.end()) {
            offset = iter->second;

            LOG_TRACE("key will be used, {} , offset , {}", key, offset);
            if (offset < devVocabSize) {
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
            keyOffsetMap[key] = currentUpdatePos;  // 更新key对应的HBM偏移
            // 记录HBM偏移之前的key
            devOffset2KeyOld.emplace_back(currentUpdatePos, devOffset2Key[currentUpdatePos]);
            auto& oldKey = devOffset2Key[currentUpdatePos];
            oldSwap.emplace_back(oldKey, key); // 记录交换的两个key oldKey:HBM->DDR key:DDR->HBM
            keyOffsetMap[oldKey] = hostOffset; // 更新被替换的key的偏移
            oldKey = key;
            notFind = false;
        }
        currentUpdatePos++; // 查找位置+1
        freeSize_--;        // HBM可用空间-1

        // 遍历完一遍整个HBM表后，从头开始遍历
        if (currentUpdatePos == devVocabSize) {
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
        const auto& iter = keyOffsetMap.find(key);
        if (iter == keyOffsetMap.end()) {
            // 淘汰依据keyProcess中的history，hashmap映射关系创建于ParseKey；两者异步，造成淘汰的值在hashmap里可能未创建
            continue;
        }
        offset = iter->second;
        keyOffsetMap.erase(iter);
        LOG_TRACE("evict embName {}, offset {}", name, offset);

        if (offset < devVocabSize) {
            // offset 在device中
            devOffset2Batch[offset] = -1;
            devOffset2KeyOld.emplace_back(offset, devOffset2Key[offset]);
            devOffset2Key[offset] = -1;
            evictDevPos.emplace_back(offset);
            evictHBMKeys.emplace_back(key);
        } else {
            // offset 在Host
            evictHostPos.emplace_back(offset);
            evictDDRKeys.emplace_back(key); // 删除映射表、初始化host表、发送dev淘汰位置
        }
    }
    if (isSSDEnabled_) {
        cacheManager_->RefreshFreqInfoCommon(name, evictHBMKeys, TransferType::HBM_2_EVICT);
        cacheManager_->RefreshFreqInfoCommon(name, evictDDRKeys, TransferType::DDR_2_EVICT);
    }

    LOG_INFO("ddr EvictDeleteEmb, emb: [{}], hostEvictSize: {}, devEvictSize: {}",
        name, evictHostPos.size(), evictDevPos.size());
    LOG_TRACE("keyOffsetMap, {}", MapToString(keyOffsetMap));
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
        const auto& iter = keyOffsetMap.find(key);
        if (iter == keyOffsetMap.end()) {
            continue;
        }
        // 淘汰依据keyProcess中的history，hashmap映射关系创建于ParseKey；两者异步，造成淘汰的值在hashmap里可能未创建
        offset = iter->second;
        keyOffsetMap.erase(iter);
        LOG_TRACE("evict embName {}, offset {}", name, offset);

        if (offset < devVocabSize) {
            devOffset2Batch[offset] = INVALID_KEY_VALUE;
            devOffset2KeyOld.emplace_back(offset, devOffset2Key[offset]);
            devOffset2Key[offset] = INVALID_KEY_VALUE;
            evictDevPos.emplace_back(offset);
        } else {
            evictHostPos.emplace_back(offset);
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
    freeSize_ = devVocabSize;
}

void EmbeddingDDR::Load(const string& savePath)
{
    LoadKey(savePath);
    LoadEmbAndOptim(savePath);
}

void EmbeddingDDR::Save(const string& savePath)
{
    SaveKey(savePath);
    SaveEmbAndOptim(savePath);
}

void EmbeddingDDR::LoadKey(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/" << name << "/key/slice.data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    size_t fileSize = fileSystemPtr->GetFileSize(ss.str());
    if (fileSize >= FILE_MAX_SIZE) {
        throw runtime_error(StringFormat("Error: Load keys failed. file {} size {} is too big.", ss.str(), fileSize));
    }

    int64_t* buf = static_cast<int64_t*>(malloc(fileSize));
    if (buf == nullptr) {
        throw runtime_error(StringFormat("Error: Load keys failed. "
                                         "failed to allocate {} bytes using malloc.", fileSize));
    }

    ssize_t res = fileSystemPtr->Read(ss.str(), reinterpret_cast<char *>(buf), fileSize);
    if (res == -1) {
        free(static_cast<void*>(buf));
        throw runtime_error(StringFormat("Error: Load keys failed. "
                                         "An error occurred while reading file: {}.", ss.str()));
    }
    if (res != fileSize) {
        free(static_cast<void*>(buf));
        throw runtime_error(StringFormat("Error: Load keys failed. Expected to read {} bytes, "
                                         "but actually read {} bytes to file {}.", fileSize, res, ss.str()));
    }

    size_t loadKeySize = fileSize / sizeof(int64_t);

    // key优先加载至device
    loadOffset.clear();
    hostLoadOffset.clear();
    int keyCount = 0;
    for (int i = 0; i < loadKeySize; i = i + 1) {
        if (buf[i] % rankSize_ != rankId_) {
            continue;
        }
        if (keyCount > devVocabSize + hostVocabSize) {
            free(static_cast<void*>(buf));
            throw runtime_error(StringFormat("Error: Load keys failed. Load key size :{} , "
                                             "exceeds the sum of device vocab size and host vocab size: {}.",
                                             keyCount, devVocabSize + hostVocabSize));
        } else if (keyCount < devVocabSize) {
            loadOffset.push_back(i);
            devOffset2Key[keyCount] = buf[i];
        } else {
            hostLoadOffset.push_back(i);
        }
        keyOffsetMap[buf[i]] = keyCount;
        keyCount++;
    }
    maxOffset = keyOffsetMap.size();
    free(static_cast<void*>(buf));
}

void EmbeddingDDR::LoadEmbAndOptim(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/" << name;

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    HostEmb *hostEmbs = Singleton<MxRec::HostEmb>::GetInstance();
    HostEmbTable &table = hostEmbs->GetEmb(name);
    if (table.embData.empty()) {
        LOG_ERROR("hostEmb data is empty");
        return;
    }

    // 读embedding
    stringstream embedStream;
    embedStream << ss.str() << "/" << "embedding/slice.data";

    size_t readSize = hostLoadOffset.size() * embSize_ * sizeof(float);
    ssize_t res = fileSystemPtr->Read(embedStream.str(), table.embData, 0, hostLoadOffset, embSize_);
    if (res == -1) {
        throw runtime_error(StringFormat("Error: Load embeddings failed. An error occurred while reading file: {}.",
                                         embedStream.str()));
    }
    if (res != readSize) {
        throw runtime_error(StringFormat("Error: Load embeddings failed. Expected to read {} bytes, "
                                         "but actually read {} bytes to file {}.", readSize, res, embedStream.str()));
    }

    // 读optim
    int64_t optimIndex = 1;
    for (const auto &param: optimParams) {
        stringstream paramStream;
        paramStream << ss.str() << "/" << optimName + "_" + param << "/slice.data";

        ssize_t res = fileSystemPtr->Read(paramStream.str(), table.embData, optimIndex, hostLoadOffset, embSize_);
        if (res == -1) {
            throw runtime_error(StringFormat("Error: Load optimizers failed. An error occurred while reading file: {}.",
                                             paramStream.str()));
        }
        if (res != readSize) {
            throw runtime_error(StringFormat("Error: Load embeddings failed. Expected to read {} bytes, "
                                             "but actually read {} bytes to file {}.",
                                             readSize, res, paramStream.str()));
        }
        optimIndex++;
    }
}

void EmbeddingDDR::SaveKey(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/" << name << "/key/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    hostKey.clear();
    hostOffset.clear();
    deviceKey.clear();
    deviceOffset.clear();

    for (const auto& it: keyOffsetMap) {
        if (it.second >= devVocabSize) {
            hostKey.push_back(it.first);
            hostOffset.push_back(it.second);
        } else {
            deviceKey.push_back(it.first);
            deviceOffset.push_back(it.second);
        }
    }

    size_t writeSize = static_cast<size_t>(hostKey.size() * sizeof(int64_t));
    ssize_t res = fileSystemPtr->Write(ss.str(), reinterpret_cast<const char *>(hostKey.data()), writeSize);
    if (res == -1) {
        throw runtime_error(StringFormat("Error: Save keys failed. "
                                         "An error occurred while writing file: {}.", ss.str()));
    }
    if (res != writeSize) {
        throw runtime_error(StringFormat("Error: Save keys failed. Expected to write {} bytes, "
                                         "but actually write {} bytes to file {}.", writeSize, res, ss.str()));
    }

    writeSize = static_cast<size_t>(deviceKey.size() * sizeof(int64_t));
    res = fileSystemPtr->Write(ss.str(), reinterpret_cast<const char *>(deviceKey.data()), writeSize);
    if (res == -1) {
        throw runtime_error(StringFormat("Error: Save keys failed. "
                                         "An error occurred while writing file: {}.", ss.str()));
    }
    if (res != writeSize) {
        throw runtime_error(StringFormat("Error: Save keys failed. Expected to write {} bytes, "
                                         "but actually write {} bytes to file {}.", writeSize, res, ss.str()));
    }
}

void EmbeddingDDR::SaveEmbData(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/" << name << "/embedding/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    size_t writeSize = embSize_ * sizeof(float) * embContent.size();
    ssize_t res = fileSystemPtr->Write(ss.str(), embContent, embSize_ * sizeof(float));
    if (res == -1) {
        throw runtime_error(StringFormat("Error: Save embeddings failed. "
                                         "An error occurred while writing file: {}.", ss.str()));
    }
    if (res != writeSize) {
        throw runtime_error(StringFormat("Error: Save embeddings failed. Expected to write {} bytes, "
                                         "but actually write {} bytes to file {}.", writeSize, res, ss.str()));
    }
}

void EmbeddingDDR::SaveOptimData(const string& savePath)
{
    for (const auto &content: optimContentMap) {
        stringstream ss;
        ss << savePath << "/" << name << "/" << optimName + "_" + content.first << "/";
        MakeDir(ss.str());
        ss << "slice_" << rankId_ << ".data";

        unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
        unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

        size_t writeSize = embSize_ * sizeof(float) * content.second.size();
        ssize_t res = fileSystemPtr->Write(ss.str(), content.second, embSize_ * sizeof(float));
        if (res == -1) {
            throw runtime_error(StringFormat("Error: Save optimizers failed. "
                                             "An error occurred while writing file: {}.", ss.str()));
        }
        if (res != writeSize) {
            throw runtime_error(StringFormat("Error: Save optimizers failed. Expected to write {} bytes, "
                                             "but actually write {} bytes to file {}.", writeSize, res, ss.str()));
        }
    }
}

void EmbeddingDDR::SaveEmbAndOptim(const string& savePath)
{
    HostEmb *hostEmbs = Singleton<MxRec::HostEmb>::GetInstance();
    HostEmbTable &table = hostEmbs->GetEmb(name);
    if (table.embData.empty()) {
        LOG_ERROR("host embedding data is empty");
    }
    embContent.clear();
    for (const string &param: optimParams) {
        optimContentMap[param].clear();
    }
    for (int64_t &offset: hostOffset) {
        embContent.push_back(table.embData[offset - devVocabSize].data());
        int optim_param_count = 1;
        for (const string &param: optimParams) {
            optimContentMap[param].push_back(table.embData[offset - devVocabSize].data() +
                                             sizeof(float) * embSize_ * optim_param_count);
            optim_param_count++;
        }
    }
    SaveEmbData(savePath);
    SaveOptimData(savePath);
}


vector<int64_t> EmbeddingDDR::GetDeviceOffset()
{
    return deviceOffset;
}

void EmbeddingDDR::SetOptimizerInfo(OptimizerInfo& optimizerInfo)
{
    optimName = optimizerInfo.optimName;
    optimParams = optimizerInfo.optimParams;
    for (const string &param: optimParams) {
        optimContentMap[param] = vector<float*>{};
    }
}

void EmbeddingDDR::SetCacheManager(CacheManager *cm)
{
    cacheManager_ = cm;
}

void EmbeddingDDR::AddKeyFreqInfo(const emb_key_t& key, RecordType type)
{
    if (!isSSDEnabled_) {
        return;
    }
    cacheManager_->PutKey(name, key, type);
}

void EmbeddingDDR::RefreshFreqInfoWithSwap()
{
    if (!isSSDEnabled_) {
        return;
    }
    // 换入换出key列表，元素为pair: pair<oldKey, key> oldKey为从HBM移出的key, key为从DDR移出的key
    LOG_DEBUG("RefreshFreqInfoWithSwap, table:{}, oldSwap Size:{}", name, oldSwap.size());
    vector<emb_key_t> enterDDRKeys;
    for (auto keyPair : oldSwap) {
        enterDDRKeys.emplace_back(keyPair.first);
    }
    cacheManager_->RefreshFreqInfoCommon(name, enterDDRKeys, TransferType::HBM_2_DDR);
    cacheManager_->RefreshFreqInfoCommon(name, ddr2HbmKeys, TransferType::DDR_2_HBM);

    AddCacheManagerTraceLog();
}

/// 记录日志：HBM和DDR换入换出后，比较hostHashMap中DDR内key和表对应的lfuCache对象中的key内容
void EmbeddingDDR::AddCacheManagerTraceLog() const
{
    if (Logger::GetLevel() != Logger::TRACE) {
        return;
    }
    auto& hostMap = keyOffsetMap;
    auto& devSize = devVocabSize;
    auto iter = cacheManager_->ddrKeyFreqMap.find(name);
    if (iter == cacheManager_->ddrKeyFreqMap.end()) {
        throw runtime_error("table not in ddrKeyFreqMap");
    }
    auto &lfu = iter->second;
    const auto& lfuTab = lfu.GetFreqTable();
    if (lfuTab.empty()) {
        return;
    }
    size_t tableKeyInDdr = 0;
    vector<emb_key_t> ddrKeys; // 获取hostHashMap中保存在DDR的key
    for (const auto& item : hostMap) {
        if (item.second < devSize) {
            continue;
        }
        ddrKeys.emplace_back(item.first);
        ++tableKeyInDdr;
    }
    vector<emb_key_t> lfuKeys;
    for (const auto& it : lfuTab) {
        lfuKeys.emplace_back(it.first);
    }
    std::sort(ddrKeys.begin(), ddrKeys.end());
    std::sort(lfuKeys.begin(), lfuKeys.end());
    std::string ddrKeysString = VectorToString(ddrKeys);
    std::string lfuKeysString = VectorToString(lfuKeys);
    if (ddrKeysString != lfuKeysString) {
        LOG_ERROR("swap HBM with DDR step error, key string not equal, table:{}, ddrKeysString:{}, lfuKeysString:{}",
                  name, ddrKeysString, lfuKeysString);
    } else {
        LOG_INFO("swap HBM with DDR step OK, table:{}, ddrKeysString == lfuKeysString, string length:{}",
                 name, lfuKeysString.length());
    }

    LOG_INFO("swap HBM with DDR step end, table:{}, tableKeyInDdr:{}, tableKeyInLfu:{}",
             name, tableKeyInDdr, lfu.keyTable.size());
}

TableInfo EmbeddingDDR::GetTableInfo()
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

void EmbeddingDDR::RefreshFreqInfoAfterLoad()
{
    vector<emb_key_t> h2d;
    vector<emb_key_t> d2h;

    for (const auto& it: cacheManager_->ddrKeyFreqMap[name].keyTable) {
        auto key = it.first;
        auto iter = keyOffsetMap.find(key);
        if (iter == keyOffsetMap.end()) {
            throw runtime_error("ddrKeyFreqMap key not in keyOffsetMap");
        }
        auto offset = iter->second;
        if (offset < devVocabSize) {
            d2h.emplace_back(key);
        }
    }
    for (const auto& it: cacheManager_->excludeDDRKeyCountMap[name]) {
        auto key = it.first;
        auto iter = keyOffsetMap.find(key);
        if (iter == keyOffsetMap.end()) {
            continue;
        }
        auto offset = iter->second;
        if (offset >= devVocabSize) {
            h2d.emplace_back(key);
        }
    }

    cacheManager_->RefreshFreqInfoCommon(name, h2d, TransferType::HBM_2_DDR);
    cacheManager_->RefreshFreqInfoCommon(name, d2h, TransferType::DDR_2_HBM);

    LOG_DEBUG("RefreshFreqInfoAfterLoad done");
}
