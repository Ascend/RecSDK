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
#include "l3_storage/cache_manager.h"
#include "ock_ctr_common/include/error_code.h"

using namespace MxRec;

EmbeddingDDR::EmbeddingDDR()
{
}

EmbeddingDDR::EmbeddingDDR(const EmbInfo& info, const RankInfo& rankInfo, int inSeed)
    : EmbeddingTable(info, rankInfo, inSeed), deviceId(rankInfo.deviceId)
{
    LOG_INFO("Init DDR table:{}, devVocabSize:{}, hostVocabSize:{}", name, devVocabSize, hostVocabSize);
}

EmbeddingDDR::~EmbeddingDDR()
{
    hdTransfer = nullptr;
    embCache = nullptr;
}

void EmbeddingDDR::Key2Offset(std::vector<emb_key_t>& splitKey, int channel)
{
}

int64_t EmbeddingDDR::capacity() const
{
    return capacity_;
}

/*
* 删除淘汰key的映射关系，并将其offset更新到evictPos，待后续复用
*/
void EmbeddingDDR::EvictDeleteEmb(const vector<emb_key_t>& keys)
{
}

/// DDR模式下的淘汰：删除映射表、初始化host表、发送dev淘汰位置
/// \param embName
/// \param keys
void EmbeddingDDR::EvictKeys(const vector<emb_key_t>& keys)
{
}

void EmbeddingDDR::Load(const string& savePath, map<string, unordered_set<emb_cache_key_t>>& trainKeySet)
{
    vector<emb_cache_key_t> keys;
    vector<vector<float>> embeddings;
    vector<vector<float>> optimizerSlots;

    LoadKey(savePath, keys);
    LoadEmbedding(savePath, embeddings);
    LoadOptimizerSlot(savePath, optimizerSlots);

    auto rc = embCache->LoadEmbTableInfos(name, keys, embeddings, optimizerSlots);
    if (rc != 0) {
        throw runtime_error("embCache->LoadEmbTableInfos failed, err code:" + to_string(rc));
    }

    trainKeySet[name].insert(keys.cbegin(), keys.cend());
}

void EmbeddingDDR::LoadKey(const string &savePath, vector<emb_cache_key_t> &keys)
{
    stringstream ss;
    ss << savePath << "/" << name << "/key/slice.data";

    if (fileSystemPtr_ == nullptr) {
        throw runtime_error("failed to obtain the file system pointer, the file system pointer is null.");
    }

    size_t fileSize = 0;
    try {
        fileSize = fileSystemPtr_->GetFileSize(ss.str());
    } catch (exception& e) {
        string errMsg = StringFormat("open file failed:%s, error code:%d", ss.str().c_str(), strerror(errno));
        throw runtime_error(errMsg);
    }
    if (fileSize >= FILE_MAX_SIZE) {
        string errMsg = StringFormat("file:%s, size:%d is too big", ss.str().c_str(), fileSize);
        throw runtime_error(errMsg);
    }

    // 暂时向HBM兼容，转成int64_t，后续再归一key类型为uint64_t
    auto buf = static_cast<int64_t*>(malloc(fileSize));
    if (buf == nullptr) {
        string errMsg = StringFormat("malloc buffer failed, error code:%d", strerror(errno));
        throw runtime_error(errMsg);
    }
    ssize_t result = fileSystemPtr_->Read(ss.str(), reinterpret_cast<char*>(buf), fileSize);
    if (result == -1) {
        free(static_cast<void*>(buf));
        string errMsg = StringFormat("read buffer failed, error code:%d", strerror(errno));
        throw runtime_error(errMsg);
    }
    if (result != fileSize) {
        free(static_cast<void*>(buf));
        throw runtime_error(StringFormat("Error: Load keys failed. Expected to read %d bytes, "
                                         "but actually read %d bytes to file %s.", fileSize, result, ss.str().c_str()));
    }

    hostLoadOffset.clear();
    size_t loadKeySize = fileSize / sizeof(int64_t);
    for (size_t i = 0; i < loadKeySize; i++) {
        // 分配到不同的卡
        if (buf[i] % rankSize_ != rankId_) {
            continue;
        }
        hostLoadOffset.emplace_back(i);
        keys.emplace_back(static_cast<emb_cache_key_t>(buf[i]));
    }

    free(static_cast<void*>(buf));
    LOG_DEBUG("load key done, table:{}", name);
}

void EmbeddingDDR::LoadEmbedding(const string &savePath, vector<vector<float>> &embeddings)
{
    // must init first
    for (size_t i = 0; i < hostLoadOffset.size(); i++) {
        vector<float> tmp(embSize_);
        embeddings.emplace_back(tmp);
    }

    stringstream ss;
    ss << savePath << "/" << name;
    stringstream embedStream;
    embedStream << ss.str() << "/" << "embedding/slice.data";

    if (fileSystemPtr_ == nullptr) {
        throw runtime_error("failed to obtain the file system pointer, the file system pointer is null.");
    }
    ssize_t res = fileSystemPtr_->Read(embedStream.str(), embeddings, 0, hostLoadOffset, embSize_);
    LOG_DEBUG("load embedding done, table:{}, read bytes:{}", name, res);
}

void EmbeddingDDR::LoadOptimizerSlot(const string &savePath, vector<vector<float>> &optimizerSlots)
{
    if (optimParams.size() == 0) {
        LOG_DEBUG("optimizer has no slot data to load");
        return;
    }

    // must init first
    for (size_t i = 0; i < hostLoadOffset.size(); i++) {
        vector<float> tmp(extEmbSize_ - embSize_);
        optimizerSlots.emplace_back(tmp);
    }

    stringstream ss;
    ss << savePath << "/" << name;

    if (fileSystemPtr_ == nullptr) {
        throw runtime_error("failed to obtain the file system pointer, the file system pointer is null.");
    }
    int64_t slotIdx = 0;
    for (const auto &param: optimParams) {
        stringstream paramStream;
        paramStream << ss.str() << "/" << optimName + "_" + param << "/slice.data";
        ssize_t res = fileSystemPtr_->Read(paramStream.str(), optimizerSlots, slotIdx, hostLoadOffset, embSize_);
        slotIdx++;
        LOG_DEBUG("load optimizer slot, table:{}, slot:{}, read bytes:{}", name, param, res);
    }

    LOG_DEBUG("load optimizer slot done, table:{}", name);
}

void EmbeddingDDR::Save(const string& savePath)
{
    vector<emb_cache_key_t> keys;
    vector<vector<float>> embeddings;
    vector<vector<float>> optimizerSlots;

    auto step = GetStepFromPath(savePath);
    if (step > 0) {
        SyncLatestEmbedding();
        embCache->GetEmbTableInfos(name, keys, embeddings, optimizerSlots);
    }

    SaveKey(savePath, keys);
    SaveEmbedding(savePath, embeddings);
    SaveOptimizerSlot(savePath, optimizerSlots, keys.size());
}

void EmbeddingDDR::SyncLatestEmbedding()
{
    // 导出host记录的存在于npu的embedding
    std::vector<std::pair<uint64_t, uint64_t>> koVec;
    int rc = embCache->ExportDeviceKeyOffsetPairs(name, koVec);
    if (rc != ock::ctr::H_OK) {
        string errMsg = StringFormat("ExportDeviceKeyOffsetPairs failed, table:%s, error code:%d", name.c_str(), rc);
        throw std::invalid_argument(errMsg);
    }
    std::vector<uint64_t> swapOutKeys;
    for (const auto& p : koVec) {
        swapOutKeys.push_back(p.first);
    }
    LOG_DEBUG("save swapOutKeys.size:{}, table:{}", swapOutKeys.size(), name);

    // 接收python save接口发送的卡内embedding
    auto size = hdTransfer->RecvAcl(TransferChannel::SAVE_D2H, TRAIN_CHANNEL_ID, name, 0, -1);
    LOG_DEBUG("save acltdtGetDatasetSize, size: {}, table:{}", size, name);
    auto aclData = acltdtGetDataItem(hdTransfer->aclDatasets[name][0], 0);
    if (aclData == nullptr) {
        throw runtime_error("Acl get tensor data from dataset failed.");
    }
    auto* ptr = reinterpret_cast<float*>(acltdtGetDataAddrFromItem(aclData));

    if (ssdVocabSize == 0) {
        // 在保存之前先更新host的embedding
        rc = embCache->EmbeddingUpdate(name, swapOutKeys, ptr);
        if (rc != ock::ctr::H_OK) {
            string errMsg = StringFormat("EmbeddingUpdate failed, table:%s, error code:%d", name.c_str(), rc);
            throw std::invalid_argument(errMsg);
        }
    } else {
        // 在保存之前先更新ddr和ssd的embedding
        HBMSwapOutInfo info;
        cacheManager_->ProcessSwapOutKeys(name, swapOutKeys, info);
        vector<float*> swapOutAddrs;
        rc = embCache->EmbeddingLookupAddrs(name, info.swapOutDDRKeys, swapOutAddrs);
        if (rc != ock::ctr::H_OK) {
            string errMsg = StringFormat("EmbeddingLookupAddrs failed, table:%s, error code:%d", name.c_str(), rc);
            throw std::invalid_argument(errMsg);
        }
        uint32_t extEmbeddingSize = embInfo_.extEmbeddingSize;
        uint32_t memSize = extEmbeddingSize * sizeof(float);
        // DDR更新
#pragma omp parallel for num_threads(MGMT_CPY_THREADS) default(none) \
    shared(swapOutAddrs, info, ptr, extEmbeddingSize, memSize)
        for (uint64_t i = 0; i < swapOutAddrs.size(); i++) {
            int errCode = memcpy_s(
                swapOutAddrs[i], memSize, ptr + info.swapOutDDRAddrOffs[i] * extEmbeddingSize, memSize);
            if (errCode != 0) {
                string errMsg = StringFormat("memcpy_s failed, table:%s, error code:%d", name.c_str(), errCode);
                throw std::invalid_argument(errMsg);
            }
        }
        cacheManager_->UpdateL3StorageEmb(name, ptr, embInfo_.extEmbeddingSize, info.swapOutL3StorageKeys,
                                          info.swapOutL3StorageAddrOffs);
    }
}

void EmbeddingDDR::SaveKey(const string& savePath, vector<emb_cache_key_t>& keys)
{
    stringstream ss;
    ss << savePath << "/" << name << "/key/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    // 暂时向HBM兼容，转成int64_t，后续再归一key类型为uint64_t
    vector<int64_t> keysCompat(keys.cbegin(), keys.cend());

    if (fileSystemPtr_ == nullptr) {
        throw runtime_error("failed to obtain the file system pointer, the file system pointer is null.");
    }
    ssize_t res = fileSystemPtr_->Write(ss.str(), reinterpret_cast<const char *>(keysCompat.data()),
                                        static_cast<size_t>(keys.size() * sizeof(int64_t)));
    if (res == -1) {
        throw runtime_error("save key failed!");
    }
}

void EmbeddingDDR::SaveEmbedding(const string& savePath, vector<vector<float>>& embeddings)
{
    stringstream ss;
    ss << savePath << "/" << name << "/embedding/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    if (fileSystemPtr_ == nullptr) {
        throw runtime_error("failed to obtain the file system pointer, the file system pointer is null.");
    }
    ssize_t writeBytesNum = fileSystemPtr_->Write(ss.str(), embeddings, embSize_);
    ssize_t expectWriteBytes = embeddings.size() * embSize_ * sizeof(float);
    if (writeBytesNum != expectWriteBytes) {
        string errMsg = StringFormat("Save embedding failed, write expect:%ld, actual:%ld, path:%s .",
                                     expectWriteBytes, writeBytesNum, savePath.c_str());
        throw runtime_error(errMsg);
    }
}

void EmbeddingDDR::SaveOptimizerSlot(const string& savePath, vector<vector<float>>& optimizerSlots, size_t keySize)
{
    if (optimizerSlots.size() == 0) {
        LOG_DEBUG("optimizer has no slot data to save");
        return;
    }
    
    if (optimizerSlots.size() != keySize) {
        string errMsg = StringFormat("optimizer slot data size not equal to key size, "
                                     "optimizerSlots.size:%d, keySize:%d",
                                     optimizerSlots.size(), keySize);
        throw runtime_error(errMsg);
    }

    size_t slotIdx = 0;
    for (const auto &slotName: optimParams) {
        stringstream ss;
        ss << savePath << "/" << name << "/" << optimName + "_" + slotName << "/";
        MakeDir(ss.str());
        ss << "slice_" << rankId_ << ".data";

        vector<vector<float>> slotData;
        for (const auto &data: optimizerSlots) {
            vector<float> tmp(data.cbegin() + slotIdx * embSize_, data.cbegin() + (slotIdx+1) * embSize_);
            slotData.emplace_back(tmp);
        }
        ssize_t writeBytesNum = fileSystemPtr_->Write(ss.str(), slotData, embSize_);
        ssize_t expectWriteBytes = slotData.size() * embSize_ * sizeof(float);
        if (writeBytesNum != expectWriteBytes) {
            string errMsg = StringFormat("save optimizer slot failed, write expect:%d, actual:%d, path:%s",
                                         expectWriteBytes, writeBytesNum, savePath.c_str());
            throw runtime_error(errMsg);
        }

        slotIdx++;
    }
}

vector<int64_t> EmbeddingDDR::GetDeviceOffset()
{
    throw runtime_error("GetDeviceOffset deprecated in ddr/ssd mode");
}

void EmbeddingDDR::SetOptimizerInfo(OptimizerInfo& optimizerInfo)
{
    optimName = optimizerInfo.optimName;
    optimParams = optimizerInfo.optimParams;
}

void EmbeddingDDR::SetCacheManager(CacheManager *cm)
{
    LOG_DEBUG("set CacheManager");
    cacheManager_ = cm;
}

TableInfo EmbeddingDDR::GetTableInfo()
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

void EmbeddingDDR::SetHDTransfer(HDTransfer *hdTransfer)
{
    this->hdTransfer = hdTransfer;
}

void EmbeddingDDR::SetEmbCache(ock::ctr::EmbCacheManagerPtr embCache)
{
    this->embCache = embCache;
}
