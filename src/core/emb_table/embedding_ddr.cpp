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

void EmbeddingDDR::Key2OffsetForDp(std::vector<emb_key_t>& keys, int channel)
{
}

int64_t EmbeddingDDR::capacity() const
{
    return capacity_.load();
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
        auto error = Error(ModuleName::M_OCK_CTR, ErrorType::LOGIC_ERROR,
                           StringFormat("Invoke embCache->LoadEmbTableInfos failed, table:%s, error code:%d.",
                                        name.c_str(), rc));
        LOG_ERROR(error.ToString());
        throw std::invalid_argument(error.ToString());
    }

    trainKeySet[name].insert(keys.cbegin(), keys.cend());
    // Reset the offsetMapper object to revert to its initialized state after loading
    auto rs = embCache->ResetOffsetMappers();
    if (rs != 0) {
        auto error = Error(ModuleName::M_OCK_CTR, ErrorType::LOGIC_ERROR,
                           StringFormat("Invoke embCache->ResetOffsetMappers failed, table:%s, error code:%d.",
                                        name.c_str(), rc));
        LOG_ERROR(error.ToString());
        throw std::invalid_argument(error.ToString());
    }
}

void EmbeddingDDR::LoadKey(const string &savePath, vector<emb_cache_key_t> &keys)
{
    stringstream ss;
    ss << savePath << "/" << name << "/key/slice.data";

    CheckFileSystemPtr();

    size_t fileSize = 0;
    try {
        fileSize = fileSystemPtr_->GetFileSize(ss.str());
    } catch (exception& e) {
        auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::IO_ERROR,
                           StringFormat("Open file failed:%s, error code:%d",
                                        ss.str().c_str(), strerror(errno)));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }
    if (fileSize >= FILE_MAX_SIZE) {
        string errMsg = StringFormat("Invalid file:%s, size:%d is too big.", ss.str().c_str(), fileSize);
        auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::INVALID_ARGUMENT, errMsg);
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }

    // 暂时向HBM兼容，转成int64_t，后续再归一key类型为uint64_t
    auto buf = static_cast<int64_t*>(malloc(fileSize));
    CheckLoadKeyMallocPtr(buf, fileSize);

    try {
        ssize_t result = fileSystemPtr_->Read(ss.str(), reinterpret_cast<char*>(buf), fileSize);
        CheckReadKeyFileBytes(result, ss.str(), fileSize);
    } catch (std::runtime_error& e) {
        free(static_cast<void*>(buf));
        auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::IO_ERROR,
                           StringFormat("Failed to read file, error is: %s.", e.what()));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }

    hostLoadOffset.clear();
    size_t loadKeySize = fileSize / sizeof(int64_t);
    for (size_t i = 0; i < loadKeySize; i++) {
        // 分配到不同的卡
        if (!embInfo_.isDp && buf[i] % rankSize_ != rankId_) {
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

    CheckFileSystemPtr();
    ssize_t res;
    try {
        res = fileSystemPtr_->Read(embedStream.str(), embeddings, 0, hostLoadOffset, embSize_);
        size_t fileSize = hostLoadOffset.size() * embSize_ * sizeof(float);
        CheckReadKeyFileBytes(res, ss.str(), fileSize);
    } catch (std::runtime_error& e) {
        auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::IO_ERROR,
                           StringFormat("Failed to read file, error is: %s.", e.what()));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }

    LOG_DEBUG("Load embedding done, table:{}, read bytes:{}.", name, res);
}

void EmbeddingDDR::LoadOptimizerSlot(const string &savePath, vector<vector<float>> &optimizerSlots)
{
    if (optimParams.size() == 0) {
        LOG_DEBUG("Optimizer has no slot data to load.");
        return;
    }

    // must init first
    for (size_t i = 0; i < hostLoadOffset.size(); i++) {
        vector<float> tmp(extEmbSize_ - embSize_);
        optimizerSlots.emplace_back(tmp);
    }

    stringstream ss;
    ss << savePath << "/" << name;

    CheckFileSystemPtr();
    int64_t slotIdx = 0;
    for (const auto &param: optimParams) {
        stringstream paramStream;
        paramStream << ss.str() << "/" << optimName + "_" + param << "/slice.data";

        ssize_t res;
        try {
            res = fileSystemPtr_->Read(paramStream.str(), optimizerSlots, slotIdx, hostLoadOffset, embSize_);
            size_t fileSize = hostLoadOffset.size() * embSize_ * sizeof(float);
            CheckReadKeyFileBytes(res, ss.str(), fileSize);
        } catch (std::runtime_error& e) {
            auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::IO_ERROR,
                               StringFormat("Failed to read file, error is: %s.", e.what()));
            LOG_ERROR(error.ToString());
            throw std::runtime_error(error.ToString());
        }

        slotIdx++;
        LOG_DEBUG("Load optimizer slot, table:{}, slot:{}, read bytes:{}.", name, param, res);
    }

    LOG_DEBUG("Load optimizer slot done, table:{}.", name);
}

void EmbeddingDDR::Save(const string& savePath, const int pythonBatchId, bool saveDelta,
                        const map<emb_key_t, KeyInfo>& keyInfo)
{
    SyncLatestEmbedding(pythonBatchId);
    vector<emb_cache_key_t> keys;
    vector<vector<float>> embeddings;
    vector<vector<float>> optimizerSlots;

    auto step = GetStepFromPath(savePath);
    embCache->GetEmbTableInfos(name, keys, embeddings, optimizerSlots);

    SaveKey(savePath, keys);
    SaveEmbedding(savePath, embeddings);
    SaveOptimizerSlot(savePath, optimizerSlots, keys.size());
}

void EmbeddingDDR::SyncLatestEmbedding(const int pythonBatchId)
{
    // 导出host记录的存在于npu的embedding
    std::vector<std::pair<uint64_t, uint64_t>> koVec;
    int rc = embCache->ExportDeviceKeyOffsetPairs(name, koVec);
    if (rc != ock::ctr::H_OK) {
        auto error = Error(ModuleName::M_OCK_CTR, ErrorType::LOGIC_ERROR,
                           StringFormat("ExportDeviceKeyOffsetPairs failed, table:%s, error code:%d.",
                                        name.c_str(), rc));
        LOG_ERROR(error.ToString());
        throw std::invalid_argument(error.ToString());
    }
    std::vector<uint64_t> swapOutKeys;
    for (const auto& p : koVec) {
        swapOutKeys.push_back(p.first);
    }
    LOG_DEBUG("Save swapOutKeys.size:{}, table:{}.", swapOutKeys.size(), name);

    // 接收python save接口发送的卡内embedding
    auto size = hdTransfer->RecvAcl(TransferChannel::SAVE_D2H, TRAIN_CHANNEL_ID, name, 0, -1);
    LOG_DEBUG("Receive D2H data end with DDR mode, size:{}, table:{}.", size, name);
    auto aclData = acltdtGetDataItem(hdTransfer->aclDatasets[name][0], 0);
    if (aclData == nullptr) {
        auto error = Error(ModuleName::M_ACL, ErrorType::NULL_PTR,
                           "Failed to get Acl DataItem pointer from Acl Dataset.");
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString());
    }
    auto* ptr = reinterpret_cast<float*>(acltdtGetDataAddrFromItem(aclData));

    // In step 0, can't update cacheEmb because key-pos mapping has been modified in hybrid_mgmt `ParseKeys` method.
    if (pythonBatchId == 0) {
        LOG_DEBUG("In step 0, skipping update cacheEmb.");
        return;
    }

    if (ssdVocabSize == 0) {
        // 在保存之前先更新host的embedding
        rc = embCache->EmbeddingUpdate(name, swapOutKeys, ptr);
        if (rc != ock::ctr::H_OK) {
            auto error = Error(ModuleName::M_OCK_CTR, ErrorType::LOGIC_ERROR,
                               StringFormat("EmbeddingUpdate failed, table:%s, error code:%d.", name.c_str(), rc));
            LOG_ERROR(error.ToString());
            throw std::invalid_argument(error.ToString());
        }
    } else {
        // SSD mode embedding update.
        EmbeddingUpdateWithSSD(swapOutKeys, ptr);
    }
}

void EmbeddingDDR::EmbeddingUpdateWithSSD(const vector<uint64_t>& swapOutKeys, float* deviceDataPtr)
{
    // 在保存之前先更新ddr和ssd的embedding
    HBMSwapOutInfo info;
    cacheManager_->ProcessSwapOutKeys(name, swapOutKeys, info);
    vector<float*> swapOutAddrs;
    int rc = embCache->EmbeddingLookupAddrs(name, info.swapOutDDRKeys, swapOutAddrs);
    if (rc != ock::ctr::H_OK) {
        auto error = Error(ModuleName::M_OCK_CTR, ErrorType::LOGIC_ERROR,
                           StringFormat("EmbeddingLookupAddrs failed, table:%s, error code:%d.", name.c_str(), rc));
        LOG_ERROR(error.ToString());
        throw std::invalid_argument(error.ToString());
    }
    uint32_t extEmbeddingSize = embInfo_.extEmbeddingSize;
    uint32_t memSize = extEmbeddingSize * sizeof(float);
    // DDR更新
#pragma omp parallel for num_threads(MGMT_CPY_THREADS) default(none) \
    shared(swapOutAddrs, info, deviceDataPtr, extEmbeddingSize, memSize)
    for (uint64_t i = 0; i < swapOutAddrs.size(); i++) {
        int errCode = memcpy_s(swapOutAddrs[i], memSize,
                               deviceDataPtr + info.swapOutDDRAddrOffs[i] * extEmbeddingSize, memSize);
        if (errCode != 0) {
            auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::MEMORY_ERROR,
                               StringFormat("Invoke memcpy_s failed, table:%s, error code:%d. You can query the "
                                            "meaning of security function error code.", name.c_str(), errCode));
            LOG_ERROR(error.ToString());
            throw std::invalid_argument(error.ToString());
        }
    }
    cacheManager_->UpdateL3StorageEmb(name, deviceDataPtr, embInfo_.extEmbeddingSize, info.swapOutL3StorageKeys,
                                      info.swapOutL3StorageAddrOffs);
}

void EmbeddingDDR::SaveKey(const string& savePath, vector<emb_cache_key_t>& keys)
{
    stringstream ss;
    ss << savePath << "/" << name << "/key/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    // 暂时向HBM兼容，转成int64_t，后续再归一key类型为uint64_t
    vector<int64_t> keysCompat(keys.cbegin(), keys.cend());

    CheckFileSystemPtr();
    ssize_t res = fileSystemPtr_->Write(ss.str(), reinterpret_cast<const char *>(keysCompat.data()),
                                        static_cast<size_t>(keys.size() * sizeof(int64_t)));
    if (res == -1) {
        auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::IO_ERROR, "Save key failed!");
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }
}

void EmbeddingDDR::SaveEmbedding(const string& savePath, vector<vector<float>>& embeddings)
{
    stringstream ss;
    ss << savePath << "/" << name << "/embedding/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    CheckFileSystemPtr();
    ssize_t writeBytesNum = fileSystemPtr_->Write(ss.str(), embeddings, embSize_);
    ssize_t expectWriteBytes = embeddings.size() * embSize_ * sizeof(float);
    if (writeBytesNum != expectWriteBytes) {
        auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::LOGIC_ERROR,
                           StringFormat("Save embedding failed, write expect:%ld, actual:%ld, path:%s.",
                                        expectWriteBytes, writeBytesNum, savePath.c_str()));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }
}

void EmbeddingDDR::SaveOptimizerSlot(const string& savePath, vector<vector<float>>& optimizerSlots, size_t keySize)
{
    if (optimizerSlots.size() == 0) {
        LOG_DEBUG("optimizer has no slot data to save");
        return;
    }
    
    if (optimizerSlots.size() != keySize) {
        string errMsg = StringFormat("Optimizer slot data size not equal to key size, "
                                     "optimizerSlots.size:%d, keySize:%d.",
                                     optimizerSlots.size(), keySize);
        auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::LOGIC_ERROR, errMsg);
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
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
            string errMsg = StringFormat("Save optimizer slot failed, write expect:%d, actual:%d, path:%s.",
                                         expectWriteBytes, writeBytesNum, savePath.c_str());
            auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::LOGIC_ERROR, errMsg);
            LOG_ERROR(error.ToString());
            throw std::runtime_error(error.ToString());
        }

        slotIdx++;
    }
}

vector<int64_t> EmbeddingDDR::GetDeviceOffset()
{
    auto error = Error(ModuleName::M_EMB_TABLE, ErrorType::LOGIC_ERROR, "GetDeviceOffset deprecated in ddr/ssd mode.");
    LOG_ERROR(error.ToString());
    throw std::runtime_error(error.ToString());
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

void EmbeddingDDR::BackUpTrainStatus()
{
    embCache->BackUpTrainStatus(name);
}

void EmbeddingDDR::RecoverTrainStatus()
{
    embCache->RecoverTrainStatus(name);
}
