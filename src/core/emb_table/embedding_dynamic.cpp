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

#include "emb_table/embedding_dynamic.h"
#include "utils/logger.h"
#include "utils/singleton.h"
#include "hd_transfer/hd_transfer.h"
#include "file_system/file_system_handler.h"
#include "utils/common.h"

using namespace MxRec;

EmbeddingDynamic::EmbeddingDynamic()
{
}

EmbeddingDynamic::EmbeddingDynamic(const EmbInfo& info, const RankInfo& rankInfo, int inSeed)
    : EmbeddingTable(info, rankInfo, inSeed)
{
    if (isDynamic_) {
        auto ret = aclrtSetDevice(static_cast<int32_t>(rankInfo.deviceId));
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("Set device failed, device_id:{}, ret={}", rankInfo.deviceId, ret);
            throw runtime_error("Acl set device failed!");
        }
        MallocEmbeddingBlock(BLOCK_EMB_NUM);
    }
}

EmbeddingDynamic::~EmbeddingDynamic()
{
    for (auto& it: memoryList_) {
        aclError ret = aclrtFree(it);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtFree failed, ret={}", ret);
        }
    }
}

void EmbeddingDynamic::Key2Offset(std::vector<emb_key_t>& keys, int channel)
{
    constexpr emb_key_t INVALID_DYNAMIC_EXPANSION_ADDR = 0; // 动态扩容算子中的无效地址是0
    std::lock_guard<std::mutex> lk(mut_); // lock for PROCESS_THREAD
    for (emb_key_t& key : keys) {
        if (key == INVALID_KEY_VALUE) {
            key = INVALID_DYNAMIC_EXPANSION_ADDR;
            continue;
        }
        const auto& iter = keyOffsetMap.find(key);
        if (iter != keyOffsetMap.end()) {
            key = iter->second;
            continue;
        }
        // 新值
        if (channel == TRAIN_CHANNEL_ID) {
            int64_t addr = GetEmptyEmbeddingAddress();
            keyOffsetMap[key] = addr;
            key = addr;
            maxOffset++;
            continue;
        }
        key = INVALID_DYNAMIC_EXPANSION_ADDR;
    }
    LOG_DEBUG("current expansion emb:{}, usage:{}/{})", name, maxOffset, devVocabSize);
}

int64_t EmbeddingDynamic::capacity() const
{
    return capacity_;
}

int64_t EmbeddingDynamic::GetEmptyEmbeddingAddress()
{
    if (embeddingList_.empty()) {
        MallocEmbeddingBlock(BLOCK_EMB_NUM);
    }
    float *addr = embeddingList_.front();
    embeddingList_.pop_front();
    return reinterpret_cast<int64_t>(addr);
}

void EmbeddingDynamic::MallocEmbeddingBlock(int embNum)
{
    void *block = nullptr;
    aclError ec = aclrtMalloc(&block, embNum * extEmbSize_ * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ec != 0) {
        throw std::bad_alloc();
    }
    RandomInit(block, embNum);
    memoryList_.push_back(block);
    for (int i = 0; i < embNum; i++) {
        float *embAddr = static_cast<float*>(block) + (i * extEmbSize_);
        embeddingList_.push_back(embAddr);
    }
    capacity_ += embNum;
}

void EmbeddingDynamic::RandomInit(void* addr, size_t embNum)
{
    LOG_INFO("Device GenerateEmbData Start, seed:{}, initializer num: {}", seed_, embInfo_.initializeInfos.size());
    vector<float> hostmem(embNum * extEmbSize_);
    for (const auto& initializeInfo: as_const(embInfo_.initializeInfos)) {
        for (size_t i = 0; i < embNum; ++i) {
            initializeInfo.initializer->GenerateData(&hostmem[i * extEmbSize_], extEmbSize_);
        }
    }
    LOG_INFO("Device GenerateEmbData End, seed:{}", seed_);

    aclError ret = aclrtMemcpy(addr, embNum * extEmbSize_ * sizeof(float),
                               hostmem.data(), embNum * extEmbSize_ * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMemcpy failed, ret={}", ret);
    }
}


void EmbeddingDynamic::Save(const string& savePath)
{
    SaveKey(savePath);
    SaveEmbAndOptim(savePath);
}

void EmbeddingDynamic::SaveKey(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/" << name << "/key/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    deviceKey.clear();
    embAddress.clear();

    for (const auto &it: keyOffsetMap) {
        deviceKey.push_back(it.first);
        embAddress.push_back(it.second);
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

void EmbeddingDynamic::SaveEmbAndOptim(const string& savePath)
{
    for (const string &param: optimParams) {
        optimAddressMap[param].clear();
    }

    for (int64_t &address: embAddress) {
        int optim_param_count = 1;
        for (const string &param: optimParams) {
            optimAddressMap[param].push_back(address + optim_param_count * embSize_ * sizeof(float));
            optim_param_count++;
        }
    }
    SaveEmbData(savePath);
    SaveOptimData(savePath);
}

void EmbeddingDynamic::SetOptimizerInfo(OptimizerInfo& optimizerInfo)
{
    optimName = optimizerInfo.optimName;
    optimParams = optimizerInfo.optimParams;
    for (const string &param: optimParams) {
        optimAddressMap[param] = vector<int64_t>{};
    }
}

void EmbeddingDynamic::SaveEmbData(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/" << name << "/embedding/";
    MakeDir(ss.str());
    ss << "slice_" << rankId_ << ".data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());
    fileSystemPtr->WriteEmbedding(ss.str(), embSize_, embAddress, rankId_);
}

void EmbeddingDynamic::SaveOptimData(const string &savePath)
{
    for (const auto &content: optimAddressMap) {
        stringstream ss;
        ss << savePath << "/" << name << "/" << optimName + "_" + content.first << "/";
        MakeDir(ss.str());
        ss << "slice_" << rankId_ << ".data";

        unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
        unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());
        fileSystemPtr->WriteEmbedding(ss.str(), embSize_, content.second, rankId_);
    }
}

void EmbeddingDynamic::Load(const string& savePath)
{
    LoadKey(savePath);
    LoadEmbAndOptim(savePath);
}

void EmbeddingDynamic::LoadEmbAndOptim(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/" << name;

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    // 读embedding
    stringstream embedStream;
    embedStream << ss.str() << "/" << "embedding/slice.data";
    EmbeddingSizeInfo embeddingSizeInfo = {embSize_, extEmbSize_};
    fileSystemPtr->ReadEmbedding(embedStream.str(), embeddingSizeInfo, firstAddress, rankId_, loadOffset);

    // 读optim
    int optimIndex = 1;
    for (const auto &param: optimParams) {
        stringstream paramStream;
        paramStream << ss.str() << "/" << optimName + "_" + param << "/slice.data";
        fileSystemPtr->ReadEmbedding(paramStream.str(), embeddingSizeInfo,
                                     firstAddress + optimIndex * embSize_ * sizeof(float), rankId_, loadOffset);
        optimIndex++;
    }
}

int EmbeddingDynamic::LoadKey(const string& savePath)
{
    stringstream ss;
    ss << savePath << "/" << name << "/key/slice.data";

    unique_ptr<FileSystemHandler> fileSystemHandler = make_unique<FileSystemHandler>();
    unique_ptr<FileSystem> fileSystemPtr = fileSystemHandler->Create(ss.str());

    size_t fileSize = fileSystemPtr->GetFileSize(ss.str());
    if (fileSize >= FILE_MAX_SIZE) {
        throw runtime_error("Error: Load keys failed. file {} size {}  is too big.", ss.str(), fileSize);
    }

    int64_t* buf = static_cast<int64_t*>(malloc(fileSize));
    if (buf == nullptr) {
        throw runtime_error("Error: Load keys failed. failed to allocate {} bytes using malloc.", fileSize);
    }

    ssize_t res = fileSystemPtr->Read(ss.str(), reinterpret_cast<char*>(buf), fileSize);
    if (res == -1) {
        throw runtime_error("Error: Load keys failed. An error occurred while reading file: {}.", ss.str());
    }
    if (res != fileSize) {
        throw runtime_error(
                "Error: Load keys failed. Expected to read {} bytes, but actually read {} bytes to file {}.",
                fileSize, res, ss.str());
    }

    size_t loadKeySize = fileSize / sizeof(int64_t);

    deviceKey.clear();
    loadOffset.clear();
    for (int i = 0; i < loadKeySize; i = i + 1) {
        if (buf[i] % rankSize_ == rankId_) {
            deviceKey.push_back(buf[i]);
            loadOffset.push_back(i);
        }
    }

    auto datasetSize = deviceKey.size() * extEmbSize_ * sizeof(float);
    void *newBlock = nullptr;
    aclError ret = aclrtMalloc(&newBlock, static_cast<int>(datasetSize), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        throw runtime_error("Error: in dynamic expansion mode,  aclrtMalloc failed, malloc size: {}.", datasetSize);
    }
    // 此处的 newBlock -> first address;
    // 对key_offset map 进行一个恢复操作
    int64_t address = reinterpret_cast<int64_t>(newBlock);
    memoryList_.push_back(newBlock);
    firstAddress = address;
    for (const auto& key : deviceKey) {
        keyOffsetMap[key] = address;
        address = address + extEmbSize_*sizeof(float);
    }

    maxOffset = keyOffsetMap.size();
    free(static_cast<void*>(buf));
}
