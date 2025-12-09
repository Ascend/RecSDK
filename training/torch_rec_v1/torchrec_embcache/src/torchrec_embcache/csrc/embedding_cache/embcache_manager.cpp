/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "embcache_manager.h"

#include <exception>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <vector>

#include <c10/util/Exception.h>
#include <ATen/Parallel.h>

#include "utils/env_config.h"
#include "utils/logger.h"
#include "utils/string_tools.h"
#include "utils/time_cost.h"

using namespace Embcache;

EmbcacheManager::EmbcacheManager(const std::vector<EmbConfig>& embConfigs, bool needAccumulateOffset)
    : embNum_(embConfigs.size()), needAccumulateOffset_(needAccumulateOffset)
{
    ConfigGlobalEnv();
    Logger::SetLevel(GlobalEnv::glogStderrthreshold);
    LogGlobalEnv();
    TORCH_CHECK(embConfigs.size() <= MAX_EMB_TABLE_NUM,
                "The number of embedding tables <= {}", MAX_EMB_TABLE_NUM);
    for (const auto& config : embConfigs) {
        auto length = config.tableName.size();
        if (config.tableName.size() > TABLE_NAME_LENGTH) {
            LOG_ERROR("The length of table name: {} is greater than {}", config.tableName, TABLE_NAME_LENGTH);
            throw std::runtime_error("the length of table name is invalid.");
        }
    }
    this->embConfigs_ = embConfigs;
    enableFastHashMap_ = EnableFastHashMap();

    // 初始化featureFilters_，大小为表数量
    featureFilters_.resize(embNum_);

    for (int32_t i = 0; i < embNum_; i++) {
        LOG_INFO("The tableName: {}, table index: {}, cacheSize is: {}", embConfigs[i].tableName, i,
                 embConfigs[i].cacheSize);
        embTableIndies_.push_back(i);
        int64_t memStartOffset = embConfigs[i].admitAndEvictConfig.IsAdmitEnabled() ? 1 : 0;
        swapManagers_.emplace_back(embConfigs[i].cacheSize, memStartOffset);

        if (enableFastHashMap_) {
            embeddingTables_.emplace_back(std::make_unique<EmbTableFastHashMap>(embConfigs[i]));
        } else {
            embeddingTables_.emplace_back(std::make_unique<EmbTableUnorderedMap>(embConfigs[i]));
        }
        TORCH_CHECK(embeddingTables_.back() != nullptr, "embeddingTables_.back() should not be nullptr");

        if (embConfigs[i].admitAndEvictConfig.IsFeatureFilterEnabled()) {
            const auto& aaeConfig = embConfigs[i].admitAndEvictConfig;
            featureFilters_[i] = std::make_unique<FeatureFilter>(
                embConfigs[i].tableName,
                aaeConfig.admitThreshold,
                aaeConfig.evictThreshold,
                aaeConfig.evictStepInterval);
        }
    }
    TORCH_CHECK(embConfigs.size() > 0, "ERROR, Size of embConfigs must > 0")
    optimNum_ = embConfigs[0].optimNum;
    for (auto& embedConfig : embConfigs) {
        TORCH_CHECK(embedConfig.optimNum == optimNum_)
    }
}

bool EmbcacheManager::EnableFastHashMap()
{
    char* enableFastHashMapStr = getenv("ENABLE_FAST_HASHMAP");
    if (!enableFastHashMapStr) {
        LOG_INFO("The env ENABLE_FAST_HASHMAP is not detected, std::unordered_map is used");
        return false;
    }

    std::string switchStr = std::string(enableFastHashMapStr);
    std::transform(switchStr.begin(), switchStr.end(), switchStr.begin(), ::tolower);
    if (switchStr == "true" || switchStr == "yes" || switchStr == "1") {
        LOG_INFO("ENABLE_FAST_HASHMAP=true, FastHashMap is used");
        return true;
    }
    LOG_INFO("The ENABLE_FAST_HASHMAP=false, std::unordered_map is used");
    return false;
}

/**
 * 根据已有的offsetPerKey获取新的offsetPerKey 处理多个feature name对应一个表的场景
 * @param offsetPerKey 当前的offsetPerKey
 * @param curTableIndices 当前的表索引
 * @return newOffsetPerKey
*/
std::vector<int64_t> EmbcacheManager::GetNewOffsetPerKey(const std::vector<int64_t>& offsetPerKey,
                                                         const std::vector<int32_t> curTableIndices) const
{
    if (curTableIndices.size() + 1 == offsetPerKey.size()) {
        std::vector<int64_t> newOffsetPerKey(offsetPerKey.cbegin(), offsetPerKey.cend());
        return newOffsetPerKey;
    }

    std::vector<int64_t> newOffsetPerKey(embTableIndies_.size() + 1, 0);
    std::vector<int64_t> featureSplitByTable(embTableIndies_.size(), 0);
    for (size_t i = 0; i < featureSplitByTable.size(); ++i) {
        auto tableIndex = curTableIndices[i];
        featureSplitByTable[i] = embConfigs_[tableIndex].num_features;
    }
    int64_t start = 0;
    for (size_t i = 0; i < featureSplitByTable.size(); ++i) {
        int64_t end = start + featureSplitByTable[i];
        TORCH_CHECK(end < offsetPerKey.size(), "end must less than offsetPerKey size");
        newOffsetPerKey[i + 1] = offsetPerKey[end];
        start = end;
    }
    return newOffsetPerKey;
}

SwapInfo EmbcacheManager::ComputeSwapInfo(const at::Tensor& batchKeys, const std::vector<int64_t>& offsetPerKey,
                                          const std::vector<int32_t>& tableIndices)
{
    TimeCost getSwapInfoTC;

    TORCH_CHECK(batchKeys.is_contiguous(), "batchKeys must be contiguous")
    TORCH_CHECK(batchKeys.dtype() == torch::kInt64, "batchKeys must be of type int64_t")
    const std::vector<int32_t>& curTableIndices = tableIndices.empty() ? embTableIndies_ : tableIndices;
    TORCH_CHECK(curTableIndices.size() == embTableIndies_.size(),
                "tableIndices size must be equal to embTableIndies_ size");
    auto newOffsetPerKey = GetNewOffsetPerKey(offsetPerKey, curTableIndices);
    TORCH_CHECK(curTableIndices.size() + 1 == newOffsetPerKey.size(),
                "tableIndices size+1 must be equal to newOffsetPerKey size");

    auto* keyPtr = batchKeys.data_ptr<int64_t>();
    TORCH_CHECK(keyPtr != nullptr, "keyPtr should not be nullptr");
    int64_t keyNum = batchKeys.numel();
    int64_t offPreSum = 0;

    std::vector<int64_t> swapoutOffs;
    std::vector<int64_t> swapinOffs;
    std::vector<int64_t> batchOffs;
    SwapInfo swapInfo;
    for (int64_t i = 0; i < curTableIndices.size(); i++) {
        int64_t idx = curTableIndices[i];
        TORCH_CHECK(idx >= 0 && idx < embNum_, "table index {} is out of range [0, {})", idx, embNum_);
        auto startIndex = newOffsetPerKey[i];
        auto endIndex = newOffsetPerKey[i + 1];
        TORCH_CHECK(startIndex >= 0 && endIndex <= keyNum && startIndex <= endIndex,
                    "Invalid offsetPerKey[{}]: {}, offsetPerKey[{}]: {}, keyNum: {}", i, startIndex, i + 1,
                    endIndex, keyNum);

        if (embConfigs_[idx].admitAndEvictConfig.IsAdmitEnabled()) {
            if (featureFilters_[idx]) {
                featureFilters_[idx]->CountFilter(keyPtr, startIndex, endIndex);
            }
        }

        // 取出每个表的 key
        std::vector<int64_t> batchKeysVec(keyPtr + startIndex, keyPtr + endIndex);
        auto tp = swapManagers_[idx].ComputeSwapInfo(batchKeysVec);

        std::vector<int64_t>& swapoutKeysi = std::get<SWAP_INFO_TUPLE_INDEX0>(tp);
        std::vector<int64_t>& swapoutOffsi = std::get<SWAP_INFO_TUPLE_INDEX1>(tp);
        std::vector<int64_t>& swapinKeysi = std::get<SWAP_INFO_TUPLE_INDEX2>(tp);
        std::vector<int64_t>& swapinOffsi = std::get<SWAP_INFO_TUPLE_INDEX3>(tp);
        std::vector<int64_t>& batchOffsi = std::get<SWAP_INFO_TUPLE_INDEX4>(tp);
        if (needAccumulateOffset_) {
            // 加上表外偏移
            for (auto& off : swapoutOffsi) {
                off += offPreSum;
            }
            for (auto& off : swapinOffsi) {
                off += offPreSum;
            }
            offPreSum += embConfigs_[idx].cacheSize;
        }

        swapInfo.swapoutKeys.emplace_back(std::move(swapoutKeysi));
        swapInfo.swapinKeys.emplace_back(std::move(swapinKeysi));
        swapoutOffs.insert(swapoutOffs.end(), swapoutOffsi.begin(), swapoutOffsi.end());
        swapinOffs.insert(swapinOffs.end(), swapinOffsi.begin(), swapinOffsi.end());
        batchOffs.insert(batchOffs.end(), batchOffsi.begin(), batchOffsi.end());
    }

    errno_t rc = EOK;
    auto longPinnedOpt = at::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true);
    swapInfo.swapoutOffs = at::empty({static_cast<int64_t>(swapoutOffs.size())}, longPinnedOpt);
    size_t swapoutOffsSize = swapoutOffs.size() * sizeof(int64_t);
    if (swapoutOffsSize > 0) {
        rc = memcpy_s(swapInfo.swapoutOffs.data_ptr<int64_t>(), swapoutOffsSize, swapoutOffs.data(), swapoutOffsSize);
        if (rc != 0) {
            LOG_ERROR("memcpy_s swapoutOffs to swapInfo.swapoutOffs failed. ret: {}", rc);
            throw std::runtime_error("memcpy_s swapoutOffs to swapInfo.swapoutOffs failed.");
        }
    }

    swapInfo.swapinOffs = at::empty({static_cast<int64_t>(swapinOffs.size())}, longPinnedOpt);
    size_t swapinOffsSize = swapinOffs.size() * sizeof(int64_t);
    if (swapinOffsSize > 0) {
        rc = memcpy_s(swapInfo.swapinOffs.data_ptr<int64_t>(), swapinOffsSize, swapinOffs.data(), swapinOffsSize);
        if (rc != 0) {
            LOG_ERROR("memcpy_s swapinOffs to swapInfo.swapinOffs failed. ret: {}", rc);
            throw std::runtime_error("memcpy_s swapinOffs to swapInfo.swapinOffs failed.");
        }
    }

    swapInfo.batchOffs = at::empty({static_cast<int64_t>(batchOffs.size())}, longPinnedOpt);
    size_t batchOffsSize = batchOffs.size() * sizeof(int64_t);
    if (batchOffsSize > 0) {
        rc = memcpy_s(swapInfo.batchOffs.data_ptr<int64_t>(), batchOffsSize, batchOffs.data(), batchOffsSize);
        if (rc != 0) {
            LOG_ERROR("memcpy_s batchOffs to swapInfo.batchOffs failed. ret: {}", rc);
            throw std::runtime_error("memcpy_s batchOffs to swapInfo.batchOffs failed.");
        }
    }

    swapCount_++;

    LOG_DEBUG("The getSwapInfoTC(ms): {}", getSwapInfoTC.ElapsedMS());

    return swapInfo;
}

AsyncTask<SwapInfo> EmbcacheManager::ComputeSwapInfoAsync(const at::Tensor& batchKeys,
                                                          const std::vector<int64_t>& offsetPerKey,
                                                          const std::vector<int32_t>& tableIndices)
{
    return AsyncTask<SwapInfo>([this, batchKeys, offsetPerKey, tableIndices]() {
        return ComputeSwapInfo(batchKeys, offsetPerKey, tableIndices);
    });
}

SwapinTensor EmbcacheManager::EmbeddingLookup(const std::vector<std::vector<int64_t>>& swapinKeys,
                                              const std::vector<int32_t>& tableIndices)
{
    TimeCost embeddingLookupTC;
    TORCH_CHECK(swapinKeys.size() == embTableIndies_.size(), "swapinKeys size must be equal to embTableIndies_ size");

    auto floatPinnedOpt = at::TensorOptions().dtype(at::kFloat).device(at::kCPU).pinned_memory(true);
    auto longPinnedOpt = at::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true);
    SwapinTensor swapinTensor;
    swapinTensor.jaggedOffs = at::empty({static_cast<int64_t>(swapinKeys.size() + 1)}, longPinnedOpt);
    auto jaggedOffsPtr = swapinTensor.jaggedOffs.data_ptr<int64_t>();

    jaggedOffsPtr[0] = 0;
    for (uint64_t i = 1; i <= swapinKeys.size(); i++) {
        jaggedOffsPtr[i] = jaggedOffsPtr[i - 1] + swapinKeys[i - 1].size() * embConfigs_[i - 1].embDim;
    }

    int64_t embsSize = jaggedOffsPtr[swapinKeys.size()];
    swapinTensor.swapinEmbs = at::empty({embsSize}, floatPinnedOpt);
    for (int32_t i = 0; i < optimNum_; i++) {
        swapinTensor.swapinOptims.emplace_back(at::empty({embsSize}, floatPinnedOpt));
    }

    const std::vector<int32_t>& curTableIndices = tableIndices.empty() ? embTableIndies_ : tableIndices;
    TORCH_CHECK(curTableIndices.size() == swapinKeys.size(),
                "tableIndices size must be equal to swapinKeys size");

    std::vector<float*> swapinOptimsPtr(optimNum_);
    for (uint64_t i = 0; i < swapinKeys.size(); i++) {
        for (int32_t j = 0; j < optimNum_; j++) {
            swapinOptimsPtr[j] = swapinTensor.swapinOptims[j].data_ptr<float>() + jaggedOffsPtr[i];
        }

        int32_t idx = curTableIndices[i];
        TORCH_CHECK(idx >= 0 && idx < embeddingTables_.size(),
                    "table index {} is out of range [0, {})", idx, embeddingTables_.size());
        embeddingTables_[idx]->FindOrInsert(swapinKeys[i], swapinTensor.swapinEmbs.data_ptr<float>() + jaggedOffsPtr[i],
                                            swapinOptimsPtr);
    }

    LOG_DEBUG("The embeddingLookupTC(ms): {}", embeddingLookupTC.ElapsedMS());
    return swapinTensor;
}

AsyncTask<SwapinTensor> EmbcacheManager::EmbeddingLookupAsync(const SwapInfo& swapInfo,
                                                              const std::vector<int32_t>& tableIndices)
{
    return AsyncTask<SwapinTensor>([this, swapinKeys = swapInfo.swapinKeys, tableIndices]() {
        return EmbeddingLookup(swapinKeys, tableIndices);
    });
}

void EmbcacheManager::EmbeddingUpdate(const std::vector<std::vector<int64_t>>& swapoutKeys,
                                      const at::Tensor& swapoutEmbs, const std::vector<at::Tensor>& swapoutOptims,
                                      const std::vector<int32_t>& tableIndices)
{
    TimeCost embeddingUpdateTC;
    for (auto& embedConfig : embConfigs_) {
        TORCH_CHECK(embedConfig.optimNum == (int32_t)swapoutOptims.size())
    }
    for (auto& optimition : swapoutOptims) {
        TORCH_CHECK(swapoutEmbs.numel() == optimition.numel())
        TORCH_CHECK(optimition.dtype() == torch::kFloat32)
    }
    TORCH_CHECK(swapoutEmbs.dtype() == torch::kFloat32)

    const std::vector<int32_t>& curTableIndices = tableIndices.empty() ? embTableIndies_ : tableIndices;
    TORCH_CHECK(swapoutKeys.size() == embTableIndies_.size(),
                "swapoutKeys size must be equal to embTableIndies_ size");
    TORCH_CHECK(curTableIndices.size() == swapoutKeys.size(),
                "tableIndices size must be equal to swapoutKeys size");

    auto* swapoutEmbsPtr = swapoutEmbs.data_ptr<float>();
    int64_t jaggedOff = 0;
    std::vector<float*> swapoutOptimPtrs(swapoutOptims.size());
    for (uint64_t i = 0; i < swapoutKeys.size(); i++) {
        for (size_t j = 0; j < swapoutOptims.size(); j++) {
            swapoutOptimPtrs[j] = swapoutOptims[j].data_ptr<float>() + jaggedOff;
        }

        int32_t idx = curTableIndices[i];
        TORCH_CHECK(idx >= 0 && idx < embeddingTables_.size(),
                    "table index {} is out of range [0, {})", idx, embeddingTables_.size());
        embeddingTables_[idx]->InsertOrAssign(swapoutKeys[i], swapoutEmbsPtr + jaggedOff, swapoutOptimPtrs);
        jaggedOff += swapoutKeys[i].size() * embConfigs_[idx].embDim;
    }

    LOG_DEBUG("The embeddingUpdateTC(ms): {}", embeddingUpdateTC.ElapsedMS());
    embUpdateCount_++;

    if (NeedEvictEmbeddingTable()) {
        RemoveEmbeddingTableInfo();
    }
}

void EmbcacheManager::Embedding2Host(const at::Tensor& weightsDev, const std::vector<at::Tensor>& momentumDevs)
{
    TORCH_CHECK(momentumDevs.size() <= MAX_MOMENTUM_NUM,
                "momentumDevs size should not be <= {}", MAX_MOMENTUM_NUM);
    for (auto& momentumDev : momentumDevs) {
        TORCH_CHECK(weightsDev.numel() == momentumDev.numel())
        TORCH_CHECK(momentumDev.dtype() == torch::kFloat32)
    }
    TORCH_CHECK(weightsDev.dtype() == torch::kFloat32)
    LOG_INFO("In Embedding2Host, weightsDev shape:{}.", GetDevWeightsShape(weightsDev));

    auto* weightsDevPtr = weightsDev.data_ptr<float>();
    int64_t jaggedOff = 0;

    for (int32_t embIndex = 0; embIndex < embNum_; embIndex++) {
        // cache中可能预留了offset 0位置，因此拷贝回host时，需先加上偏移
        auto start = swapManagers_[embIndex].GetMemStartOffset();
        int64_t currentTableOffset = jaggedOff + start * embConfigs_[embIndex].embDim;
        auto end = swapManagers_[embIndex].GetOccupiedNum();
        std::vector<int64_t> keys;
        keys.reserve(end - start);
        for (int64_t off = start; off < end; off++) {
            keys.emplace_back(swapManagers_[embIndex].GetKey(off));
        }
        std::vector<float*> momentumDataPtrs(momentumDevs.size());
        for (size_t i = 0; i < momentumDevs.size(); i++) {
            momentumDataPtrs[i] = momentumDevs[i].data_ptr<float>() + currentTableOffset;
        }
        embeddingTables_[embIndex]->InsertOrAssign(keys, weightsDevPtr + currentTableOffset, momentumDataPtrs);

        // Here, GetOccupiedNum is less than embConfigs_[embIndex].cacheSize = weightsDev.shape[0],
        // and we need to skip the unnecessary weight indices.
        jaggedOff += embConfigs_[embIndex].cacheSize * embConfigs_[embIndex].embDim;
        LOG_INFO("Embedding2Host, embIndex:{}, update key size:{}, jaggedOff:{}, currentTableOffset:{}.", embIndex,
                 keys.size(), jaggedOff, currentTableOffset);
    }
}

std::shared_ptr<FileSystem> EmbcacheManager::GetFileSystem(const std::string& path)
{
    FileSystemHandler handler;
    return handler.Create(path);
}

void EmbcacheManager::CreateMomentumDir(const std::string& pathPrefix,
                                        const std::shared_ptr<FileSystem>& fileSystemPtr) const
{
    if (optimNum_ == 0) {
        return;
    }
    if (optimNum_ > 0) {
        std::string fileMomentum1SliceData = pathPrefix + MOMENTUM1_STR_PATH + SLICE_DATA_PATH;
        fileSystemPtr->CreateFileDir(fileMomentum1SliceData);
    }
    if (optimNum_ > 1) {
        std::string fileMomentum2SliceData = pathPrefix + MOMENTUM2_STR_PATH + SLICE_DATA_PATH;
        fileSystemPtr->CreateFileDir(fileMomentum2SliceData);
    }
}

void EmbcacheManager::Save(const std::string& path, const int rank)
{
    LOG_INFO("open file once for a table");
    auto fileSystemPtr = GetFileSystem(path);
    Check4Write(fileSystemPtr, path, rank);
    for (int32_t i = 0; i < embNum_; i++) {
        std::string tableName = embConfigs_[i].tableName;
        std::string pathPrefix = path + "/" + tableName + RANK_STR_PATH + std::to_string(rank);
        std::string embDataFile = pathPrefix + EMBEDDING_STR_PATH + SLICE_DATA_PATH;
        fileSystemPtr->CreateFileDir(embDataFile);
        std::string keyDataFile = pathPrefix + KEY_STR_PATH + SLICE_DATA_PATH;
        fileSystemPtr->CreateFileDir(keyDataFile);
        CreateMomentumDir(pathPrefix, fileSystemPtr);

        auto keyFd = open(keyDataFile.c_str(), O_RDWR | O_CREAT | O_APPEND, 0640);
        auto embFd = open(embDataFile.c_str(), O_RDWR | O_CREAT | O_APPEND, 0640);
        if (embFd == -1) {
            throw std::runtime_error("Failed to open embedding file.");
        }
        std::string momentum1DataFile = pathPrefix + MOMENTUM1_STR_PATH + SLICE_DATA_PATH;
        int m1Fd = 0;
        if (optimNum_ > 0) {
            m1Fd = open(momentum1DataFile.c_str(), O_RDWR | O_CREAT | O_APPEND, 0640);
            if (m1Fd == -1) {
                throw std::runtime_error("Failed to open momentum1 file.");
            }
        }
        std::string momentum2DataFile = pathPrefix + MOMENTUM2_STR_PATH + SLICE_DATA_PATH;
        int m2Fd = 0;
        if (optimNum_ > 1) {
            m2Fd = open(momentum2DataFile.c_str(), O_RDWR | O_CREAT | O_APPEND, 0640);
            if (m2Fd == -1) {
                throw std::runtime_error("Failed to open momentum2 file.");
            }
        }

        size_t count = 0;
        std::vector<int64_t> saveKeys;
        LOG_INFO("Start save table:{}.", tableName);
        auto embDim = embConfigs_[i].embDim;
        embeddingTables_[i]->ForEachKey([&](const int64_t key, const float* value) {
            ++count;
            // 1. write key
            WriteData(fileSystemPtr, keyDataFile, reinterpret_cast<const char*>(&key), sizeof(int64_t), keyFd);
            if (embConfigs_[i].admitAndEvictConfig.IsAdmitEnabled()) {
                saveKeys.emplace_back(key);
            }
            // 2. write embedding
            WriteData(fileSystemPtr, embDataFile, reinterpret_cast<const char*>(value), embDim * sizeof(float), embFd);
            LOG_TRACE("In save, table:{}, key:{}, embedding.dim:{}.", tableName, key, embDim);

            // 3. write momentum
            if (optimNum_ > 0) {
                WriteData(fileSystemPtr, momentum1DataFile, reinterpret_cast<const char*>(value + embDim),
                          embDim * sizeof(float), m1Fd);
            }
            if (optimNum_ > 1) {
                WriteData(fileSystemPtr, momentum2DataFile, reinterpret_cast<const char*>(value + optimNum_ * embDim),
                          embDim * sizeof(float), m2Fd);
            }
        });
        WriteAttributeFile(i, pathPrefix, count, fileSystemPtr);

        // save feature filter related data
        SaveFeatureAdmitAndEvictInfo(fileSystemPtr, i, pathPrefix, saveKeys);
        LOG_INFO("In save, table:{}, save data shape: [{}, {}].", tableName, count, embDim);
    }
}

void EmbcacheManager::Check4Write(const std::shared_ptr<FileSystem>& fileSystemPtr, const std::string& filePath,
                                  int rank)
{
    std::filesystem::path pathObj(filePath);
    if (std::filesystem::absolute(pathObj) != filePath) {
        auto errMsg = Logger::Format(
            "File path is invalid, it is not an absolute path:{}.", filePath);
        throw std::runtime_error(errMsg);
    }
    if (fileSystemPtr == nullptr) {
        auto errMsg = Logger::Format(
            "Failed to get file system pointer, the fileSystemPtr is nullptr. Current rank:{}.", rank);
        throw std::runtime_error(errMsg);
    }
    fileSystemPtr->CreateFileDir(filePath + "/file");  // only create file parent dir if not exist
    fileSystemPtr->Valid4WriteDir(filePath);
}

void EmbcacheManager::WriteData(const std::shared_ptr<FileSystem>& fileSystemPtr, const std::string& filePath,
                                const char* dataAddr, size_t dataSize)
{
    auto writeBytes = fileSystemPtr->Write(filePath, dataAddr, dataSize);
    if (writeBytes != static_cast<ssize_t>(dataSize)) {
        auto errMsg = Logger::Format(
            "Write data to file error, expect write bytes:{}, actual write bytes:{}, file:{}."
            " Please check whether the available disk space is sufficient.",
            dataSize, writeBytes, filePath);
        LOG_ERROR(errMsg);
        throw std::runtime_error(errMsg);
    }
}

void EmbcacheManager::WriteData(const std::shared_ptr<FileSystem>& fileSystemPtr, const std::string& filePath,
                                const char* dataAddr, size_t dataSize, int fd)
{
    auto writeBytes = fileSystemPtr->Write(filePath, dataAddr, dataSize, fd);
    if (writeBytes != static_cast<ssize_t>(dataSize)) {
        auto errMsg = Logger::Format(
            "Write data to file error, expect write bytes:{}, actual write bytes:{}, file:{}."
            " Please check whether the available disk space is sufficient.",
            dataSize, writeBytes, filePath);
        LOG_ERROR(errMsg);
        throw std::runtime_error(errMsg);
    }
}

void EmbcacheManager::WriteAttributeFile(int32_t tableIndex, const std::string& pathPrefix, size_t count,
                                         const std::shared_ptr<FileSystem>& fileSystemPtr)
{
    // key emb attribute
    std::string keyAttrFile = pathPrefix + KEY_STR_PATH + SLICE_ATTR_PATH;
    fileSystemPtr->CreateFileDir(keyAttrFile);
    std::vector<int64_t> keyAttribute = {sizeof(int64_t), count};
    WriteData(fileSystemPtr, keyAttrFile, reinterpret_cast<const char*>(keyAttribute.data()),
              keyAttribute.size() * sizeof(int64_t));

    std::string embAttrFile = pathPrefix + EMBEDDING_STR_PATH + SLICE_ATTR_PATH;
    fileSystemPtr->CreateFileDir(embAttrFile);
    std::vector<int64_t> embedAttribute = {sizeof(float), count, embConfigs_[tableIndex].embDim};
    WriteData(fileSystemPtr, embAttrFile, reinterpret_cast<const char*>(embedAttribute.data()),
              embedAttribute.size() * sizeof(int64_t));

    // optimizer attribute
    if (optimNum_ == 0) {
        return;
    }
    std::vector<int64_t> momentumAttrData = {sizeof(float), static_cast<int64_t>(count),
                                             embConfigs_[tableIndex].embDim};
    if (optimNum_ > 0) {
        std::string m1AttrFile = pathPrefix + MOMENTUM1_STR_PATH + SLICE_ATTR_PATH;
        fileSystemPtr->CreateFileDir(m1AttrFile);
        WriteData(fileSystemPtr, m1AttrFile, reinterpret_cast<const char*>(momentumAttrData.data()),
                  momentumAttrData.size() * sizeof(int64_t));
    }
    if (optimNum_ > 1) {
        std::string m2AttrFile = pathPrefix + MOMENTUM2_STR_PATH + SLICE_ATTR_PATH;
        fileSystemPtr->CreateFileDir(m2AttrFile);
        // 目前momentum2Attribute和momentum1Attribute是一致的
        WriteData(fileSystemPtr, m2AttrFile, reinterpret_cast<const char*>(momentumAttrData.data()),
                  momentumAttrData.size() * sizeof(int64_t));
    }
}

void EmbcacheManager::Load(const std::string& path, int rank)
{
    auto fileSystemPtr = GetFileSystem(path);
    TORCH_CHECK(fileSystemPtr != nullptr, "fileSystemPtr should not be nullptr");

    for (int32_t i = 0; i < embNum_; i++) {
        std::string tableName = embConfigs_[i].tableName;
        LOG_INFO("Start load, rank:{}, table:{}.", rank, tableName);
        TableRankParam tableParams(embConfigs_[i].tableName, i, embConfigs_[i].embDim, rank);
        std::string filePrefix = path + "/" + tableName + "/rank" + std::to_string(rank);

        // load key
        std::vector<int64_t> keys;
        std::string keyAttrFile = filePrefix + "/key/slice.attribute";
        std::string keysDataFile = filePrefix + "/key/slice.data";
        ReadKeysData(fileSystemPtr, keys, keyAttrFile, keysDataFile);

        // load embedding and optimizer
        auto loadCountOneTime = GetOneTimeLoadCount(embConfigs_[i].embDim);
        for (size_t j = 0; j < keys.size(); j += loadCountOneTime) {
            auto start = keys.begin() + static_cast<int64_t>(j);
            auto end = keys.begin() + std::min(j + loadCountOneTime, keys.size());
            std::vector<int64_t> sliceKeys(start, end);
            tableParams.loadEmbeddingOffset = static_cast<int64_t>(j);
            LoadEmbeddingAndOptimizer(fileSystemPtr, i, filePrefix, sliceKeys, tableParams);
        }

        // load feature filter related data
        LoadFeatureAdmitAndEvictInfo(fileSystemPtr, i, filePrefix, keys);
    }
}

int32_t EmbcacheManager::GetOneTimeLoadCount(int32_t embDim)
{
    if (embDim == 0) {
        throw std::runtime_error("embDim is zero.");
    }
    return MAX_EMB_DIM / embDim * ONE_TIME_LOAD_DIM_4096;
}

void EmbcacheManager::LoadEmbeddingAndOptimizer(const shared_ptr<FileSystem>& fileSystemPtr, int32_t tableIndex,
                                                const string& filePrefix, const vector<int64_t>& keys,
                                                const TableRankParam& tableParams)
{
    vector<vector<float>> embeddings;
    string embFilePath = filePrefix + "/embedding/slice.data";
    ReadEmbeddings(fileSystemPtr, embeddings, embFilePath, keys.size(), tableParams);

    // load optimizer parameter
    vector<vector<float>> momentum1;
    if (optimNum_ > 0) {
        string momentum1FilePath = filePrefix + "/momentum1/slice.data";
        ReadEmbeddings(fileSystemPtr, momentum1, momentum1FilePath, keys.size(), tableParams);
    }
    vector<vector<float>> momentum2;
    if (optimNum_ > 1) {
            string momentum2FilePath = filePrefix + "/momentum2/slice.data";
            ReadEmbeddings(fileSystemPtr, momentum2, momentum2FilePath, keys.size(), tableParams);
    }
    RecordLoadDebugInfo(keys, embeddings, momentum1, momentum2, tableParams);

    for (size_t k = 0; k < keys.size(); k++) {
        vector<int64_t> insertKey = {keys[k]};
        vector<float*> momentum = {};
        if (optimNum_ > 0) {
            momentum.emplace_back(momentum1[k].data());
        }
        if (optimNum_ > 1) {
            momentum.emplace_back(momentum2[k].data());
        }
        embeddingTables_[tableIndex]->InsertOrAssign(insertKey, embeddings[k].data(), momentum);
    }
}

void EmbcacheManager::RecordLoadDebugInfo(const vector<int64_t>& keys, const vector<std::vector<float>>& embeddings,
                                          const vector<std::vector<float>>& momentum1,
                                          const vector<std::vector<float>>& momentum2,
                                          const TableRankParam& tableParams)
{
    if (Logger::GetLevel() > Logger::TRACE) {
        return;
    }
    for (size_t j = 0; j < keys.size(); ++j) {
        LOG_TRACE("In load, rank:{}, table:{}, current key:{}.",
                  tableParams.rank, tableParams.tableName, keys[j]);
    }
}

void EmbcacheManager::ReadAttributeData(const std::shared_ptr<FileSystem>& fileSystemPtr, const string& filePath,
                                        std::vector<int64_t>& dataVec, int dataCount)
{
    dataVec.resize(dataCount, ATTR_VEC_INIT_VALUE);
    auto attrBytes = dataCount * sizeof(int64_t);
    auto readBytes = fileSystemPtr->Read(filePath, reinterpret_cast<char*>(dataVec.data()),
                                         dataCount * sizeof(int64_t));
    if (readBytes != static_cast<ssize_t>(attrBytes)) {
        auto errMsg =
            Logger::Format("Read key attribute file error, expect read bytes:{}, actual read bytes:{}, file:{}",
                           attrBytes, readBytes, filePath);
        throw std::runtime_error(errMsg);
    }
}

template <class T>
void EmbcacheManager::ReadKeysData(const std::shared_ptr<FileSystem>& fileSystemPtr, std::vector<T>& keys,
                                   const string& keyAttrFile, const string& keyDataFile)
{
    // check key attribute
    std::vector<int64_t> keyAttrVec;
    ReadAttributeData(fileSystemPtr, keyAttrFile, keyAttrVec, KEY_ATTRIBUTE_DATA_LEN);
    if (keyAttrVec[1] == ATTR_VEC_INIT_VALUE || keyAttrVec[1] > KEY_SIZE_MAX) {
        auto errMsg =
            Logger::Format("Read key attribute file error, keys count is invalid:{}, file:{}.",
                           keyAttrVec[1], keyAttrFile);
        throw std::runtime_error(errMsg);
    }

    // key data
    size_t keyFileBytes = fileSystemPtr->GetFileSize(keyDataFile);
    if (keyFileBytes % sizeof(T) != 0 || keyFileBytes / sizeof(T) != keyAttrVec[1]) {
        auto errMsg =
            Logger::Format("Key file bytes is not an integer multiple of type int64_t, key file:{}", keyDataFile);
        LOG_ERROR(errMsg);
        throw std::runtime_error(errMsg);
    }
    keys.resize(keyAttrVec[1], INVALID_KEY);
    auto readBytes = fileSystemPtr->Read(keyDataFile, reinterpret_cast<char*>(keys.data()), keyFileBytes);
    if (readBytes != static_cast<ssize_t>(keyFileBytes)) {
        auto errMsg = Logger::Format("Read key data file error, expect read bytes:{}, actual read bytes:{}, file:{}",
            keyFileBytes, readBytes, keyDataFile);
        throw std::runtime_error(errMsg);
    }
}

void EmbcacheManager::CheckEmbeddingDim(const std::shared_ptr<FileSystem>& fileSystemPtr, const string& dataFilePath,
                                        const TableRankParam& tableParams)
{
    if (dataFilePath.substr(dataFilePath.size() - DATA_SUFFIX.length()) != DATA_SUFFIX) {
        LOG_ERROR("Check embedding data file dim error, dataFilePath is not end with `data`.");
        throw std::runtime_error("Check embedding data file dim error, dataFilePath is not end with `data`.");
    }
    std::string embAttrFile = dataFilePath.substr(0, dataFilePath.size() - DATA_SUFFIX.length()) + ATTR_SUFFIX;
    std::vector<int64_t> embAttrVec;
    ReadAttributeData(fileSystemPtr, embAttrFile, embAttrVec, EMB_ATTRIBUTE_DATA_LEN);
    auto embDimFromFile = embAttrVec[EMB_ATTRIBUTE_DATA_LEN - 1];
    if (embDimFromFile == ATTR_VEC_INIT_VALUE || embDimFromFile != tableParams.embDim) {
        auto errMsg = Logger::Format(
            "Embedding or momentum dim error, load data dim from attribute:{}, current table dim:{}, file:{}.",
            embDimFromFile, tableParams.embDim, embAttrFile);
        throw std::runtime_error(errMsg);
    }
}

void EmbcacheManager::ReadEmbeddings(const std::shared_ptr<FileSystem>& fileSystemPtr,
                                     std::vector<std::vector<float>>& embeddings, const string& filePath,
                                     size_t vectorSize, const TableRankParam& tableParams)
{
    LOG_INFO("In load, rank:{}, table:{}, start load file data:{}.", tableParams.rank, tableParams.tableName, filePath);
    int32_t embDim = tableParams.embDim;
    CheckEmbeddingDim(fileSystemPtr, filePath, tableParams);
    for (size_t i = 0; i < vectorSize; ++i) {
        std::vector<float> tmp(embDim);
        embeddings.emplace_back(tmp);
    }
    std::vector<int64_t> offsetVec(vectorSize);
    std::iota(offsetVec.begin(), offsetVec.end(), tableParams.loadEmbeddingOffset);
    ssize_t readBytes;
    try {
        readBytes = fileSystemPtr->Read(filePath, embeddings, 0, offsetVec, embDim);
    } catch (std::runtime_error& e) {
        auto errMsg = Logger::Format("In load, rank:{}, table:{}, load file error: {}.", tableParams.rank,
            tableParams.tableName, filePath);
        LOG_ERROR(errMsg);
        throw std::runtime_error(errMsg);
    }
    auto expectReadBytes = static_cast<ssize_t>(embeddings.size() * embDim * sizeof(float));
    if (readBytes != expectReadBytes) {
        auto errMsg = Logger::Format("Read data to file error, expect read bytes:{}, actual read bytes:{}, file:{}",
            filePath, expectReadBytes, readBytes);
        LOG_ERROR(errMsg);
        throw std::runtime_error(errMsg);
    }
    LOG_INFO("In load, rank:{}, table:{}, load file end, embeddings size:{}, file:{}.", tableParams.rank,
             tableParams.tableName, embeddings.size(), filePath);
}

std::string EmbcacheManager::GetDevWeightsShape(const at::Tensor& weightsDev)
{
    std::stringstream ss;
    ss << "weightsDev shape:[";
    auto shape = weightsDev.sizes();
    for (auto i : shape) {
        ss << " ";
        ss << i;
    }
    ss << "].";
    return ss.str();
}

// input dist 之前，调用 RecordTimestamp. 后面淘汰时，要判断key是否在当前卡， 当前只能记录到当前卡上原始batch中的key
// timestamp
void EmbcacheManager::RecordTimestamp(const at::Tensor& batchKeys, const std::vector<int64_t>& offsetPerKey,
                                      const at::Tensor& timestamps, const std::vector<int32_t>& tableIndices)
{
    LOG_INFO("Start invoke mgmt RecordTimestamp");
    TimeCost recordTimestampTC;
    const auto* keyPtr = batchKeys.data_ptr<int64_t>();
    const auto* timestampsPtr = timestamps.data_ptr<int64_t>();

    TORCH_CHECK(keyPtr != nullptr, "keyPtr should not be nullptr");
    TORCH_CHECK(timestampsPtr != nullptr, "timestampsPtr should not be nullptr");
    const std::vector<int32_t>& curTableIndices = tableIndices.empty() ? embTableIndies_ : tableIndices;
    TORCH_CHECK(curTableIndices.size() == embTableIndies_.size(),
                "tableIndices size must be equal to embTableIndies_ size");
    auto newOffsetPerKey = GetNewOffsetPerKey(offsetPerKey, curTableIndices);
    TORCH_CHECK(curTableIndices.size() + 1 == newOffsetPerKey.size(),
                "tableIndices size+1 must be equal to offsetPerKey size");

    for (int64_t i = 0; i < curTableIndices.size(); ++i) {
        int32_t idx = curTableIndices[i];
        TORCH_CHECK(idx >= 0 && idx < embNum_, "table index {} is out of range [0, {})", idx, embNum_);

        if (embConfigs_[idx].admitAndEvictConfig.IsEvictEnabled()) {
            if (featureFilters_[idx]) {
                auto startIndex = newOffsetPerKey[i];
                auto endIndex = newOffsetPerKey[i + 1];
                TORCH_CHECK(startIndex >= 0, "startIndex should >= 0");
                TORCH_CHECK(endIndex >= startIndex, "endIndex should >= startIndex");
                TORCH_CHECK(endIndex <= batchKeys.numel(), "endIndex should <= batchKeys.numel()");
                TORCH_CHECK(endIndex <= timestamps.numel(), "endIndex should <= timestamps.numel()");
                featureFilters_[idx]->RecordTimestamp(keyPtr, startIndex, endIndex, timestampsPtr);
            }
        }
    }
    LOG_INFO("RecordTimestamp execution time: {} ms", recordTimestampTC.ElapsedMS());
}

void EmbcacheManager::EvictFeatures()
{
    LOG_INFO("Start invoke EvictFeatures method, ComputeSwapInfo execute times: {}", swapCount_);
    TimeCost evictFeaturesTC;
    size_t evictKeyCount = 0;
    for (int32_t i = 0; i < embNum_; ++i) {
        if (!embConfigs_[i].admitAndEvictConfig.IsEvictEnabled()) {
            LOG_INFO("The table: {} doesn't enable evict, skip feature evict.", embConfigs_[i].tableName);
            continue;
        }

        // 获取当前表要淘汰的keys
        if (!featureFilters_[i]) {
            continue;
        }

        const std::vector<int64_t>& evictFeatures = featureFilters_[i]->evictFeatureRecord_.GetEvictKeys();
        // 调用swapManager删除映射信息
        // 删除embeddingTables中的embedding待对应step的swap out emb update执行完成后触发
        swapManagers_[i].RemoveKeys(evictFeatures);
        featureFilters_[i]->evictFeatureRecord_.SetSwapCount(swapCount_);
        evictKeyCount += evictFeatures.size();
    }
    LOG_INFO("EvictFeatures execution time: {} ms, all table evictKeyCount: {}", evictFeaturesTC.ElapsedMS(),
             evictKeyCount);
}

void EmbcacheManager::RecordEmbeddingUpdateTimes()
{
    embUpdateCount_++;

    if (NeedEvictEmbeddingTable()) {
        RemoveEmbeddingTableInfo();
    }
}

AsyncTask<void> EmbcacheManager::EmbeddingUpdateAsync(const SwapInfo& swapInfo, const at::Tensor& swapoutEmbs,
                                                      const std::vector<at::Tensor>& swapoutOptims,
                                                      const std::vector<int32_t>& tableIndices)
{
    return AsyncTask<void>([this, swapoutKeys = swapInfo.swapoutKeys, swapoutEmbs, swapoutOptims, tableIndices]() {
        EmbeddingUpdate(swapoutKeys, swapoutEmbs, swapoutOptims, tableIndices);
    });
}
bool EmbcacheManager::NeedEvictEmbeddingTable()
{
    for (int32_t i = 0; i < embNum_; ++i) {
        // 开启淘汰
        if (!embConfigs_[i].admitAndEvictConfig.IsEvictEnabled()) {
            continue;
        }
        if (!featureFilters_[i]) {
            continue;
        }

        // 待删除embTable的keys非空且达到和GetSwapInfo相同的步数
        if (!featureFilters_[i]->evictFeatureRecord_.GetEvictKeys().empty() &&
            featureFilters_[i]->evictFeatureRecord_.CanRemoveFromEmbTable(embUpdateCount_)) {
            return true;
        }
    }
    return false;
}

void EmbcacheManager::RemoveEmbeddingTableInfo()
{
    LOG_INFO("Start invoke RemoveEmbeddingTableInfo, embUpdateCount_: {}", embUpdateCount_);
    TimeCost removeEmbeddingTableTC;
    for (int32_t i = 0; i < embNum_; ++i) {
        if (!featureFilters_[i]) {
            continue;
        }

        auto& keys = featureFilters_[i]->evictFeatureRecord_.GetEvictKeys();
        if (keys.empty()) {
            LOG_INFO("Feature keys list is empty, skip to remove embedding from table: {}", embConfigs_[i].tableName);
            continue;
        }

        embeddingTables_[i]->RemoveEmbedding(keys);
        LOG_TRACE("Remove table embedding info, tableName: {}, remove key size: {}, detail keys: {}",
                  embConfigs_[i].tableName, keys.size(), StringTools::ToString(keys));
        featureFilters_[i]->evictFeatureRecord_.ClearEvictInfo();
    }
    LOG_INFO("RemoveEmbeddingTableInfo execution time: {} ms", removeEmbeddingTableTC.ElapsedMS());
}

void EmbcacheManager::StatisticsKeyCount(const at::Tensor& batchKeys, const torch::Tensor& offset,
                                         const at::Tensor& batchKeyCounts, int64_t tableIndex)
{
    // 添加表索引边界检查和详细调试信息
    LOG_INFO("StatisticsKeyCount called with tableIndex: {}, embNum_: {}", tableIndex, embNum_);
    TORCH_CHECK(tableIndex >= 0 && tableIndex < embNum_,
                "table index {} is out of range [0, {}). embNum_={}, "
                "This error indicates that the tableIndex parameter passed from Python exceeds "
                "the number of tables configured in EmbcacheManager.",
                tableIndex, embNum_, embNum_);

    LOG_INFO("StatisticsKeyCount, tableName: {}, isAdmit: {}",
             embConfigs_[tableIndex].tableName, embConfigs_[tableIndex].admitAndEvictConfig.IsAdmitEnabled());

    // 只有开启了准入功能的表才需要记录key count统计信息
    if (!embConfigs_[tableIndex].admitAndEvictConfig.IsAdmitEnabled()) {
        LOG_INFO("Table {} does not have admit enabled, skipping StatisticsKeyCount", tableIndex);
        return;
    }
    TORCH_CHECK(offset.numel() > tableIndex + 1, "param error, tableIndex need be smaller than offset length,"
                " but got equal or greater than offset length.")

    bool isCountDataEmpty = batchKeyCounts.numel() == 0;
    if (!isCountDataEmpty) {
        TORCH_CHECK(batchKeys.numel() == batchKeyCounts.numel(),
                    "batchKeys length should equal with batchKeyCounts length when batchKeyCounts is not empty.")
    }
    auto* featureDataPtr = batchKeys.data_ptr<int64_t>();
    auto* countDataPtr = batchKeyCounts.data_ptr<int64_t>();
    auto* offsetDataPtr = offset.data_ptr<int64_t>();

    TORCH_CHECK(featureDataPtr != nullptr, "featureDataPtr should not be nullptr");
    TORCH_CHECK(offsetDataPtr != nullptr, "offsetDataPtr should not be nullptr");
    if (!isCountDataEmpty) {
        TORCH_CHECK(countDataPtr != nullptr, "countDataPtr should not be nullptr when counts data is not empty");
    }
    int64_t start = offsetDataPtr[tableIndex];
    int64_t end = offsetDataPtr[tableIndex + 1];
    TORCH_CHECK(start >= 0 && end >= start && end <= batchKeys.numel(),
                "param error, start and end should meet 0 <= start <= end <= batchKeys.numel(), "
                "but got start: {}, end: {}, batchKeys.numel(): {}.",
                start, end, batchKeys.numel());

    if (!featureFilters_[tableIndex]) {
        return;
    }

    featureFilters_[tableIndex]->StatisticsKeyCount(featureDataPtr, countDataPtr, start, end, isCountDataEmpty);
}

void EmbcacheManager::SaveFeatureAdmitAndEvictInfo(const std::shared_ptr<FileSystem>& fileSystemPtr,
                                                   int32_t tableIndex, const std::string& filePrefix,
                                                   const std::vector<int64_t>& saveKeys)
{
    TimeCost saveFeatureFilterDataTC;
    if (embConfigs_[tableIndex].admitAndEvictConfig.IsAdmitEnabled()) {
        SaveFeatureCount(fileSystemPtr, tableIndex, filePrefix, saveKeys);
    }
    if (embConfigs_[tableIndex].admitAndEvictConfig.IsEvictEnabled()) {
        SaveFeatureTimestamp(fileSystemPtr, tableIndex, filePrefix);
    }
    LOG_INFO("saveFeatureFilterDataTC(ms):", saveFeatureFilterDataTC.ElapsedMS());
}

void EmbcacheManager::SaveFeatureTimestamp(const std::shared_ptr<FileSystem>& fileSystemPtr,
                                           int32_t tableIndex, const std::string& filePrefix)
{
    // 时间戳数据 key数量和当前卡的key不一样；需要单独保存key数据
    auto& featureTimestampMap = featureFilters_[tableIndex]->GetFeatureTimestampMap();

    // write attribute
    std::string attributeFile = filePrefix + EVICT_STR_PATH + SLICE_ATTR_PATH;
    fileSystemPtr->CreateFileDir(attributeFile);
    std::vector<int64_t> attrVec = {sizeof(int64_t), static_cast<long>(featureTimestampMap.size())};
    WriteData(fileSystemPtr, attributeFile, reinterpret_cast<const char*>(attrVec.data()),
              attrVec.size() * sizeof(int64_t));

    // write evict record key
    std::string evictKeyFile = filePrefix + EVICT_STR_PATH + SLICE_EVICT_KEY_DATA_PATH;
    fileSystemPtr->CreateFileDir(evictKeyFile);
    std::vector<int64_t> evictRecordKeys;
    evictRecordKeys.reserve(ONE_TIME_IO_WRITE);
    // write evict record timestamp
    std::string evictTsFile = filePrefix + EVICT_STR_PATH + SLICE_EVICT_TS_DATA_PATH;
    fileSystemPtr->CreateFileDir(evictTsFile);
    std::vector<int64_t> evictRecordTs;
    evictRecordTs.reserve(ONE_TIME_IO_WRITE);

    size_t loopCount = 0;
    for (auto iter : featureTimestampMap) {
        evictRecordKeys.emplace_back(iter.first);
        evictRecordTs.emplace_back(static_cast<int64_t>(iter.second));
        if (loopCount > 0 && (loopCount % ONE_TIME_IO_WRITE == 0 || loopCount == (featureTimestampMap.size() - 1))) {
            WriteData(fileSystemPtr, evictKeyFile, reinterpret_cast<const char*>(evictRecordKeys.data()),
                      evictRecordKeys.size() * sizeof(int64_t));
            evictRecordKeys.clear();
            WriteData(fileSystemPtr, evictTsFile, reinterpret_cast<const char*>(evictRecordTs.data()),
                      evictRecordTs.size() * sizeof(int64_t));
            evictRecordTs.clear();
        }
        loopCount++;
    }
}

void EmbcacheManager::SaveFeatureCount(const std::shared_ptr<FileSystem>& fileSystemPtr,
                                       int32_t tableIndex, const std::string& filePrefix,
                                       const std::vector<int64_t>& saveKeys)
{
    // write attribute
    std::string attributeFile = filePrefix + ADMIT_STR_PATH + SLICE_ATTR_PATH;
    fileSystemPtr->CreateFileDir(attributeFile);
    std::vector<int64_t> attrVec = {sizeof(int64_t), static_cast<long>(saveKeys.size())};
    WriteData(fileSystemPtr, attributeFile, reinterpret_cast<const char*>(attrVec.data()),
              attrVec.size() * sizeof(int64_t));

    // write key count data.
    std::string dataFile = filePrefix + ADMIT_STR_PATH + SLICE_DATA_PATH;
    fileSystemPtr->CreateFileDir(dataFile);
    const auto& featureCountMap = featureFilters_[tableIndex]->GetFeatureCountMap();
    std::vector<int64_t> keyCountVec;
    keyCountVec.reserve(ONE_TIME_IO_WRITE);
    size_t count = 0;
    for (size_t i = 0; i < saveKeys.size(); ++i) {
        auto key = saveKeys[i];
        auto ret = featureCountMap.find(key);
        if (ret != featureCountMap.end()) {
            keyCountVec.emplace_back(ret->second.count);
        } else {
            keyCountVec.emplace_back(-1);
        }
        LOG_TRACE("In save key count, tableIndex:{}, key:{}, count:{}.", tableIndex, key, keyCountVec[i]);

        if (i > 0 && (i % ONE_TIME_IO_WRITE == 0 || i == (saveKeys.size() - 1))) {
            WriteData(fileSystemPtr, dataFile, reinterpret_cast<const char*>(keyCountVec.data()),
                      keyCountVec.size() * sizeof(int64_t));
            count += keyCountVec.size();
            keyCountVec.clear();
        }
    }
    LOG_INFO("In save key count, tableIndex:{}, save key count size:{}, featureCountMap size:{}.",
             tableIndex, count, featureCountMap.size());
}

void EmbcacheManager::LoadFeatureAdmitAndEvictInfo(const std::shared_ptr<FileSystem>& fileSystemPtr,
                                                   int32_t tableIndex, const std::string& filePrefix,
                                                   const std::vector<int64_t>& saveKeys)
{
    TimeCost loadFeatureFilterDataTC;
    if (embConfigs_[tableIndex].admitAndEvictConfig.IsAdmitEnabled()) {
        // read key count data
        std::vector<uint64_t> keyCountVec;
        std::string keyAttrFile = filePrefix + ADMIT_STR_PATH + SLICE_ATTR_PATH;
        std::string keysDataFile = filePrefix + ADMIT_STR_PATH + SLICE_DATA_PATH;
        ReadKeysData(fileSystemPtr, keyCountVec, keyAttrFile, keysDataFile);
        featureFilters_[tableIndex]->LoadFeatureRecords(saveKeys, keyCountVec);
    }
    if (embConfigs_[tableIndex].admitAndEvictConfig.IsEvictEnabled()) {
        // 时间戳数据 key数量和当前卡的key不一样；需要分别读取 evict key， evict timestamp 信息
        // read key
        std::vector<int64_t> keysVec;
        std::string keyAttrFile = filePrefix + EVICT_STR_PATH + SLICE_ATTR_PATH;
        std::string keysDataFile = filePrefix + EVICT_STR_PATH + SLICE_EVICT_KEY_DATA_PATH;
        ReadKeysData(fileSystemPtr, keysVec, keyAttrFile, keysDataFile);

        // read timestamp
        std::vector<int64_t> keyTimestampVec;
        std::string evictTsDataFile = filePrefix + EVICT_STR_PATH + SLICE_EVICT_TS_DATA_PATH;
        ReadKeysData(fileSystemPtr, keyTimestampVec, keyAttrFile, evictTsDataFile);

        // data load
        featureFilters_[tableIndex]->LoadTimestampRecords(keysVec, keyTimestampVec);
    }
    LOG_INFO("The loadFeatureFilterDataTC(ms):{}.", loadFeatureFilterDataTC.ElapsedMS());
}
