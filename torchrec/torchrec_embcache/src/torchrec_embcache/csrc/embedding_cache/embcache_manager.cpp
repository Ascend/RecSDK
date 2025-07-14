/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "embcache_manager.h"

#include <c10/util/Exception.h>
#include <exception>
#include <fstream>
#include <filesystem>
#include <glog/logging.h>
#include <sstream>
#include <vector>

#include "glogger.h"

#include "utils/singleton.h"
#include "utils/time_cost.h"

using namespace Embcache;

EmbcacheManager::EmbcacheManager(const std::vector<EmbConfig>& embConfigs, bool needAccumulateOffset)
    : embNum(embConfigs.size()), needAccumulateOffset(needAccumulateOffset)
{
    for (const auto& config : embConfigs) {
        auto length = config.tableName.size();
        if (config.tableName.size() > TABLE_NAME_LENGTH) {
            LOG(ERROR) << "The length of table name:" << config.tableName << " is grater than max length "
                       << TABLE_NAME_LENGTH;
            throw std::runtime_error("the length of table name is invalid.");
        }
    }
    this->embConfigs = embConfigs;
    auto length = embConfigs[0].tableName.size();
    Singleton<Glogger>::GetInstance()->Init();
    enableFastHashMap = EnableFastHashMap();

    for (int32_t i = 0; i < embNum; i++) {
        LOG(INFO) << "The tableName:" << embConfigs[i].tableName << ", table index:" << i
                  << ", cacheSize is:" << embConfigs[i].cacheSize;
        embTableIndexMap_[embConfigs[i].tableName] = i;
        int64_t memStartOffset = embConfigs[i].admitAndEvictConfig.IsAdmitEnabled() ? 1 : 0;
        swapManagers.emplace_back(embConfigs[i].cacheSize, memStartOffset);

        if (enableFastHashMap) {
            embeddingTables.emplace_back(std::make_unique<EmbTableFastHashMap>(embConfigs[i]));
        } else {
            embeddingTables.emplace_back(std::make_unique<EmbTableUnorderedMap>(embConfigs[i]));
        }

        if (embConfigs[i].admitAndEvictConfig.IsFeatureFilterEnabled()) {
            auto& aaeConfig = embConfigs[i].admitAndEvictConfig;
            featureFilters.emplace_back(FeatureFilter(embConfigs[i].tableName, aaeConfig.admitThreshold,
                                                      aaeConfig.evictThreshold, aaeConfig.evictStepInterval));
        }
    }
    TORCH_CHECK(embConfigs.size() > 0, "ERROR, Size of embConfigs must > 0")
    optimNum = embConfigs[0].optimNum;
    for (auto& embedConfig : embConfigs) {
        TORCH_CHECK(embedConfig.optimNum == optimNum)
    }
}

int32_t EmbcacheManager::GetEmbTableIndex(const std::string& tableName)
{
    auto it = embTableIndexMap_.find(tableName);
    TORCH_CHECK(it != embTableIndexMap_.end(), "The table name:" + tableName + " is not found");
    return it->second;
}

bool EmbcacheManager::EnableFastHashMap()
{
    char* enableFastHashMapStr = getenv("ENABLE_FAST_HASHMAP");
    if (!enableFastHashMapStr) {
        LOG(WARNING) << "The env ENABLE_FAST_HASHMAP is not detected, std::unordered_map is used";
        return false;
    }

    std::string switchStr = std::string(enableFastHashMapStr);
    std::transform(switchStr.begin(), switchStr.end(), switchStr.begin(), ::tolower);
    if (switchStr == "true" || switchStr == "yes" || switchStr == "1") {
        LOG(WARNING) << "ENABLE_FAST_HASHMAP=true, FastHashMap is used";
        return true;
    }
    LOG(WARNING) << "The ENABLE_FAST_HASHMAP=false, std::unordered_map is used";
    return false;
}

SwapInfo EmbcacheManager::ComputeSwapInfo(const at::Tensor& batchKeys, const std::vector<int64_t>& offsetPerKey,
                                          const std::vector<std::string>& tableNames = {})
{
    TimeCost getSwapInfoTC;

    TORCH_CHECK(batchKeys.is_contiguous(), "batchKeys must be contiguous")
    TORCH_CHECK(batchKeys.dtype() == torch::kInt64, "batchKeys must be of type int64_t")
    std::vector<std::string>& curTableNames = tableNames.empty() ? embTableIndexMap_.keys() : tableNames;
    TORCH_CHECK(curTableNames.size() == offsetPerKey.size(), "tableNames size must be equal to offsetPerKey size");

    auto* keyPtr = batchKeys.data_ptr<int64_t>();
    int64_t keyNum = batchKeys.numel();
    int64_t offPreSum = 0;

    std::vector<int64_t> swapoutOffs;
    std::vector<int64_t> swapinOffs;
    std::vector<int64_t> batchOffs;
    SwapInfo swapInfo;
    for (int64_t i = 0; i < curTableNames.size(); i++) {
        int64_t idx = GetEmbTableIndex(curTableNames[i]);
        if (embConfigs[idx].admitAndEvictConfig.IsAdmitEnabled()) {
            featureFilters[idx].CountFilter(keyPtr, offsetPerKey[i], offsetPerKey[i + 1]);
        }

        // 取出每个表的 key
        std::vector<int64_t> batchKeysVec(keyPtr + offsetPerKey[i], keyPtr + offsetPerKey[i + 1]);
        auto tp = swapManagers[idx].ComputeSwapInfo(batchKeysVec);

        std::vector<int64_t>& swapoutKeysi = std::get<SWAP_INFO_TUPLE_INDEX0>(tp);
        std::vector<int64_t>& swapoutOffsi = std::get<SWAP_INFO_TUPLE_INDEX1>(tp);
        std::vector<int64_t>& swapinKeysi = std::get<SWAP_INFO_TUPLE_INDEX2>(tp);
        std::vector<int64_t>& swapinOffsi = std::get<SWAP_INFO_TUPLE_INDEX3>(tp);
        std::vector<int64_t>& batchOffsi = std::get<SWAP_INFO_TUPLE_INDEX4>(tp);
        if (needAccumulateOffset) {
            // 加上表外偏移
            for (auto& off : swapoutOffsi) {
                off += offPreSum;
            }
            for (auto& off : swapinOffsi) {
                off += offPreSum;
            }
            offPreSum += embConfigs[idx].cacheSize;
        }

        swapInfo.swapoutKeys.emplace_back(std::move(swapoutKeysi));
        swapInfo.swapinKeys.emplace_back(std::move(swapinKeysi));
        swapoutOffs.insert(swapoutOffs.end(), swapoutOffsi.begin(), swapoutOffsi.end());
        swapinOffs.insert(swapinOffs.end(), swapinOffsi.begin(), swapinOffsi.end());
        batchOffs.insert(batchOffs.end(), batchOffsi.begin(), batchOffsi.end());
    }

    auto longPinnedOpt = at::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true);
    swapInfo.swapoutOffs = at::empty({static_cast<int64_t>(swapoutOffs.size())}, longPinnedOpt);
    std::memcpy(swapInfo.swapoutOffs.data_ptr<int64_t>(), swapoutOffs.data(), sizeof(int64_t) * swapoutOffs.size());

    swapInfo.swapinOffs = at::empty({static_cast<int64_t>(swapinOffs.size())}, longPinnedOpt);
    std::memcpy(swapInfo.swapinOffs.data_ptr<int64_t>(), swapinOffs.data(), sizeof(int64_t) * swapinOffs.size());

    swapInfo.batchOffs = at::empty({static_cast<int64_t>(batchOffs.size())}, longPinnedOpt);
    std::memcpy(swapInfo.batchOffs.data_ptr<int64_t>(), batchOffs.data(), sizeof(int64_t) * batchOffs.size());

    swapCount++;

    LOG(INFO) << "The getSwapInfoTC(ms):" << getSwapInfoTC.ElapsedMS();

    return swapInfo;
}

AsyncTask<SwapInfo> EmbcacheManager::ComputeSwapInfoAsync(const at::Tensor& batchKeys,
                                                          const std::vector<int64_t>& offsetPerKey)
{
    return AsyncTask<SwapInfo>([this, batchKeys, offsetPerKey]() { return ComputeSwapInfo(batchKeys, offsetPerKey); });
}

SwapinTensor EmbcacheManager::EmbeddingLookup(const std::vector<std::vector<int64_t>>& swapinKeys,
                                              const std::vector<std::string>& tableNames = {})
{
    TimeCost embeddingLookupTC;

    auto floatPinnedOpt = at::TensorOptions().dtype(at::kFloat).device(at::kCPU).pinned_memory(true);
    auto longPinnedOpt = at::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true);
    SwapinTensor swapinTensor;
    swapinTensor.jaggedOffs = at::empty({static_cast<int64_t>(swapinKeys.size() + 1)}, longPinnedOpt);
    auto jaggedOffsPtr = swapinTensor.jaggedOffs.data_ptr<int64_t>();

    jaggedOffsPtr[0] = 0;
    for (uint64_t i = 1; i <= swapinKeys.size(); i++) {
        jaggedOffsPtr[i] = jaggedOffsPtr[i - 1] + swapinKeys[i - 1].size() * embConfigs[i - 1].embDim;
    }

    int64_t embsSize = jaggedOffsPtr[swapinKeys.size()];
    swapinTensor.swapinEmbs = at::empty({embsSize}, floatPinnedOpt);
    for (int32_t i = 0; i < optimNum; i++) {
        swapinTensor.swapinOptims.emplace_back(at::empty({embsSize}, floatPinnedOpt));
    }

    std::vector<std::string>& curTableNames = tableNames.empty() ? embTableIndexMap_.keys() : tableNames;
    TORCH_CHECK(curTableNames.size() == swapinKeys.size(), "tableNames size must be equal to swapinKeys size");

    std::vector<float*> swapinOptimsPtr(optimNum);
    for (uint64_t i = 0; i < swapinKeys.size(); i++) {
        for (int32_t j = 0; j < optimNum; j++) {
            swapinOptimsPtr[j] = swapinTensor.swapinOptims[j].data_ptr<float>() + jaggedOffsPtr[i];
        }

        int32_t idx = GetEmbTableIndex(curTableNames[i]);
        embeddingTables[idx]->FindOrInsert(swapinKeys[i], swapinTensor.swapinEmbs.data_ptr<float>() + jaggedOffsPtr[i],
                                           swapinOptimsPtr);
    }

    LOG(INFO) << "The embeddingLookupTC(ms):" << embeddingLookupTC.ElapsedMS();
    return swapinTensor;
}

AsyncTask<SwapinTensor> EmbcacheManager::EmbeddingLookupAsync(const SwapInfo& swapInfo)
{
    return AsyncTask<SwapinTensor>([this, swapinKeys = swapInfo.swapinKeys]() { return EmbeddingLookup(swapinKeys); });
}

void EmbcacheManager::EmbeddingUpdate(const std::vector<std::vector<int64_t>>& swapoutKeys,
                                      const at::Tensor& swapoutEmbs, const std::vector<at::Tensor>& swapoutOptims,
                                      const std::vector<std::string>& tableNames = {})
{
    TimeCost embeddingUpdateTC;
    for (auto& embedConfig : embConfigs) {
        TORCH_CHECK(embedConfig.optimNum == (int32_t)swapoutOptims.size())
    }
    for (auto& optimition : swapoutOptims) {
        TORCH_CHECK(swapoutEmbs.numel() == optimition.numel())
        TORCH_CHECK(optimition.dtype() == torch::kFloat32)
    }
    TORCH_CHECK(swapoutEmbs.dtype() == torch::kFloat32)

    std::vector<std::string>& curTableNames = tableNames.empty() ? embTableIndexMap_.keys() : tableNames;
    TORCH_CHECK(curTableNames.size() == swapoutKeys.size(), "tableNames size must be equal to swapoutKeys size");

    auto* swapoutEmbsPtr = swapoutEmbs.data_ptr<float>();
    int64_t jaggedOff = 0;
    std::vector<float*> swapoutOptimPtrs(swapoutOptims.size());
    for (uint64_t i = 0; i < swapoutKeys.size(); i++) {
        for (size_t j = 0; j < swapoutOptims.size(); j++) {
            swapoutOptimPtrs[j] = swapoutOptims[j].data_ptr<float>() + jaggedOff;
        }

        int32_t idx = GetEmbTableIndex(curTableNames[i]);
        embeddingTables[idx]->InsertOrAssign(swapoutKeys[i], swapoutEmbsPtr + jaggedOff, swapoutOptimPtrs);
        jaggedOff += swapoutKeys[i].size() * embConfigs[idx].embDim;
    }

    LOG(INFO) << "The embeddingUpdateTC(ms):" << embeddingUpdateTC.ElapsedMS();
    embUpdateCount++;

    if (NeedEvictEmbeddingTable()) {
        RemoveEmbeddingTableInfo();
    }
}

// input dist 之前，调用 RecordTimestamp. 后面淘汰时，要判断key是否在当前卡， 当前只能记录到当前卡上原始batch中的key
// timestamp
void EmbcacheManager::RecordTimestamp(const at::Tensor& batchKeys, const std::vector<int64_t>& offsetPerKey,
                                      const at::Tensor& timestamps, const std::vector<std::string>& tableNames = {})
{
    LOG(INFO) << "Start invoke mgmt RecordTimestamp";
    TimeCost recordTimestampTC;
    const auto* keyPtr = batchKeys.data_ptr<int64_t>();
    const auto* timestampsPtr = timestamps.data_ptr<int64_t>();
    std::vector<std::string>& curTableNames = tableNames.empty() ? embTableIndexMap_.keys() : tableNames;
    TORCH_CHECK(curTableNames.size() == offsetPerKey.size(), "tableNames size must be equal to offsetPerKey size");

    for (int64_t i = 0; i < embNum; ++i) {
        int32_t idx = GetEmbTableIndex(curTableNames[i]);
        if (embConfigs[idx].admitAndEvictConfig.IsEvictEnabled()) {
            featureFilters[idx].RecordTimestamp(keyPtr, offsetPerKey[i], offsetPerKey[i + 1], timestampsPtr);
        }
    }
    LOG(INFO) << "The recordTimestampTC(ms):" << recordTimestampTC.ElapsedMS();
}

void EmbcacheManager::EvictFeatures()
{
    LOG(INFO) << "Start invoke EvictFeatures method, ComputeSwapInfo execute times:" << swapCount;
    TimeCost evictFeaturesTC;
    size_t evictKeyCount = 0;
    for (int32_t i = 0; i < embNum; ++i) {
        if (!embConfigs[i].admitAndEvictConfig.IsEvictEnabled()) {
            LOG(INFO) << "The table：" << embConfigs[i].tableName << " doesn't enable evict, skip feature evict.";
            continue;
        }

        // 获取当前表要淘汰的keys
        const std::vector<int64_t>& evictFeatures = featureFilters[i].evictFeatureRecord.GetEvictKeys();
        // 调用swapManager删除映射信息
        // 删除embeddingTables中的embedding待对应step的swap out emb update执行完成后触发
        swapManagers[i].RemoveKeys(evictFeatures);
        featureFilters[i].evictFeatureRecord.SetSwapCount(swapCount);
        evictKeyCount += evictFeatures.size();
    }
    LOG(INFO) << "The evictFeaturesTC(ms):" << evictFeaturesTC.ElapsedMS()
              << ", all table evictKeyCount:" << evictKeyCount;
}

void EmbcacheManager::RecordEmbeddingUpdateTimes()
{
    embUpdateCount++;

    if (NeedEvictEmbeddingTable()) {
        RemoveEmbeddingTableInfo();
    }
}

AsyncTask<void> EmbcacheManager::EmbeddingUpdateAsync(const SwapInfo& swapInfo, const at::Tensor& swapoutEmbs,
                                                      const std::vector<at::Tensor>& swapoutOptims)
{
    return AsyncTask<void>([this, swapoutKeys = swapInfo.swapoutKeys, swapoutEmbs, swapoutOptims]() {
        EmbeddingUpdate(swapoutKeys, swapoutEmbs, swapoutOptims);
    });
}

void EmbcacheManager::Embedding2Host(const at::Tensor& weightsDev, const std::vector<at::Tensor>& momentumDevs)
{
    for (auto& momentumDev : momentumDevs) {
        TORCH_CHECK(weightsDev.numel() == momentumDev.numel())
        TORCH_CHECK(momentumDev.dtype() == torch::kFloat32)
    }
    TORCH_CHECK(weightsDev.dtype() == torch::kFloat32)

    auto* weightsDevPtr = weightsDev.data_ptr<float>();
    int64_t jaggedOff = 0;
    std::vector<int64_t> keys;

    std::string weightShapeStr = GetDevWeightsShape(weightsDev);
    LOG(INFO) << "In Embedding2Host, weightsDev shape:" << weightShapeStr;

    for (int32_t embIndex = 0; embIndex < embNum; embIndex++) {
        keys.clear();
        // cache中可能预留了offset 0位置，因此拷贝回host时，需先加上偏移
        auto start = swapManagers[embIndex].GetMemStartOffset();
        int64_t currentTableOffset = jaggedOff + start * embConfigs[embIndex].embDim;
        auto end = swapManagers[embIndex].GetOccupiedNum();
        for (int64_t off = start; off < end; off++) {
            keys.emplace_back(swapManagers[embIndex].GetKey(off));
        }
        LOG(INFO) << "keys.size: " << keys.size();
        std::vector<float*> momentum1DevPtrs(momentumDevs.size());
        for (size_t i = 0; i < momentumDevs.size(); i++) {
            momentum1DevPtrs[i] = momentumDevs[i].data_ptr<float>() + currentTableOffset;
        }
        embeddingTables[embIndex]->InsertOrAssign(keys, weightsDevPtr + currentTableOffset, momentum1DevPtrs);

        // Here, GetOccupiedNum is less than embConfigs[embIndex].cacheSize = weightsDev.shape[0],
        // and we need to skip the unnecessary weight indices.
        jaggedOff += embConfigs[embIndex].cacheSize * embConfigs[embIndex].embDim;
        LOG(INFO) << "Embedding2Host, embIndex:" << embIndex << ", update key size:" << keys.size()
                  << ", jaggedOff:" << jaggedOff << ", currentTableOffset:" << currentTableOffset;
    }
}

std::string EmbcacheManager::GetDevWeightsShape(const at::Tensor& weightsDev) const
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

void EmbcacheManager::Save(const std::string path, const int rank)
{
    for (int32_t i = 0; i < embNum; i++) {
        std::string tableName = embConfigs[i].tableName;

        std::string midPath = path + "/" + tableName + RANK_STR_PATH + std::to_string(rank);

        std::ofstream fileEmbeddingSliceAttr = OpenFile(midPath + EMBEDDING_STR_PATH + SLICE_ATTR_PATH);
        std::ofstream fileEmbeddingSliceData = OpenFile(midPath + EMBEDDING_STR_PATH + SLICE_DATA_PATH);
        std::ofstream fileKeySliceAttr = OpenFile(midPath + KEY_STR_PATH + SLICE_ATTR_PATH);
        std::ofstream fileKeySliceData = OpenFile(midPath + KEY_STR_PATH + SLICE_DATA_PATH);
        std::ofstream fileMomentum1SliceAttr = OpenFile(midPath + MOMENTUM1_STR_PATH + SLICE_ATTR_PATH);
        std::ofstream fileMomentum1SliceData = OpenFile(midPath + MOMENTUM1_STR_PATH + SLICE_DATA_PATH);
        std::ofstream fileMomentum2SliceAttr = OpenFile(midPath + MOMENTUM2_STR_PATH + SLICE_ATTR_PATH);
        std::ofstream fileMomentum2SliceData = OpenFile(midPath + MOMENTUM2_STR_PATH + SLICE_DATA_PATH);

        size_t count = 0;
        std::vector<int64_t> saveKeys;
        LOG(INFO) << "Start save table:" << tableName;
        embeddingTables[i]->ForEachKey([&](const int64_t key, const float* value) {
            ++count;
            // 1. write key
            WriteData(fileKeySliceData, reinterpret_cast<const char*>(&key), 1 * sizeof(int64_t));
            if (embConfigs[i].admitAndEvictConfig.IsAdmitEnabled()) {
                saveKeys.emplace_back(key);
            }
            // 2. write embedding
            WriteData(fileEmbeddingSliceData, reinterpret_cast<const char*>(value),
                      embConfigs[i].embDim * sizeof(float));
            LOG(INFO) << "In save, table:" << i << ", key:" << key << ", embedding.dim:" << embConfigs[i].embDim
                      << ", embedding " << StringTools::ToString(value, embConfigs[i].embDim);

            // 3. write momentum
            if (optimNum > 0) {
                WriteData(fileMomentum1SliceData, reinterpret_cast<const char*>(value + embConfigs[i].embDim),
                          embConfigs[i].embDim * sizeof(float));
                LOG(INFO) << "In save, table:" << i << ", key:" << key << ", momentum1.dim " << embConfigs[i].embDim
                          << " momentum1 "
                          << StringTools::ToString(value + 1 * embConfigs[i].embDim, embConfigs[i].embDim);
            }
            if (optimNum > 1) {
                WriteData(fileMomentum2SliceData, reinterpret_cast<const char*>(value + 2 * embConfigs[i].embDim),
                          embConfigs[i].embDim * sizeof(float));
                LOG(INFO) << "In save, table:" << i << ", key:" << key << ", momentum2.dim " << embConfigs[i].embDim
                          << " momentum2 "
                          << StringTools::ToString(value + OPTIMIZER_SLOT_INDEX2 * embConfigs[i].embDim,
                                                   embConfigs[i].embDim);
            }
        });
        LOG(INFO) << "The tableName: " << tableName << " shape: " << count << ", " << embConfigs[i].embDim;
        std::vector<int64_t> keyAttribute = {sizeof(int64_t), count};
        WriteData(fileKeySliceAttr, reinterpret_cast<const char*>(keyAttribute.data()),
                  keyAttribute.size() * sizeof(int64_t));
        std::vector<int64_t> embedAttribute = {sizeof(int64_t), count, embConfigs[i].embDim};
        WriteData(fileEmbeddingSliceAttr, reinterpret_cast<const char*>(embedAttribute.data()),
                  embedAttribute.size() * sizeof(int64_t));
        WriteOptimizerAttributeFile(i, fileMomentum1SliceAttr, fileMomentum2SliceAttr, count);

        // 4 保存准入淘汰数据
        SaveFeatureAdmitAndEvictInfo(i, midPath, saveKeys);
    }
}

void EmbcacheManager::WriteOptimizerAttributeFile(int32_t i, std::ofstream& fileMomentum1SliceAttr,
                                                  std::ofstream& fileMomentum2SliceAttr, size_t count)
{
    std::vector<int64_t> momentum1Attribute = {sizeof(int64_t), count, embConfigs[i].embDim};

    WriteData(fileMomentum1SliceAttr, reinterpret_cast<const char*>(momentum1Attribute.data()),
              momentum1Attribute.size() * sizeof(int64_t));

    // 目前momentum2Attribute和momentum1Attribute是一致的
    WriteData(fileMomentum2SliceAttr, reinterpret_cast<const char*>(momentum1Attribute.data()),
              momentum1Attribute.size() * sizeof(int64_t));
}

std::ofstream EmbcacheManager::OpenFile(std::string path)
{
    std::filesystem::path filepath(path);
    std::filesystem::path dir = filepath.parent_path();
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOG(ERROR) << path << " can not open";
        TORCH_CHECK(-1, path + "can not open")
    }

    file.exceptions(std::ios::failbit | std::ios::badbit);
    LOG(INFO) << "Open file: " << path;
    return file;
}

void EmbcacheManager::WriteData(std::ofstream& file, const char* dataPtr, size_t bytes)
{
    try {
        file.write(dataPtr, bytes);
    } catch (const std::ios_base::failure& e) {
        LOG(ERROR) << e.what();
        throw std::runtime_error(e.what());
    }
    if (!file.good()) {
        LOG(ERROR) << "File status is abnormal after write data, maybe write bytes not meet the expectation.";
        TORCH_CHECK(-1, "File status is abnormal after write data, maybe write bytes not meet the expectation.")
    }
}

void EmbcacheManager::Load(const std::string& path, int rank)
{
    for (int32_t i = 0; i < embNum; i++) {
        std::string tableName = embConfigs[i].tableName;
        LOG(INFO) << "Start load, rank:" << rank << ", tableName:" << tableName;
        std::vector<int64_t> keys;
        std::string filePath = path + "/" + tableName + "/" + "rank" + std::to_string(rank);
        EmbcacheManager::ReadFile(filePath, keys, "key");

        std::vector<std::vector<float>> embeddings;
        int32_t embDim = embConfigs[i].embDim;
        EmbcacheManager::ReadFile(filePath, embeddings, "embedding", embDim);
        LOG(INFO) << "In load, rank:" << rank << ", tableName:" << tableName << ", keys size:" << keys.size()
                  << ", embeddings size:" << embeddings.size();
        TORCH_CHECK(keys.size() == embeddings.size(), "In load scene, keys size:", keys.size(),
                    " is not equal with embedding size:", embeddings.size())

        std::vector<std::vector<float>> momentum1;
        if (optimNum > 0) {
            int retCode = EmbcacheManager::ReadFile(filePath, momentum1, "momentum1", embDim);
            TORCH_CHECK(retCode == 0, "Failed to read optimizer momentum1 file data.")
        }

        std::vector<std::vector<float>> momentum2;
        if (optimNum > 1) {
            int retCode = EmbcacheManager::ReadFile(filePath, momentum2, "momentum2", embDim);
            TORCH_CHECK(retCode == 0, "Failed to read optimizer momentum2 file data.")
        }

        std::vector<float> emptyList = {};
        for (size_t j = 0; j < keys.size(); ++j) {
            std::vector<float> m1 = momentum1.empty() ? emptyList : momentum1[j];
            std::vector<float> m2 = momentum2.empty() ? emptyList : momentum2[j];
            LOG(INFO) << "In load, rank:" << rank << ", tableName:" << tableName << ", key:" << keys[j]
                      << ", embedding:" << StringTools::ToString(embeddings[j])
                      << ", momentum1:" << StringTools::ToString(m1) << ", momentum2:" << StringTools::ToString(m2);
        }

        for (size_t k = 0; k < keys.size(); k++) {
            std::vector<int64_t> insertKey = {keys[k]};
            std::vector<float*> momentum = {};
            if (optimNum > 0) {
                momentum.emplace_back(momentum1[k].data());
            }
            if (optimNum > 1) {
                momentum.emplace_back(momentum2[k].data());
            }
            embeddingTables[i]->InsertOrAssign(insertKey, embeddings[k].data(), momentum);
        }

        // 加载准入淘汰数据
        LoadFeatureAdmitAndEvictInfo(i, filePath, keys);
    }
}

template <class T>
int32_t EmbcacheManager::ReadFile(const std::string& filePath, std::vector<T>& dataOutputs,
                                  const std::string& loadItemName, const std::string& detailFileName)
{
    std::stringstream ss;
    ss << filePath << "/" << loadItemName << detailFileName;

    std::ifstream file(ss.str(), std::ios::binary);
    if (!file.is_open()) {
        LOG(ERROR) << "Open file: " << ss.str() << " failed.";
        return -1;
    }

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize == 0) {
        LOG(INFO) << "Read file: " << ss.str() << " size: 0.";
        file.close();
        return 0;
    }

    const auto elementCount = static_cast<size_t>(fileSize / sizeof(T));
    if (elementCount * sizeof(T) != fileSize) {
        LOG(ERROR) << "File size " << fileSize << " does not match data type.";
        file.close();
        return -1;
    }

    dataOutputs.reserve(elementCount);

    std::vector<char> buffer(READ_AND_WRITE_SIZE_PEER_TIME);
    while (file.read(buffer.data(), READ_AND_WRITE_SIZE_PEER_TIME)) {
        size_t readBytes = file.gcount();
        size_t readElements = readBytes / sizeof(T);
        T* elements = reinterpret_cast<T*>(buffer.data());
        dataOutputs.insert(dataOutputs.end(), elements, elements + readElements);
    }

    size_t remainingBytes = file.gcount();
    if (remainingBytes > 0) {
        size_t readElements = remainingBytes / sizeof(T);
        T* elements = reinterpret_cast<T*>(buffer.data());
        dataOutputs.insert(dataOutputs.end(), elements, elements + readElements);
    }

    file.close();
    return 0;
}

int32_t EmbcacheManager::ReadFile(const std::string& filePath, std::vector<std::vector<float>>& embedding,
                                  const std::string& loadItemName, int32_t embDim)
{
    std::stringstream ss;
    ss << filePath << "/" << loadItemName << "/slice.data";
    std::ifstream file(ss.str(), std::ios::binary);
    if (!file.is_open()) {
        LOG(ERROR) << "Open file: " << ss.str() << " failed.";
        return -1;
    }

    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    if (fileSize == 0) {
        LOG(INFO) << "Read file: " << ss.str() << " size: 0.";
        file.close();
        return 0;
    }

    const auto totalElements = static_cast<size_t>(fileSize / sizeof(float));
    if (static_cast<std::streampos>(totalElements * sizeof(float)) != fileSize) {
        LOG(ERROR) << "File size " << fileSize << " does not match data type.";
        file.close();
        return -1;
    }
    TORCH_CHECK(embDim != 0, "table embedding dim must be non zero value.")
    const uint32_t rows = totalElements / embDim;
    if (rows * embDim != totalElements) {
        LOG(ERROR) << "Invalid embedding dimension " << embDim << " for file size " << fileSize;
        file.close();
        return -1;
    }

    embedding.resize(rows, std::vector<float>(embDim));
    try {
        for (size_t i = 0; i < rows; ++i) {
            file.read(reinterpret_cast<char*>(embedding[i].data()), embDim * sizeof(float));
            if (file.gcount() != embDim * sizeof(float)) {
                LOG(ERROR) << "Failed to read enough bytes.";
                file.close();
                return -1;
            }
        }
    } catch (const std::ios_base::failure& e) {
        LOG(ERROR) << "File read error: " << e.what();
        file.close();
        return -1;
    }
    file.close();
    return 0;
}

bool EmbcacheManager::NeedEvictEmbeddingTable()
{
    for (int32_t i = 0; i < embNum; ++i) {
        // 开启淘汰
        if (!embConfigs[i].admitAndEvictConfig.IsEvictEnabled()) {
            continue;
        }
        // 待删除embTable的keys非空且达到和GetSwapInfo相同的步数
        if (!featureFilters[i].evictFeatureRecord.GetEvictKeys().empty() &&
            featureFilters[i].evictFeatureRecord.CanRemoveFromEmbTable(embUpdateCount)) {
            return true;
        }
    }
    return false;
}

void EmbcacheManager::RemoveEmbeddingTableInfo()
{
    LOG(INFO) << "Start invoke RemoveEmbeddingTableInfo, embUpdateCount:" << embUpdateCount;
    TimeCost removeEmbeddingTableTC;
    for (int32_t i = 0; i < embNum; ++i) {
        auto& keys = featureFilters[i].evictFeatureRecord.GetEvictKeys();
        if (keys.empty()) {
            LOG(INFO) << "Feature keys list is empty, skip to remove embedding from table:" << embConfigs[i].tableName;
            continue;
        }

        // 调用embTable Remove
        embeddingTables[i]->RemoveEmbedding(keys);
        LOG(INFO) << "Remove table embedding info, table:" << embConfigs[i].tableName
                  << ", remove key size:" << keys.size() << ", detail keys:" << StringTools::ToString(keys);
        featureFilters[i].evictFeatureRecord.ClearEvictInfo();
    }
    LOG(INFO) << "The removeEmbeddingTableTC(ms):" << removeEmbeddingTableTC.ElapsedMS();
}

void EmbcacheManager::SaveFeatureAdmitAndEvictInfo(int32_t tableIndex, const std::string& filePrefix,
                                                   const std::vector<int64_t>& saveKeys)
{
    TimeCost saveFeatureFilterDataTC;
    if (embConfigs[tableIndex].admitAndEvictConfig.IsAdmitEnabled()) {
        SaveFeatureCount(tableIndex, filePrefix, saveKeys);
    }
    if (embConfigs[tableIndex].admitAndEvictConfig.IsEvictEnabled()) {
        SaveFeatureTimestamp(tableIndex, filePrefix);
    }
    LOG(INFO) << "saveFeatureFilterDataTC(ms):" << saveFeatureFilterDataTC.ElapsedMS();
}

void EmbcacheManager::SaveFeatureTimestamp(int32_t tableIndex, const std::string& filePrefix)
{
    // 时间戳数据 key数量和当前卡的key不一样；需要单独保存key数据
    auto& featureTimestampMap = featureFilters[tableIndex].GetFeatureTimestampMap();

    // write attribute
    std::ofstream attributeFile = OpenFile(filePrefix + EVICT_STR_PATH + SLICE_ATTR_PATH);
    std::vector<int64_t> attrVec = {sizeof(int64_t), featureTimestampMap.size()};
    WriteData(attributeFile, reinterpret_cast<const char*>(attrVec.data()), attrVec.size() * sizeof(int64_t));

    // write evict record key
    std::ofstream evictKeyFile = OpenFile(filePrefix + EVICT_STR_PATH + SLICE_EVICT_KEY_DATA_PATH);
    std::vector<int64_t> evictRecordKeys;
    // write evict record timestamp
    std::ofstream evictTsFile = OpenFile(filePrefix + EVICT_STR_PATH + SLICE_EVICT_TS_DATA_PATH);
    std::vector<int64_t> evictRecordTs;

    size_t loopCount = 0;
    for (auto iter : featureTimestampMap) {
        evictRecordKeys.emplace_back(iter.first);
        evictRecordTs.emplace_back(static_cast<int64_t>(iter.second));
        if (loopCount > 0 &&
            (loopCount % ONE_TIME_IO_WRITE == 0 || loopCount == (featureTimestampMap.size() - 1))) {
            WriteData(evictKeyFile, reinterpret_cast<const char*>(evictRecordKeys.data()),
                      evictRecordKeys.size() * sizeof(int64_t));
            evictRecordKeys.clear();
            WriteData(evictTsFile, reinterpret_cast<const char*>(evictRecordTs.data()),
                      evictRecordTs.size() * sizeof(int64_t));
            evictRecordTs.clear();
        }
        loopCount++;
    }
}

void EmbcacheManager::SaveFeatureCount(int32_t tableIndex, const std::string& filePrefix,
                                       const std::vector<int64_t>& saveKeys)
{
    // write attribute
    std::ofstream attributeFile = OpenFile(filePrefix + ADMIT_STR_PATH + SLICE_ATTR_PATH);
    std::vector<int64_t> attrVec = {sizeof(int64_t), saveKeys.size()};
    WriteData(attributeFile, reinterpret_cast<const char*>(attrVec.data()), attrVec.size() * sizeof(int64_t));

    // write key count data.
    std::ofstream dataFile = OpenFile(filePrefix + ADMIT_STR_PATH + SLICE_DATA_PATH);
    const auto& featureCountMap = featureFilters[tableIndex].GetFeatureCountMap();
    std::vector<int64_t> keyCountVec;
    size_t count = 0;
    for (size_t i = 0; i < saveKeys.size(); ++i) {
        auto key = saveKeys[i];
        auto ret = featureCountMap.find(key);
        if (ret != featureCountMap.end()) {
            keyCountVec.emplace_back(ret->second.count);
        } else {
            keyCountVec.emplace_back(-1);
        }
        LOG(INFO) << "In save key count, tableIndex:" << tableIndex << ", key:" << key
                  << ", count:" << keyCountVec[i];

        if (i > 0 && (i % ONE_TIME_IO_WRITE == 0 || i == (saveKeys.size() - 1))) {
            WriteData(dataFile, reinterpret_cast<const char*>(keyCountVec.data()),
                      keyCountVec.size() * sizeof(int64_t));
            count += keyCountVec.size();
            keyCountVec.clear();
        }
    }
    LOG(INFO) << "In save key count, tableIndex:" << tableIndex << ", save key count size:" << count
              << ", featureCountMap size:" << featureCountMap.size();
}

void EmbcacheManager::LoadFeatureAdmitAndEvictInfo(int32_t tableIndex, const std::string& filePrefix,
                                                   const std::vector<int64_t>& saveKeys)
{
    TimeCost loadFeatureFilterDataTC;
    if (embConfigs[tableIndex].admitAndEvictConfig.IsAdmitEnabled()) {
        // read key count data
        std::vector<uint64_t> keyCountVec;
        auto retCode = ReadFile(filePrefix, keyCountVec, ADMIT_STR_PATH);
        if (retCode != 0) {
            LOG(ERROR) << "Failed to read feature count file, error code:" + std::to_string(retCode);
            TORCH_CHECK(-1, "Failed to read feature count file, error code:" + std::to_string(retCode))
        }
        featureFilters[tableIndex].LoadFeatureRecords(saveKeys, keyCountVec);
    }
    if (embConfigs[tableIndex].admitAndEvictConfig.IsEvictEnabled()) {
        // 时间戳数据 key数量和当前卡的key不一样；需要分别读取 evict key， evict timestamp 信息
        // read key
        std::vector<int64_t> keysVec;
        auto retCode = ReadFile(filePrefix, keysVec, EVICT_STR_PATH, SLICE_EVICT_KEY_DATA_PATH);
        if (retCode != 0) {
            LOG(ERROR) << "Failed to read feature evict key file, error code:" + std::to_string(retCode);
            TORCH_CHECK(-1, "Failed to read feature evict key file, error code:" + std::to_string(retCode))
        }

        // read timestamp
        std::vector<int64_t> keyTimestampVec;
        retCode = ReadFile(filePrefix, keyTimestampVec, EVICT_STR_PATH, SLICE_EVICT_TS_DATA_PATH);
        if (retCode != 0) {
            LOG(ERROR) << "Failed to read feature evict timestamp file, error code:" + std::to_string(retCode);
            TORCH_CHECK(-1, "Failed to read feature evict timestamp file, error code:" + std::to_string(retCode))
        }

        // data load
        featureFilters[tableIndex].LoadTimestampRecords(keysVec, keyTimestampVec);
    }
    LOG(INFO) << "The loadFeatureFilterDataTC(ms):" << loadFeatureFilterDataTC.ElapsedMS();
}

void EmbcacheManager::StatisticsKeyCount(const at::Tensor& batchKeys, const torch::Tensor& offset,
                                         const at::Tensor& batchKeyCounts, int64_t tableIndex)
{
    LOG(INFO) << "StatisticsKeyCount, tableIndex:" << tableIndex
              << ", isAdmit:" << embConfigs[tableIndex].admitAndEvictConfig.IsAdmitEnabled();
    if (!embConfigs[tableIndex].admitAndEvictConfig.IsAdmitEnabled()) {
        return;
    }
    TORCH_CHECK(offset.numel() > tableIndex, "param error, tableIndex need be smaller than offset length,"
                                             " but got equal or greater than offset length.")
    // 未开启local unique时，counts为空tensor，处理时默认key对应count为1
    bool isCountDataEmpty = batchKeyCounts.numel() == 0;
    if (!isCountDataEmpty) {
        TORCH_CHECK(batchKeys.numel() == batchKeyCounts.numel(),
                    "batchKeys length should equal with batchKeyCounts length when batchKeyCounts is not empty.")
    }
    auto* featureDataPtr = batchKeys.data_ptr<int64_t>();
    auto* countDataPtr = batchKeyCounts.data_ptr<int64_t>();
    auto* offsetDataPtr = offset.data_ptr<int64_t>();
    int64_t start = offsetDataPtr[tableIndex];
    int64_t end = offsetDataPtr[tableIndex + 1];
    TORCH_CHECK(end <= batchKeys.numel())
    featureFilters[tableIndex].StatisticsKeyCount(featureDataPtr, countDataPtr, start, end, isCountDataEmpty);
}
