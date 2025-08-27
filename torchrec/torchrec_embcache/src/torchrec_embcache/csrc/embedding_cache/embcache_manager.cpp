/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
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

#include "utils/logger.h"
#include "utils/time_cost.h"
#include "utils/string_tools.h"

using namespace Embcache;

EmbcacheManager::EmbcacheManager(const std::vector<EmbConfig>& embConfigs, bool needAccumulateOffset)
    : embNum_(embConfigs.size()), needAccumulateOffset_(needAccumulateOffset)
{
    for (const auto& config : embConfigs) {
        auto length = config.tableName.size();
        if (config.tableName.size() > TABLE_NAME_LENGTH) {
            LOG_ERROR("The length of table name: {} is greater than {}", config.tableName, TABLE_NAME_LENGTH);
            throw std::runtime_error("the length of table name is invalid.");
        }
    }
    this->embConfigs_ = embConfigs;
    auto length = embConfigs[0].tableName.size();
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

SwapInfo EmbcacheManager::ComputeSwapInfo(const at::Tensor& batchKeys, const std::vector<int64_t>& offsetPerKey,
                                          const std::vector<int32_t>& tableIndices)
{
    TimeCost getSwapInfoTC;

    TORCH_CHECK(batchKeys.is_contiguous(), "batchKeys must be contiguous")
    TORCH_CHECK(batchKeys.dtype() == torch::kInt64, "batchKeys must be of type int64_t")
    const std::vector<int32_t>& curTableIndices = tableIndices.empty() ? embTableIndies_ : tableIndices;
    TORCH_CHECK(curTableIndices.size() + 1 == offsetPerKey.size(),
                "tableIndices size+1 must be equal to offsetPerKey size");

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

        if (embConfigs_[idx].admitAndEvictConfig.IsAdmitEnabled()) {
            if (featureFilters_[idx]) {
                featureFilters_[idx]->CountFilter(keyPtr, offsetPerKey[i], offsetPerKey[i + 1]);
            }
        }

        // 取出每个表的 key
        TORCH_CHECK(offsetPerKey[i] <= keyNum, "offsetPerKey[{}] is greater than keyNum", i);
        TORCH_CHECK(offsetPerKey[i + 1] <= keyNum, "offsetPerKey[{}] is greater than keyNum", i + 1);
        std::vector<int64_t> batchKeysVec(keyPtr + offsetPerKey[i], keyPtr + offsetPerKey[i + 1]);
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

    LOG_INFO("The getSwapInfoTC(ms): {}", getSwapInfoTC.ElapsedMS());

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
        embeddingTables_[idx]->FindOrInsert(swapinKeys[i], swapinTensor.swapinEmbs.data_ptr<float>() + jaggedOffsPtr[i],
                                            swapinOptimsPtr);
    }

    LOG_INFO("The embeddingLookupTC(ms): {}", embeddingLookupTC.ElapsedMS());
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
        embeddingTables_[idx]->InsertOrAssign(swapoutKeys[i], swapoutEmbsPtr + jaggedOff, swapoutOptimPtrs);
        jaggedOff += swapoutKeys[i].size() * embConfigs_[idx].embDim;
    }

    LOG_INFO("The embeddingUpdateTC(ms): {}", embeddingUpdateTC.ElapsedMS());
    embUpdateCount_++;

    if (NeedEvictEmbeddingTable()) {
        RemoveEmbeddingTableInfo();
    }
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
    TORCH_CHECK(curTableIndices.size() + 1 == offsetPerKey.size(),
                "tableIndices size+1 must be equal to offsetPerKey size");

    for (int64_t i = 0; i < embNum_; ++i) {
        int32_t idx = curTableIndices[i];
        TORCH_CHECK(idx >= 0 && idx < embNum_, "table index {} is out of range [0, {})", idx, embNum_);

        if (embConfigs_[idx].admitAndEvictConfig.IsEvictEnabled()) {
            if (featureFilters_[idx]) {
                featureFilters_[idx]->RecordTimestamp(keyPtr, offsetPerKey[i], offsetPerKey[i + 1], timestampsPtr);
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
        LOG_INFO("Remove table embedding info, tableName: {}, remove key size: {}, detail keys: {}",
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
    TORCH_CHECK(end <= batchKeys.numel())

    if (!featureFilters_[tableIndex]) {
        return;
    }

    featureFilters_[tableIndex]->StatisticsKeyCount(featureDataPtr, countDataPtr, start, end, isCountDataEmpty);
}
