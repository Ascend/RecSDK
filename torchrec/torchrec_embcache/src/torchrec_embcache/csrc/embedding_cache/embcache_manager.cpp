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
#include "utils/singleton.h"
#include "utils/time_cost.h"

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
            // 待补充 feature filter 初始化
            // auto& aaeConfig = embConfigs[i].admitAndEvictConfig;
            // featureFilters.emplace_back(FeatureFilter(embConfigs[i].tableName, aaeConfig.admitThreshold,
            //                                           aaeConfig.evictThreshold, aaeConfig.evictStepInterval));
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
    int64_t keyNum = batchKeys.numel();
    int64_t offPreSum = 0;

    std::vector<int64_t> swapoutOffs;
    std::vector<int64_t> swapinOffs;
    std::vector<int64_t> batchOffs;
    SwapInfo swapInfo;
    for (int64_t i = 0; i < curTableIndices.size(); i++) {
        int64_t idx = curTableIndices[i];
        if (embConfigs_[idx].admitAndEvictConfig.IsAdmitEnabled()) {
            // 待补充 feature filter 统计
            // featureFilters[idx].CountFilter(keyPtr, offsetPerKey[i], offsetPerKey[i + 1]);
        }

        // 取出每个表的 key
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

    auto longPinnedOpt = at::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true);
    swapInfo.swapoutOffs = at::empty({static_cast<int64_t>(swapoutOffs.size())}, longPinnedOpt);
    size_t swapoutOffsSize = swapoutOffs.size() * sizeof(int64_t);
    auto rc = memcpy_s(swapInfo.swapoutOffs.data_ptr<int64_t>(), swapoutOffsSize, swapoutOffs.data(), swapoutOffsSize);
    if (rc != 0) {
        LOG_ERROR("memcpy_s swapoutOffs to swapInfo.swapoutOffs failed. ret: {}", rc);
        throw std::runtime_error("memcpy_s swapoutOffs to swapInfo.swapoutOffs failed.");
    }

    swapInfo.swapinOffs = at::empty({static_cast<int64_t>(swapinOffs.size())}, longPinnedOpt);
    size_t swapinOffsSize = swapinOffs.size() * sizeof(int64_t);
    rc = memcpy_s(swapInfo.swapinOffs.data_ptr<int64_t>(), swapinOffsSize, swapinOffs.data(), swapinOffsSize);
    if (rc != 0) {
        LOG_ERROR("memcpy_s swapinOffs to swapInfo.swapinOffs failed. ret: {}", rc);
        throw std::runtime_error("memcpy_s swapinOffs to swapInfo.swapinOffs failed.");
    }

    swapInfo.batchOffs = at::empty({static_cast<int64_t>(batchOffs.size())}, longPinnedOpt);
    size_t batchOffsSize = batchOffs.size() * sizeof(int64_t);
    rc = memcpy_s(swapInfo.batchOffs.data_ptr<int64_t>(), batchOffsSize, batchOffs.data(), batchOffsSize);
    if (rc != 0) {
        LOG_ERROR("memcpy_s batchOffs to swapInfo.batchOffs failed. ret: {}", rc);
        throw std::runtime_error("memcpy_s batchOffs to swapInfo.batchOffs failed.");
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
    // LOG_INFO("Start invoke mgmt RecordTimestamp");
    // TimeCost recordTimestampTC;
    // const auto* keyPtr = batchKeys.data_ptr<int64_t>();
    // const auto* timestampsPtr = timestamps.data_ptr<int64_t>();
    // const std::vector<int32_t>& curTableIndices = tableIndices.empty() ? embTableIndies_ : tableIndices;
    // TORCH_CHECK(curTableIndices.size() + 1 == offsetPerKey.size(),
    //             "tableIndices size+1 must be equal to offsetPerKey size");

    // for (int64_t i = 0; i < embNum_; ++i) {
    //     int32_t idx = curTableIndices[i];
    //     if (embConfigs_[idx].admitAndEvictConfig.IsEvictEnabled()) {
    //         featureFilters[idx].RecordTimestamp(keyPtr, offsetPerKey[i], offsetPerKey[i + 1], timestampsPtr);
    //     }
    // }
    // LOG_INFO("The recordTimestampTC(ms): {}", recordTimestampTC.ElapsedMS());
}

void EmbcacheManager::EvictFeatures()
{
    // LOG_INFO("Start invoke EvictFeatures method, ComputeSwapInfo execute times: {}", swapCount_);
    // TimeCost evictFeaturesTC;
    // size_t evictKeyCount = 0;
    // for (int32_t i = 0; i < embNum_; ++i) {
    //     if (!embConfigs_[i].admitAndEvictConfig.IsEvictEnabled()) {
    //         LOG_INFO("The table: {} doesn't enable evict, skip feature evict.", embConfigs_[i].tableName);
    //         continue;
    //     }

    //     // 获取当前表要淘汰的keys
    //     const std::vector<int64_t>& evictFeatures = featureFilters[i].evictFeatureRecord.GetEvictKeys();
    //     // 调用swapManager删除映射信息
    //     // 删除embeddingTables中的embedding待对应step的swap out emb update执行完成后触发
    //     swapManagers_[i].RemoveKeys(evictFeatures);
    //     featureFilters[i].evictFeatureRecord.SetSwapCount(swapCount_);
    //     evictKeyCount += evictFeatures.size();
    // }
    // LOG_INFO("The evictFeaturesTC(ms): {}, all table evictKeyCount: {}", evictFeaturesTC.ElapsedMS(), evictKeyCount);
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

/*
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
    LOG_INFO("In Embedding2Host, weightsDev shape: {}", weightShapeStr);

    for (int32_t embIndex = 0; embIndex < embNum_; embIndex++) {
        keys.clear();
        // cache中可能预留了offset 0位置，因此拷贝回host时，需先加上偏移
        auto start = swapManagers_[embIndex].GetMemStartOffset();
        int64_t currentTableOffset = jaggedOff + start * embConfigs_[embIndex].embDim;
        auto end = swapManagers_[embIndex].GetOccupiedNum();
        for (int64_t off = start; off < end; off++) {
            keys.emplace_back(swapManagers_[embIndex].GetKey(off));
        }
        LOG_INFO("keys.size: {}", keys.size());
        std::vector<float*> momentum1DevPtrs(momentumDevs.size());
        for (size_t i = 0; i < momentumDevs.size(); i++) {
            momentum1DevPtrs[i] = momentumDevs[i].data_ptr<float>() + currentTableOffset;
        }
        embeddingTables_[embIndex]->InsertOrAssign(keys, weightsDevPtr + currentTableOffset, momentum1DevPtrs);

        // Here, GetOccupiedNum is less than embConfigs_[embIndex].cacheSize = weightsDev.shape[0],
        // and we need to skip the unnecessary weight indices.
        jaggedOff += embConfigs_[embIndex].cacheSize * embConfigs_[embIndex].embDim;
        LOG_INFO("Embedding2Host, embIndex: {}, update key size: {}, jaggedOff: {}, currentTableOffset: {}", embIndex,
                 keys.size(), jaggedOff, currentTableOffset);
    }
}
*/

/*
std::tuple<at::Tensor, std::vector<at::Tensor>> EmbcacheManager::GetDeviceSwapOutData(SwapInfo& swapInfo,
    const at::Tensor& swapoutOffs, const std::vector<at::Tensor>& weightsDevs,
    const std::vector<at::Tensor>& momentum1Devs, const std::vector<at::Tensor>& momentum2Devs,
    const std::vector<int32_t>& tableIndices)
{
    const std::vector<int64_t>& keysLengthPreSum = swapInfo.GetSwapoutKeysLengthPreSum();
    std::vector<int64_t> outEmbPreSumByDim(embConfigs_.size() + 1, 0);
    const std::vector<int32_t>& curTableIndices = tableIndices.empty() ? embTableIndies_ : tableIndices;
    for (size_t i = 0; i < swapInfo.swapoutKeys.size(); ++i) {
        auto keysSize = swapInfo.swapoutKeys[i].size();
        auto tableIdx = curTableIndices[i];
        outEmbPreSumByDim[i + 1] = outEmbPreSumByDim[i] + keysSize * embConfigs_[tableIdx].embDim;
    }
    int64_t outEmbShapeFlatSize = outEmbPreSumByDim[outEmbPreSumByDim.size() - 1];

    // output emb and optimizer tensor
    at::Tensor outEmbTensor = at::empty({outEmbShapeFlatSize}, weightsDevs[0].options());
    int64_t outOptimizerShapeFlat = embConfigs_[0].optimNum == 0 ? 0 : outEmbShapeFlatSize;
    std::vector<at::Tensor> outOptimizers(embConfigs_[0].optimNum);
    for (int32_t i = 0; i < embConfigs_[0].optimNum; ++i) {
        outOptimizers[i] = at::empty({outOptimizerShapeFlat}, weightsDevs[0].options());
    }

    // dispatch swap out
    for (size_t i = 0; i < curTableIndices.size(); ++i) {
        at::Tensor indices = swapoutOffs.slice(0, keysLengthPreSum[i], keysLengthPreSum[i + 1]);
        if (keysLengthPreSum[i] == 0 && keysLengthPreSum[i + 1] == 0) {
            // Current table don't need swap out, skip.
            continue;
        }
        auto tableIdx = curTableIndices[i];
        auto tableDim = embConfigs_[tableIdx].embDim;
        auto optimizerNum = embConfigs_[tableIdx].optimNum;
        at::Tensor outEmbedding = outEmbTensor.slice(0, outEmbPreSumByDim[i], outEmbPreSumByDim[i + 1]);
        torch::index_select_out(outEmbedding, weightsDevs[tableIdx].view({-1, tableDim}), 0, indices);
        if (optimizerNum > 0) {
            at::Tensor outMomentum1 = outOptimizers[0].slice(0, outEmbPreSumByDim[i], outEmbPreSumByDim[i + 1]);
            torch::index_select_out(outMomentum1, momentum1Devs[tableIdx].view({-1, tableDim}), 0, indices);
        }
        if (optimizerNum > 1) {
            at::Tensor outMomentum2 = outOptimizers[1].slice(0, outEmbPreSumByDim[i], outEmbPreSumByDim[i + 1]);
            torch::index_select_out(outMomentum2, momentum2Devs[tableIdx].view({-1, tableDim}), 0, indices);
        }
    }

    return {outEmbTensor, outOptimizers};
}

void EmbcacheManager::SwapInEmbAndOptimizer(SwapInfo& swapInfo, const SwapinTensor& swapInTensor,
    const at::Tensor& swapInOffsTensor, std::vector<at::Tensor>& weightsDevs,
    std::vector<at::Tensor>& momentum1Devs, std::vector<at::Tensor>& momentum2Devs,
    const std::vector<int32_t>& tableIndices)
{
    const auto& swapInEmbeddings = swapInTensor.swapinEmbs;
    const auto& swapInOptimizers = swapInTensor.swapinOptims;
    const auto& jaggedOffs = swapInTensor.jaggedOffs;
    const auto* jaggedOffsPtr = jaggedOffs.data_ptr<int64_t>();
    const auto& keysLengthPreSum = swapInfo.GetSwapinKeysLengthPreSum();
    const auto& tbConfigs = this->embConfigs_;
    const std::vector<int32_t>& curTableIndices = tableIndices.empty() ? embTableIndies_ : tableIndices;
    // swap in to device
    for (size_t i = 0; i < curTableIndices.size(); ++i) {
        if (jaggedOffsPtr[i] == 0 && jaggedOffsPtr[i + 1] == 0) {
            continue;
        }
        auto tableIdx = curTableIndices[i];
        auto tableDim = embConfigs_[tableIdx].embDim;
        auto optimizerNum = embConfigs_[tableIdx].optimNum;
        at::Tensor swapInIndices = swapInOffsTensor.slice(0, keysLengthPreSum[i], keysLengthPreSum[i + 1]);
        at::Tensor swapInEmb = swapInEmbeddings.slice(0, jaggedOffsPtr[i], jaggedOffsPtr[i + 1])
                                   .view({-1, tableDim});
        weightsDevs[tableIdx].view({-1, tableDim}).index_put_({swapInIndices}, swapInEmb);
        if (optimizerNum > 0) {
            at::Tensor swapInMomentum1 = swapInOptimizers[0]
                                             .slice(0, jaggedOffsPtr[i], jaggedOffsPtr[i + 1])
                                             .view({-1, tableDim});
            momentum1Devs[tableIdx].view({-1, tableDim}).index_put_({swapInIndices}, swapInMomentum1);
        }
        if (optimizerNum > 1) {
            at::Tensor swapInMomentum2 = swapInOptimizers[1]
                                             .slice(0, jaggedOffsPtr[i], jaggedOffsPtr[i + 1])
                                             .view({-1, tableDim});
            momentum2Devs[tableIdx].view({-1, tableDim}).index_put_({swapInIndices}, swapInMomentum2);
        }
    }
}
*/

/*
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
*/

/*
void EmbcacheManager::Save(const std::string path, const int rank)
{
    for (int32_t i = 0; i < embNum_; i++) {
        std::string tableName = embConfigs_[i].tableName;

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
        embeddingTables_[i]->ForEachKey([&](const int64_t key, const float* value) {
            ++count;
            // 1. write key
            WriteData(fileKeySliceData, reinterpret_cast<const char*>(&key), 1 * sizeof(int64_t));
            if (embConfigs_[i].admitAndEvictConfig.IsAdmitEnabled()) {
                saveKeys.emplace_back(key);
            }
            // 2. write embedding
            WriteData(fileEmbeddingSliceData, reinterpret_cast<const char*>(value),
                      embConfigs_[i].embDim * sizeof(float));
            LOG(INFO) << "In save, table:" << i << ", key:" << key << ", embedding.dim:" << embConfigs_[i].embDim
                      << ", embedding " << StringTools::ToString(value, embConfigs_[i].embDim);

            // 3. write momentum
            if (optimNum_ > 0) {
                WriteData(fileMomentum1SliceData, reinterpret_cast<const char*>(value + embConfigs_[i].embDim),
                          embConfigs_[i].embDim * sizeof(float));
                LOG(INFO) << "In save, table:" << i << ", key:" << key << ", momentum1.dim " << embConfigs_[i].embDim
                          << " momentum1 "
                          << StringTools::ToString(value + 1 * embConfigs_[i].embDim, embConfigs_[i].embDim);
            }
            if (optimNum_ > 1) {
                WriteData(fileMomentum2SliceData, reinterpret_cast<const char*>(value + 2 * embConfigs_[i].embDim),
                          embConfigs_[i].embDim * sizeof(float));
                LOG(INFO) << "In save, table:" << i << ", key:" << key << ", momentum2.dim " << embConfigs_[i].embDim
                          << " momentum2 "
                          << StringTools::ToString(value + OPTIMIZER_SLOT_INDEX2 * embConfigs_[i].embDim,
                                                   embConfigs_[i].embDim);
            }
        });
        LOG(INFO) << "The tableName: " << tableName << " shape: " << count << ", " << embConfigs_[i].embDim;
        std::vector<int64_t> keyAttribute = {sizeof(int64_t), count};
        WriteData(fileKeySliceAttr, reinterpret_cast<const char*>(keyAttribute.data()),
                  keyAttribute.size() * sizeof(int64_t));
        std::vector<int64_t> embedAttribute = {sizeof(int64_t), count, embConfigs_[i].embDim};
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
    std::vector<int64_t> momentum1Attribute = {sizeof(int64_t), count, embConfigs_[i].embDim};

    WriteData(fileMomentum1SliceAttr, reinterpret_cast<const char*>(momentum1Attribute.data()),
              momentum1Attribute.size() * sizeof(int64_t));

    // 目前momentum2Attribute和momentum1Attribute是一致的
    WriteData(fileMomentum2SliceAttr, reinterpret_cast<const char*>(momentum1Attribute.data()),
              momentum1Attribute.size() * sizeof(int64_t));
}
*/

/*
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
*/

/*
void EmbcacheManager::Load(const std::string& path, int rank)
{
    for (int32_t i = 0; i < embNum_; i++) {
        std::string tableName = embConfigs_[i].tableName;
        LOG(INFO) << "Start load, rank:" << rank << ", tableName:" << tableName;
        std::vector<int64_t> keys;
        std::string filePath = path + "/" + tableName + "/" + "rank" + std::to_string(rank);
        EmbcacheManager::ReadFile(filePath, keys, "key");

        std::vector<std::vector<float>> embeddings;
        int32_t embDim = embConfigs_[i].embDim;
        EmbcacheManager::ReadFile(filePath, embeddings, "embedding", embDim);
        LOG(INFO) << "In load, rank:" << rank << ", tableName:" << tableName << ", keys size:" << keys.size()
                  << ", embeddings size:" << embeddings.size();
        TORCH_CHECK(keys.size() == embeddings.size(), "In load scene, keys size:", keys.size(),
                    " is not equal with embedding size:", embeddings.size())

        std::vector<std::vector<float>> momentum1;
        if (optimNum_ > 0) {
            int retCode = EmbcacheManager::ReadFile(filePath, momentum1, "momentum1", embDim);
            TORCH_CHECK(retCode == 0, "Failed to read optimizer momentum1 file data.")
        }

        std::vector<std::vector<float>> momentum2;
        if (optimNum_ > 1) {
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
            if (optimNum_ > 0) {
                momentum.emplace_back(momentum1[k].data());
            }
            if (optimNum_ > 1) {
                momentum.emplace_back(momentum2[k].data());
            }
            embeddingTables_[i]->InsertOrAssign(insertKey, embeddings[k].data(), momentum);
        }

        // 加载准入淘汰数据
        LoadFeatureAdmitAndEvictInfo(i, filePath, keys);
    }
}
*/

/*
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
*/

bool EmbcacheManager::NeedEvictEmbeddingTable()
{
    // for (int32_t i = 0; i < embNum_; ++i) {
    //     // 开启淘汰
    //     if (!embConfigs_[i].admitAndEvictConfig.IsEvictEnabled()) {
    //         continue;
    //     }
    //     // 待删除embTable的keys非空且达到和GetSwapInfo相同的步数
    //     if (!featureFilters[i].evictFeatureRecord.GetEvictKeys().empty() &&
    //         featureFilters[i].evictFeatureRecord.CanRemoveFromEmbTable(embUpdateCount_)) {
    //         return true;
    //     }
    // }
    return false;
}

void EmbcacheManager::RemoveEmbeddingTableInfo()
{
    // LOG(INFO) << "Start invoke RemoveEmbeddingTableInfo, embUpdateCount_:" << embUpdateCount_;
    // TimeCost removeEmbeddingTableTC;
    // for (int32_t i = 0; i < embNum_; ++i) {
    //     auto& keys = featureFilters[i].evictFeatureRecord.GetEvictKeys();
    //     if (keys.empty()) {
    //         LOG(INFO) << "Feature keys list is empty, skip to remove embedding from table:" << embConfigs_[i].tableName;
    //         continue;
    //     }

    //     // 调用embTable Remove
    //     embeddingTables_[i]->RemoveEmbedding(keys);
    //     LOG(INFO) << "Remove table embedding info, table:" << embConfigs_[i].tableName
    //               << ", remove key size:" << keys.size() << ", detail keys:" << StringTools::ToString(keys);
    //     featureFilters[i].evictFeatureRecord.ClearEvictInfo();
    // }
    // LOG(INFO) << "The removeEmbeddingTableTC(ms):" << removeEmbeddingTableTC.ElapsedMS();
}
/*
void EmbcacheManager::SaveFeatureAdmitAndEvictInfo(int32_t tableIndex, const std::string& filePrefix,
                                                   const std::vector<int64_t>& saveKeys)
{
    TimeCost saveFeatureFilterDataTC;
    if (embConfigs_[tableIndex].admitAndEvictConfig.IsAdmitEnabled()) {
        SaveFeatureCount(tableIndex, filePrefix, saveKeys);
    }
    if (embConfigs_[tableIndex].admitAndEvictConfig.IsEvictEnabled()) {
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
*/

/*
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
*/

/*
void EmbcacheManager::LoadFeatureAdmitAndEvictInfo(int32_t tableIndex, const std::string& filePrefix,
                                                   const std::vector<int64_t>& saveKeys)
{
    TimeCost loadFeatureFilterDataTC;
    if (embConfigs_[tableIndex].admitAndEvictConfig.IsAdmitEnabled()) {
        // read key count data
        std::vector<uint64_t> keyCountVec;
        auto retCode = ReadFile(filePrefix, keyCountVec, ADMIT_STR_PATH);
        if (retCode != 0) {
            LOG(ERROR) << "Failed to read feature count file, error code:" + std::to_string(retCode);
            TORCH_CHECK(-1, "Failed to read feature count file, error code:" + std::to_string(retCode))
        }
        featureFilters[tableIndex].LoadFeatureRecords(saveKeys, keyCountVec);
    }
    if (embConfigs_[tableIndex].admitAndEvictConfig.IsEvictEnabled()) {
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
*/

void EmbcacheManager::StatisticsKeyCount(const at::Tensor& batchKeys, const torch::Tensor& offset,
                                         const at::Tensor& batchKeyCounts, int64_t tableIndex)
{
    // LOG_INFO("StatisticsKeyCount, tableIndex: {}, isAdmit: {}", tableIndex,
    //          embConfigs_[tableIndex].admitAndEvictConfig.IsAdmitEnabled());
    // if (!embConfigs_[tableIndex].admitAndEvictConfig.IsAdmitEnabled()) {
    //     return;
    // }
    // TORCH_CHECK(offset.numel() > tableIndex, "param error, tableIndex need be smaller than offset length,"
    //                                          " but got equal or greater than offset length.")
    // // 未开启local unique时，counts为空tensor，处理时默认key对应count为1
    // bool isCountDataEmpty = batchKeyCounts.numel() == 0;
    // if (!isCountDataEmpty) {
    //     TORCH_CHECK(batchKeys.numel() == batchKeyCounts.numel(),
    //                 "batchKeys length should equal with batchKeyCounts length when batchKeyCounts is not empty.")
    // }
    // auto* featureDataPtr = batchKeys.data_ptr<int64_t>();
    // auto* countDataPtr = batchKeyCounts.data_ptr<int64_t>();
    // auto* offsetDataPtr = offset.data_ptr<int64_t>();
    // int64_t start = offsetDataPtr[tableIndex];
    // int64_t end = offsetDataPtr[tableIndex + 1];
    // TORCH_CHECK(end <= batchKeys.numel())
    // featureFilters[tableIndex].StatisticsKeyCount(featureDataPtr, countDataPtr, start, end, isCountDataEmpty);
}
