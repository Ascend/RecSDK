/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: operator module
 * Author: MindX SDK
 * Date: 2022/11/23
 */

#include "feature_admit_and_evict.h"
#include <chrono>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <spdlog/fmt/chrono.h>
#include <spdlog/fmt/bundled/ranges.h>
#include "checkpoint/checkpoint.h"

using namespace MxRec;

std::vector<ThresholdValue> FeatureAdmitAndEvict::m_cfgThresholds {};
absl::flat_hash_map<std::string, SingleEmbTableStatus> FeatureAdmitAndEvict::m_embStatus {};

FeatureAdmitAndEvict::FeatureAdmitAndEvict(int recordsInitSize) : m_recordsInitSize(recordsInitSize) {}

FeatureAdmitAndEvict::~FeatureAdmitAndEvict()
{
    m_isEnableFunction = false;
    m_isExit = true;
    if (m_evictThread.joinable()) {
        m_evictThread.join();
    }
}

bool FeatureAdmitAndEvict::Init(const std::vector<ThresholdValue>& thresholdValues)
{
    if (!ParseThresholdCfg(thresholdValues)) {
        m_isEnableFunction = false;
        spdlog::error("Config is error, feature admin-and-evict function is not available ...\n");
        return false;
    }

    return true;
}

// 以下为类的公共接口
FeatureAdmitReturnType FeatureAdmitAndEvict::FeatureAdmit(int channel,
    const std::unique_ptr<emb_batch_t>& batch, keys_t& splitKey, std::vector<uint32_t>& keyCount)
{
    if (splitKey.size() != keyCount.size()) {
        spdlog::error("splitKey.size {} != keyCount.size {}", splitKey.size(), keyCount.size());
        return FeatureAdmitReturnType::FEATURE_ADMIT_RETURN_ERROR;
    }

    // 如果当前 tensorName 不在准入范围之内，则不进行“特征准入”逻辑
    std::string tensorName = batch->name;
    absl::flat_hash_map<int64_t, uint32_t> mergeKeys;
    mergeKeys.reserve(splitKey.size());
    PreProcessKeys(splitKey, keyCount, mergeKeys);

    std::lock_guard<std::mutex> lock(m_syncMutexs);
    auto iter = m_recordsData.historyRecords.find(tensorName);
    if (iter == m_recordsData.historyRecords.end()) { // 之前tensorName没出现过时，数据初始化
        absl::flat_hash_map<int64_t, FeatureItemInfo> records(m_recordsInitSize);
        m_recordsData.historyRecords[tensorName] = records;
    }
    spdlog::debug("FeatureAdmitAndEvict PrintSize, name:[{}], history key:[{}] ...", tensorName,
                  m_recordsData.historyRecords[tensorName].size());

    m_recordsData.timestamps[tensorName] = batch->timestamp;
    absl::flat_hash_map<int64_t, bool> visitedRecords;
    for (auto& key : splitKey) {
        if (key == -1) {
            continue;
        }

        // 特征准入&特征淘汰
        auto it = visitedRecords.find(key);
        if (it == visitedRecords.end()) {
            visitedRecords[key] = true;
            if (FeatureAdmitHelper(channel, tensorName, key, mergeKeys[key]) ==
                FeatureAdmitType::FEATURE_ADMIT_FAILED) {
                visitedRecords[key] = false;
                key = -1; // 被淘汰的Feature ID
            }
            continue;
        }

        if (visitedRecords[key] == false) {
            key = -1;
        }
    }

    return FeatureAdmitReturnType::FEATURE_ADMIT_RETURN_OK;
}

FeatureAdmitType FeatureAdmitAndEvict::FeatureAdmitHelper(int channel, const std::string& tensorName,
                                                          int64_t featureId, uint32_t featureCnt)
{
    // “特征准入”逻辑
    uint32_t currKeyCount = 0;
    absl::flat_hash_map<int64_t, FeatureItemInfo>& historyRecordInfos = m_recordsData.historyRecords[tensorName];
    auto innerIt = historyRecordInfos.find(featureId);

    if (channel == TRAIN_CHANNEL_ID) {
        if (innerIt == historyRecordInfos.end()) {
            // 维护 m_historyRecords
            FeatureItemInfo info(featureId, featureCnt, tensorName, m_recordsData.timestamps[tensorName]);
            historyRecordInfos[featureId] = info;
            currKeyCount = featureCnt;
        } else {
            // 维护 m_historyRecords
            FeatureItemInfo &info = historyRecordInfos[featureId];
            info.count += featureCnt;
            info.lastTime = m_recordsData.timestamps[tensorName];
            currKeyCount = info.count;
        }
    } else if (channel == EVAL_CHANNEL_ID) { // eval
        if (innerIt != historyRecordInfos.end()) {
            currKeyCount = historyRecordInfos[featureId].count;
        }
    }

    // 准入条件判断
    if (currKeyCount >= static_cast<uint32_t>(m_tensor2Threshold[tensorName].countThreshold)) {
        return FeatureAdmitType::FEATURE_ADMIT_OK;
    }

    return FeatureAdmitType::FEATURE_ADMIT_FAILED;
}

// 特征淘汰接口
void FeatureAdmitAndEvict::FeatureEvict(map<std::string, std::vector<emb_key_t>>& evictKeyMap)
{
    std::vector<std::string> tensorNames = GetAllNeedEvictTensorNames();
    if (tensorNames.empty()) {
        spdlog::info("EmbNames is empty, no evict function ...");
        return ;
    }
    if (!m_isEnableFunction) {
        spdlog::warn("m_isEnableFunction switch is false, no evict function ...");
        return ;
    }
    std::lock_guard<std::mutex> lock(m_syncMutexs);
    // 从 m_historyRecords 中淘汰删除
    size_t tensorCnt = tensorNames.size();
    for (size_t i = 0; i < tensorCnt; ++i) {
        FeatureEvictHelper(tensorNames[i], evictKeyMap[tensorNames[i]]);
    }
}

void FeatureAdmitAndEvict::FeatureEvictHelper(const std::string& embName, std::vector<emb_key_t>& evictKey)
{
    // 从 m_historyRecords 中淘汰删除
    time_t currTime = m_recordsData.timestamps[embName];
    // 从 m_tensor2SortedLastTime 获取当前要淘汰的featureId
    SortedRecords lastTimePriority;
    for (auto& item : m_recordsData.historyRecords[embName]) {
        lastTimePriority.push(item.second);
    }
    while (!lastTimePriority.empty()) {
        if (currTime - lastTimePriority.top().lastTime < m_tensor2Threshold[embName].timeThreshold) {
            break;
        }
        evictKey.emplace_back(lastTimePriority.top().featureId);
        lastTimePriority.pop();
    }

    if (evictKey.size() == 0) {
        spdlog::info("tensor-name[{}]'s lastTime[{}], had no key to delete ...", embName, currTime);
        return;
    }
    spdlog::info("tensor-name[{}]'s lastTime[{}], had size[{}] keys to delete ...", embName, currTime,
                 evictKey.size());

    // 真正从 m_historyRecords 中淘汰
    absl::flat_hash_map<int64_t, FeatureItemInfo>& historyRecords = m_recordsData.historyRecords[embName];
    for (size_t k = 0; k < evictKey.size(); ++k) {
        historyRecords.erase(evictKey[k]);
    }
    if (historyRecords.empty()) {
        m_recordsData.historyRecords.erase(embName);
    }
}

// 特征淘汰的使能接口
void FeatureAdmitAndEvict::SetFunctionSwitch(bool isEnableEvict)
{
    if (isEnableEvict) {
        spdlog::info("feature admit-and-evict switch is opened ...");
    } else {
        spdlog::info("feature admit-and-evict switch is closed ...");
    }
    m_isEnableFunction = isEnableEvict;
}
bool FeatureAdmitAndEvict::GetFunctionSwitch() const
{
    return m_isEnableFunction;
}

void FeatureAdmitAndEvict::PreProcessKeys(const std::vector<int64_t>& splitKey, std::vector<uint32_t>& keyCount,
    absl::flat_hash_map<int64_t, uint32_t>& mergeKeys)
{
    for (size_t i = 0; i < splitKey.size(); ++i) {
        if (splitKey[i] == -1) {
            continue;
        }

        auto it = mergeKeys.find(splitKey[i]);
        if (it == mergeKeys.end()) {
            mergeKeys[splitKey[i]] = keyCount[i];
        } else {
            mergeKeys[splitKey[i]] += keyCount[i];
        }
    }
}

bool FeatureAdmitAndEvict::IsThresholdCfgOK(const std::vector<ThresholdValue>& thresholds,
    const std::vector<std::string>& embNames, bool isTimestamp)
{
    for (size_t i = 0; i < thresholds.size(); ++i) {
        auto it = std::find(embNames.begin(), embNames.end(), thresholds[i].tensorName);
        if (it == embNames.end()) { // 配置不存在于当前跑的模型，也要报错
            spdlog::error("embName[{}] is not exist at current model ...", thresholds[i].tensorName);
            return false;
        } else {
            // 同时支持“准入&淘汰”，却没有传时间戳
            if (m_embStatus[*it] == SingleEmbTableStatus::SETS_ERROR) {
                spdlog::error("embName[{}] config error, please check ...", embNames[i]);
                return false;
            } else if (m_embStatus[*it] == SingleEmbTableStatus::SETS_BOTH && !isTimestamp) {
                spdlog::error("embName[{}] admit and evict, but no timestamp", embNames[i]);
                return false;
            }
        }
    }

    return true;
}

auto FeatureAdmitAndEvict::GetTensorThresholds() -> tensor_2_thresh_mem_t
{
    std::lock_guard<std::mutex> lock(m_syncMutexs);
    return m_tensor2Threshold;
}

auto FeatureAdmitAndEvict::GetHistoryRecords() -> AdmitAndEvictData&
{
    std::lock_guard<std::mutex> lock(m_syncMutexs);
    return m_recordsData;
}

void FeatureAdmitAndEvict::LoadTensorThresholds(tensor_2_thresh_mem_t& loadData)
{
    std::lock_guard<std::mutex> lock(m_syncMutexs);
    m_tensor2Threshold = std::move(loadData);
}

void FeatureAdmitAndEvict::LoadHistoryRecords(AdmitAndEvictData& loadData)
{
    std::lock_guard<std::mutex> lock(m_syncMutexs);
    m_recordsData = std::move(loadData);
}

// 解析m_tensor2Threshold
bool FeatureAdmitAndEvict::ParseThresholdCfg(const std::vector<ThresholdValue>& thresholdValues)
{
    if (thresholdValues.empty()) {
        spdlog::error("thresholdValues is empty ...");
        return false;
    }

    m_cfgThresholds = thresholdValues;
    for (const auto& value : thresholdValues) {
        spdlog::info("embName[{}], count[{}], time[{}] ...",
                     value.tensorName, value.countThreshold, value.timeThreshold);
        auto it = m_tensor2Threshold.find(value.tensorName);
        if (it != m_tensor2Threshold.end()) {
            // train和eval同时开启，会出现表重复配置
            spdlog::info("[{}] is repeated configuration ...", value.tensorName);
            return true;
        }
        m_tensor2Threshold[value.tensorName] = value;

        if (value.countThreshold != -1 && value.timeThreshold != -1) {
            m_embStatus[value.tensorName] = SingleEmbTableStatus::SETS_BOTH;
        } else if (value.countThreshold != -1 && value.timeThreshold == -1) {
            m_embStatus[value.tensorName] = SingleEmbTableStatus::SETS_ONLY_ADMIT;
        } else {
            spdlog::error("[{}] config error, have evict but no admit ...", value.tensorName);
            m_embStatus[value.tensorName] = SingleEmbTableStatus::SETS_ERROR;
            return false;
        }
    }

    return true;
}

std::vector<std::string> FeatureAdmitAndEvict::GetAllNeedEvictTensorNames()
{
    std::vector<std::string> names;
    std::lock_guard<std::mutex> lock(m_syncMutexs);
    for (const auto& record : m_recordsData.historyRecords) {
        // 只获取支持特征准入的embName
        if (m_embStatus[record.first] == SingleEmbTableStatus::SETS_BOTH) {
            names.emplace_back(record.first);
        }
    }
    return names;
}

void FeatureAdmitAndEvict::ResetAllRecords()
{
    std::lock_guard<std::mutex> lock(m_syncMutexs);
    for (auto& record : m_recordsData.historyRecords) {
        record.second.clear();
    }
    m_recordsData.historyRecords.clear();
    m_recordsData.timestamps.clear();
}