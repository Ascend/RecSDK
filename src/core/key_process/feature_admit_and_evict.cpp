/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: operator module
 * Author: MindX SDK
 * Date: 2022/11/23
 */

#include "feature_admit_and_evict.h"
#include <chrono>

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
        LOG(ERROR) << "Config is error, feature admin-and-evict function is not available ...\n";
        return false;
    }
    SetCombineSwitch();

    return true;
}

// 以下为类的公共接口
FeatureAdmitReturnType FeatureAdmitAndEvict::FeatureAdmit(int channel,
    const std::unique_ptr<emb_batch_t>& batch, keys_t& splitKey, std::vector<uint32_t>& keyCount)
{
    if (splitKey.size() != keyCount.size()) {
        LOG(ERROR) << StringFormat("splitKey.size %d != keyCount.size %d", splitKey.size(), keyCount.size());
        return FeatureAdmitReturnType::FEATURE_ADMIT_RETURN_ERROR;
    }

    std::string tableName = batch->name;
    if (m_isCombine) {
        tableName = COMBINE_HISTORY_NAME;
    }
    absl::flat_hash_map<int64_t, uint32_t> mergeKeys;
    mergeKeys.reserve(splitKey.size());
    PreProcessKeys(splitKey, keyCount, mergeKeys);

    std::lock_guard<std::mutex> lock(m_syncMutexs);
    // 如果当前 tableName 不在准入范围之内，则不进行“特征准入”逻辑
    auto iter = m_recordsData.historyRecords.find(tableName);
    if (iter == m_recordsData.historyRecords.end()) { // 之前tableName没出现过时，数据初始化
        absl::flat_hash_map<int64_t, FeatureItemInfo> records(m_recordsInitSize);
        m_recordsData.historyRecords[tableName] = records;
    }
    VLOG(GLOG_DEBUG) << StringFormat(
        "FeatureAdmitAndEvict PrintSize, name:[%s], history key:[%d] ...", tableName.c_str(),
        m_recordsData.historyRecords[tableName].size());

    if (batch->timestamp > m_recordsData.timestamps[tableName]) {
        m_recordsData.timestamps[tableName] = batch->timestamp;
    }
    absl::flat_hash_map<int64_t, bool> visitedRecords;
    for (auto& key : splitKey) {
        if (key == -1) {
            continue;
        }

        // 特征准入&特征淘汰
        auto it = visitedRecords.find(key);
        if (it == visitedRecords.end()) {
            visitedRecords[key] = true;
            if (FeatureAdmitHelper(channel, batch->name, key, mergeKeys[key]) ==
                FeatureAdmitType::FEATURE_ADMIT_FAILED) {
                visitedRecords[key] = false;
                key = -1; // 被淘汰的Feature ID
            }
            continue;
        }

        if (!visitedRecords[key]) {
            key = -1;
        }
    }
    if (VLOG_IS_ON(GLOG_TRACE)) {
        VLOG(GLOG_TRACE) << StringFormat(
            "FeatureAdmit, name:[%s], channel:[%d], after admit, splitKey:[%s] ...", tableName.c_str(), channel,
            VectorToString(splitKey).c_str());
    }

    return FeatureAdmitReturnType::FEATURE_ADMIT_RETURN_OK;
}

FeatureAdmitType FeatureAdmitAndEvict::FeatureAdmitHelper(const int channel, const std::string& tableNameOrigin,
                                                          const int64_t featureId, const uint32_t featureCnt)
{
    // “特征准入”逻辑
    uint32_t currKeyCount = 0;
    std::string tableName = tableNameOrigin;
    if (m_isCombine) {
        tableName = COMBINE_HISTORY_NAME;
    }

    absl::flat_hash_map<int64_t, FeatureItemInfo>& historyRecordInfos = m_recordsData.historyRecords[tableName];
    auto innerIt = historyRecordInfos.find(featureId);

    if (channel == TRAIN_CHANNEL_ID) {
        if (innerIt == historyRecordInfos.end()) {
            // 维护 m_historyRecords
            FeatureItemInfo info(featureCnt, m_recordsData.timestamps[tableName]);
            info.count *= m_table2Threshold[tableNameOrigin].faaeCoefficient;
            historyRecordInfos[featureId] = info;
            currKeyCount = info.count;
        } else {
            // 维护 m_historyRecords
            FeatureItemInfo &info = historyRecordInfos[featureId];
            info.count += m_table2Threshold[tableNameOrigin].faaeCoefficient * featureCnt;
            info.lastTime = m_recordsData.timestamps[tableName];
            currKeyCount = info.count;
        }
    } else if (channel == EVAL_CHANNEL_ID) { // eval
        if (innerIt != historyRecordInfos.end()) {
            currKeyCount = historyRecordInfos[featureId].count;
        }
    }

    // 准入条件判断
    if (currKeyCount >= static_cast<uint32_t>(m_table2Threshold[tableNameOrigin].countThreshold)) {
        return FeatureAdmitType::FEATURE_ADMIT_OK;
    }

    return FeatureAdmitType::FEATURE_ADMIT_FAILED;
}

// 特征淘汰接口
void FeatureAdmitAndEvict::FeatureEvict(map<std::string, std::vector<emb_key_t>>& evictKeyMap)
{
    std::vector<std::string> tableNames = GetAllNeedEvictTableNames();
    if (tableNames.empty()) {
        LOG(INFO) << "EmbNames is empty, no evict function ...";
        return ;
    }
    if (!m_isEnableFunction) {
        LOG(WARNING) << "m_isEnableFunction switch is false, no evict function ...";
        return ;
    }
    std::lock_guard<std::mutex> lock(m_syncMutexs);
    // 从 m_historyRecords 中淘汰删除
    size_t tableCnt = tableNames.size();
    for (size_t i = 0; i < tableCnt; ++i) {
        FeatureEvictHelper(tableNames[i], evictKeyMap[tableNames[i]]);
    }
}

void FeatureAdmitAndEvict::FeatureEvictHelper(const std::string& embName, std::vector<emb_key_t>& evictKey)
{
    // 从 m_historyRecords 中淘汰删除
    time_t currTime = m_recordsData.timestamps[embName];
    // 从 m_table2SortedLastTime 获取当前要淘汰的featureId
    auto cmp = [](const auto& a, const auto& b) { return a.second.lastTime > b.second.lastTime; };
    std::priority_queue<std::pair<int64_t, FeatureItemInfo>,
            std::vector<std::pair<int64_t, FeatureItemInfo>>, decltype(cmp)> lastTimePriority(cmp);
    for (auto& item : m_recordsData.historyRecords[embName]) {
        lastTimePriority.emplace(item);
    }
    while (!lastTimePriority.empty()) {
        if (currTime - lastTimePriority.top().second.lastTime < m_table2Threshold[embName].timeThreshold) {
            break;
        }
        evictKey.emplace_back(lastTimePriority.top().first);
        lastTimePriority.pop();
    }

    if (evictKey.size() == 0) {
        LOG(INFO) << StringFormat(
            "table-name[%s]'s lastTime[%d], had no key to delete ...", embName.c_str(), currTime);
        return;
    }
    LOG(INFO) << StringFormat(
        "table-name[%s]'s lastTime[%d], had size[%d] keys to delete ...", embName.c_str(), currTime, evictKey.size());

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
        LOG(INFO) << "feature admit-and-evict switch is opened ...";
    } else {
        LOG(INFO) << "feature admit-and-evict switch is closed ...";
    }
    m_isEnableFunction = isEnableEvict;
}

void FeatureAdmitAndEvict::SetCombineSwitch()
{
    m_isCombine = GetCombineSwitch();
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
        auto it = std::find(embNames.begin(), embNames.end(), thresholds[i].tableName);
        if (it == embNames.end()) { // 配置不存在于当前跑的模型，也要报错
            LOG(ERROR) << StringFormat(
                "embName[%s] is not exist at current model ...", thresholds[i].tableName.c_str());
            return false;
        } else {
            // 同时支持“准入&淘汰”，却没有传时间戳
            if (m_embStatus[*it] == SingleEmbTableStatus::SETS_ERROR) {
                LOG(ERROR) << StringFormat("embName[%s] config error, please check ...", embNames[i].c_str());
                return false;
            } else if (m_embStatus[*it] == SingleEmbTableStatus::SETS_BOTH && !isTimestamp) {
                LOG(ERROR) << StringFormat("embName[%s] admit and evict, but no timestamp", embNames[i].c_str());
                return false;
            }
        }
    }

    return true;
}

auto FeatureAdmitAndEvict::GetTableThresholds() -> table_2_thresh_mem_t
{
    std::lock_guard<std::mutex> lock(m_syncMutexs);
    return m_table2Threshold;
}

auto FeatureAdmitAndEvict::GetHistoryRecords() -> AdmitAndEvictData&
{
    std::lock_guard<std::mutex> lock(m_syncMutexs);
    return m_recordsData;
}

void FeatureAdmitAndEvict::LoadTableThresholds(table_2_thresh_mem_t& loadData)
{
    std::lock_guard<std::mutex> lock(m_syncMutexs);
    m_table2Threshold = std::move(loadData);
}

void FeatureAdmitAndEvict::LoadHistoryRecords(AdmitAndEvictData& loadData)
{
    std::lock_guard<std::mutex> lock(m_syncMutexs);
    m_recordsData = std::move(loadData);
}

// 解析m_table2Threshold
bool FeatureAdmitAndEvict::ParseThresholdCfg(const std::vector<ThresholdValue>& thresholdValues)
{
    if (thresholdValues.empty()) {
        LOG(ERROR) << "thresholdValues is empty ...";
        return false;
    }

    m_cfgThresholds = thresholdValues;
    for (const auto& value : thresholdValues) {
        LOG(INFO) << StringFormat(
            "embName[%s], count[%d], time[%d], coefficient[%d] ...",
            value.tableName.c_str(), value.countThreshold, value.timeThreshold, value.faaeCoefficient);
        auto it = m_table2Threshold.find(value.tableName);
        if (it != m_table2Threshold.end()) {
            // train和eval同时开启，会出现表重复配置
            LOG(INFO) << StringFormat("[%s] is repeated configuration ...", value.tableName.c_str());
            return true;
        }
        m_table2Threshold[value.tableName] = value;

        if (value.faaeCoefficient < 1) {
            LOG(ERROR) << StringFormat("[%s] config error, coefficient smaller than 1 ...", value.tableName.c_str());
        }
        if (value.countThreshold != -1 && value.timeThreshold != -1) {
            m_embStatus[value.tableName] = SingleEmbTableStatus::SETS_BOTH;
        } else if (value.countThreshold != -1 && value.timeThreshold == -1) {
            m_embStatus[value.tableName] = SingleEmbTableStatus::SETS_ONLY_ADMIT;
        } else {
            LOG(ERROR) << StringFormat("[%s] config error, have evict but no admit ...", value.tableName.c_str());
            m_embStatus[value.tableName] = SingleEmbTableStatus::SETS_ERROR;
            return false;
        }
    }

    return true;
}

std::vector<std::string> FeatureAdmitAndEvict::GetAllNeedEvictTableNames()
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