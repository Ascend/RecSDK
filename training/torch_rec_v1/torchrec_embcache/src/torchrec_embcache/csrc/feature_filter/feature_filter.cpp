/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "feature_filter.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <unordered_set>

#include <c10/util/Exception.h>
#include "common/constants.h"
#include "utils/logger.h"
#include "score_strategy.h"

namespace Embcache {

void FeatureFilter::RecordTimestamp(const int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex,
                                    const int64_t* timestampDataPtr)
{
    // 添加空指针校验
    TORCH_CHECK(featureDataPtr != nullptr, "featureDataPtr should not be nullptr");
    TORCH_CHECK(timestampDataPtr != nullptr, "timestampDataPtr should not be nullptr");
    TORCH_CHECK(startIndex >= 0, "startIndex should be >= 0");

    auto beforeRecordSize = timestampRecordMap_.size();
    for (int64_t i = startIndex; i < endIndex; ++i) {
        auto feature = *(featureDataPtr + i);
        auto timestampData = *(timestampDataPtr + i);
        auto timestamp = static_cast<std::time_t>(timestampData);
        timestampRecordMap_.insert_or_assign(feature, timestamp);
        latestTimestamp_ = std::max(latestTimestamp_, timestamp);
    }
    auto afterRecordSize = timestampRecordMap_.size();
    LOG_DEBUG("Enter RecordTimestamp, beforeRecordSize: {}, afterRecordSize: {}", beforeRecordSize, afterRecordSize);

    // 因记录timestamp和计算swap info存在步数差异，因此记录timestamp时需同时记录淘汰keys
    if (recordTsBatchId_ > 0 && (recordTsBatchId_ + 1) % evictStepInterval_ == 0) {
        FeatureEvict();
    }
    recordTsBatchId_++;
}

void FeatureFilter::FeatureEvict()
{
    std::vector<int64_t>& evictKeys = evictFeatureRecord_.GetEvictKeys();
    if (evictThreshold_ == 0) {
        LOG_DEBUG("Current table evictThreshold is 0, will skip.");
        return;
    }

    LOG_DEBUG("The latestTimestamp for current table: {}, evictThreshold: {}", latestTimestamp_, evictThreshold_);
    auto tempEvictThreshold = static_cast<std::time_t>(evictThreshold_);
    for (const auto& iter : timestampRecordMap_) {
        auto feature = iter.first;
        if (feature == INVALID_KEY) {
            continue;
        }
        if (latestTimestamp_ - iter.second > tempEvictThreshold) {
            evictKeys.emplace_back(feature);
        }
    }
    // 淘汰掉的key从timestampRecordMap中移出
    for (const auto& feature : evictKeys) {
        timestampRecordMap_.erase(feature);
        if (IsAdmitEnabled()) {
            // 开启准入时同时移出准入map中的key
            featureRecordMap_.erase(feature);
        }
    }
    LOG_INFO("The table name: {}, get evict keys size: {}", tableName_, evictKeys.size());
}

void FeatureFilter::FeatureScoreEvict()
{
    std::vector<int64_t>& evictKeys = evictFeatureRecord_.GetEvictKeys();
    if (evictScoreRecordMap_.empty()) {
        LOG_DEBUG("evictScoreRecordMap_ is empty, no features to evict.");
        return;
    }

    size_t totalSize = evictScoreRecordMap_.size();
    size_t evictCount = totalSize * admitAndEvictConfig_.showClickParams.evictPercentage;  // 按比例淘汰

    // 如果计算出的淘汰数量为0，但map不为空，则至少淘汰1个元素
    if (evictCount == 0 && totalSize > 0) {
        evictCount = 1;
    }

    std::vector<std::pair<int64_t, double>> sortedScores(evictScoreRecordMap_.begin(), evictScoreRecordMap_.end());

    // 按值升序排序 - 值小的排在前面 a < b 为 true
    std::sort(sortedScores.begin(), sortedScores.end(),
              [](const std::pair<int64_t, double>& a, const std::pair<int64_t, double>& b) {
                  return CompareShowClickEvictScore(a, b);
              });

    evictKeys.clear();
    for (size_t i = 0; i < evictCount; ++i) {
        // 从头开始取（值最小的元素）
        int64_t feature = sortedScores[i].first;
        if (feature != INVALID_KEY) {
            evictKeys.emplace_back(feature);
        }
    }

    for (const auto& feature : evictKeys) {
        evictScoreRecordMap_.erase(feature);

        // showclick只要开启淘汰都会记录数据到featureRecordMap_中，所以当feature被淘汰时也需要移除key
        auto featureIt = featureRecordMap_.find(feature);
        if (featureIt != featureRecordMap_.end()) {
            featureRecordMap_.erase(feature);
        }
    }
    LOG_INFO("The table name: {}, get evict keys size: {}", tableName_, evictKeys.size());
}

const std::unordered_map<int64_t, FeatureRecord>& FeatureFilter::GetFeatureCountMap()
{
    return featureRecordMap_;
}

const std::unordered_map<int64_t, FeatureRecord>& FeatureFilter::GetFeatureRecordMap()
{
    return featureRecordMap_;
}

const std::unordered_map<int64_t, std::time_t>& FeatureFilter::GetFeatureTimestampMap()
{
    return timestampRecordMap_;
}

void FeatureFilter::ClearFeatureCountMap()
{
    featureRecordMap_.clear();
}

void FeatureFilter::ClearFeatureTimestampMap()
{
    timestampRecordMap_.clear();
}

const std::unordered_map<int64_t, double>& FeatureFilter::GetScoreRecordMap()
{
    return evictScoreRecordMap_;
}

void FeatureFilter::LoadFeatureRecords(const std::vector<int64_t>& keys, std::vector<uint64_t>& counts)
{
    if (keys.size() != counts.size()) {
        throw std::runtime_error("Failed to load key count info, vector size is not same between keys and counts.");
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        featureRecordMap_[keys[i]].count = counts[i];
    }
}

void FeatureFilter::LoadTimestampRecords(const std::vector<int64_t>& keys, std::vector<int64_t>& timestamps,
                                         std::vector<int64_t>& offsets)
{
    if (keys.size() != timestamps.size()) {
        throw std::runtime_error("Failed to load timestamp info, vector size is not same between keys and timestamps.");
    }
    for (size_t i = 0; i < offsets.size(); ++i) {
        auto offset = offsets[i];
        if (offset < 0 || offset >= static_cast<int64_t>(keys.size())) {
            throw std::runtime_error("Offset value " + std::to_string(offset) +
                                     " is out of range [0, " + std::to_string(keys.size()) + ").");
        }
        auto key = keys[offset];
        auto timestamp = static_cast<std::time_t>(timestamps[offset]);
        auto& currentTimestamp = timestampRecordMap_[key];
        if (timestamp > currentTimestamp) {
            timestampRecordMap_[key] = timestamp;
        }
    }
}

void FeatureFilter::StatisticsKeyCount(const int64_t* featureDataPtr, const int64_t* countDataPtr, int64_t startIndex,
                                       int64_t endIndex, bool isCountDataEmpty, int64_t countDim)
{
    // 添加空指针校验
    TORCH_CHECK(featureDataPtr != nullptr, "featureDataPtr should not be nullptr");
    TORCH_CHECK(isCountDataEmpty || countDataPtr != nullptr,
                "countDataPtr should not be nullptr when counts data is not empty");
    TORCH_CHECK(startIndex >= 0, "startIndex should be >= 0");

    for (int64_t i = startIndex; i < endIndex; ++i) {
        auto feature = *(featureDataPtr + i);
        int64_t rawCount;
        int64_t rawLabel;
        if (isCountDataEmpty) {
            rawCount = 1;
            rawLabel = 0;
        } else {
            if (countDim == 2) {
                rawCount = *(countDataPtr + i * 2);
                rawLabel = *(countDataPtr + i * 2 + 1);
            } else if (countDim == 1) {
                rawCount = *(countDataPtr + i);
                rawLabel = 0;
            } else {
                LOG_WARN("Invalid value countDim for countData, setting count to 1 and label to 0");
                rawCount = 1;
                rawLabel = 0;
            }
        }
        if (rawCount < 0) {
            rawCount = 0;
        }
        if (rawLabel < 0) {
            rawLabel = 0;
        }
        auto count = static_cast<uint64_t>(rawCount);
        auto label = static_cast<uint64_t>(rawLabel);
        auto iter = featureRecordMap_.find(feature);
        if (iter != featureRecordMap_.end()) {
            iter->second.count += count;
            iter->second.label += label;
        } else {
            FeatureRecord featureRecord = {count, label};
            featureRecordMap_[feature] = featureRecord;
        }
    }
}

void FeatureFilter::CountFilter(int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex)
{
    // 添加空指针校验
    TORCH_CHECK(featureDataPtr != nullptr, "featureDataPtr should not be nullptr");
    TORCH_CHECK(startIndex >= 0, "startIndex should be >= 0");

    // 准入检查，将未准入的特征置为INVALID_KEY
    if (!IsAdmitEnabled()) {
        return;
    }

    TORCH_CHECK(admitThreshold_ >= 0, "admitThreshold_ should be >= 0");
    auto thresholdCount = static_cast<uint64_t>(admitThreshold_);
    for (int64_t i = startIndex; i < endIndex; ++i) {
        auto feature = *(featureDataPtr + i);
        auto iter = featureRecordMap_.find(feature);
        if (iter != featureRecordMap_.end() && iter->second.count < thresholdCount) {
            LOG_DEBUG("Feature filtered out due to insufficient count. TableName : {}, Feature : {}, Count : {}, "
                      "Threshold : {}", tableName_, feature, iter->second.count, thresholdCount);
            *(featureDataPtr + i) = INVALID_KEY;
        }
    }
}

void FeatureFilter::ShowClickFilter(int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex)
{
    // 添加空指针校验
    TORCH_CHECK(featureDataPtr != nullptr, "featureDataPtr should not be nullptr");
    TORCH_CHECK(startIndex >= 0, "startIndex should be >= 0");

    // 准入检查，将未准入的特征置为INVALID_KEY
    if (!admitAndEvictConfig_.IsFeatureFilterEnabled()) {
        return;
    }

    TORCH_CHECK(admitAndEvictConfig_.showClickParams.admitThreshold >= SHOWCLICK_OPEN_THRESHOLD,
                "admitThreshold should be >= 0");
    TORCH_CHECK(admitAndEvictConfig_.showClickParams.scoreDecay >= 0.0f &&
                admitAndEvictConfig_.showClickParams.scoreDecay <= 1.0f,
                "scoreDecay should be in [0, 1]");
                
    auto thresholdScore = static_cast<double>(admitAndEvictConfig_.showClickParams.admitThreshold);
    std::unordered_set<int64_t> unAdmittedKeys;
    for (int64_t i = startIndex; i < endIndex; ++i) {
        auto feature = *(featureDataPtr + i);
        auto iter = featureRecordMap_.find(feature);
        if (iter != featureRecordMap_.end()) {
            auto count = iter->second.count;
            auto label = iter->second.label;
            // 准入分数计算
            if (admitAndEvictConfig_.IsAdmitEnabled()) {
                auto score = ComputeShowClickAdmitScore(count, label, admitAndEvictConfig_.showClickParams);
                if (score < thresholdScore) {
                    LOG_DEBUG(
                        "Feature filtered out due to insufficient score. TableName : {}, Feature : {}, Score : {}, "
                        "Threshold : {}",
                        tableName_, feature, score, thresholdScore);
                    *(featureDataPtr + i) = INVALID_KEY;
                    // 已经准入失败的key，不再计算淘汰分数
                    unAdmittedKeys.emplace(feature);
                    continue;
                }
            }
            if (admitAndEvictConfig_.IsEvictEnabled()) {
                // 淘汰分数计算
                if (evictScoreRecordMap_.find(feature) == evictScoreRecordMap_.end()) {
                    evictScoreRecordMap_[feature] =
                        ComputeShowClickAdmitScore(count, label, admitAndEvictConfig_.showClickParams);
                } else {
                    auto oldScore = evictScoreRecordMap_[feature];
                    auto evictScore =
                        ComputeShowClickEvictScore(oldScore, count, label, admitAndEvictConfig_.showClickParams);
                    evictScoreRecordMap_[feature] = evictScore;
                }
            }
        }
    }

    // 将未准入的key从准入记录中移除
    for (const auto& key : unAdmittedKeys) {
        featureRecordMap_.erase(key);
    }

    if (admitAndEvictConfig_.IsEvictEnabled()) {
        // 周期触发淘汰计数
        if (recordTsBatchId_ > 0 && (recordTsBatchId_ + 1) % admitAndEvictConfig_.evictStepInterval == 0) {
            FeatureScoreEvict();
        }
    }
    recordTsBatchId_++;
}

bool FeatureFilter::IsAdmitEnabled() const
{
    return admitThreshold_ != INVALID_KEY;
}

}  // namespace Embcache