/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "feature_filter.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include <c10/util/Exception.h>
#include "common/constants.h"
#include "utils/logger.h"

namespace Embcache {

FeatureFilter::FeatureFilter(const std::string& tableName, int64_t admitThreshold,
                             uint64_t evictThreshold, uint64_t evictStepInterval)
    : tableName_(tableName), admitThreshold_(admitThreshold),
      evictThreshold_(evictThreshold), evictStepInterval_(evictStepInterval)
{
}

void FeatureFilter::RecordTimestamp(const int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex,
                                    const int64_t* timestampDataPtr)
{
    // 添加空指针校验
    TORCH_CHECK(featureDataPtr != nullptr, "featureDataPtr should not be nullptr");
    TORCH_CHECK(timestampDataPtr != nullptr, "timestampDataPtr should not be nullptr");

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

const std::unordered_map<int64_t, FeatureRecord>& FeatureFilter::GetFeatureCountMap()
{
    return featureRecordMap_;
}

const std::unordered_map<int64_t, std::time_t>& FeatureFilter::GetFeatureTimestampMap()
{
    return timestampRecordMap_;
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

void FeatureFilter::LoadTimestampRecords(const std::vector<int64_t>& keys, std::vector<int64_t>& timestamps)
{
    if (keys.size() != timestamps.size()) {
        throw std::runtime_error("Failed to load timestamp info, vector size is not same between keys and timestamps.");
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        timestampRecordMap_[keys[i]] = static_cast<std::time_t>(timestamps[i]);
    }
}

void FeatureFilter::StatisticsKeyCount(const int64_t* featureDataPtr, const int64_t* countDataPtr, int64_t startIndex,
                                       int64_t endIndex, bool isCountDataEmpty)
{
    // 添加空指针校验
    TORCH_CHECK(featureDataPtr != nullptr, "featureDataPtr should not be nullptr");
    TORCH_CHECK(isCountDataEmpty || countDataPtr != nullptr,
                "countDataPtr should not be nullptr when counts data is not empty");

    for (int64_t i = startIndex; i < endIndex; ++i) {
        auto feature = *(featureDataPtr + i);
        // 确保 count 为非负数，并转换为 uint64_t 类型
        int64_t rawCount = isCountDataEmpty ? 1 : *(countDataPtr + i);
        if (rawCount < 0) {
            LOG_WARN("Negative count {} detected for feature {}, setting to 0", rawCount, feature);
            rawCount = 0;
        }
        auto count = static_cast<uint64_t>(rawCount);

        auto iter = featureRecordMap_.find(feature);
        if (iter != featureRecordMap_.end()) {
            iter->second.count += count;
        } else {
            FeatureRecord featureRecord = {count};
            featureRecordMap_[feature] = featureRecord;
        }
    }
}

void FeatureFilter::CountFilter(int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex)
{
    // 添加空指针校验
    TORCH_CHECK(featureDataPtr != nullptr, "featureDataPtr should not be nullptr");

    // 准入检查，将未准入的特征置为INVALID_KEY
    if (!IsAdmitEnabled()) {
        return;
    }

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

bool FeatureFilter::IsAdmitEnabled() const
{
    return admitThreshold_ != INVALID_KEY;
}

}  // namespace Embcache