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

#include "common/constants.h"

namespace Embcache {

FeatureFilter::FeatureFilter(const std::string& tableName, int32_t admitThreshold,
                             uint64_t evictThreshold, uint64_t evictStepInterval)
    : tableName(tableName), admitThreshold(admitThreshold),
      evictThreshold(evictThreshold), evictStepInterval(evictStepInterval)
{
}

void FeatureFilter::RecordTimestamp(const int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex,
                                    const int64_t* timestampDataPtr)
{
    auto beforeRecordSize = timestampRecordMap.size();
    for (int64_t i = startIndex; i < endIndex; ++i) {
        auto feature = *(featureDataPtr + i);
        auto timestampData = *(timestampDataPtr + i);
        auto timestamp = static_cast<std::time_t>(timestampData);
        timestampRecordMap.insert_or_assign(feature, timestamp);
        latestTimestamp = std::max(latestTimestamp, timestamp);
    }
    auto afterRecordSize = timestampRecordMap.size();
    LOG(INFO) << "Enter RecordTimestamp, beforeRecordSize:" << beforeRecordSize
              << ", afterRecordSize:" << afterRecordSize;

    // 因记录timestamp和计算swap info存在步数差异，因此记录timestamp时需同时记录淘汰keys
    if (recordTsBatchId > 0 && (recordTsBatchId + 1) % evictStepInterval == 0) {
        FeatureEvict();
    }
    recordTsBatchId++;
}

void FeatureFilter::FeatureEvict()
{
    std::vector<int64_t>& evictKeys = evictFeatureRecord.GetEvictKeys();
    if (evictThreshold == 0) {
        LOG(INFO) << "Current table evictThreshold is 0, will skip.";
        return;
    }

    LOG(INFO) << "The latestTimestamp for current table:" << latestTimestamp << ", evictThreshold:" << evictThreshold;
    auto tempEvictThreshold = static_cast<std::time_t>(evictThreshold);
    for (auto iter : timestampRecordMap) {
        auto feature = iter.first;
        if (feature == -1) {
            continue;
        }
        bool needEvict = false;
        if (latestTimestamp - iter.second > tempEvictThreshold) {
            evictKeys.emplace_back(feature);
            needEvict = true;
        }
    }
    // 淘汰掉的key从timestampRecordMap中移出
    bool isAdmitEnabled = admitThreshold != -1;
    for (auto feature : evictKeys) {
        timestampRecordMap.erase(feature);
        if (isAdmitEnabled) {
            // 开启准入时同时移出准入map中的key
            featureRecordMap.erase(feature);
        }
    }
    LOG(INFO) << "The table name:" << tableName << ", get evict keys size:" << evictKeys.size();
}

const std::unordered_map<int64_t, FeatureRecord>& FeatureFilter::GetFeatureCountMap()
{
    return featureRecordMap;
}

const std::unordered_map<int64_t, std::time_t>& FeatureFilter::GetFeatureTimestampMap()
{
    return timestampRecordMap;
}

void FeatureFilter::LoadFeatureRecords(const std::vector<int64_t>& keys, std::vector<uint64_t>& counts)
{
    if (keys.size() != counts.size()) {
        throw std::runtime_error("Failed to load key count info, vector size is not same between keys and counts.");
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        featureRecordMap[keys[i]].count = counts[i];
    }
}

void FeatureFilter::LoadTimestampRecords(const std::vector<int64_t>& keys, std::vector<int64_t>& timestamps)
{
    if (keys.size() != timestamps.size()) {
        throw std::runtime_error("Failed to load timestamp info, vector size is not same between keys and timestamps.");
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        timestampRecordMap[keys[i]] = static_cast<std::time_t>(timestamps[i]);
    }
}

void FeatureFilter::StatisticsKeyCount(const int64_t* featureDataPtr, const int64_t* countDataPtr, int64_t startIndex,
                                       int64_t endIndex, bool isCountDataEmpty)
{
    for (int64_t i = startIndex; i < endIndex; ++i) {
        auto feature = *(featureDataPtr + i);
        auto count = isCountDataEmpty ? 1 : *(countDataPtr + i);
        auto iter = featureRecordMap.find(feature);
        if (iter != featureRecordMap.end()) {
            iter->second.count += count;
        } else {
            FeatureRecord featureRecord = {count};
            featureRecordMap[feature] = featureRecord;
        }
    }
}

void FeatureFilter::CountFilter(int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex)
{
    // 准入检查，将未准入的特征置为-1
    auto thresholdCount = static_cast<uint64_t>(admitThreshold);
    for (int64_t i = startIndex; i < endIndex; ++i) {
        auto feature = *(featureDataPtr + i);
        auto iter = featureRecordMap.find(feature);
        if (iter != featureRecordMap.end() && iter->second.count < thresholdCount) {
            *(featureDataPtr + i) = INVALID_KEY;
        }
    }
}

}  // namespace Embcache
