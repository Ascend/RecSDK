/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef FEATURE_FILTER_H
#define FEATURE_FILTER_H

#include <cstdint>
#include <ctime>
#include <unordered_map>
#include <vector>
#include <string>

#include "evict_feature_record.h"
#include "common/constants.h"

namespace Embcache {

struct FeatureRecord {
    uint64_t count;
};

class FeatureFilter {
public:
    FeatureFilter(const std::string& tableName, int64_t admitThreshold, uint64_t evictThreshold,
                  uint64_t evictStepInterval);
    void StatisticsKeyCount(const int64_t* featureDataPtr, const int64_t* countDataPtr, int64_t startIndex,
                            int64_t endIndex, bool isCountDataEmpty);
    void CountFilter(int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex);
    void RecordTimestamp(const int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex,
                         const int64_t* timestampDataPtr);
    void FeatureEvict();

    // 要从embTable中删除的key信息，待lookup执行到和GetSwapInfo相同步数后删除key对应emb
    EvictFeatureRecord evictFeatureRecord_;

    const std::unordered_map<int64_t, FeatureRecord>& GetFeatureCountMap();
    const std::unordered_map<int64_t, std::time_t>& GetFeatureTimestampMap();

    void LoadFeatureRecords(const std::vector<int64_t>& keys, std::vector<uint64_t>& counts);
    void LoadTimestampRecords(const std::vector<int64_t>& keys, std::vector<int64_t>& timestamps);

    // 清空准入统计、淘汰统计信息
    void ClearFeatureCountMap();
    void ClearFeatureTimestampMap();

private:
    std::string tableName_;

    // 准入相关配置
    int64_t admitThreshold_ = INVALID_KEY;         // 准入阈值，默认值表示未开启准入
    std::unordered_map<int64_t, FeatureRecord> featureRecordMap_;  // 准入，记录key次数

    // 淘汰相关配置
    uint64_t evictThreshold_ = 0;                                  // unit: second
    uint64_t evictStepInterval_ = 0;                               // 淘汰间隔步数
    uint64_t recordTsBatchId_ = 0;
    std::time_t latestTimestamp_ = 0;                              // 当前表最新的时间戳，用于判断淘汰
    std::unordered_map<int64_t, std::time_t> timestampRecordMap_;  // 淘汰，记录key时间戳

    bool IsAdmitEnabled() const;
};

}  // namespace Embcache

#endif  // FEATURE_FILTER_H