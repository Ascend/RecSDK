/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
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

#include <glog/logging.h>

#include "evict_feature_record.h"

namespace Embcache {

const int64_t INVALID_KEY = -1;

struct FeatureRecord {
    uint64_t count;
};

class FeatureFilter {
public:
    FeatureFilter(int32_t admitThreshold, uint64_t evictThreshold);

    void StatisticsKeyCount(const int64_t* featureDataPtr, const int64_t* countDataPtr, int64_t startIndex,
                            int64_t endIndex, bool isCountDataEmpty);

    void CountFilter(int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex);

    void RecordTimestamp(const int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex,
                         const int64_t* timestampDataPtr);

    std::vector<int64_t> FeatureEvict();

    // 要从embTable中删除的key信息，待lookup执行到和GetSwapInfo相同步数后删除key对应emb
    EvictFeatureRecord evictFeatureRecord;

    const std::unordered_map<int64_t, FeatureRecord>& GetFeatureCountMap();
    const std::unordered_map<int64_t, std::time_t>& GetFeatureTimestampMap();

    void LoadFeatureRecords(const std::vector<int64_t>& keys, std::vector<uint64_t>& counts);
    void LoadTimestampRecords(const std::vector<int64_t>& keys, std::vector<int64_t>& timestamps);

private:
    int32_t admitThreshold = -1;                                  // 准入阈值，默认值表示未开启准入
    uint64_t evictThreshold = 0;                                  // unit: second
    std::unordered_map<int64_t, FeatureRecord> featureRecordMap;  // 准入，记录key次数
    std::unordered_map<int64_t, std::time_t> timestampRecordMap;  // 淘汰，记录key时间戳
    std::time_t latestTimestamp = 0;                              // 当前表最新的时间戳，用于判断淘汰
};

}  // namespace Embcache

#endif  // FEATURE_FILTER_H
