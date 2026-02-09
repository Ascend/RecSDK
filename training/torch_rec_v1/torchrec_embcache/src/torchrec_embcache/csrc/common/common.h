/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_COMMON_COMMON_H
#define EMBEDDING_CACHE_COMMON_COMMON_H

#include <cstdint>
#include <cstddef>
#include <string>

#include "constants.h"

namespace Embcache {

#ifndef HM_UNLIKELY
#define HM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

#ifndef HM_LIKELY
#define HM_LIKELY(x) __builtin_expect(!!(x), 1)
#endif

enum class FkvState : uint8_t {
    FKV_EXIST = 0,
    FKV_NOT_EXIST = 1,
    FKV_KEY_CONFLICT = 2,
    FKV_BEFORE_PUT_FUNC_FAIL = 3,
    FKV_BEFORE_REMOVE_FUNC_FAIL = 4,
    FKV_NO_SPACE = 5,
    FKV_FAIL = 6,
};

enum class InitializerType : uint8_t {
    LINEAR = 0,
    TRUNCATED_NORMAL = 1,
    UNIFORM = 2,
};

enum class BeforePutFuncState {
    BEFORE_SUCCESS,
    BEFORE_NO_SPACE,
    BEFORE_FAIL,
};

enum class BeforeRemoveFuncState {
    BEFORE_SUCCESS,
    BEFORE_FAIL,
};

enum class AdmitAndEvictPolicyType {
    NONE, // 无准入淘汰策略
    POLICY_COUNT,
    POLICY_SHOWCLICK,
};

struct ShowClickParams {
    // 准入分数计算 admitScore = alpha * count + beta * label
    // 淘汰分数     evictScore = (oldEvictScore + alpha * count + beta * label) * scoreDecay
    float alpha = 1.0;
    float beta = 0.0;
    // 准入阈值 开启准入时,小于此分数的则丢弃  当此分数大于0时表示开启准入功能
    float admitThreshold = 0.0;
    // 淘汰比例 开启淘汰时,分数较小且在此比例中的则淘汰  当此值大于0时表示开启淘汰功能
    float evictPercentage = 0.0;
    // 分数衰减系数 用于淘汰分数计算和更新 [0,1] 1表示不衰减 0表示全衰减
    float scoreDecay = 1.0;
};

struct AdmitAndEvictConfig {
    int64_t admitThreshold = INVALID_KEY;
    float notAdmittedDefaultValue = 0.0;

    uint64_t evictThreshold = 0;  // unit: seconds
    uint64_t evictStepInterval = 0;

    ShowClickParams showClickParams;
    AdmitAndEvictPolicyType policyType;

    AdmitAndEvictConfig() = default;
    AdmitAndEvictConfig(int64_t admitThreshold, float notAdmittedDefaultValue, uint64_t evictThreshold,
                        uint64_t evictStepInterval)
        : admitThreshold(admitThreshold),
          notAdmittedDefaultValue(notAdmittedDefaultValue),
          evictThreshold(evictThreshold),
          evictStepInterval(evictStepInterval)
    {
        policyType = AdmitAndEvictPolicyType::POLICY_COUNT;
    };

    AdmitAndEvictConfig(int64_t admitThreshold, float notAdmittedDefaultValue, uint64_t evictThreshold,
                        uint64_t evictStepInterval, const ShowClickParams& showClickParams,
                        const AdmitAndEvictPolicyType& policyType)
        : admitThreshold(admitThreshold),
          notAdmittedDefaultValue(notAdmittedDefaultValue),
          evictThreshold(evictThreshold),
          evictStepInterval(evictStepInterval),
          showClickParams(showClickParams),
          policyType(policyType) {};

    bool IsAdmitEnabled() const
    {
        if (policyType == AdmitAndEvictPolicyType::POLICY_COUNT) {
            return admitThreshold != INVALID_KEY;
        }
        if (policyType == AdmitAndEvictPolicyType::POLICY_SHOWCLICK) {
            return showClickParams.admitThreshold > SHOWCLICK_OPEN_THRESHOLD;
        }
        return false;
    }

    bool IsEvictEnabled() const
    {
        if (policyType == AdmitAndEvictPolicyType::POLICY_COUNT) {
            return evictThreshold != 0;
        }
        if (policyType == AdmitAndEvictPolicyType::POLICY_SHOWCLICK) {
            return showClickParams.evictPercentage > SHOWCLICK_OPEN_THRESHOLD;
        }
        return false;
    }

    bool IsFeatureFilterEnabled() const
    {
        return IsAdmitEnabled() || IsEvictEnabled();
    }
};

struct EmbConfig {
    std::string tableName;
    InitializerType initializerType;
    int32_t embDim;
    int32_t optimNum;   // 使用的优化器参数数量
    int64_t cacheSize;  // cache 可以存放的 Embedding 数量
    float weightInitMin;
    float weightInitMax;
    float weightInitMean;    // 仅TRUNCATED_NORMAL使用
    float weightInitStddev;  // 仅TRUNCATED_NORMAL使用
    AdmitAndEvictConfig admitAndEvictConfig;
    int32_t initializerRandomPoolSize;
    int32_t seed;
    int64_t num_features;  // 每个表对应的feature name个数
    bool isIncremental; // 是否增量存储
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_COMMON_COMMON_H
