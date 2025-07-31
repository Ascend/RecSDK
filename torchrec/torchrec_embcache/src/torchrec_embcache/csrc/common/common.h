/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
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

// const char* const FVK_STATE_STR[] = {
//     "FKV_EXIST",    "FKV_NOT_EXIST", "FKV_KEY_CONFLICT", "FKV_BEFORE_PUT_FUNC_FAIL", "FKV_BEFORE_REMOVE_FUNC_FAIL",
//     "FKV_NO_SPACE", "FKV_FAIL"};

enum class BeforePutFuncState {
    BEFORE_SUCCESS,
    BEFORE_NO_SPACE,
    BEFORE_FAIL,
};

enum class BeforeRemoveFuncState {
    BEFORE_SUCCESS,
    BEFORE_FAIL,
};

struct AdmitAndEvictConfig {
    int32_t admitThreshold = -1;
    float notAdmittedDefaultValue = 0.0;

    uint64_t evictThreshold = 0;  // unit: seconds
    uint64_t evictStepInterval = 0;

    AdmitAndEvictConfig() = default;
    AdmitAndEvictConfig(int32_t admitThreshold, float notAdmittedDefaultValue, uint64_t evictThreshold,
                        uint64_t evictStepInterval)
        : admitThreshold(admitThreshold),
          notAdmittedDefaultValue(notAdmittedDefaultValue),
          evictThreshold(evictThreshold),
          evictStepInterval(evictStepInterval) {};

    bool IsAdmitEnabled() const
    {
        return admitThreshold != -1;
    }

    bool IsEvictEnabled() const
    {
        return evictThreshold != 0;
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
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_COMMON_COMMON_H
