/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: truncated normal initializer module
 * Author: MindX SDK
 * Date: 2022/12/22
 */

#include <algorithm>
#include "utils/common.h"
#include "truncated_normal_initializer.h"

using namespace MxRec;

TruncatedNormalInitializer::TruncatedNormalInitializer(int start, int len, NormalInitializerInfo& initInfo)
    : start(start), len(len), seed(initInfo.seed)
{
    initParam = initInfo.initK;
    // 校验stddev mean值范围
    if (initInfo.mean > TRUNCATED_NORMAL_MEAN_MAX) {
        LOG_WARN("truncated normal mean param is greater than 1e9, and will use 10e9.");
        mean = TRUNCATED_NORMAL_MEAN_MAX;
    } else if (initInfo.mean < TRUNCATED_NORMAL_MEAN_MIN) {
        LOG_WARN("truncated normal mean param is less than -1e9, and will use -10e9.");
        mean = TRUNCATED_NORMAL_MEAN_MIN;
    } else {
        mean = initInfo.mean;
    }
    if (initInfo.stddev > TRUNCATED_NORMAL_STDDEV_MAX) {
        LOG_WARN("truncated normal stddev param is greater than 100, and will use 100.");
        stddev = TRUNCATED_NORMAL_STDDEV_MAX;
    } else if (initInfo.stddev < TRUNCATED_NORMAL_STDDEV_MIN) {
        LOG_WARN("truncated normal stddev param is less than -100, and will use -100.");
        stddev = TRUNCATED_NORMAL_STDDEV_MIN;
    } else {
        stddev = initInfo.stddev;
    }

    generator = std::default_random_engine(seed);
    distribution = std::normal_distribution<float>(mean, stddev);
    minBound = initParam * (mean - static_cast<float>(boundNum) * stddev);
    maxBound = initParam * (mean + static_cast<float>(boundNum) * stddev);
}


void TruncatedNormalInitializer::GenerateData(float* const emb, const int embSize)
{
    if (len == 0) {
        return;
    }
    if (embSize < (start + len)) {
        LOG_WARN("InitializeInfo start {} + len {} is larger than embedding size {}.", start, len, embSize);
        return;
    }
    std::generate_n(emb + start, len, [this]() {
        float tmp = initParam * distribution(generator);
        while (tmp < minBound || tmp > maxBound) {
            tmp = initParam * distribution(generator);
        }
        return tmp;
    });
}
