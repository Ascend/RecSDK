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
    : start(start), len(len), mean(initInfo.mean), stddev(initInfo.stddev), seed(initInfo.seed)
{
    initParam = initInfo.initK;

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
