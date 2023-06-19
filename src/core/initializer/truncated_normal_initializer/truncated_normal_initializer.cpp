/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: truncated normal initializer module
 * Author: MindX SDK
 * Date: 2022/12/22
 */

#include <algorithm>
#include <tuple>
#include <spdlog/spdlog.h>
#include "truncated_normal_initializer.h"

using namespace MxRec;

TruncatedNormalInitializer::TruncatedNormalInitializer(int start, int len, std::tuple<float, float, int, float> ret)
    : start(start), len(len)
{
    std::tie(mean, stddev, seed, initParam) = ret;

    generator = std::default_random_engine(seed);
    distribution = std::normal_distribution<float>(mean, stddev);
    minBound = mean - static_cast<float>(boundNum) * stddev;
    maxBound = mean + static_cast<float>(boundNum) * stddev;
}


void TruncatedNormalInitializer::GenerateData(float* const emb, const int embSize)
{
    if (len == 0) {
        return;
    }
    if (embSize < (start + len)) {
        spdlog::warn(
            "InitializeInfo start {} + len {} is larger than embedding size {}.",
            start, len, embSize);
        return;
    }
    std::generate_n(emb + start, len, [&]() {
        float tmp = initParam * distribution(generator);
        while (tmp < minBound || tmp > maxBound) {
            tmp = initParam * distribution(generator);
        }
        return tmp;
    });
}
