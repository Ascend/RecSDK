/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: random normal initializer module
 * Author: MindX SDK
 * Date: 2022/12/23
 */

#include <algorithm>
#include "utils/common.h"
#include "random_normal_initializer.h"

using namespace MxRec;

RandomNormalInitializer::RandomNormalInitializer(int start, int len, NormalInitializerInfo& initInfo)
    : start(start), len(len), mean(initInfo.mean), stddev(initInfo.stddev), seed(initInfo.seed)
{
    initParam = initInfo.initK;
    generator = std::default_random_engine(seed);
    distribution = std::normal_distribution<float>(mean, stddev);
}

void RandomNormalInitializer::GenerateData(float* const emb, const int embSize)
{
    if (len == 0) {
        return;
    }
    if (embSize < (start + len)) {
        LOG_WARN("InitializeInfo start {} + len {} is larger than embedding size {}.", start, len, embSize);
        return;
    }
    std::generate_n(emb + start, len, [this]() { return initParam * distribution(generator); });
}