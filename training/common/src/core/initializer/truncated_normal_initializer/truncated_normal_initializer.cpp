/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include <algorithm>
#include "log/logger.h"
#include "truncated_normal_initializer.h"

using namespace MxRec;

TruncatedNormalInitializer::TruncatedNormalInitializer(int start, int len, NormalInitializerInfo& initInfo)
    : start(start), len(len), seed(initInfo.seed), generator(std::default_random_engine(initInfo.seed))
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
        LOG_WARN("truncated normal stddev param is less than {}, and will use {}.",
            TRUNCATED_NORMAL_STDDEV_MIN, TRUNCATED_NORMAL_STDDEV_MIN);
        stddev = TRUNCATED_NORMAL_STDDEV_MIN;
    } else {
        stddev = initInfo.stddev;
    }

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
