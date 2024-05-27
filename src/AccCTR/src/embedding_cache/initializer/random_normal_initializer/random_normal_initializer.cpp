/* Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
#include <iostream>
#include "embedding_cache.h"
#include "embedding_cache/common.h"
#include "external_logger.h"

using namespace EmbCache;
using namespace ock;

RandomNormalInitializer::RandomNormalInitializer(uint32_t start, uint32_t len, NormalInitializerInfo &initInfo)
    : start(start), len(len), mean(initInfo.mean), stddev(initInfo.stddev), seed(initInfo.seed)
{
    // 校验stddev mean及initK值范围
    if (initInfo.mean > NORMAL_MEAN_MAX) {
        ExternalLogger::PrintLog(LogLevel::WARN, "random normal mean param is greater than " +
            std::to_string(NORMAL_MEAN_MAX) + ", and will use " + std::to_string(NORMAL_MEAN_MAX) + ".");
        mean = NORMAL_MEAN_MAX;
    } else if (initInfo.mean < NORMAL_MEAN_MIN) {
        ExternalLogger::PrintLog(LogLevel::WARN, "random normal mean param is less than " +
            std::to_string(NORMAL_MEAN_MIN) + ", and will use " + std::to_string(NORMAL_MEAN_MIN) + ".");
        mean = NORMAL_MEAN_MIN;
    } else {
        mean = initInfo.mean;
    }
    if (initInfo.stddev > NORMAL_STDDEV_MAX) {
        ExternalLogger::PrintLog(LogLevel::WARN, "random normal stddev param is greater than " +
            std::to_string(NORMAL_STDDEV_MAX) + ", and will use " + std::to_string(NORMAL_STDDEV_MAX) + ".");
        stddev = NORMAL_STDDEV_MAX;
    } else if (initInfo.stddev < NORMAL_STDDEV_MIN) {
        ExternalLogger::PrintLog(LogLevel::WARN, "random normal stddev param is less than " +
            std::to_string(NORMAL_STDDEV_MIN) + ", and will use " + std::to_string(NORMAL_STDDEV_MIN) + ".");
        stddev = NORMAL_STDDEV_MIN;
    } else {
        stddev = initInfo.stddev;
    }
    if (initInfo.initK > INIT_K_MAX) {
        ExternalLogger::PrintLog(LogLevel::WARN, "random normal initK is greater than " + std::to_string(INIT_K_MAX) +
            ", and will use " + std::to_string(INIT_K_MAX) + ".");
        initParam = INIT_K_MAX;
    } else if (initInfo.initK < INIT_K_MIN) {
        ExternalLogger::PrintLog(LogLevel::WARN, "random normal initK is less than " + std::to_string(INIT_K_MIN) +
            ", and will use " + std::to_string(INIT_K_MIN) + ".");
        initParam = INIT_K_MIN;
    } else {
        initParam = initInfo.initK;
    }

    generator = std::default_random_engine(seed);
    distribution = std::normal_distribution<float>(mean, stddev);
}

void RandomNormalInitializer::GenerateData(float* emb, int embSize)
{
    if (len == 0) {
        return;
    }
    if (embSize != INVALID_EMB_SIZE && embSize < static_cast<int>(start + len)) {
        ExternalLogger::PrintLog(LogLevel::WARN,
                                 "InitializeInfo start " + std::to_string(start) + " + len " + std::to_string(len) +
                                 " is larger than embedding size " + std::to_string(embSize));
        return;
    }
    std::generate_n(emb + start, len, [this]() { return initParam * distribution(generator); });
}