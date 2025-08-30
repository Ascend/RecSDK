/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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
#include "random_normal_initializer.h"

using namespace MxRec;

RandomNormalInitializer::RandomNormalInitializer(int start, int len, NormalInitializerInfo& initInfo)
    : start(start), len(len), mean(initInfo.mean), stddev(initInfo.stddev), seed(initInfo.seed),
      initParam(initInfo.initK), generator(std::default_random_engine(seed)),
      distribution(std::normal_distribution<float>(mean, stddev))
{
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