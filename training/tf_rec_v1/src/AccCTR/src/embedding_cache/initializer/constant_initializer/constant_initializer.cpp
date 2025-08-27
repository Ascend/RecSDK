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

#include "embedding_cache.h"
#include "embedding_cache/common.h"
#include "external_logger.h"

using namespace std;
using namespace EmbCache;
using namespace ock;

ConstantInitializer::ConstantInitializer(uint32_t start, uint32_t len, float value, float initK)
    : start(start), len(len)
{
    if (value > CONSTANT_VALUE_MAX) {
        ExternalLogger::PrintLog(LogLevel::WARN, "constant value is greater than " +
            std::to_string(CONSTANT_VALUE_MAX) + ", and will use " + std::to_string(CONSTANT_VALUE_MAX) + ".");
        constantValue = CONSTANT_VALUE_MAX;
    } else if (value < CONSTANT_VALUE_MIN) {
        ExternalLogger::PrintLog(LogLevel::WARN, "constant value is less than " + std::to_string(CONSTANT_VALUE_MIN) +
            ", and will use " + std::to_string(CONSTANT_VALUE_MIN) + ".");
        constantValue = CONSTANT_VALUE_MIN;
    } else {
        constantValue = value;
    }
    if (initK > INIT_K_MAX) {
        ExternalLogger::PrintLog(LogLevel::WARN, "constant initK is greater than " + std::to_string(INIT_K_MAX) +
            ", and will use " + std::to_string(INIT_K_MAX) + ".");
        initParam = INIT_K_MAX;
    } else if (initK < INIT_K_MIN) {
        ExternalLogger::PrintLog(LogLevel::WARN, "constant initK is less than " + std::to_string(INIT_K_MIN) +
            ", and will use " + std::to_string(INIT_K_MIN) + ".");
        initParam = INIT_K_MIN;
    } else {
        initParam = initK;
    }
}

void ConstantInitializer::GenerateData(float* emb, int embSize)
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
    std::fill_n(emb + start, len, initParam * constantValue);
}
