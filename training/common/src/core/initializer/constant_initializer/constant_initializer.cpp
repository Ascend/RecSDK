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

#include "constant_initializer.h"
#include "log/logger.h"

using namespace std;
using namespace MxRec;

ConstantInitializer::ConstantInitializer(int start, int len, float value, float initK)
    : start(start), len(len), value(value)
{
    initParam = initK;
}

void ConstantInitializer::GenerateData(float* const emb, const int embSize)
{
    if (!RangeValidate(start, len)) {
        throw runtime_error("input params is illegal");
    }

    if (emb == nullptr) {
        throw runtime_error("Input emb address is null!. ");
    }

    if (embSize < (start + len)) {
        LOG_WARN("InitializeInfo start {} + len {} is larger than embedding size {}.", start, len, embSize);
        return;
    }
    std::fill_n(emb + start, len, initParam * value);
}