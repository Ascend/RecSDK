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

#ifndef MX_REC_TRUNCATED_NORMAL_INITIALIZER_H
#define MX_REC_TRUNCATED_NORMAL_INITIALIZER_H

#include <random>

#include "initializer/initializer.h"

namespace MxRec {
    using namespace std;

    constexpr float TRUNCATED_NORMAL_STDDEV_MAX = 100;
    constexpr float TRUNCATED_NORMAL_STDDEV_MIN = -100;
    constexpr float TRUNCATED_NORMAL_MEAN_MAX = 1e9;
    constexpr float TRUNCATED_NORMAL_MEAN_MIN = -1e9;

    class TruncatedNormalInitializer : public Initializer {
    public:
        TruncatedNormalInitializer() = default;
        TruncatedNormalInitializer(int start, int len, NormalInitializerInfo& initInfo);

        ~TruncatedNormalInitializer() override {};

        void GenerateData(float* const emb, const int embSize) override;

        int boundNum = 2;

        int start;
        int len;
        float mean;
        float stddev;
        int seed;

        std::default_random_engine generator;
        std::normal_distribution<float> distribution;
        float minBound = 0;
        float maxBound = 0;
    };
}

#endif // MX_REC_TRUNCATED_NORMAL_INITIALIZER_H