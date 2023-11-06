/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: truncated normal initializer module
 * Author: MindX SDK
 * Date: 2022/12/22
 */

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