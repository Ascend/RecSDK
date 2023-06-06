/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: random normal initializer module
 * Author: MindX SDK
 * Date: 2022/12/23
 */

#ifndef MX_REC_RANDOM_NORMAL_INITIALIZER_H
#define MX_REC_RANDOM_NORMAL_INITIALIZER_H

#include <vector>
#include <random>

#include "initializer/initializer.h"

namespace MxRec {
    using namespace std;

    class RandomNormalInitializer : public Initializer {
    public:
        RandomNormalInitializer() = default;
        RandomNormalInitializer(int start, int len, float mean, float stddev, int seed, float initK);

        ~RandomNormalInitializer() override {};

        void GenerateData(float *const emb, const int embSize) override;

        int start;
        int len;
        float mean;
        float stddev;
        int seed;

        std::default_random_engine generator;
        std::normal_distribution<float> distribution;
    };
}

#endif // MX_REC_RANDOM_NORMAL_INITIALIZER_H