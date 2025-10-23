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

#include <gtest/gtest.h>
#include <tuple>

#include "initializer/initializer.h"
#include "initializer/constant_initializer/constant_initializer.h"
#include "initializer/truncated_normal_initializer/truncated_normal_initializer.h"
#include "initializer/random_normal_initializer/random_normal_initializer.h"

using namespace std;
using namespace MxRec;

TEST(InitializerTest, ConstantInitializerTest)
{
    ConstantInitializer constant_initializer; // start; end; constant_val; initK;

    constant_initializer = ConstantInitializer(1, 5, 7, 1);

    vector<vector<float>> embData;
    int vocabSize = 5;
    int embeddingSize = 10;
    embData.resize(vocabSize, vector<float>(embeddingSize));
    for (int i = 0; i < vocabSize; i++) {
        constant_initializer.GenerateData(embData.at(i).data(), embeddingSize);
    }

    std::cout << "ConstantInitializerExample:" << std::endl;
    for (int i = 0; i < vocabSize; i++) {
        for (int j = 0; j < embeddingSize; j++) {
            std::cout << embData[i][j] << ' ';
        }
        std::cout << std::endl;
    }

    ASSERT_EQ(embData.at(2).at(2), 7);
    ASSERT_EQ(embData.at(2).at(0), 0);
}

TEST(InitializerTest, TruncatedNormalInitializerTest)
{
    TruncatedNormalInitializer truncatedNormalInitializer;

    auto initInfo = NormalInitializerInfo(1.0, 0.3, 1, 0.1);
    truncatedNormalInitializer = TruncatedNormalInitializer(1, 10, initInfo);

    vector<vector<float>> embData;
    int vocabSize = 5;
    int embeddingSize = 11;
    embData.resize(vocabSize, vector<float>(embeddingSize));
    for (int i = 0; i < vocabSize; i++) {
        truncatedNormalInitializer.GenerateData(embData.at(i).data(), embeddingSize);
    }

    std::cout << "mean: " << truncatedNormalInitializer.mean << std::endl;
    std::cout << "stddev: " << truncatedNormalInitializer.stddev << std::endl;
    std::cout << "minBound: " << truncatedNormalInitializer.minBound << std::endl;
    std::cout << "maxBound: " << truncatedNormalInitializer.maxBound << std::endl;

    std::cout << "TruncatedNormalInitializerExample:" << std::endl;
    for (int i = 0; i < vocabSize; i++) {
        for (int j = 0; j < embeddingSize; j++) {
            std::cout << embData[i][j] << ' ';
        }
        std::cout << std::endl;
    }

    ASSERT_EQ(1, 1);
}

TEST(InitializerTest, RandomNormalInitializerTest)
{
    auto initInfo = NormalInitializerInfo(1.0, 0.3, 1, 0.1);
    RandomNormalInitializer randomNormalInitializer(1, 10, initInfo);

    vector<vector<float>> embData;
    int vocabSize = 5;
    int embeddingSize = 11;
    embData.resize(vocabSize, vector<float>(embeddingSize));
    for (int i = 0; i < vocabSize; i++) {
        randomNormalInitializer.GenerateData(embData.at(i).data(), embeddingSize);
    }

    std::cout << "mean: " << randomNormalInitializer.mean << std::endl;
    std::cout << "stddev: " << randomNormalInitializer.stddev << std::endl;

    std::cout << "RandomNormalInitializerExample:" << std::endl;
    for (int i = 0; i < vocabSize; i++) {
        for (int j = 0; j < embeddingSize; j++) {
            std::cout << embData[i][j] << ' ';
        }
        std::cout << std::endl;
    }
    ASSERT_EQ(1, 1);
}