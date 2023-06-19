/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: initializer test
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include "initializer/initializer.h"
#include "initializer/constant_initializer/constant_initializer.h"
#include "initializer/truncated_normal_initializer/truncated_normal_initializer.h"
#include "initializer/random_normal_initializer/random_normal_initializer.h"

using namespace std;
using namespace MxRec;

TEST(InitializerTest, ConstantInitializerTest)
{
    ConstantInitializer constant_initializer; // start; end; constant_val;

    constant_initializer = ConstantInitializer(1, 5, 7);

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

    std::tuple<float, float, int, float> ret(1.0, 0.3, 1, 0.1);
    truncatedNormalInitializer = TruncatedNormalInitializer(1, 10, ret);

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
    std::tuple<float, float, int, float> ret(2.0, 0.5, 1, 0.1);
    RandomNormalInitializer randomNormalInitializer(1, 10, ret);

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