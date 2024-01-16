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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <easy/profiler.h>

#include "utils/common.h"

using namespace std;
using namespace MxRec;
using namespace testing;

TEST(common, InitializeInfo)
{
    NormalInitializerInfo nInfoTruncatedNormal;
    string nameTruncatedNormal = "truncated_normal_initializer";
    InitializeInfo iInfo = InitializeInfo(nameTruncatedNormal, 0, 1, nInfoTruncatedNormal);
    ASSERT_EQ(iInfo.initializerType, InitializerType::TRUNCATED_NORMAL);

    NormalInitializerInfo nInfoRandomNormal;
    string nameRandomNormal = "random_normal_initializer";
    iInfo = InitializeInfo(nameRandomNormal, 0, 1, nInfoRandomNormal);
    ASSERT_EQ(iInfo.initializerType, InitializerType::RANDOM_NORMAL);

    NormalInitializerInfo nInfoInvalid;
    string nameInvalid = "x";
    bool isExceptionThrow = { false };
    try {
        iInfo = InitializeInfo(nameInvalid, 0, 1, nInfoInvalid);
    } catch (const std::invalid_argument& e) {
        isExceptionThrow = true;
    }
    ASSERT_EQ(isExceptionThrow, true);
}

// 测试 RandomInfo 构造函数
TEST(TestRandomInfo, Basic)
{
    MxRec::RandomInfo info(0, 10, 1.0f, 0.0f, 1.0f);
    EXPECT_EQ(info.start, 0);
    EXPECT_EQ(info.len, 10);
    EXPECT_EQ(info.constantVal, 1.0f);
    EXPECT_EQ(info.randomMin, 0.0f);
    EXPECT_EQ(info.randomMax, 1.0f);
}

TEST(TestSetLog, Basic)
{
    // 在每次测试之前重置 gRankId
    MxRec::GlogConfig::gRankId = "";

    // 假设 GlobalEnv::glogStderrthreshold 已经被设置为一个有效的值
    MxRec::SetLog(0);

    // 检查 gGlogLevel 是否被正确设置
    EXPECT_EQ(MxRec::GlogConfig::gGlogLevel, GlobalEnv::glogStderrthreshold);

    // 检查 gRankId 是否被正确设置
    EXPECT_EQ(MxRec::GlogConfig::gRankId, "0");
}

TEST(TestGetThreadNumEnv, Basic)
{
    // 假设 GlobalEnv::keyProcessThreadNum 已经被设置为一个有效的值
    int num = MxRec::GetThreadNumEnv();
    // 检查返回的线程数是否正确
    EXPECT_EQ(num, GlobalEnv::keyProcessThreadNum);
}

TEST(TestValidateReadFile, Basic)
{
    EXPECT_NO_THROW(MxRec::ValidateReadFile("/home/slice_0.data", 28000000));
}
