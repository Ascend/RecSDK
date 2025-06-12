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
    bool isExceptionThrow = {false};
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

TEST(TestSetLog, GRankIdNotEmpty)
{
    MxRec::GlogConfig::gRankId = "1";
    MxRec::SetLog(0);
    EXPECT_EQ(MxRec::GlogConfig::gRankId, "1");
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

TEST(TestValidateReadFile, InvalidDatasetSize_Error)
{
    long long invalidSize = 1LL << 40 + 1;
    EXPECT_THROW(MxRec::ValidateReadFile("/home/slice_0.data", invalidSize), invalid_argument);
}


TEST(RankInfoTest, ConstructOk)
{
    auto maxSteps = vector<int>{100};
    auto rankInfo = RankInfo(1, 1, maxSteps);
    EXPECT_EQ(rankInfo.localRankSize, 1);
}

TEST(RankInfoTest, LocalRankSizeZero)
{
    auto maxSteps = vector<int>{100};
    auto rankInfo = RankInfo(0, 1, maxSteps);
    EXPECT_EQ(rankInfo.localRankSize, 0);
}

TEST(RankInfoTest, LocalRankSizeNotZero)
{
    auto ctrlSteps = vector<int>{100};
    auto rankInfo = RankInfo(1, 1, 1, 1, ctrlSteps);
    EXPECT_EQ(rankInfo.localRankSize, 1);
}

TEST(CommTest, CheckFilePermissionErr)
{
    auto filePath = string("/etc/os-release");
    EXPECT_EQ(CheckFilePermission(filePath), false);

    filePath = string("/etc/unknown");
    EXPECT_EQ(CheckFilePermission(filePath), false);
}

TEST(CommTest, FloatPtrToLimitStrOk)
{
    float val = 0;
    auto ptrVal = &val;
    auto strVal = FloatPtrToLimitStr(ptrVal, 1);
    EXPECT_EQ(strVal, "0.000000 ");
}

TEST(CommTest, GetStepFromPathOk) {
    const auto loadPath = SAVE_SPARSE_PATH_PREFIX + "-ckpt-0";
    auto step = GetStepFromPath(loadPath);
    EXPECT_EQ(step, 0);
}

TEST(CommTest, CheckFileExistOk) {
    const auto filePath = "invalid_path"s;
    EXPECT_FALSE(CheckFileExist(filePath));
}
