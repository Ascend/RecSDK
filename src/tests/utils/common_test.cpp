/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: key process test
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <easy/profiler.h>

#include "utils/common.h"

using namespace std;
using namespace MxRec;
using namespace testing;

TEST(common, SetLog)
{
    int rankId = 0;
    SetLog(rankId);
    ASSERT_EQ(g_glogLevel, 0);

    putenv("GLOG_stderrthreshold=1");
    SetLog(rankId);
    ASSERT_EQ(g_glogLevel, 1);
}

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