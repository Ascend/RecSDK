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

#include "utils/config.h"
#include "utils/logger.h"

using namespace std;
using namespace MxRec;

void SetEnvironmentVariables()
{
    setenv(RecEnvNames::ACL_TIMEOUT, "100", 1);
    setenv(RecEnvNames::HD_CHANNEL_SIZE, "50", 1);
    setenv(RecEnvNames::KEY_PROCESS_THREAD_NUM, "8", 1);
    setenv(RecEnvNames::MAX_UNIQUE_THREAD_NUM, "10", 1);
    setenv(RecEnvNames::FAST_UNIQUE, "1", 1);
    setenv(RecEnvNames::UPDATE_EMB_V2, "1", 1);
    setenv(RecEnvNames::HOT_EMB_UPDATE_STEP, "2000", 1);
    setenv(RecEnvNames::GLOG_STDERR_THRESHOLD, "1", 1);
    setenv(RecEnvNames::USE_COMBINE_FAAE, "1", 1);
    setenv(RecEnvNames::STAT_ON, "1", 1);
    setenv(RecEnvNames::RECORD_KEY_COUNT, "1", 1);
}

void UnsetEnvironmentVariables()
{
    unsetenv(RecEnvNames::ACL_TIMEOUT);
    unsetenv(RecEnvNames::HD_CHANNEL_SIZE);
    unsetenv(RecEnvNames::KEY_PROCESS_THREAD_NUM);
    unsetenv(RecEnvNames::MAX_UNIQUE_THREAD_NUM);
    unsetenv(RecEnvNames::FAST_UNIQUE);
    unsetenv(RecEnvNames::UPDATE_EMB_V2);
    unsetenv(RecEnvNames::HOT_EMB_UPDATE_STEP);
    unsetenv(RecEnvNames::GLOG_STDERR_THRESHOLD);
    unsetenv(RecEnvNames::USE_COMBINE_FAAE);
    unsetenv(RecEnvNames::STAT_ON);
    unsetenv(RecEnvNames::RECORD_KEY_COUNT);
}

TEST(GlobalEnv, DefaultValues)
{
    ASSERT_EQ(GlobalEnv::aclTimeout, -1);
    ASSERT_EQ(GlobalEnv::hdChannelSize, 40);
    ASSERT_EQ(GlobalEnv::keyProcessThreadNum, 6);
    ASSERT_EQ(GlobalEnv::maxUniqueThreadNum, 8);
    ASSERT_EQ(GlobalEnv::fastUnique, false);
    ASSERT_EQ(GlobalEnv::updateEmbV2, false);
    ASSERT_EQ(GlobalEnv::hotEmbUpdateStep, 1000);
    ASSERT_EQ(GlobalEnv::glogStderrthreshold, 0);
    ASSERT_EQ(GlobalEnv::useCombineFaae, false);
    ASSERT_EQ(GlobalEnv::statOn, false);
    ASSERT_EQ(GlobalEnv::recordKeyCount, false);
}

TEST(GlobalEnv, ConfigGlobalEnv)
{
    SetEnvironmentVariables();

    ConfigGlobalEnv();

    // 验证环境变量是否已经被正确配置
    ASSERT_EQ(GlobalEnv::aclTimeout, 100);
    ASSERT_EQ(GlobalEnv::hdChannelSize, 50);
    ASSERT_EQ(GlobalEnv::keyProcessThreadNum, 8);
    ASSERT_EQ(GlobalEnv::maxUniqueThreadNum, 10);
    ASSERT_EQ(GlobalEnv::fastUnique, true);
    ASSERT_EQ(GlobalEnv::updateEmbV2, true);
    ASSERT_EQ(GlobalEnv::hotEmbUpdateStep, 2000);
    ASSERT_EQ(GlobalEnv::glogStderrthreshold, 1);
    ASSERT_EQ(GlobalEnv::useCombineFaae, true);
    ASSERT_EQ(GlobalEnv::statOn, true);
    ASSERT_EQ(GlobalEnv::recordKeyCount, true);

    // 清除环境变量
    UnsetEnvironmentVariables();
}

TEST(LogTest, LogGlobalEnv)
{
    SetEnvironmentVariables();

    ConfigGlobalEnv();

    // 使用ASSERT_NO_THROW宏，断言LogGlobalEnv函数是否没有抛出异常
    ASSERT_NO_THROW(LogGlobalEnv());

    UnsetEnvironmentVariables();
}
