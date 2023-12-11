/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: log test
 * Author: MindX SDK
 * Create: 2023
 * History: NA
 */

#include <gtest/gtest.h>
#include "utils/common.h"

using namespace std;
using namespace MxRec;
using namespace testing;

TEST(Log, Format)
{
    string test = Logger::Format("{}{}{}", 1, 2, 3);
    EXPECT_STREQ(test.c_str(), "123");
}

TEST(Log, LogLevel)
{
    MxRec::Logger::SetLevel(Logger::DEBUG);
    testing::internal::CaptureStdout();
    LOG_DEBUG("debug log {}", "hellow");
    LOG_INFO("info log {}", "hellow");
    LOG_WARN("warn log {}", "hellow");
    LOG_ERROR("error log {}", "hellow");
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("debug log hellow"), string::npos);
    EXPECT_NE(output.find("info log hellow"), string::npos);
    EXPECT_NE(output.find("warn log hellow"), string::npos);
    EXPECT_NE(output.find("error log hellow"), string::npos);

    MxRec::Logger::SetLevel(Logger::INFO);
    testing::internal::CaptureStdout();
    LOG_DEBUG("debug log {}", "hellow");
    LOG_INFO("info log {}", "hellow");
    LOG_WARN("warn log {}", "hellow");
    LOG_ERROR("error log {}", "hellow");
    output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output.find("debug log hellow"), string::npos);
    EXPECT_NE(output.find("info log hellow"), string::npos);
    EXPECT_NE(output.find("warn log hellow"), string::npos);
    EXPECT_NE(output.find("error log hellow"), string::npos);

    MxRec::Logger::SetLevel(Logger::WARN);
    testing::internal::CaptureStdout();
    LOG_DEBUG("debug log {}", "hellow");
    LOG_INFO("info log {}", "hellow");
    LOG_WARN("warn log {}", "hellow");
    LOG_ERROR("error log {}", "hellow");
    output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output.find("debug log hellow"), string::npos);
    EXPECT_EQ(output.find("info log hellow"), string::npos);
    EXPECT_NE(output.find("warn log hellow"), string::npos);
    EXPECT_NE(output.find("error log hellow"), string::npos);

    MxRec::Logger::SetLevel(Logger::ERROR);
    testing::internal::CaptureStdout();
    LOG_DEBUG("debug log {}", "hellow");
    LOG_INFO("info log {}", "hellow");
    LOG_WARN("warn log {}", "hellow");
    LOG_ERROR("error log {}", "hellow");
    output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output.find("debug log hellow"), string::npos);
    EXPECT_EQ(output.find("info log hellow"), string::npos);
    EXPECT_EQ(output.find("warn log hellow"), string::npos);
    EXPECT_NE(output.find("error log hellow"), string::npos);
}

TEST(Log, LayzEvalution)
{
    MxRec::Logger::SetLevel(Logger::WARN);
    testing::internal::CaptureStdout();
    int flag1 = 0;
    int flag2 = 0;
    LOG_INFO("info log {} {}", "hellow", [&] {
        flag1 = 1;
        return "hellow";
    }());
    LOG_WARN("warn log {} {}", "hellow", [&] {
        flag2 = 1;
        return "hellow";
    }());
    LOG_ERROR("error log {}", "hellow");
    LOG_DEBUG("debug log {}", "hellow");
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output.find("debug log hellow"), string::npos);
    EXPECT_EQ(output.find("info log hellow hellow"), string::npos);
    EXPECT_NE(output.find("warn log hellow hellow"), string::npos);
    EXPECT_NE(output.find("error log hellow"), string::npos);
    EXPECT_EQ(flag1, 0);
    EXPECT_EQ(flag2, 1);
}

TEST(Log, Basic)
{
    MxRec::Logger::SetLevel(Logger::INFO);
    testing::internal::CaptureStdout();
    LOG_INFO("basictest");
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("basictest"), string::npos);
}

TEST(Log, TooManyArgs1)
{
    MxRec::Logger::SetLevel(Logger::INFO);
    testing::internal::CaptureStdout();
    LOG_INFO("{} {} {}", 0.1f, 'h', 'e', "llow");
    std::string output = testing::internal::GetCapturedStdout();
    cout << output << endl;
    EXPECT_NE(output.find("0.1 h ellow"), string::npos);
}

TEST(Log, TooManyArgs2)
{
    MxRec::Logger::SetLevel(Logger::INFO);
    testing::internal::CaptureStdout();
    LOG_INFO("{}", "h", "h", "h", "h", "h", "h", "h");
    std::string output = testing::internal::GetCapturedStdout();
    cout << output << endl;
    EXPECT_NE(output.find("hhhhhhh"), string::npos);
}

TEST(Log, FewArgs)
{
    MxRec::Logger::SetLevel(Logger::INFO);
    testing::internal::CaptureStdout();
    LOG_INFO("{} {} {} {} {} {}", "hellow", "hellow");
    std::string output = testing::internal::GetCapturedStdout();
    cout << output << endl;
    EXPECT_NE(output.find("hellow hellow"), string::npos);
}

TEST(Log, CkptType)
{
    MxRec::Logger::SetLevel(Logger::INFO);
    testing::internal::CaptureStdout();
    LOG_INFO("ckpt type={}", CkptDataType::EMB_DATA);
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("ckpt type=1"), string::npos);

    testing::internal::CaptureStdout();
    LOG_INFO("ckpt type={}", CkptDataType::NDDR_OFFSET);
    output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("ckpt type=5"), string::npos);
}