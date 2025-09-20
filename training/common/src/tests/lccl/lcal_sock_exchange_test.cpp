/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "lccl/src/tools/socket/lcal_sock_exchange.h"

#include <vector>
#include <mpi.h>
#include <gtest/gtest.h>
#include <emock/emock.hpp>

#include "lcal_api.h"
#include "lcal_comm.h"
#include "lcal_types.h"

namespace Lcal {
using std::string;
using std::vector;

class LcalSockExchangeTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        emock::GlobalMockObject::reset();
    }
};

TEST_F(LcalSockExchangeTest, CheckValidOK)
{
    auto id = LcalUniqueId();
    auto res = LcalSockExchange::CheckValid(id);
    ASSERT_EQ(res, LCAL_SUCCESS);
}

TEST_F(LcalSockExchangeTest, CleanupOK)
{
    auto ranks = vector<int>{0};
    auto sock = LcalSockExchange(0, 1, ranks);
}

TEST_F(LcalSockExchangeTest, AllGatherOK)
{
    class MockLcalSockExchange : public LcalSockExchange {
    public:
        MockLcalSockExchange(int rank, int rankSize, LcalUniqueId lcalCommId)
            : LcalSockExchange(rank, rankSize, lcalCommId) {};

        int Prepare() override
        {
            return LCAL_SUCCESS;
        }
    };

    auto sock = MockLcalSockExchange(0, 1, LcalUniqueId{});

    auto sendBuf = string("test");
    auto recvBuf = new char[sendBuf.length()];

    auto res = sock.AllGather(sendBuf.c_str(), 4, recvBuf);
    ASSERT_EQ(res, LCAL_SUCCESS);

    delete[] recvBuf;
}

// tf DT用例环境出错，torch需求转测暂时屏蔽。
TEST_F(LcalSockExchangeTest, DISABLED_GetNodeNumOK)
{
    auto ranks = vector<int>{0};
    auto sock = LcalSockExchange(0, 1, ranks);

    auto res = sock.GetNodeNum();
    ASSERT_EQ(res, 1);
}

TEST(LcalSockExchange, ParseIpAndPortOK)
{
    auto ip = string();
    uint16_t port = 0;
    auto input = "127.0.0.1:8080";
    auto res = ParseIpAndPort(input, ip, port);
    EXPECT_EQ(res, LCAL_SUCCESS);
}

TEST_F(LcalSockExchangeTest, GetAddrFromStringOK)
{
    auto ua = LcalSocketAddress();
    auto ipPortPair = "127.0.0.1:8080";

    auto res = GetAddrFromString(&ua, ipPortPair);
    ASSERT_EQ(res, LCAL_SUCCESS);
}

TEST_F(LcalSockExchangeTest, BootstrapGetServerIpOK)
{
    auto ua = LcalSocketAddress();
    auto res = BootstrapGetServerIp(ua);
    ASSERT_EQ(res, LCAL_SUCCESS);
}

TEST(LcalSockExchange, ValidateIPv4Address_ValidIPs)
{
    const string valid_ips[] = {
        "127.0.0.1",
        "1.2.3.4",
        "10.0.0.1",
        "192.168.0.1",
        "172.16.0.1",
        "255.255.255.255",
        "0.0.0.1"
    };
    for (const auto& ip : valid_ips) {
        EXPECT_EQ(ValidateIPv4Address(ip), LCAL_SUCCESS);
    }
}

TEST(LcalSockExchange, ValidateIPv4Address_InvalidIPs_NoThrow)
{
    const string invalid_ips[] = {
        "",                 // 空字符串
        " ",                // 仅空格
        "1.2.3",            // 段数不足
        "1.2.3.4.5",        // 段数过多
        "256.0.0.1",        // 超范围
        "1.2.3.-1",         // 非法字符
        "1.2.3.4 ",         // 尾部空格
        " 1.2.3.4",         // 头部空格
        "localhost",        // 主机名
        "fe80::1",          // IPv6
        "::",               // IPv6
        "::1",              // IPv6
        "2001:db8::",       // IPv6
        "[::1]"             // bracket IPv6
        "01.002.003.004"    // 前导零，inet_pton 认为无效IP
        "00.0.0.0"          // 前导零，inet_pton 认为无效IP
    };
    for (const auto& ip : invalid_ips) {
        EXPECT_EQ(ValidateIPv4Address(ip), LCAL_ERROR_INTERNAL);
    }
}

TEST(LcalSockExchange, ValidateIPv4Address_ZeroIP_Throws)
{
    EXPECT_THROW(ValidateIPv4Address("0.0.0.0"), std::invalid_argument);
}

}  // namespace Lcal
