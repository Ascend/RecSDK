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

#include "lcal_sock_exchange.h"

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

TEST_F(LcalSockExchangeTest, GetNodeNumOK)
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

}  // namespace Lcal
