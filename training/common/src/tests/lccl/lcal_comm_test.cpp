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

#include <cstdint>
#include <string>

#include <emock/emock.hpp>
#include <gtest/gtest.h>
#include <mpi.h>
#include <runtime/base.h>
#include <runtime/dev.h>
#include <runtime/mem.h>

#include "lccl/include/lcal_comm.h"
#include "lccl/include/comm_args.h"
#include "lccl/include/lcal_api.h"
#include "lccl/src/tools/socket/lcal_sock_exchange.h"
#include "lccl/include/lcal_types.h"

namespace Lcal {

using std::string;
using std::vector;

class LcalCommTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        emock::GlobalMockObject::reset();
    }
};

TEST_F(LcalCommTest, ConstructAndGetOk)
{
    auto ranks = vector<int>{0};
    auto comm0 = LcalComm(0, 1, ranks);
    EXPECT_EQ(comm0.GetRank(), 0);
    EXPECT_EQ(comm0.GetRankSize(), 1);

    auto comm1 = LcalComm(0, 1, LcalUniqueId{});
    EXPECT_EQ(comm1.GetCommSize(), 0);
    EXPECT_EQ(comm1.GetCommArgsPtr(), nullptr);
    EXPECT_EQ(comm1.GetPhysicalInfo().coreNum, 0);
}

TEST_F(LcalCommTest, InitOk)
{
    class MockLcalComm : public LcalComm {
    public:
        MockLcalComm(int rank, int rankSize) : LcalComm(rank, rankSize) {};

        int GatherDevId() override
        {
            return LCAL_SUCCESS;
        }

        int InitCommMem() override
        {
            return LCAL_SUCCESS;
        }

        int SyncCommArgs() override
        {
            return LCAL_SUCCESS;
        }
    };

    auto comm = MockLcalComm(0, 1);
    EXPECT_EQ(comm.Init(), LCAL_SUCCESS);
}

TEST_F(LcalCommTest, InitErr_InvalidRank)
{
    auto comm = LcalComm(-1, 1);
    EXPECT_EQ(comm.Init(), LCAL_ERROR_PARA_CHECK_FAIL);
}

TEST_F(LcalCommTest, InitErr_GatherDevIdFailure)
{
    class MockLcalComm : public LcalComm {
    public:
        MockLcalComm(int rank, int rankSize) : LcalComm(rank, rankSize) {};

        int GatherDevId() override
        {
            return LCAL_ERROR_INTERNAL;
        }
    };

    auto comm = MockLcalComm(0, 1);
    EXPECT_EQ(comm.Init(), LCAL_ERROR_INTERNAL);
}

TEST_F(LcalCommTest, InitErr_InitCommFailure)
{
    class MockLcalComm : public LcalComm {
    public:
        MockLcalComm(int rank, int rankSize) : LcalComm(rank, rankSize) {};

        int GatherDevId() override
        {
            return LCAL_SUCCESS;
        }

        int InitCommon() override
        {
            return LCAL_ERROR_INTERNAL;
        }
    };

    auto comm = MockLcalComm(0, 1);
    EXPECT_EQ(comm.Init(), LCAL_ERROR_INTERNAL);
}

TEST_F(LcalCommTest, InitErr_InitCommMemFailure)
{
    class MockLcalComm : public LcalComm {
    public:
        MockLcalComm(int rank, int rankSize) : LcalComm(rank, rankSize) {};

        int GatherDevId() override
        {
            return LCAL_SUCCESS;
        }

        int InitCommMem() override
        {
            return LCAL_ERROR_INTERNAL;
        }
    };

    auto comm = MockLcalComm(0, 1);
    EXPECT_EQ(comm.Init(), LCAL_ERROR_INTERNAL);
}

TEST_F(LcalCommTest, InitCommErr)
{
    class MockLcalComm : public LcalComm {
    public:
        MockLcalComm(int rank, int rankSize) : LcalComm(rank, rankSize) {};

        int EnablePeerAccess() override
        {
            return LCAL_ERROR_INTERNAL;
        }
    };

    auto comm = MockLcalComm(0, 1);
    EXPECT_EQ(comm.InitCommon(), LCAL_ERROR_INTERNAL);
}

TEST_F(LcalCommTest, InitThreadErr_aclrtMallocFailure)
{
    EMOCK(aclrtMalloc).stubs().will(returnValue(ACL_ERROR_RT_MEMORY_ALLOCATION));

    auto comm = LcalComm(0, 1);
    EXPECT_EQ(comm.InitThread(), LCAL_ERROR_INTERNAL);
}

TEST_F(LcalCommTest, InitThreadErr_InvalidRank)
{
    auto comm = LcalComm(-1, 1);
    EXPECT_EQ(comm.InitThread(), LCAL_ERROR_PARA_CHECK_FAIL);
}

TEST_F(LcalCommTest, InitThreadErr_GatherDevIdFailure)
{
    class MockLcalComm : public LcalComm {
    public:
        MockLcalComm(int rank, int rankSize) : LcalComm(rank, rankSize) {};

        int GatherDevIdThread() override
        {
            return LCAL_ERROR_INTERNAL;
        }
    };

    auto comm = MockLcalComm(0, 1);
    EXPECT_EQ(comm.InitThread(), LCAL_ERROR_INTERNAL);
}

TEST_F(LcalCommTest, InitThreadErr_InitCommonFailure)
{
    class MockLcalComm : public LcalComm {
    public:
        MockLcalComm(int rank, int rankSize) : LcalComm(rank, rankSize) {};

        int GatherDevIdThread() override
        {
            return LCAL_SUCCESS;
        }

        int InitCommon() override
        {
            return LCAL_ERROR_INTERNAL;
        }
    };

    auto comm = MockLcalComm(0, 1);
    EXPECT_EQ(comm.InitThread(), LCAL_ERROR_INTERNAL);
}

TEST_F(LcalCommTest, EnablePeerAccessOk)
{
    EMOCK(aclrtDeviceEnablePeerAccess).stubs().with(any()).will(returnValue(ACL_SUCCESS));

    auto comm = LcalComm(0, 1);
    comm.devId_ = 1;
    comm.devList_ = vector<int>{0};
    EXPECT_EQ(comm.EnablePeerAccess(), LCAL_SUCCESS);
}

aclError mockRtGetPairDevicesInfo(uint32_t devId, uint32_t peerDevId, int32_t flags, int64_t* value)
{
    constexpr auto kInvalidTopologyType = -1;
    *value = kInvalidTopologyType;
    return 0;
}

TEST_F(LcalCommTest, EnablePeerAccessErr_InvalidRankSizeForPCIE)
{
    EMOCK(SkipUnusedChannel910B2C).stubs().with(any()).will(returnValue(false));
    EMOCK(rtGetPairDevicesInfo).stubs().will(invoke(mockRtGetPairDevicesInfo));
    EMOCK(aclrtDeviceEnablePeerAccess).stubs().with(any()).will(returnValue(ACL_SUCCESS));

    auto comm = LcalComm(0, 1);
    comm.devId_ = 1;
    comm.devList_ = vector<int>{0};
    comm.physicalInfo_.physicalLink = PhysicalLink::RESERVED;
    comm.rankSize_ = PING_PONG_SIZE + 1;
    EXPECT_EQ(comm.EnablePeerAccess(), LCAL_ERROR_INTERNAL);
}

TEST_F(LcalCommTest, CommArgsSetBuffOk)
{
    auto commArgs = CommArgs();
    auto buff = new int8_t* [1];
    buff[0] = nullptr;

    commArgs.SetBuff(buff);
    EXPECT_EQ(commArgs.peerMems[0], nullptr);
}

TEST_F(LcalCommTest, CommArgsSetBuffErr)
{
    auto commArgs = CommArgs();
    auto buff = new int8_t* [1];
    buff[0] = nullptr;

    commArgs.rankSize = LCAL_MAX_RANK_SIZE + 1;
    EXPECT_THROW(commArgs.SetBuff(buff), std::invalid_argument);
}

TEST_F(LcalCommTest, GatherDevId)
{
    class MockLcalSockExchange : public LcalSockExchange {
    public:
        MockLcalSockExchange(int rank, int rankSize, LcalUniqueId lcalCommId)
            : LcalSockExchange(rank, rankSize, lcalCommId) {};

        int GetNodeNum() override
        {
            return 1;
        }

        int AllGather(const void* sendBuf, size_t sendSize, void* recvBuf) override
        {
            return LCAL_SUCCESS;
        }
    };
    EMOCK(aclrtGetDevice).stubs().with(any()).will(returnValue(ACL_SUCCESS));

    auto comm = LcalComm(0, 1);
    comm.socketExchange_ = new MockLcalSockExchange(0, 1, LcalUniqueId{});

    EXPECT_EQ(comm.GatherDevId(), 0);
}

TEST_F(LcalCommTest, InitMemOk)
{
    EMOCK(aclrtMalloc).stubs().will(returnValue(ACL_SUCCESS));
    EMOCK(aclrtMemset).stubs().will(returnValue(ACL_SUCCESS));

    auto comm = LcalComm(0, 1);
    EXPECT_EQ(comm.InitMem(), LCAL_SUCCESS);
}

TEST_F(LcalCommTest, GetSidIdOk)
{
    class MockLcalSockExchange : public LcalSockExchange {
    public:
        MockLcalSockExchange(int rank, int rankSize, LcalUniqueId lcalCommId)
            : LcalSockExchange(rank, rankSize, lcalCommId) {};

        int GetNodeNum() override
        {
            return 1;
        }

        int AllGather(const void* sendBuf, size_t sendSize, void* recvBuf) override
        {
            return LCAL_SUCCESS;
        }
    };
    class MockLcalComm : public LcalComm {
    public:
        MockLcalComm(int rank, int rankSize) : LcalComm(rank, rankSize) {};

        void CloseIpcMem() override
        {
            return;
        }
    };
    EMOCK(rtGetDeviceInfo).stubs().with(any()).will(returnValue(ACL_SUCCESS));

    auto comm = LcalComm(0, 1);
    comm.socketExchange_ = new MockLcalSockExchange(0, 1, LcalUniqueId{});
    comm.devList_ = vector<int>{0};
    comm.physicalInfo_.chipName = ChipName::CHIP_910_9391;

    int64_t sdids[LCAL_MAX_RANK_SIZE] = {0};
    EXPECT_EQ(comm.GetSidId(sdids), LCAL_SUCCESS);
}

TEST_F(LcalCommTest, GetPidOk)
{
    class MockLcalSockExchange : public LcalSockExchange {
    public:
        MockLcalSockExchange(int rank, int rankSize, LcalUniqueId lcalCommId)
            : LcalSockExchange(rank, rankSize, lcalCommId) {};

        int AllGather(const void* sendBuf, size_t sendSize, void* recvBuf) override
        {
            return LCAL_SUCCESS;
        }
    };

    EMOCK(rtDeviceGetBareTgid).stubs().with(any()).will(returnValue(ACL_SUCCESS));
    auto comm = LcalComm(0, 1);
    comm.socketExchange_ = new MockLcalSockExchange(0, 1, LcalUniqueId{});

    auto pid = new uint32_t[LCAL_MAX_RANK_SIZE]();
    EXPECT_EQ(comm.GetPid(pid), LCAL_SUCCESS);
}

TEST_F(LcalCommTest, GetNameOk)
{
    class MockLcalSockExchange : public LcalSockExchange {
    public:
        MockLcalSockExchange(int rank, int rankSize, LcalUniqueId lcalCommId)
            : LcalSockExchange(rank, rankSize, lcalCommId) {};

        int AllGather(const void* sendBuf, size_t sendSize, void* recvBuf) override
        {
            return LCAL_SUCCESS;
        }
    };

    auto comm = LcalComm(0, 1);
    comm.socketExchange_ = new MockLcalSockExchange(0, 1, LcalUniqueId{});

    auto name = string("test");
    auto names = new char[LCAL_MAX_RANK_SIZE][IPC_NAME_SIZE]();
    EXPECT_EQ(comm.GetName(name, names), LCAL_SUCCESS);
}

TEST_F(LcalCommTest, InitCommMemOk)
{
    class MockLcalComm : public LcalComm {
    public:
        MockLcalComm(int rank, int rankSize) : LcalComm(rank, rankSize) {};
        int InitMem() override
        {
            return LCAL_SUCCESS;
        }
        int GetPid(uint32_t pids[LCAL_MAX_RANK_SIZE]) override
        {
            return LCAL_SUCCESS;
        }
        int GetSidId(int64_t sdids[LCAL_MAX_RANK_SIZE]) override
        {
            return LCAL_SUCCESS;
        }
        int SetMemoryName(std::string& name) override
        {
            return LCAL_SUCCESS;
        }
        int SetIpcPidSdid(std::string& name, const uint32_t* pids, const int64_t* sdids) const override
        {
            return LCAL_SUCCESS;
        }
        int GetName(std::string& name, char names[LCAL_MAX_RANK_SIZE][IPC_NAME_SIZE]) override
        {
            return LCAL_SUCCESS;
        }
        int OpenIpcMem(const char names[LCAL_MAX_RANK_SIZE][IPC_NAME_SIZE]) override
        {
            return LCAL_SUCCESS;
        }
    };
    auto comm = MockLcalComm(0, 1);
    EXPECT_EQ(comm.InitCommMem(), ACL_SUCCESS);
}

TEST_F(LcalCommTest, CloseIpcMemOk)
{
    EMOCK(rtIpcCloseMemory).expects(once()).will(returnValue(ACL_SUCCESS));

    auto comm = LcalComm(0, 2);
    comm.peerMem_[1] = new int8_t[0];
    comm.CloseIpcMem();
}

TEST_F(LcalCommTest, OpenIpcMemOk)
{
    EMOCK(GetChipName)
        .expects(exactly(3))
        .will(returnObjectList(ChipName::CHIP_910B2C, ChipName::CHIP_910_9361, ChipName::CHIP_910_9361));
    EMOCK(SkipUnusedChannel910B2C).stubs().will(returnValue(false));
    EMOCK(rtIpcOpenMemory).stubs().with(any()).will(returnValue(ACL_SUCCESS));

    auto comm = LcalComm(0, 2);
    auto names = new char[LCAL_MAX_RANK_SIZE][IPC_NAME_SIZE]();

    EXPECT_EQ(comm.OpenIpcMem(names), LCAL_SUCCESS);
    EXPECT_EQ(comm.OpenIpcMem(names), LCAL_SUCCESS);
}

TEST_F(LcalCommTest, SetMemoryNameOk)
{
    EMOCK(GetChipName).expects(once()).will(returnValue(ChipName::CHIP_910_9382));
    EMOCK(rtIpcSetMemoryName).expects(once()).will(returnValue(ACL_SUCCESS));

    auto comm = LcalComm(0, 1);
    auto memName = string("test");
    auto res = comm.SetMemoryName(memName);

    EXPECT_EQ(res, LCAL_SUCCESS);
}

TEST_F(LcalCommTest, SetIpcPidSdidOk)
{
    EMOCK(rtSetIpcMemPid).expects(exactly(2)).will(returnValue(ACL_SUCCESS));
    EMOCK(rtSetIpcMemorySuperPodPid).expects(exactly(2)).will(returnValue(ACL_SUCCESS));

    auto name = string("test");
    auto pids = new uint32_t[LCAL_MAX_RANK_SIZE]();
    auto sdids = new int64_t[LCAL_MAX_RANK_SIZE]();

    auto comm = LcalComm(1, 2);
    comm.physicalInfo_.chipName = ChipName::CHIP_910B2C;

    auto res = comm.SetIpcPidSdid(name, pids, sdids);
    EXPECT_EQ(res, LCAL_SUCCESS);

    comm.physicalInfo_.chipName = ChipName::CHIP_910_9361;
    res = comm.SetIpcPidSdid(name, pids, sdids);
    EXPECT_EQ(res, LCAL_SUCCESS);
}

TEST_F(LcalCommTest, SyncCommArgsErr_aclrtMallocFailure)
{
    EMOCK(aclrtMalloc).expects(once()).will(returnValue(ACL_ERROR_RT_MEMORY_ALLOCATION));

    auto comm = LcalComm(0, 1);
    EXPECT_EQ(comm.SyncCommArgs(), LCAL_ERROR_INTERNAL);
}

TEST_F(LcalCommTest, SyncCommArgsErr_aclrtMemcpyFailure)
{
    EMOCK(aclrtMalloc).expects(once()).will(returnValue(ACL_SUCCESS));
    EMOCK(aclrtMemcpy).expects(once()).will(returnValue(ACL_ERROR_RT_PARAM_INVALID));

    auto comm = LcalComm(0, 1);
    EXPECT_EQ(comm.SyncCommArgs(), LCAL_ERROR_INTERNAL);
}

TEST_F(LcalCommTest, GetCoreNumOk)
{
    constexpr int AI_CORE_NUM_24 = 24;
    constexpr int AI_CORE_NUM_20 = 20;
    constexpr int AI_CORE_NUM_2 = 2;

    auto res = GetCoreNum(ChipName::CHIP_910B2C);
    EXPECT_EQ(res, AI_CORE_NUM_24);

    res = GetCoreNum(ChipName::CHIP_910B3);
    EXPECT_EQ(res, AI_CORE_NUM_20);

    res = GetCoreNum(ChipName::CHIP_310P3);
    EXPECT_EQ(res, AI_CORE_NUM_2);
}

TEST_F(LcalCommTest, SkipUnusedChannel910B2COk)
{
    auto comm = LcalComm(0, 2);
    EXPECT_TRUE(SkipUnusedChannel910B2C(1, 8, ChipName::CHIP_910B2C));
}

}  // namespace Lcal
