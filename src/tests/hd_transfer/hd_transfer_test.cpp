/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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
#include <emock/emock.hpp>

#include "hd_transfer/hd_transfer.h"

using namespace MxRec;
using namespace testing;

const size_t TEST_SIZE = 1024;

struct TransferChannelCompare {
    bool operator()(const TransferChannel& lhs, const TransferChannel& rhs) const
    {
        return static_cast<int>(lhs) < static_cast<int>(rhs);
    }
};

class HdTransferTest : public testing::Test {
public:
    HDTransfer m_hdTransfer;
    uint32_t m_localRankId{0};
    std::string m_embTableName = "table1";
    std::map<TransferChannel, std::string, TransferChannelCompare> m_channel2Str = {
        {TransferChannel::RESTORE_SECOND, "restore_second"},
        {TransferChannel::D2H,            "d2h"},
        {TransferChannel::RESTORE,        "restore"},
        {TransferChannel::ALL2ALL,        "all2all"},
        {TransferChannel::UNIQKEYS,       "uniquekeys"},
        {TransferChannel::LOOKUP,         "lookup"},
        {TransferChannel::MASK,           "mask"},
        {TransferChannel::EVICT,          "evict"},
        {TransferChannel::H2D,            "h2d"},
        {TransferChannel::SWAP,           "swap"},
        {TransferChannel::SAVE_D2H,       "save_d2h"},
        {TransferChannel::SAVE_H2D,       "save_h2d"},
        {TransferChannel::KEY_D2H,        "key_d2h"},
        {TransferChannel::RECVSHAPE,      "recvshape"}
    };

    void TearDown()
    {
        GlobalMockObject::reset();
    }
};

TEST_F(HdTransferTest, FullProcessHdTransferTest)
{
    std::vector<EmbInfo> embInfos;
    m_hdTransfer.Init(embInfos, m_localRankId, false, false);

    int channelNum{0};
    std::vector<Tensor> tensors;
    int batchId{0};
    m_hdTransfer.Send(TransferChannel::H2D, tensors, channelNum, m_embTableName, batchId);
    auto ret = m_hdTransfer.RecvAcl(TransferChannel::H2D, channelNum, m_embTableName, 0, batchId);
    EXPECT_EQ(ret, 0);

    ret = m_hdTransfer.RecvOffsetsAcl(TransferChannel::H2D, channelNum, m_embTableName);
    EXPECT_EQ(ret, 0);

    auto channels = m_hdTransfer.GetTransChannel();
    EXPECT_EQ(channels.size(), 0);

    auto usedChannels = m_hdTransfer.GetUsedTransChannel();
    EXPECT_EQ(usedChannels.size(), 0);

    m_hdTransfer.ClearTransChannel(channelNum);

    m_hdTransfer.Destroy();
}

TEST_F(HdTransferTest, RecvMteShm_Ok)
{
    HDTransfer hdTransfer;
    std::vector<EmbInfo> embInfos;
    hdTransfer.Init(embInfos, m_localRankId, false, false);

    std::string testName = "test";
    float* testPtr = nullptr;
    int64_t testDim0;
    int batchId{0};
    char dummy[TEST_SIZE];

    void* fakePtr = &dummy;
    EMOCK(GetHostAddr).stubs().will(returnValue(fakePtr));
    auto* mockData = new RmaShmData();
    EMOCK(ShmDequeuePre).stubs().will(returnValue(mockData));

    auto ret = hdTransfer.RecvMteShm(testName, testPtr, testDim0, batchId);
    EXPECT_EQ(ret, 0);

    hdTransfer.Destroy();
}

TEST_F(HdTransferTest, RecvMteShm_TrueEmptyFlag)
{
    HDTransfer hdTransfer;
    std::vector<EmbInfo> embInfos;
    hdTransfer.Init(embInfos, m_localRankId, false, false);

    std::string testName = "test";
    float* testPtr = nullptr;
    int64_t testDim;
    int batchId{0};
    char dummy[TEST_SIZE];

    void* fakePtr = &dummy;
    EMOCK(GetHostAddr).stubs().will(returnValue(fakePtr));
    EMOCK(ShmDequeuePre).stubs().will(returnValue((RmaShmData*)nullptr));

    auto ret = hdTransfer.RecvMteShm(testName, testPtr, testDim, batchId);
    EXPECT_EQ(ret, 0);

    hdTransfer.Destroy();
}

TEST_F(HdTransferTest, RecvMteShm_NullptrShmAddrError)
{
    HDTransfer hdTransfer;
    std::vector<EmbInfo> embInfos;
    hdTransfer.Init(embInfos, m_localRankId, false, false);
    std::string testName = "test";
    float* testPtr = nullptr;
    int64_t testDim;
    int batchId{0};

    EMOCK(GetHostAddr).stubs().will(returnValue((void*)nullptr));
    EXPECT_THROW(hdTransfer.RecvMteShm(testName, testPtr, testDim, batchId), std::exception);
}

TEST_F(HdTransferTest, DequeueShm_NullptrQueueHeader)
{
    int channelNum{0};
    EXPECT_THROW(m_hdTransfer.DequeueShm(TransferChannel::H2D, channelNum, m_embTableName), std::runtime_error);
}

TEST_F(HdTransferTest, TransferChannel2Str)
{
    for (const auto& it : m_channel2Str) {
        auto ret = TransferChannel2Str(it.first);
        EXPECT_EQ(ret, it.second);
    }
}

TEST_F(HdTransferTest, GetShmAddr)
{
    std::string name = "test";
    int deviceId = 1;
    int capacity = 51;

    EMOCK(GetChipName).stubs().will(returnValue(std::string("testDeviceName")));
    EXPECT_THROW(GetShmAddr(name, deviceId, capacity), std::runtime_error);
}