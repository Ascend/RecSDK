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

#include <random>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <easy/profiler.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <limits>
#include "utils/common.h"
#include "emb_table/emb_table.h"

using namespace std;
using namespace MxRec;
using namespace testing;
using namespace tensorflow;

class EmbTableTest : public testing::Test {
protected:
    void SetUp()
    {
        // 设置测试用的EmbInfo
        embInfo.extEmbeddingSize = embTable.TEST_EMB_SIZE;
        LOG_INFO("EmbTable BLOCK_EMB_COUNT {} INIT_BLOCK_COUNT {}",
            embTable.BLOCK_EMB_COUNT, embTable.INIT_BLOCK_COUNT);
        rankInfo.rankId = 0;
        rankInfo.rankSize = 1;
        rankInfo.localRankSize = 1;
        rankInfo.useStatic = true;
        rankInfo.localRankId = 0;
        rankInfo.isDDR = true;
        rankInfo.maxStep = { 1, -1 };
        rankInfo.deviceId = 0;
        // 初始化EmbeddingTable
#ifndef GTEST
        LOG_INFO("rank {} running", rankInfo.deviceId);
        aclInit(nullptr);
#endif
    }

    EmbTable embTable;
    EmbInfo embInfo;
    RankInfo rankInfo;
    aclrtContext  context;

    void TearDown() {
    }
};

// 测试初始化是否正常
TEST_F(EmbTableTest, Init)
{
#ifndef GTEST
    // 测试初始化是否出现异常
    EXPECT_NO_THROW(embTable.Init(embInfo, rankInfo, 0));
    LOG_INFO("embTable Init succeed!");
    ASSERT_EQ(embTable.rankInfo.g_rankId, rankInfo.g_rankId);
    ASSERT_EQ(embTable.rankInfo.rankSize, rankInfo.rankSize);
    ASSERT_EQ(embTable.rankInfo.localRankSize, rankInfo.localRankSize);
    ASSERT_EQ(embTable.rankInfo.useStatic, rankInfo.useStatic);
    ASSERT_EQ(embTable.rankInfo.localRankId, rankInfo.localRankId);
    // 测试容量是否正常
    LOG_INFO("totalCapacity {}, INIT_BLOCK_COUNT {}", embTable.totalCapacity, embTable.INIT_BLOCK_COUNT);
    EXPECT_EQ(embTable.totalCapacity, embTable.INIT_BLOCK_COUNT * embTable.BLOCK_EMB_COUNT);
#endif
}

// 测试embedding list为空时的情况
TEST_F(EmbTableTest, GetEmbAddressEmptyList)
{
#ifndef GTEST
    embTable.Init(embInfo, rankInfo, 0);
    while (!embTable.embeddingList.empty()) {
        float *embAddr = reinterpret_cast<float*>(embTable.GetEmbAddress());
        EXPECT_NE(embAddr, nullptr);
    }
    ASSERT_EQ(embTable.embeddingList.size(), 0);

    float *curAddr = nullptr;
    int usedCapacityBefore = embTable.usedCapacity;
    ASSERT_NO_THROW({
        curAddr= reinterpret_cast<float*>(embTable.GetEmbAddress());
    });
    EXPECT_NE(curAddr, nullptr);
    EXPECT_EQ(embTable.usedCapacity, usedCapacityBefore + 1);
#endif
}

// 测试正常情况
TEST_F(EmbTableTest, GetEmbAddressNormal)
{
#ifndef GTEST
    embTable.Init(embInfo, rankInfo, 0);
    ASSERT_EQ(embTable.totalCapacity, embTable.INIT_BLOCK_COUNT);
    float *curAddr = nullptr;
    int totalCapacityBefore = embTable.totalCapacity;
    int usedCapacityBefore = embTable.usedCapacity;
    ASSERT_NO_THROW({
        curAddr = reinterpret_cast<float*>(embTable.GetEmbAddress());
    });
    EXPECT_NE(curAddr, nullptr);
    EXPECT_EQ(embTable.totalCapacity, totalCapacityBefore);
    EXPECT_EQ(embTable.usedCapacity, usedCapacityBefore + 1);
#endif
}

// 测试将一个emb地址放入embeddingList中,是否成功
TEST_F(EmbTableTest, PutEmbAddress)
{
#ifndef GTEST
    embTable.Init(embInfo, rankInfo, 0);
    int64_t curAddr;
    int usedCapacityBefore = embTable.usedCapacity;
    ASSERT_NO_THROW({
        curAddr = embTable.GetEmbAddress();
    });
    EXPECT_EQ(embTable.usedCapacity, usedCapacityBefore + 1);
    embTable.PutEmbAddress(curAddr);
    EXPECT_EQ(embTable.usedCapacity, usedCapacityBefore);
    EXPECT_EQ(curAddr, reinterpret_cast<int64_t>(embTable.embeddingList.back()));
#endif
}
