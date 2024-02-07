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
#include <mpi.h>
#include "utils/common.h"
#include "emb_table/emb_table.h"
#include "emb_table/embedding_ddr.h"
#include "host_emb/host_emb.h"

using namespace std;
using namespace MxRec;
using namespace testing;
using namespace tensorflow;

class EmbeddingDDRTest : public testing::Test {
protected:
    EmbeddingDDRTest()
    {
        struct EmbInfoParams embParam(string("test1"), 0, 1000, 2000, true, true);
        std::vector<size_t> vocabsize = {100};
        std::vector<InitializeInfo> initializeInfos = {};
        std::vector<std::string> ssdDataPath = {""};
        vector<int> maxStep = {1000};
        embInfo_ = EmbInfo(embParam, vocabsize, initializeInfos, ssdDataPath);
        int rankId;
        MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
        rankInfo_ = RankInfo(rankId, 0, 0, 1, maxStep);
    }

    void SetUp() {
    }
    void TearDown() {
    }

    static void SetupTestCase()
    {
        if (access("test_dir", F_OK) == 0) {
            system("rm -rf test_dir");
        }
    }

    static void TearDownTestCase()
    {
        if (access("test_dir", F_OK) == 0) {
            system("rm -rf test_dir");
        }
    }

    EmbInfo embInfo_;
    RankInfo rankInfo_;
};

TEST_F(EmbeddingDDRTest, SaveLoadBasic)
{
    vector<EmbInfo> embInfos = {embInfo_};
    HostEmb* hostEmbs = Singleton<MxRec::HostEmb>::GetInstance();
    hostEmbs->Initialize(embInfos, 0);
    HostEmbTable& table = hostEmbs->GetEmb("test1");

    shared_ptr<EmbeddingDDR> ddr1 = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    shared_ptr<EmbeddingDDR> ddr2 = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);

    // 使用时间构造测试数据
    ddr1->extEmbSize_ = time(nullptr);
    ddr1->devVocabSize_ = time(nullptr);
    ddr1->hostVocabSize_ = time(nullptr);
    ddr1->currentUpdatePos = time(nullptr);
    ddr1->maxOffset_ = time(nullptr);

    vector<emb_key_t> devOffset2KeyTestData;
    for (int i = 0; i < 10; ++i) {
        devOffset2KeyTestData.push_back(static_cast<emb_key_t>(i));
        ddr1->keyOffsetMap_[i] = i;
        ddr1->evictPos_.push_back(i);
    }

    ddr1->devOffset2Key = devOffset2KeyTestData;

    ddr1->Save("test_dir");
    ddr2->Load("test_dir");

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(ddr1->evictPos_[i], ddr2->evictPos_[i]);
    }

    EXPECT_EQ(ddr1->extEmbSize_, ddr2->extEmbSize_);
    EXPECT_EQ(ddr1->devVocabSize_, ddr2->devVocabSize_);
}

/**
 * 测试host侧 embedding数据的保存和加载
 */
TEST_F(EmbeddingDDRTest, SaveLoadEmbeddingData)
{
    vector<EmbInfo> embInfos = {embInfo_};
    HostEmb* hostEmbs = Singleton<MxRec::HostEmb>::GetInstance();
    hostEmbs->Initialize(embInfos, 0);
    HostEmbTable& table = hostEmbs->GetEmb("test1");

    vector<float> tmp1 {1.1, 2.1, 3.1};
    vector<float> tmp2 {1.2, 2.2, 3.2};
    vector<float> tmp3 {1.3, 2.3, 3.3};
    vector<vector<float>> testData;
    testData.push_back(tmp1);
    testData.push_back(tmp2);
    testData.push_back(tmp3);

    for (vector<float>& tmp : testData) {
        table.embData.push_back(tmp);
    }

    shared_ptr<EmbeddingDDR> ddr1 = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    shared_ptr<EmbeddingDDR> ddr2 = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    ddr1->Save("test_dir");
    // 修改成0
    for (vector<float>& tmp: table.embData) {
        for (float& t : tmp) {
            t = 0;
        }
    }
    ddr2->Load("test_dir");
    for (size_t i = 0; i < table.embData.size(); ++i) {
        for (size_t j = 0; j < table.embData[i].size(); ++j) {
            EXPECT_EQ(testData[i][j], table.embData[i][j]);
        }
    }
}

/**
 * 测试基本查找
 */
TEST_F(EmbeddingDDRTest, DDRBasic)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    const size_t testNum = 100;
    vector<emb_key_t> testKeys;
    vector<size_t> testSwap;
    for (size_t i = 0; i < testNum; ++i) {
        testKeys.push_back(i);
    }
    table->FindOffset(testKeys, 0, TRAIN_CHANNEL_ID, testSwap);
    EXPECT_EQ(testKeys.size(), 100);
    EXPECT_EQ(testSwap.size(), 0);
}

TEST_F(EmbeddingDDRTest, evict)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    const size_t testNum = 100;
    vector<emb_key_t> testKeys;
    vector<size_t> testSwap;
    for (size_t i = 0; i < testNum; ++i) {
        testKeys.push_back(i);
    }
    table->FindOffset(testKeys, 0, TRAIN_CHANNEL_ID, testSwap);
    table->EvictKeys(testKeys);
    EXPECT_EQ(table->evictPos_.size(), 100);
    EXPECT_EQ(testKeys.size(), 100);
    EXPECT_EQ(testSwap.size(), 0);
}

TEST_F(EmbeddingDDRTest, FindSwap)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    const size_t testNum = 100;
    vector<size_t> testSwap;
    table->FindSwapPosOld(0, 0, 0, testSwap);
    EXPECT_EQ(testSwap.size(), 1);
}

TEST_F(EmbeddingDDRTest, EvictDeleteEmb)
{
    shared_ptr<EmbeddingDDR> table = std::make_shared<EmbeddingDDR>(embInfo_, rankInfo_, 0);
    const size_t testNum = 100;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < testNum; ++i) {
        testKeys.push_back(i);
    }
    table->EvictDeleteEmb(testKeys);
    EXPECT_EQ(testKeys.size(), 100);
}
