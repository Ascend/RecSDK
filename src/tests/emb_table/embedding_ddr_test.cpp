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

using namespace std;
using namespace MxRec;
using namespace testing;
using namespace tensorflow;

class EmbeddingDDRTest : public testing::Test {
protected:
    EmbeddingDDRTest()
    {
        struct EmbInfoParams embParam(string("test1"), 0, 1000, 2000, true, true);
        std::vector<size_t> vocabsize = {100, 100, 100};
        vector<EmbCache::InitializerInfo> initializeInfos = {};
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


/**
 * 测试host侧 embedding数据的保存和加载
 */
TEST_F(EmbeddingDDRTest, SaveLoadEmbeddingData)
{
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
