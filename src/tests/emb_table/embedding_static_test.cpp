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
#include "emb_table/embedding_static.h"

using namespace std;
using namespace MxRec;
using namespace testing;
using namespace tensorflow;

class EmbeddingStaticTest : public testing::Test {
protected:
    EmbeddingStaticTest()
    {
        struct EmbInfoParams embParam(string("test1"), 0, 1000, 2000, true, true);
        std::vector<size_t> vocabsize = {100, 100, 100};
        vector<EmbCache::InitializerInfo> initializeInfos = {};
        std::vector<std::string> ssdDataPath = {""};
        vector<int> maxStep = {1000};
        embInfo_ = EmbInfo(embParam, vocabsize, initializeInfos, ssdDataPath);
        rankInfo_ = RankInfo(0, 0, 0, 1, maxStep);
    }

    void SetUp() {
    }
    void TearDown() {
    }
    static void TearDownTestCase() {
    }

    EmbInfo embInfo_;
    RankInfo rankInfo_;
};

/**
 * 正常情况，将表装满
 */
TEST_F(EmbeddingStaticTest, Key2OffsetBasic)
{
    vector<EmbInfo> embInfos = {embInfo_};

    shared_ptr<EmbeddingStatic> table = std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);

    vector<emb_key_t> tmp1;
    for (size_t i = 0; i < 100; ++i) {
        tmp1.push_back(i);
    }

    table->Key2Offset(tmp1, TRAIN_CHANNEL_ID);
    for (size_t i = 0; i < tmp1.size(); ++i) {
        EXPECT_EQ(tmp1[i], i);
    }
    EXPECT_EQ(table->size(), 100);
    EXPECT_EQ(table->size(), table->GetMaxOffset());
    EXPECT_EQ(table->capacity(), 100);
}

/**
 * 边界条件:101超过100容量
 */
TEST_F(EmbeddingStaticTest, Key2OffsetOverflow)
{
    vector<EmbInfo> embInfos = {embInfo_};

    shared_ptr<EmbeddingStatic> table= std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);

    vector<emb_key_t> tmp1;
    for (size_t i = 0; i < 101; ++i) {
        tmp1.push_back(i);
    }
    int exp = 0;
    try {
        table->Key2Offset(tmp1, TRAIN_CHANNEL_ID);
    } catch (exception& e) {
        exp = 1;
    }

    EXPECT_EQ(table->capacity(), 100);
    EXPECT_EQ(exp, 1);
}

/**
 * 异常1: 使用eval channel
 */
TEST_F(EmbeddingStaticTest, Key2OffsetEvalChannel)
{
    vector<EmbInfo> embInfos = {embInfo_};

    shared_ptr<EmbeddingStatic> table = std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);

    vector<emb_key_t> testData;
    for (size_t i = 0; i < 100; ++i) {
        testData.push_back(i);
    }
    table->Key2Offset(testData, EVAL_CHANNEL_ID);
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(testData[i], INVALID_KEY_VALUE);
    }
}

/**
 * 正常: 使用淘汰的位置
 */
TEST_F(EmbeddingStaticTest, Key2OffsetEvict)
{
    vector<EmbInfo> embInfos = {embInfo_};
    shared_ptr<EmbeddingStatic> table = std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);

    constexpr size_t tableNum = 100;
    constexpr size_t testNum = 10;

    vector<emb_key_t> testData;
    for (size_t i = 0; i < tableNum; ++i) {
        testData.push_back(i);
    }
    table->Key2Offset(testData, TRAIN_CHANNEL_ID);
    // 全部淘汰
    vector<emb_cache_key_t> testDataAdapt(testData.cbegin(), testData.cend());
    table->EvictKeys(testDataAdapt);

    vector<emb_key_t> new_data;
    for (size_t i = 0; i < testNum; ++i) {
        new_data.push_back(i);
    }
    table->Key2Offset(new_data, TRAIN_CHANNEL_ID);
    // 查看是否淘汰
    std::vector<int64_t> evicted_keys = table->GetEvictedKeys();
    EXPECT_EQ(evicted_keys.size(), tableNum - testNum);
}

/**
 * 测试key数据的保存和加载
 */
TEST_F(EmbeddingStaticTest, SaveKeyData)
{
    vector<EmbInfo> embInfos = {embInfo_};
    shared_ptr<EmbeddingStatic> hbm = std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);
    hbm->SetFileSystemPtr("test_dir");
    hbm->Save("test_dir");
    bool fileExist = false;
    if (access("./test_dir/test1/key", F_OK) == 0) {
        fileExist = true;
    }
    EXPECT_EQ(fileExist, true);
}