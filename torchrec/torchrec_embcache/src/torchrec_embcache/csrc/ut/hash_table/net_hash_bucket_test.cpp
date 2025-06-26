/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "hash_table/hash_bucket.h"

#include "../common_main.h"

using namespace Embcache;

class NetHashBucketTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        uint64_t key = 1;
        uint64_t value = 1;
        std::function<BeforePutFuncState()> beforePutFunc = []() {
            return BeforePutFuncState::BEFORE_SUCCESS;
        };
        FkvState state = bucket.Put(key, value, beforePutFunc);
        LOG(INFO) << "after Put, state:" << FkvStateStr[(int)state];
    }

    void TearDown() override {}

    NetHashBucket bucket;
};

TEST_F(NetHashBucketTest, Put)
{
    uint64_t key = 1;
    uint64_t value = 256;
    std::function<BeforePutFuncState()> beforePutFunc = []() {
        return BeforePutFuncState::BEFORE_SUCCESS;
    };
    FkvState state = bucket.Put(key, value, beforePutFunc);
    LOG(INFO) << "after Put (with the same key), state:" << FkvStateStr[(int)state];
    ASSERT_EQ(state, FkvState::FKV_NOT_EXIST);
}

TEST_F(NetHashBucketTest, Find)
{
    uint64_t key = 1;
    uint64_t valueFound = -1;
    bool isFound = bucket.Find(key, valueFound);
    LOG(INFO) << std::boolalpha << "after Find, isFound:" << isFound << ", valueFound:" << valueFound;
    ASSERT_EQ(valueFound, 1);
}

TEST_F(NetHashBucketTest, Remove)
{
    uint64_t key = 1;
    FkvState state = bucket.Remove(key);
    LOG(INFO) << "after Remove, state:" << FkvStateStr[(int)state];
    ASSERT_EQ(state, FkvState::FKV_EXIST);
}

int main(int argc, char* argv[])
{
    return CommonMain(argc, argv);
}