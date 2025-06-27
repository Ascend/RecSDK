/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "hash_table/fast_hashmap.h"
#include "utils/string_tools.h"

#include "../common_main.h"

using namespace Embcache;

class FastHashMapTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        bool success = hashmap.Init(reserve);
        LOG(INFO) << std::boolalpha << "after Init, success:" << success;
    }

    void TearDown() override
    {
        hashmap.Destroy();
        LOG(INFO) << "Destroy finished.";
    }

    FastHashMap hashmap;
    uint64_t reserve = 256;
};

TEST_F(FastHashMapTest, FindOrInsert)
{
    std::function<BeforePutFuncState()> beforePutFunc = []() {
        return BeforePutFuncState::BEFORE_SUCCESS;
    };
    uint64_t key = 1;
    uint64_t value = 1;
    FkvState state = hashmap.FindOrInsert(key, value, beforePutFunc);
    LOG(INFO) << "FindOrInsert (first time, actually PutKeyValue), state:" << FkvStateStr[(int)state];
    ASSERT_EQ(state, FkvState::FKV_NOT_EXIST);

    value = -1;
    state = hashmap.FindOrInsert(key, value, beforePutFunc);
    LOG(INFO) << "FindOrInsert (second time, actually Find), state:" << FkvStateStr[(int)state] << ", value:" << value;
    ASSERT_EQ(state, FkvState::FKV_EXIST);
    ASSERT_EQ(value, 1);
}

TEST_F(FastHashMapTest, Find1)
{
    uint64_t key = 1;
    uint64_t value = -1;
    bool success = hashmap.Find(key, value);
    LOG(INFO) << std::boolalpha << "Find1, success:" << success;
    ASSERT_EQ(success, false);
}

TEST_F(FastHashMapTest, Find2)
{
    std::function<BeforePutFuncState()> beforePutFunc = []() {
        return BeforePutFuncState::BEFORE_SUCCESS;
    };
    uint64_t key = 1;
    uint64_t value = 1;
    FkvState state = hashmap.FindOrInsert(key, value, beforePutFunc);
    LOG(INFO) << "FindOrInsert (first time, actually PutKeyValue), state:" << FkvStateStr[(int)state];
    ASSERT_EQ(state, FkvState::FKV_NOT_EXIST);

    value = -1;
    bool success = hashmap.Find(key, value);
    LOG(INFO) << std::boolalpha << "Find2, success:" << success << ", value:" << value;
    ASSERT_EQ(success, true);
    ASSERT_EQ(value, 1);
}

TEST_F(FastHashMapTest, Remove_Not_EXIST)
{
    uint64_t key = 999;
    FkvState state = hashmap.Remove(key);
    LOG(INFO) << "Remove_Not_EXIST, state:" << FkvStateStr[(int)state];
    ASSERT_EQ(state, FkvState::FKV_NOT_EXIST);
}

TEST_F(FastHashMapTest, Remove_EXIST)
{
    std::function<BeforePutFuncState()> beforePutFunc = []() {
        return BeforePutFuncState::BEFORE_SUCCESS;
    };
    uint64_t key = 1;
    uint64_t value = 1;
    FkvState state = hashmap.FindOrInsert(key, value, beforePutFunc);
    LOG(INFO) << "FindOrInsert (first time, actually PutKeyValue), state:" << FkvStateStr[(int)state];
    ASSERT_EQ(state, FkvState::FKV_NOT_EXIST);

    state = hashmap.Remove(key);
    LOG(INFO) << "Remove_EXIST, state:" << FkvStateStr[(int)state];
    ASSERT_EQ(state, FkvState::FKV_EXIST);
}

TEST_F(FastHashMapTest, Export)
{
    std::function<BeforePutFuncState()> beforePutFunc = []() {
        return BeforePutFuncState::BEFORE_SUCCESS;
    };

    uint64_t key = 1;
    uint64_t value = 1;
    FkvState state = hashmap.FindOrInsert(key, value, beforePutFunc);
    LOG(INFO) << "FindOrInsert (first time, actually PutKeyValue), state:" << FkvStateStr[(int)state];
    ASSERT_EQ(state, FkvState::FKV_NOT_EXIST);

    key = 2;
    value = 2;
    state = hashmap.FindOrInsert(key, value, beforePutFunc);
    LOG(INFO) << "FindOrInsert (second time, actually PutKeyValue), state:" << FkvStateStr[(int)state];
    ASSERT_EQ(state, FkvState::FKV_NOT_EXIST);

    std::vector<std::pair<uint64_t, uint64_t>> results = hashmap.Export();
    LOG(INFO) << "Export, results:" << StringTools::ToString(results);
    std::vector<std::pair<uint64_t, uint64_t>> expected = {{1, 1}, {2, 2}};
    ASSERT_EQ(expected, results);
}

int main(int argc, char* argv[])
{
    return CommonMain(argc, argv);
}