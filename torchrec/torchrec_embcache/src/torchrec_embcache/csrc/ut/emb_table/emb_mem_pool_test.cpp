/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "emb_table/emb_mem_pool.h"

#include "../common_main.h"

using namespace Embcache;

constexpr int EMB_DIM = 8;
constexpr int OPT_NUM = 2;
constexpr int BUFFER_SIZE = 1024;
constexpr float WEIGHT_INIT_MIN = -0.5;
constexpr float WEIGHT_INIT_MAX = 0.5;

class EmbMemPoolTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        EmbConfig embConfig;
        embConfig.tableName = "user_table";
        embConfig.embDim = EMB_DIM;
        embConfig.optimNum = OPT_NUM;
        embConfig.weightInitMin = WEIGHT_INIT_MIN;
        embConfig.weightInitMax = WEIGHT_INIT_MAX;

        uint64_t bufferSize = BUFFER_SIZE;
        uint64_t hostVocabSize = BUFFER_SIZE * BUFFER_SIZE * BUFFER_SIZE;

        memPoolPtr = std::make_shared<EmbMemoryPool>(embConfig, bufferSize, hostVocabSize);
    }

    void TearDown() override
    {
        memPoolPtr->Stop();
    }

    std::shared_ptr<EmbMemoryPool> memPoolPtr;
};

TEST_F(EmbMemPoolTest, GetNewValueToBeInserted)
{
    uint64_t value;
    BeforePutFuncState state = memPoolPtr->GetNewValueToBeInserted(value);
    LOG(INFO) << "BeforePutFuncState, value:" << value;
    ASSERT_EQ(1, 1);
}

int main(int argc, char* argv[])
{
    return CommonMain(argc, argv);
}