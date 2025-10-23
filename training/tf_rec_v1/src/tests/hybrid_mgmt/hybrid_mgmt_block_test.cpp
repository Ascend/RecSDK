/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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

#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <thread>

#include "hybrid_mgmt/hybrid_mgmt_block.h"
#include "utils/common.h"
using namespace MxRec;
using namespace std::chrono_literals;

class HybridMgmtBlockTest : public testing::Test {
public:
    std::unique_ptr<HybridMgmtBlock> hybridMgmtBlock;
    std::vector<std::unique_ptr<std::thread>> procThreads {};
    bool isRunning = true;
protected:
    void SetUp()
    {
        LOG_DEBUG("start initialize") ;
    }
    int loop = 10;
};

TEST_F(HybridMgmtBlockTest, CheckAndDoBlock)
{
    int steps[] = {-1, 1};
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->SetStepInterval(1, 1);
    hybridMgmtBlock->CheckAndSetBlock(0);
    hybridMgmtBlock->CheckAndSetBlock(1);
    ASSERT_EQ(hybridMgmtBlock->GetBlockStatus(0), true);
}

TEST_F(HybridMgmtBlockTest, CheckAndSetBlock_NotDivisible)
{
    auto batchId = 3;
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->hybridBatchId[TRAIN_CHANNEL_ID] = batchId;
    auto trainStep = 2;
    hybridMgmtBlock->SetStepInterval(trainStep, 1);
    hybridMgmtBlock->CheckAndSetBlock(TRAIN_CHANNEL_ID);
    ASSERT_EQ(hybridMgmtBlock->GetBlockStatus(TRAIN_CHANNEL_ID), true);
}

TEST_F(HybridMgmtBlockTest, CountAndNotifyWake)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->SetStepInterval(1, 1);
    hybridMgmtBlock->CheckAndNotifyWake(0);
    hybridMgmtBlock->CountPythonStep(0, 1);
    hybridMgmtBlock->pythonBatchId[0] = 1;
    hybridMgmtBlock->hybridBatchId[0] = 0;
    auto fn = [this](int channelId) {
        hybridMgmtBlock->CheckAndNotifyWake(channelId);
        hybridMgmtBlock->CountPythonStep(0, 1);
        return 0;
    };
    procThreads.emplace_back(std::make_unique<std::thread>(fn, 0));
    std::this_thread::sleep_for(std::chrono::milliseconds(2ms));
    hybridMgmtBlock->hybridBatchId[0] = 1;
    for (auto p = procThreads.begin(); p != procThreads.end(); p++) {
        (*p)->join();
    }
}

TEST_F(HybridMgmtBlockTest, CheckAndNotifyWake_IsBlock)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->SetStepInterval(1, 1);
    hybridMgmtBlock->pythonBatchId[TRAIN_CHANNEL_ID] = 0;
    hybridMgmtBlock->hybridBatchId[TRAIN_CHANNEL_ID] = 1;
    hybridMgmtBlock->CheckAndNotifyWake(TRAIN_CHANNEL_ID);
    EXPECT_EQ(hybridMgmtBlock->GetBlockStatus(TRAIN_CHANNEL_ID), true);
}

TEST_F(HybridMgmtBlockTest, DoBlock)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->SetStepInterval(1, 1);
    hybridMgmtBlock->pythonBatchId[0] = 1;
    hybridMgmtBlock->hybridBatchId[0] = 1;
    auto fn = [this](int channelId) {
        hybridMgmtBlock->DoBlock(channelId);
        return 0;
    };
    procThreads.emplace_back(std::make_unique<std::thread>(fn, 0));
    std::this_thread::sleep_for(std::chrono::milliseconds(2ms));
    hybridMgmtBlock->SetBlockStatus(0, false);
    for (auto p = procThreads.begin(); p != procThreads.end(); p++) {
        (*p)->join();
    }
}

TEST_F(HybridMgmtBlockTest, DoBlock_NotRunning)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->isRunning = false;
    hybridMgmtBlock->SetStepInterval(1, 1);
    hybridMgmtBlock->pythonBatchId[0] = 1;
    hybridMgmtBlock->hybridBatchId[0] = 1;
    hybridMgmtBlock->DoBlock(TRAIN_CHANNEL_ID);
}

TEST_F(HybridMgmtBlockTest, ResetAll)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->SetStepInterval(1, 1);
    hybridMgmtBlock->ResetAll(0);
    ASSERT_EQ(hybridMgmtBlock->hybridBatchId[0], 0);
}

TEST_F(HybridMgmtBlockTest, CountPythonStep)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();

    hybridMgmtBlock->pythonBatchId[0] = 1;
    hybridMgmtBlock->loop[0] = 1;

    hybridMgmtBlock->CountPythonStep(0, loop);

    ASSERT_EQ(hybridMgmtBlock->pythonBatchId[0], loop + 1);
    ASSERT_EQ(hybridMgmtBlock->loop[0], loop);
}

TEST_F(HybridMgmtBlockTest, CheckAndSetBlockOnSave)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    RankInfo ri;
    ri.ctrlSteps = {1, 1, 1, 100};
    hybridMgmtBlock->SetRankInfo(ri);
    hybridMgmtBlock->hybridBatchId[TRAIN_CHANNEL_ID] = 1;
    hybridMgmtBlock->CheckAndSetBlock(TRAIN_CHANNEL_ID);
    EXPECT_EQ(hybridMgmtBlock->GetBlockStatus(TRAIN_CHANNEL_ID), true);
}

TEST_F(HybridMgmtBlockTest, CheckAndDonotBlockByInvalidStepsInterval)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->SetStepInterval(0, -1);
    hybridMgmtBlock->hybridBatchId[TRAIN_CHANNEL_ID] = 1;
    hybridMgmtBlock->hybridBatchId[EVAL_CHANNEL_ID] = 1;

    hybridMgmtBlock->CheckAndSetBlock(0);
    hybridMgmtBlock->CheckAndSetBlock(1);
    EXPECT_EQ(hybridMgmtBlock->GetBlockStatus(0), true);
    EXPECT_EQ(hybridMgmtBlock->GetBlockStatus(1), true);
}

TEST_F(HybridMgmtBlockTest, WaitValid)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->hybridBatchId[TRAIN_CHANNEL_ID] = 1;
    hybridMgmtBlock->pythonBatchId[TRAIN_CHANNEL_ID] = 0;
    auto ret = hybridMgmtBlock->WaitValid(TRAIN_CHANNEL_ID);
    EXPECT_EQ(ret, false);
}

TEST_F(HybridMgmtBlockTest, WaitValid_ReturnTrue)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->isRunning = false;
    hybridMgmtBlock->hybridBatchId[TRAIN_CHANNEL_ID] = 1;
    hybridMgmtBlock->pythonBatchId[TRAIN_CHANNEL_ID] = 1;

    EXPECT_EQ(hybridMgmtBlock->WaitValid(TRAIN_CHANNEL_ID), true);
}

TEST_F(HybridMgmtBlockTest, CheckValid_ChangeChennal_SameBatch)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->lastRunChannelId = TRAIN_CHANNEL_ID;
    hybridMgmtBlock->hybridBatchId[TRAIN_CHANNEL_ID] = 1;
    hybridMgmtBlock->pythonBatchId[TRAIN_CHANNEL_ID] = 1;

    hybridMgmtBlock->CheckValid(EVAL_CHANNEL_ID);
    EXPECT_EQ(hybridMgmtBlock->lastRunChannelId, EVAL_CHANNEL_ID);
}

TEST_F(HybridMgmtBlockTest, CheckValid_ChangeChennal_Hybrid)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->lastRunChannelId = TRAIN_CHANNEL_ID;
    hybridMgmtBlock->hybridBatchId[TRAIN_CHANNEL_ID] = 1;
    hybridMgmtBlock->pythonBatchId[TRAIN_CHANNEL_ID] = 0;

    RankInfo ri;
    ri.isDDR = true;
    ri.ctrlSteps = {1, 1, 1, 1};
    hybridMgmtBlock->SetRankInfo(ri);

    EXPECT_THROW(hybridMgmtBlock->CheckValid(EVAL_CHANNEL_ID), HybridMgmtBlockingException);
}

TEST_F(HybridMgmtBlockTest, CheckValid_ChangeChennal_Python)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->lastRunChannelId = TRAIN_CHANNEL_ID;
    hybridMgmtBlock->hybridBatchId[TRAIN_CHANNEL_ID] = 0;
    hybridMgmtBlock->pythonBatchId[TRAIN_CHANNEL_ID] = 1;

    hybridMgmtBlock->CheckValid(EVAL_CHANNEL_ID);
    EXPECT_EQ(hybridMgmtBlock->lastRunChannelId, EVAL_CHANNEL_ID);
}

TEST_F(HybridMgmtBlockTest, HybridMgmtBlockingExceptionShouldConstructExceptionWhenSceneIsValid)
{
    std::string scene = "TestScene";
    HybridMgmtBlockingException exception(scene);
}