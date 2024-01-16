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

TEST_F(HybridMgmtBlockTest, CheckValid)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->SetStepInterval(1, 1);
    hybridMgmtBlock->pythonBatchId[0] = 0;
    hybridMgmtBlock->hybridBatchId[0] = 0;
    hybridMgmtBlock->CheckValid(0);
    hybridMgmtBlock->CheckValid(0);

    int step2 = 2;
    hybridMgmtBlock->pythonBatchId[0] = 0;
    hybridMgmtBlock->hybridBatchId[0] = step2;
    hybridMgmtBlock->lastRunChannelId = 0;
    try {
        hybridMgmtBlock->CheckValid(1);
        ASSERT_EQ(-1, 0);
    } catch (HybridMgmtBlockingException e) {
        LOG_INFO(HYBRID_BLOCKING + "sucess");
        ASSERT_EQ(0, 0);
    }
    hybridMgmtBlock->pythonBatchId[0] = 0;
    hybridMgmtBlock->hybridBatchId[0] = 1;
    hybridMgmtBlock->CheckValid(0);
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

TEST_F(HybridMgmtBlockTest, ResetAll)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->SetStepInterval(1, 1);
    hybridMgmtBlock->ResetAll(0);
    ASSERT_EQ(hybridMgmtBlock->hybridBatchId[0], 0);
}

TEST_F(HybridMgmtBlockTest, CheckSaveEmbMapValid)
{
    hybridMgmtBlock = std::make_unique<HybridMgmtBlock>();
    hybridMgmtBlock->SetStepInterval(1, 1);
    hybridMgmtBlock->lastRunChannelId = 0;

    hybridMgmtBlock->pythonBatchId[0] = 0;
    hybridMgmtBlock->hybridBatchId[0] = 0;
    hybridMgmtBlock->CheckSaveEmbMapValid();
    int status0 = hybridMgmtBlock->CheckSaveEmbMapValid();

    hybridMgmtBlock->pythonBatchId[0] = 0;
    hybridMgmtBlock->hybridBatchId[0] = 1;
    hybridMgmtBlock->CheckSaveEmbMapValid();
    int status1 = hybridMgmtBlock->CheckSaveEmbMapValid();

    int step2 = 2;
    hybridMgmtBlock->pythonBatchId[0] = 0;
    hybridMgmtBlock->hybridBatchId[0] = step2;
    int status2 = hybridMgmtBlock->CheckSaveEmbMapValid();
    ASSERT_EQ(status0, 0);
    ASSERT_EQ(status1, 1);
    ASSERT_EQ(status2, -1);
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