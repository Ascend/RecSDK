/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <unistd.h>
#include <thread>

#include "utils/safe_queue.h"
#include "utils/string_tools.h"

#include "../common_main.h"

using namespace Embcache;

class SafeQueueTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        std::vector<int> nums{1, 2, 3, 4, 5};
        for (auto num : nums) {
            sQueue.push(num);
        }
    }

    void TearDown() override {}

    SafeQueue<int> sQueue;
};

TEST_F(SafeQueueTest, PushPop)
{
    std::vector<int> toPushNums{6, 7, 8, 9, 10};
    std::thread pushThread([&toPushNums, &sQueue]() {
        for (auto num : toPushNums) {
            usleep(1000);
            sQueue.push(num);
        }
    });

    std::vector<int> toPopNums;
    std::thread popThread([&toPopNums, &sQueue]() {
        while (toPopNums.size() != 10) {
            int data;
            bool success = sQueue.pop(data);
            if (success) {
                toPopNums.push_back(data);
            }
            usleep(100);
        }
    });

    pushThread.join();
    popThread.join();

    LOG(INFO) << "pop_results:" << StringTools::ToString(toPopNums);
    std::vector<int> expected{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    ASSERT_EQ(expected, toPopNums);
}

int main(int argc, char* argv[])
{
    return CommonMain(argc, argv);
}