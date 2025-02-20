/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include "utils/singleton.h"
#include "utils/task_queue.h"
#include "utils/thread_pool.h"
#include "utils/time_cost.h"

namespace MxRec {

TEST(TimeCostTest, ElapsedSecOk)
{
    auto duration = TimeCost();
    EXPECT_GE(duration.ElapsedSec(), 0);
}

TEST(TimeCostTest, ElapsedMSOk)
{
    auto duration = TimeCost();
    EXPECT_GE(duration.ElapsedMS(), 0);
}

TEST(SingletonTest, GetInstanceOk)
{
    auto instance = Singleton<int>::GetInstance();
    EXPECT_NE(instance, nullptr);
}

TEST(TaskQueueTest, PushAndPopOk)
{
    auto queue = Common::TaskQueue<int>();
    auto val = 0;

    queue.Pushv(val);
    EXPECT_EQ(queue.WaitAndPop(), 0);

    queue.Pushv(0);
    EXPECT_EQ(queue.WaitAndPop(), 0);

    EXPECT_EQ(queue.Size(), 0);
    queue.DestroyQueue();
}

TEST(ThreadPoolTest, EnqueueOk)
{
    auto pool = new ThreadPool(1);
    auto isExec = false;

    pool->enqueue([&isExec]() { isExec = true; });
    delete pool;

    EXPECT_TRUE(isExec);
}

}  // namespace MxRec
