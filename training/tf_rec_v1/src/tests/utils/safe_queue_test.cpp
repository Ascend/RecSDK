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

#include <thread>
#include <iostream>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "utils/safe_queue.h"

void TestSafeQueue()
{
    MxRec::SafeQueue<int> sq;

    // 测试入队操作
    sq.Pushv(std::make_unique<int>(1));
    assert(sq.Size() == 1);

    // 测试出队操作
    auto item = sq.TryPop();
    assert(*item == 1);
    assert(sq.Size() == 0);

    // 测试空队列出队
    item = sq.TryPop();
    assert(item == nullptr);

    // 测试多线程环境
    std::thread t1([&sq]() {
        const int maxCount = 100;
        for (int i = 0; i < maxCount; ++i) {
            sq.Pushv(std::make_unique<int>(i));
        }
    });

    std::thread t2([&sq]() {
        const int maxCount = 100;
        for (int i = 0; i < maxCount; ++i) {
            auto item = sq.TryPop();
            if (item != nullptr) {
                std::cout << *item << std::endl;
            }
        }
    });

    t1.join();
    t2.join();
}

void TestSingletonQueue()
{
    auto sq1 = MxRec::SingletonQueue<int>::GetInstances(0);
    auto sq2 = MxRec::SingletonQueue<int>::GetInstances(0);

    // 测试是否为单例
    assert(sq1 == sq2);

    // 测试超出范围
    auto sq3 = MxRec::SingletonQueue<int>::GetInstances(MxRec::MAX_QUEUE_NUM);
    assert(sq3 == nullptr);
}

void TestPutDirty()
{
    MxRec::SafeQueue<int> sq;

    // 创建一个智能指针对象ptr
    std::unique_ptr<int> ptr = std::make_unique<int>(1);

    // 使用 PutDirty 方法将 unique_ptr 放入 SafeQueue
    sq.PutDirty(std::move(ptr));

    // 使用 GetOne 方法检查 unique_ptr 是否已经被正确地放入 SafeQueue
    auto item = sq.GetOne();
    assert(*item == 1);
}

void TestGetOneFromEmptyQueue()
{
    MxRec::SafeQueue<int> sq;

    // 直接从新创建的 SafeQueue 中调用 GetOne
    auto item = sq.GetOne();

    // 检查返回的 unique_ptr 是否非空
    assert(item != nullptr);

    // 检查返回的 unique_ptr 指向的值是否为默认构造的 int
    assert(*item == int());
}

TEST(LogTest, TestHybridMgmt)
{
    TestSafeQueue();
    TestSingletonQueue();
    TestPutDirty();
    TestGetOneFromEmptyQueue();
}