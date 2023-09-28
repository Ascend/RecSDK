/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: checkpoint module
 * Author: MindX SDK
 * Date: 2023/9/28
 * History: NA
 */

#include "buffer_queue.h"

using namespace MxRec;
using namespace std;

void BufferQueue::push(std::vector<char>&& buffer)
{
    std::unique_lock<std::mutex> lock(mtx);
    bufferQueue.push(std::move(buffer));
    cv.notify_one();
}

std::vector<char> BufferQueue::pop()
{
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] {
        return !bufferQueue.empty();
    });
    auto buffer = std::move(bufferQueue.front());
    bufferQueue.pop();
    return buffer;
}