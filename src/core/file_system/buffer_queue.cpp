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

#include "buffer_queue.h"

using namespace MxRec;
using namespace std;

void BufferQueue::Push(std::vector<char> &&buffer)
{
    std::unique_lock<std::mutex> lock(mtx);
    bufferQueue.push(std::move(buffer));
    cv.notify_one();
}

void BufferQueue::Pop(std::vector<char>& buffer)
{
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] {
        return !bufferQueue.empty();
    });
    buffer = std::move(bufferQueue.front());
    bufferQueue.pop();
}