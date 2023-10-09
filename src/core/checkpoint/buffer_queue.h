/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: checkpoint module
 * Author: MindX SDK
 * Date: 2023/9/28
 * History: NA
 */

#ifndef MXREC_BUFFER_QUEUE_H
#define MXREC_BUFFER_QUEUE_H

#include <vector>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace MxRec {
    class BufferQueue {
    public:
        void Push(std::vector<char> &&buffer);
        void Pop(std::vector<char>& buffer);
    private:
        std::queue<std::vector<char>> bufferQueue;
        std::mutex mtx;
        std::condition_variable cv;
    };
}

#endif
