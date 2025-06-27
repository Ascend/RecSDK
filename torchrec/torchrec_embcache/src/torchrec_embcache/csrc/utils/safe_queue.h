/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_UTILS_SAFE_QUEUE_H
#define EMBEDDING_CACHE_UTILS_SAFE_QUEUE_H

#include <queue>
#include <mutex>

namespace Embcache {

template <typename T>
class SafeQueue {
public:
    bool pop(T& e)
    {
        std::lock_guard<std::mutex> lk(mtx);
        if (dataQueue.empty()) {
            return false;
        }
        e = dataQueue.front();
        dataQueue.pop();
        return true;
    }

    void push(const T& e)
    {
        std::lock_guard<std::mutex> lk(mtx);
        dataQueue.push(e);
    }

    uint64_t GetLength()
    {
        std::lock_guard<std::mutex> lk(mtx);
        return dataQueue.size();
    }

private:
    std::mutex mtx;
    std::queue<T> dataQueue;
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_UTILS_SAFE_QUEUE_H