/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * All rights reserved.
 *
 * Threc_embcachehis source code is licensed under the BSD-style license found in the
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
    void push(const T& e)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        dataQueue_.push(e);
    }

    bool pop(T& e)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (dataQueue_.empty()) {
            return false;
        }
        e = dataQueue_.front();
        dataQueue_.pop();
        return true;
    }

private:
    std::mutex mtx_;
    std::queue<T> dataQueue_;
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_UTILS_SAFE_QUEUE_H