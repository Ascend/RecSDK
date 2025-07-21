/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_THREAD_POOL_H
#define EMBEDDING_CACHE_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <memory>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>
#include "common/constants.h"

namespace Embcache {

class ThreadPool {
public:
    explicit ThreadPool(size_t threads) : stopped_(false)
    {
        if (threads == 0) {
            threads = 1;
        }
        for (size_t i = 0; i < threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->condition_.wait(lock, [this] { return this->stopped_ || !this->tasks_.empty(); });
                        if (this->stopped_ && this->tasks_.empty()) {
                            return;
                        }
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    template <class F>
    void enqueue(F&& f)
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stopped_) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks_.emplace(std::forward<F>(f));
        }
        condition_.notify_one();
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stopped_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stopped_;
};

inline uint64_t GetEmbMemoryPoolThreadNum()
{
    uint64_t embMemoryPoolThreadNum = EmbMemPoolConfigConstants::refillThreadNum;
    char* threadNumStr = getenv("EMB_MEMORY_POOL_THREAD_NUM");
    if (threadNumStr) {
        embMemoryPoolThreadNum = atoi(threadNumStr);
    }
    return embMemoryPoolThreadNum;
}

inline ThreadPool& GetEmbMemoryPool()
{
    static ThreadPool instance(GetEmbMemoryPoolThreadNum());
    return instance;
}

inline ThreadPool& GetAsyncTaskPool()
{
    static ThreadPool instance(10);
    return instance;
}

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_THREAD_POOL_H
