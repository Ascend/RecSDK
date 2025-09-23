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

#ifndef MXREC_THREAD_POOL_H
#define MXREC_THREAD_POOL_H

#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <utility>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>

#include "common_func/common_func.h"
#include "error/error.h"
#include "log/logger.h"

namespace MxRec {

class ThreadPool {
public:
    explicit ThreadPool(size_t num) : stop(false)
    {
        if (num == 0) {
            auto error = Error(ModuleName::M_UTILS, ErrorType::INVALID_ARGUMENT,
                               StringFormat("ThreadPool size must be greater than 0"));
            LOG_ERROR(error.ToString());
            throw std::invalid_argument(error.ToString());
        }
        LOG_INFO("ThreadPool init num: {}", num);
        for (size_t i = 0; i < num; i++) {
            threads.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });

                        if (this->stop && this->tasks.empty()) {
                            return;
                        }

                        task = std::move(this->tasks.front());
                        this->tasks.pop();

                        LOG_TRACE("ThreadPool pop one task!");
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& thread : threads) {
            thread.join();
        }
        LOG_INFO("ThreadPool finish!");
    }

    ThreadPool(const ThreadPool&) = delete;

    ThreadPool& operator=(const ThreadPool&) = delete;

    // Enqueue task without return value.
    template <class F>
    void enqueue(F&& f)
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            tasks.emplace(std::forward<F>(f));
            LOG_TRACE("ThreadPool enqueue one task!");
        }
        condition.notify_one();
    }

    // Enqueue task with return value.
    template <typename F, typename... A>
    auto enqueueWithFuture(F&& f, A&&... args) -> std::future<decltype(f(std::forward<A>(args)...))>
    {
        using ReturnType = decltype(std::declval<F>()(std::declval<A>()...));
        auto task =
            std::make_shared<std::packaged_task<ReturnType()>>(std::bind(std::forward<F>(f), std::forward<A>(args)...));

        // Get task return value.
        std::future<ReturnType> futureRes = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (stop) {
                throw std::runtime_error("enqueue on stopped thread pool");
            }
            tasks.emplace([task]() { (*task)(); });
            LOG_TRACE("ThreadPool enqueue one task!");
        }
        condition.notify_one();
        return futureRes;
    }

private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;

    std::mutex mutex;
    std::condition_variable condition;
    bool stop;
};

}  // namespace MxRec

#endif  // MXREC_THREAD_POOL_H
