/* Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
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

#ifndef OCK_EXTERNAL_THREADER_H
#define OCK_EXTERNAL_THREADER_H

#include <mutex>
#include <iostream>
#include <sstream>
#include <vector>
#include <future>
#include <queue>
#include <thread>
#include <condition_variable>
#include <functional>
#include "singleton.h"

using ExternalThread = void (*)(const std::vector<std::function<void()>> &tasks);

namespace ock {
class ThreadPoolAsync {
public:
    ThreadPoolAsync() : stop(false) {}

    ~ThreadPoolAsync()
    {
        {
            std::lock_guard<std::mutex> lock(taskMutex);
            stop = true;
        }
        taskCv.notify_all();
        for (auto &t : workerThreads) {
            t.join();
        }
    }

    void SetNumThreads(int n)
    {
        if (n < 1) {
            return;
        }

        for (int i = 0; i < n; ++i) {
            workerThreads.emplace_back(std::bind(&ThreadPoolAsync::WorkerThread, this));
        }
    }

    template <typename F> std::future<int> AddTask(F &&f)
    {
        std::lock_guard<std::mutex> lock(taskMutex);

        auto pt = std::make_unique<std::packaged_task<int()>>(std::forward<F>(f));
        auto fut = pt->get_future();
        tasks.emplace(std::move(pt));
        taskCv.notify_one();
        return fut;
    }

private:
    std::vector<std::thread> workerThreads;
    std::queue<std::unique_ptr<std::packaged_task<int()>>> tasks;
    std::mutex taskMutex;
    std::condition_variable taskCv;
    std::atomic<bool> stop = false;

    void WorkerThread()
    {
        while (true) {
            std::unique_ptr<std::packaged_task<int()>> task;
            {
                std::unique_lock<std::mutex> lock(taskMutex);
                while (tasks.empty() && !stop) {
                    taskCv.wait(lock);
                }
                if (stop) {
                    break;
                }
                task = std::move(tasks.front());
                tasks.pop();
            }
            (*task)();
        }
    }
};


class SimpleThreadPool {
public:
    static void SyncRun(const std::vector<std::function<void()>> &tasks)
    {
        std::vector<std::future<void>> futs;
        for (auto &task : tasks) {
            futs.push_back(std::async(task));
        }
        for (auto &fut : futs) {
            fut.wait();
        }
    }
};

class ExternalThreader {
public:
    ExternalThreader() = default;

    static ExternalThreader *Instance()
    {
        return Singleton<ExternalThreader>::GetInstance();
    }

    void SetExternalLogFunction(ExternalThread func);

    void Run(const std::vector<std::function<void()>> &tasks) const;

    ExternalThreader(const ExternalThreader &) = delete;
    ExternalThreader &operator = (const ExternalThreader &) = delete;
    ExternalThreader(ExternalThreader &&) = delete;
    ExternalThreader &operator = (const ExternalThreader &&) = delete;

    ~ExternalThreader()
    {
        mThreadFunc = nullptr;
    }

private:
    static ExternalThreader *gThread;
    static std::mutex gMutex;

    ExternalThread mThreadFunc = nullptr;
};
}

#endif // OCK_EXTERNAL_THREADER_H
