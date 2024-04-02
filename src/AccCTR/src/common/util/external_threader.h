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
#include "singleton.h"

using ExternalThread = void (*)(const std::vector<std::function<void()>> &tasks);

namespace ock {
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
