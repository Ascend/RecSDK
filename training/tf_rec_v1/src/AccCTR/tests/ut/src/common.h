/* Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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

#ifndef CTR_COMMON_H
#define CTR_COMMON_H
#include <iostream>

#include "factory.h"

extern ock::ctr::FactoryPtr factory;

enum CTRLogLevel {
    DEBUG = 0,
    INFO,
    WARN,
    ERROR,
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

static void CTRLog(int level, const char *msg)
{
    switch (level) {
        case CTRLogLevel::DEBUG:
            std::cout << "DEBUG:" << msg << std::endl;
            break;
        case CTRLogLevel::INFO:
            std::cout << "INFO:" << msg << std::endl;
            break;
        case CTRLogLevel::WARN:
            std::cout << "WARN:" << msg << std::endl;
            break;
        case CTRLogLevel::ERROR:
            std::cout << "ERROR:" << msg << std::endl;
            break;
        default:
            break;
    }
}

#endif // CTR_COMMON_H
