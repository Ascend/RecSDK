/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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

#ifndef RECBASE_EXTERNAL_LOGGER_H
#define RECBASE_EXTERNAL_LOGGER_H
#include <mutex>
#include <iostream>
#include <sstream>

#include "singleton.h"

using ExternalLog = void (*)(int level, const char *msg);

namespace ock {
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
};

class ExternalLogger {
public:
    ExternalLogger() = default;

    static ExternalLogger *Instance()
    {
        return Singleton<ExternalLogger>::GetInstance();
    }

    void SetExternalLogFunction(ExternalLog func);

    void Log(const int level, const std::ostringstream &oss) const;

    static void PrintLog(LogLevel level, const std::string &message);

    static void PrintLog(LogLevel level, const std::string &message, bool flag);

    ExternalLogger(const ExternalLogger &) = delete;

    ExternalLogger &operator = (const ExternalLogger &) = delete;

    ExternalLogger(ExternalLogger &&) = delete;

    ExternalLogger &operator = (const ExternalLogger &&) = delete;

    ~ExternalLogger()
    {
        mLogFunc = nullptr;
    }

private:
    static ExternalLogger *gLogger;
    static std::mutex gMutex;

    ExternalLog mLogFunc = nullptr;
};
}

#endif // RECBASE_EXTERNAL_LOGGER_H