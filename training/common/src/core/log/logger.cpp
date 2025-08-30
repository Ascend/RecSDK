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


#include "logger.h"

namespace MxRec {

int MxRec::Logger::level = MxRec::Logger::INFO;
int MxRec::Logger::rank = 0;

void Logger::SetRank(int logRank)
{
    Logger::rank = logRank;
}

void Logger::SetLevel(int logLevel)
{
    Logger::level = logLevel;
}

int Logger::GetLevel()
{
    return Logger::level;
}

const char* Logger::LevelToStr(int logLevel)
{
    if (logLevel < TRACE || logLevel > ERROR) {
        return "INVALID LEVEL";
    }
    static const char* msg[] = {
        "TRACE",
        "DEBUG",
        "INFO",
        "WARN",
        "ERROR",
    };
    constexpr int levelOffset = 2;
    return msg[logLevel + levelOffset];
}

void Logger::LogUnpack(std::queue<std::string>& fmt, std::stringstream &ss)
{
    while (!fmt.empty()) {
        ss << fmt.front();
        fmt.pop();
    }
    return;
}

}