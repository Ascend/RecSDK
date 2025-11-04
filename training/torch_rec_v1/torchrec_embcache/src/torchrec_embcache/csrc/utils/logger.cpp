/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "utils/logger.h"

namespace Embcache {

int Embcache::Logger::level = Embcache::Logger::INFO;
int Embcache::Logger::rank = 0;

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