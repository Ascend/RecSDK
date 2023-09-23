/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Create: 2023
 * History: NA
 */


#include "utils/logger.h"

namespace MxRec {

int MxRec::Logger::level = MxRec::Logger::info;
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
    if (logLevel < trace || logLevel > error) {
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
    return msg[level + levelOffset];
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