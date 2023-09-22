/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Create: 2023
 * History: NA
 */


#include "utils/log.h"

namespace MxRec {

int MxRec::Log::level = MxRec::Log::info;
int MxRec::Log::rank = 0;

void Log::SetRank(int rank)
{
    Log::rank = rank;
}

void Log::SetLevel(int level)
{
    Log::level = level;
}

int Log::GetLevel()
{
    return Log::level;
}

const char* Log::LevelToStr(int level)
{
    if (level < trace || level > error) {
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

void Log::LogUnpack(std::queue<std::string>& fmt, std::stringstream &ss)
{
    while (!fmt.empty()) {
        ss << fmt.front();
        fmt.pop();
    }
    return;
}

}