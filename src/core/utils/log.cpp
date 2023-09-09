/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Create: 2023
 * History: NA
 */


#include "utils/log.h"

namespace MxRec {

int MxRec::Log::_level = MxRec::Log::INFO;
int MxRec::Log::_rank = 0;

void Log::SetRank(int rank)
{
    Log::_rank = rank;
}

void Log::SetLevel(int level)
{
    Log::_level = level;
}

int Log::GetLevel()
{
    return Log::_level;
}

const char* Log::LevelToStr(int level)
{
    if (level < TRACE || level > ERROR) {
        return "INVALID LEVEL";
    }
    static const char* msg[] = {
        "TRACE",
        "DEBUG",
        "INFO",
        "WARN",
        "ERROR",
    };
    return msg[level];
}

void Log::LogUnpack(queue<string>& fmt, stringstream &ss)
{
    while (!fmt.empty()) {
        ss << fmt.front();
        fmt.pop();
    }
    return;
}

}