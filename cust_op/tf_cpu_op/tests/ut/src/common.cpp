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

#include "common.h"

#include <chrono>

std::string GetCurrentTime()
{
    auto nowTime = std::chrono::system_clock::now();
    auto inTimeT = std::chrono::system_clock::to_time_t(nowTime);
    std::stringstream string;
    string << std::put_time(std::localtime(&inTimeT), "%Y-%m-%d %X");
    return string.str();
}

void RecBaseLog(int messageLevel, const char *message)
{
    std::string color = "";
    std::string levelString;
    switch (messageLevel) {
        case DEBUG:
            levelString = "DEBUG";
            break;
        case INFO:
            levelString = "INFO";
            break;
        case WARNING:
            color = "\x1b[33m";
            levelString = "WARNING";
            break;
        case ERROR:
            color = "\x1b[31m";
            levelString = "ERROR";
            break;
        default:
            break;
    }
    std::cout << color << "[" << GetCurrentTime() << "][" << levelString << "] " << message << "\x1b[0m" << std::endl;
}