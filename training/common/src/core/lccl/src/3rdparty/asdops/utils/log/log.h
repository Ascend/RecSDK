/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef COMMON_LOG_LOG_H
#define COMMON_LOG_LOG_H

#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

#define ASD_LOG(level) ASD_LOG_##level
namespace AsdOps {
struct LogLevel {
    static constexpr int TRACE = 0;
    static constexpr int DEBUG = 1;
    static constexpr int INFO = 2;
    static constexpr int WARN = 3;
    static constexpr int ERROR = 4;
};

// 定义一个辅助类，用于输出日志并在析构时自动添加换行
class Log {
public:
    Log() = default;

    ~Log()
    {
        // 当对象析构时，输出整个缓存的内容，并添加换行
        std::cout << stream.str() << std::endl;
    }

    // 重载 << 操作符，用于接收日志内容
    template <typename T> Log &operator << (const T &msg)
    {
        stream << msg;
        return *this;
    }

    static int GetLogLevel()
    {
        static int level = -1;
        if (level != -1) {
            return level;
        }

        const char *env_val = std::getenv("MXREC_LOG_LEVEL");
        if (env_val == nullptr) {
            level = LogLevel::INFO;
        } else {
            std::string log_level_str = env_val;
            if (log_level_str == "TRACE") {
                level = LogLevel::TRACE;
            } else if (log_level_str == "DEBUG") {
                level = LogLevel::DEBUG;
            } else if (log_level_str == "INFO") {
                level = LogLevel::INFO;
            } else if (log_level_str == "WARN") {
                level = LogLevel::WARN;
            } else if (log_level_str == "ERROR") {
                level = LogLevel::ERROR;
            }
        }
        return level;
    }

    // Function to extract the filename from a path
    static const char *ExtractFileName(const char *path)
    {
        // Find the last '/' or '\\' in the path
        const char *file = strrchr(path, '/');
        if (!file) {
            file = strrchr(path, '\\');
        }
        // If no '/' or '\\' was found, return the original path
        return file ? file + 1 : path;
    }

private:
    std::ostringstream stream; // 使用字符串流来缓存输出内容
};

#define ASD_LOG_TRACE                                          \
    if (AsdOps::LogLevel::TRACE >= AsdOps::Log::GetLogLevel()) \
    AsdOps::Log() << "\033[36mTRACE \033[0m" << AsdOps::Log::ExtractFileName(__FILE__) << ":" << __LINE__ << " "
#define ASD_LOG_DEBUG                                          \
    if (AsdOps::LogLevel::DEBUG >= AsdOps::Log::GetLogLevel()) \
    AsdOps::Log() << "\033[32mDEBUG \033[0m" << AsdOps::Log::ExtractFileName(__FILE__) << ":" << __LINE__ << " "
#define ASD_LOG_INFO                                          \
    if (AsdOps::LogLevel::INFO >= AsdOps::Log::GetLogLevel()) \
    AsdOps::Log() << "\033[37mINFO \033[0m" << AsdOps::Log::ExtractFileName(__FILE__) << ":" << __LINE__ << " "
#define ASD_LOG_WARN                                          \
    if (AsdOps::LogLevel::WARN >= AsdOps::Log::GetLogLevel()) \
    AsdOps::Log() << "\033[33mWARN \033[0m" << AsdOps::Log::ExtractFileName(__FILE__) << ":" << __LINE__ << " "
#define ASD_LOG_ERROR                                          \
    if (AsdOps::LogLevel::ERROR >= AsdOps::Log::GetLogLevel()) \
    AsdOps::Log() << "\033[31mERROR \033[0m" << __FILE__ << ":" << __LINE__ << " "
}
#endif