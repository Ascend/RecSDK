/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Create: 2023
 * History: NA
 */

#ifndef  MXREC_LOGGER_H
#define  MXREC_LOGGER_H

#include <cstdio>
#include <ctime>
#include <cstring>
#include <sys/time.h>
#include <string>
#include <sstream>
#include <iostream>
#include <queue>


namespace MxRec {

constexpr int YEAR_BASE = 1900;
constexpr size_t DELIM_LEN = 2;

class Logger {
public:

    static constexpr int TRACE = -2;
    static constexpr int DEBUG = -1;
    static constexpr int INFO = 0;
    static constexpr int WARN = 1;
    static constexpr int ERROR = 2;

    static void SetRank(int logRank);

    static void SetLevel(int logLevel);

    static int GetLevel();

    template<typename... Args>
    static void Format(std::stringstream& ss, const char* fmt, Args &&...args)
    {
        std::queue<std::string> formats;
        std::string tmp(fmt);
        for (size_t pos = tmp.find_first_of("{}"); pos != std::string::npos; pos = tmp.find_first_of("{}")) {
            std::string x = tmp.substr(0, pos);
            formats.push(x);
            tmp = tmp.substr(pos + DELIM_LEN);
        }
        formats.push(tmp);
        LogUnpack(formats, ss, args...);
    }

    template<typename... Args>
    static std::string Format(const char* fmt, Args &&...args)
    {
        std::stringstream ss;
        Logger::Format(ss, fmt, args...);
        return ss.str();
    }

    template<typename... Args>
    static void Log(const char* file, int line, int level, const char* fmt, Args &&...args)
    {
        std::stringstream ss;
        struct tm t;
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        localtime_r(&tv.tv_sec, &t);
        ss << "[MxRec][" << YEAR_BASE + t.tm_year << "/" << 1 + t.tm_mon << "/" << t.tm_mday<< " "
           << t.tm_hour << ":" << t.tm_min << ":" << t.tm_sec << "." << tv.tv_usec << "] ["
           << Logger::rank << "] ["<< Logger::LevelToStr(level) << "] ["
           << (strrchr(file, '/') ? strrchr(file, '/') + 1 : file) << ":" << line << "] ";
        Logger::Format(ss, fmt, args...);
        ss << std::endl;
        std::cout << ss.str();
    }

    template<typename... Args>
    static void Log(const char* file, int line, int level, const std::string& fmt, Args &&...args)
    {
        Logger::Log(file, line, level, fmt.c_str(), args...);
    }

private:
    static const char* LevelToStr(int logLevel);

    static void LogUnpack(std::queue<std::string>& fmt, std::stringstream &ss);

    template<typename head, typename... tail>
    static void LogUnpack(std::queue<std::string>& fmt, std::stringstream &ss, head &h, tail &&...tails)
    {
        if (!fmt.empty()) {
            ss << fmt.front();
            fmt.pop();
        }
        ss << h;
        LogUnpack(fmt, ss, tails...);
    };
    static int level;
    static int rank;
};

#define LOG_TRACE(args...) if (MxRec::Logger::GetLevel() <= MxRec::Logger::TRACE) \
MxRec::Logger::Log(__FILE__, __LINE__, MxRec::Logger::TRACE, args)

#define LOG_DEBUG(args...) if (MxRec::Logger::GetLevel() <= MxRec::Logger::DEBUG) \
MxRec::Logger::Log(__FILE__, __LINE__, MxRec::Logger::DEBUG, args)

#define LOG_INFO(args...) if (MxRec::Logger::GetLevel() <= MxRec::Logger::INFO) \
MxRec::Logger::Log(__FILE__, __LINE__, MxRec::Logger::INFO, args)

#define LOG_WARN(args...) if (MxRec::Logger::GetLevel() <= MxRec::Logger::WARN) \
MxRec::Logger::Log(__FILE__, __LINE__, MxRec::Logger::WARN, args)

#define LOG_ERROR(args...) if (MxRec::Logger::GetLevel() <= MxRec::Logger::ERROR) \
MxRec::Logger::Log(__FILE__, __LINE__, MxRec::Logger::ERROR, args)

}

#endif  // MXREC_LOGGER_H