/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Create: 2023
 * History: NA
 */

#ifndef  MXREC_LOG_H
#define  MXREC_LOG_H

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

class Log {
public:

    static constexpr int trace = -2;
    static constexpr int debug = -1;
    static constexpr int info = 0;
    static constexpr int warn = 1;
    static constexpr int error = 2;

    static void SetRank(int rank);

    static void SetLevel(int level);

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
        Log::Format(ss, fmt, args...);
        return ss.str();
    }

    template<typename... Args>
    static void log(const char* file, int line, int level, const char* fmt, Args &&...args)
    {
        std::stringstream ss;
        struct tm t;
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        localtime_r(&tv.tv_sec, &t);
        ss << "[MxRec][" << YEAR_BASE + t.tm_year << "/" << t.tm_mon << "/" << t.tm_mday<< " "
           << t.tm_hour << ":" << t.tm_min << ":" << t.tm_sec << "." << tv.tv_usec << "] ["
           << Log::rank << "] ["<< Log::LevelToStr(level) << "] ["
           << (strrchr(file, '/') ? strrchr(file, '/') + 1 : file) << ":" << line << "] ";
        Log::Format(ss, fmt, args...);
        ss << std::endl;
        std::cout << ss.str();
    }

    template<typename... Args>
    static void log(const char* file, int line, int level, const std::string& fmt, Args &&...args)
    {
        Log::log(file, line, level, fmt.c_str(), args...);
    }

private:
    static const char* LevelToStr(int level);

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


#define LOG_TRACE(args...) if (MxRec::Log::GetLevel() <= MxRec::Log::trace) \
MxRec::Log::log(__FILE__, __LINE__, MxRec::Log::trace, args)

#define LOG_DEBUG(args...) if (MxRec::Log::GetLevel() <= MxRec::Log::debug) \
MxRec::Log::log(__FILE__, __LINE__, MxRec::Log::debug, args)

#define LOG_INFO(args...) if (MxRec::Log::GetLevel() <= MxRec::Log::info) \
MxRec::Log::log(__FILE__, __LINE__, MxRec::Log::info, args)

#define LOG_WARN(args...) if (MxRec::Log::GetLevel() <= MxRec::Log::warn) \
MxRec::Log::log(__FILE__, __LINE__, MxRec::Log::warn, args)

#define LOG_ERROR(args...) if (MxRec::Log::GetLevel() <= MxRec::Log::error) \
MxRec::Log::log(__FILE__, __LINE__, MxRec::Log::error, args)

}

#endif  // MXREC_LOG_H