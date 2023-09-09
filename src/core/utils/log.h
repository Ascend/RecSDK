/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Create: 2023
 * History: NA
 */

#ifndef  MXREC_LOG_H_
#define  MXREC_LOG_H_

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <string>
#include <sstream>
#include <iostream>
#include <queue>

using namespace std;

namespace MxRec {

constexpr int YEAR_BASE = 1900;
constexpr size_t DELIM_LEN = 2;

class Log {
public:

    static constexpr int TRACE = 0;
    static constexpr int DEBUG = 1;
    static constexpr int INFO = 2;
    static constexpr int WARN = 3;
    static constexpr int ERROR = 4;

    static void SetRank(int rank);

    static void SetLevel(int level);

    static int GetLevel();

    template<typename... Args>
    static void Format(stringstream& ss, const char* fmt, Args &&...args)
    {
        queue<string> formats;
        string tmp(fmt);
        for (size_t pos = tmp.find_first_of("{}"); pos != string::npos; pos = tmp.find_first_of("{}")) {
            string x = tmp.substr(0, pos);
            formats.push(x);
            tmp = tmp.substr(pos + DELIM_LEN);
        }
        formats.push(tmp);
        LogUnpack(formats, ss, args...);
    }

    template<typename... Args>
    static string Format(const char* fmt, Args &&...args)
    {
        stringstream ss;
        Log::Format(ss, fmt, args...);
        return ss.str();
    }

    template<typename... Args>
    static void log(const char* file, int line, int level, const char* fmt, Args &&...args)
    {
        stringstream ss;
        struct tm t;
        struct timeval tv;
        gettimeofday(&tv, NULL);
        localtime_r(&tv.tv_sec, &t);
        ss << "[MxRec][" << YEAR_BASE + t.tm_year << "/" << t.tm_mon << "/" << t.tm_mday<< " "
           << t.tm_hour << ":" << t.tm_min << ":" << t.tm_sec << "." << tv.tv_usec << "] ["
           << Log::_rank << "] ["<< Log::LevelToStr(level) << "] ["
           << (strrchr(file, '/') ? strrchr(file, '/') + 1 : file) << ":" << line << "] ";
        Log::Format(ss, fmt, args...);
        ss << std::endl;
        std::cout << ss.str();
    }

    template<typename... Args>
    static void log(const char* file, int line, int level, const string& fmt, Args &&...args)
    {
        Log::log(file, line, level, fmt.c_str(), args...);
    }

private:
    static const char* LevelToStr(int level);

    static void LogUnpack(queue<string>& fmt, stringstream &ss);

    template<typename head, typename... tail>
    static void LogUnpack(queue<string>& fmt, stringstream &ss, head &h, tail &&...tails)
    {
        if (!fmt.empty()) {
            ss << fmt.front();
            fmt.pop();
        }
        ss << h;
        LogUnpack(fmt, ss, tails...);
    };
    static int _level;
    static int _rank;
};


#define LOG_TRACE(args...) if (MxRec::Log::GetLevel() <= MxRec::Log::TRACE) \
MxRec::Log::log(__FILE__, __LINE__, MxRec::Log::TRACE, args)

#define LOG_DEBUG(args...) if (MxRec::Log::GetLevel() <= MxRec::Log::DEBUG) \
MxRec::Log::log(__FILE__, __LINE__, MxRec::Log::DEBUG, args)

#define LOG_INFO(args...) if (MxRec::Log::GetLevel() <= MxRec::Log::INFO) \
MxRec::Log::log(__FILE__, __LINE__, MxRec::Log::INFO, args)

#define LOG_WARN(args...) if (MxRec::Log::GetLevel() <= MxRec::Log::WARN) \
MxRec::Log::log(__FILE__, __LINE__, MxRec::Log::WARN, args)

#define LOG_ERROR(args...) if (MxRec::Log::GetLevel() <= MxRec::Log::ERROR) \
MxRec::Log::log(__FILE__, __LINE__, MxRec::Log::ERROR, args)

}

#endif  // MXREC_LOG_H_