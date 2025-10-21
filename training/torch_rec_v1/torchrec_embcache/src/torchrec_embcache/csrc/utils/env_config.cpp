/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "env_config.h"

#include <cstdlib>
#include <stdexcept>

#include "logger.h"

namespace Embcache {
    // 日志级别环境变量 默认INFO
    int GlobalEnv::glogStderrthreshold = Logger::INFO;

    bool ParseEnv2Int(const char* envName, int& output)
    {
        const char* envVariablePtr = std::getenv(envName);
        if (envVariablePtr == nullptr) {
            return false;
        }
        std::string envNameStr = envName;
        try {
            output = std::stoi(envVariablePtr);
            return true;
        } catch (std::invalid_argument const& ex) {
            LOG_ERROR("Parse environment variable to int error, env variable is invalid");
            throw std::runtime_error("Parse environment variable error.");
        } catch (std::out_of_range const& ex) {
            LOG_ERROR("Parse environment variable to int error, env variable is out of range");
            throw std::runtime_error("Parse environment variable error.");
        }
        return false;
    }

    void ConfigGlobalEnv()
    {
        // 设置日志级别
        int logLevel = Logger::INFO;
        auto flag = ParseEnv2Int(EnvVariableNames::GLOG_STDERRTHRESHOLD, logLevel);
        if (flag) {
            if (logLevel < Logger::TRACE || logLevel > Logger::ERROR) {
                auto errMsg = Logger::Format("log level by env value:{} is invalid, it must be in range:[{}, {}]",
                                             logLevel, Logger::TRACE, Logger::ERROR);
                throw std::runtime_error(errMsg);
            }
            GlobalEnv::glogStderrthreshold = logLevel;
        }
    }

    void LogGlobalEnv()
    {
        LOG_DEBUG("Environment variables: [{}: {}].",
                  EnvVariableNames::GLOG_STDERRTHRESHOLD, GlobalEnv::glogStderrthreshold);
    }
}