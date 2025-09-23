/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "external_logger.h"

#include "set_external_logger.h"
#include "error_code.h"

namespace ock {
void ExternalLogger::SetExternalLogFunction(ExternalLog func)
{
    if (mLogFunc == nullptr) {
        mLogFunc = func;
    }
}

void ExternalLogger::Log(const int level, const std::ostringstream &oss) const
{
    if (mLogFunc != nullptr) {
        mLogFunc(level, oss.str().c_str());
    }
}

void ExternalLogger::PrintLog(LogLevel level, const std::string &message)
{
    std::ostringstream oss;
    oss << message;
    auto logger = ExternalLogger::Instance();
    if (logger != nullptr) {
        logger->Log(static_cast<int>(level), oss);
    }
}

int SetExternalLogFunc(ExternalLog logFunc)
{
    if (logFunc == nullptr) {
        std::cout << "Invalid external log function pointer" << std::endl;
        return H_POINTER_NULL;
    }

    auto logger = ExternalLogger::Instance();
    if (logger == nullptr) {
        std::cout << "Failed to create logger instance" << std::endl;
        return H_ERROR;
    }

    logger->SetExternalLogFunction(logFunc);
    return H_OK;
}
}