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

#include "error.h"

#include <string>

#include "absl/strings/str_format.h"

namespace MxRec {
std::string Error::ToString() const
{
    std::string modName = this->ModAsString();
    std::string errType = this->TypeAsString();

    return absl::StrFormat("Module: %s, Error Type: %s, Error Message: %s", modName, errType, this->msg_);
}

std::string Error::TypeAsString() const noexcept
{
    switch (this->err_) {
        case ErrorType::NOT_FOUND:
            return "NotFound";
        case ErrorType::FILE_NOT_EXIST:
            return "FileNotExist";
        case ErrorType::NOT_SUPPORTED:
            return "NotSupported";
        case ErrorType::INVALID_ARGUMENT:
            return "InvalidArgument";
        case ErrorType::IO_ERROR:
            return "IOError";
        case ErrorType::ACL_ERROR:
            return "AscendCLError";
        case ErrorType::MPI_ERROR:
            return "MPIError";
        case ErrorType::CONSTRUCT_ERROR:
            return "ConstructError";
        case ErrorType::EXECUTION_ORDER_ERROR:
            return "ExecuteOrderError";
        case ErrorType::LOGIC_ERROR:
            return "LogicError";
        case ErrorType::LIST_EMPTY:
            return "ListEmpty";
        case ErrorType::NULL_PTR:
            return "NullPtr";
        case ErrorType::MEMORY_ERROR:
            return "MemoryError";
        case ErrorType::RESOURCE_NOT_ENOUGH:
            return "ResourceNotEnough";
        case ErrorType::HDFS_ERROR:
            return "HdfsError";
        default:
            return "UnknownError";
    }
}

std::string Error::ModAsString() const noexcept
{
    switch (this->mod_) {
        case ModuleName::M_CHECK_POINT:
            return "CheckPoint";
        case ModuleName::M_EMB_TABLE:
            return "EmbTable";
        case ModuleName::M_FILE_SYSTEM:
            return "FileSystem";
        case ModuleName::M_HD_TRANSFER:
            return "HdTransfer";
        case ModuleName::M_HYBRID_MGMT:
            return "HybridMgmt";
        case ModuleName::M_KEY_PROCESS:
            return "KeyProcess";
        case ModuleName::M_L3_STORAGE:
            return "L3Storage";
        case ModuleName::M_SSD_ENGINE:
            return "SsdEngine";
        case ModuleName::M_UTILS:
            return "Utils";
        case ModuleName::M_OCK_CTR:
            return "AccCTR";
        case ModuleName::M_ACL:
            return "AscendCL";
        default:
            return "UnknownModule";
    }
}
}  // namespace MxRec
