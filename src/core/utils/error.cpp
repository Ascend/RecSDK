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
#include <unordered_map>

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
    static const auto errorTypeMap = std::unordered_map<ErrorType, std::string>{
        {ErrorType::NOT_FOUND, "NotFound"},
        {ErrorType::FILE_NOT_EXIST, "FileNotExist"},
        {ErrorType::NOT_SUPPORTED, "NotSupported"},
        {ErrorType::INVALID_ARGUMENT, "InvalidArgument"},
        {ErrorType::IO_ERROR, "IOError"},
        {ErrorType::ACL_ERROR, "AscendCLError"},
        {ErrorType::MPI_ERROR, "MPIError"},
        {ErrorType::CONSTRUCT_ERROR, "ConstructError"},
        {ErrorType::EXECUTION_ORDER_ERROR, "ExecuteOrderError"},
        {ErrorType::LOGIC_ERROR, "LogicError"},
        {ErrorType::LIST_EMPTY, "ListEmpty"},
        {ErrorType::NULL_PTR, "NullPtr"},
        {ErrorType::MEMORY_ERROR, "MemoryError"},
        {ErrorType::RESOURCE_NOT_ENOUGH, "ResourceNotEnough"},
        {ErrorType::HDFS_ERROR, "HdfsError"}
    };

    auto it = errorTypeMap.find(this->err_);
    if (it != errorTypeMap.end()) {
        return it->second;
    }

    return "UnknownError";
}

std::string Error::ModAsString() const noexcept
{
    static const auto moduleNameMap = std::unordered_map<ModuleName, std::string>{
        {ModuleName::M_CHECK_POINT, "CheckPoint"},
        {ModuleName::M_EMB_TABLE, "EmbTable"},
        {ModuleName::M_FILE_SYSTEM, "FileSystem"},
        {ModuleName::M_HD_TRANSFER, "HdTransfer"},
        {ModuleName::M_HYBRID_MGMT, "HybridMgmt"},
        {ModuleName::M_KEY_PROCESS, "KeyProcess"},
        {ModuleName::M_L3_STORAGE, "L3Storage"},
        {ModuleName::M_SSD_ENGINE, "SsdEngine"},
        {ModuleName::M_UTILS, "Utils"},
        {ModuleName::M_OCK_CTR, "AccCTR"},
        {ModuleName::M_ACL, "AscendCL"},
        {ModuleName::M_HYBRID_MGMT_BLOCK, "HybridMgmtBlock"},
        {ModuleName::M_FEATURE_ADMIT_AND_EVICT, "FeatureAdmitAndEvict"},
        {ModuleName::M_DATASET_OPS, "DatasetOps"},
        {ModuleName::M_RMA_SHM_SVM, "RmaShmSvm"}
    };

    auto it = moduleNameMap.find(this->mod_);
    if (it != moduleNameMap.end()) {
        return it->second;
    }

    return "UnknownModule";
}
}  // namespace MxRec
