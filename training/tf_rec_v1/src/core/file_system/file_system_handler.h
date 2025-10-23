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

#ifndef MX_REC_FILE_SYSTEM_HANDLER_H
#define MX_REC_FILE_SYSTEM_HANDLER_H

#include "hdfs_file_system/hdfs_file_system.h"
#include "local_file_system/local_file_system.h"

namespace MxRec {
    using namespace std;

    class FileSystemHandler {
    public:
        unique_ptr<FileSystem> Create(const string& filePath);
    private:
        const vector<string> hdfsPrefixes = {"hdfs://", "viewfs://"};
    };
}

#endif // MX_REC_FILE_SYSTEM_HANDLER_H