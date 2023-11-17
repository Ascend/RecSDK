/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2023-10-19
 */

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