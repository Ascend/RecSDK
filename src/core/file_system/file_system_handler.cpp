/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2023-11-16
 */

#include "file_system_handler.h"

using namespace std;
using namespace MxRec;

unique_ptr<FileSystem> FileSystemHandler::Create(const string& filePath)
{
    if (filePath.empty()) {
        throw runtime_error("dataDir is Null. The pointer of the file system cannot be created.");
    }
    for (const auto &prefix: hdfsPrefixes) {
        if (filePath.substr(0, prefix.length()) == prefix) {
            return make_unique<HdfsFileSystem>();
        }
    }
    return make_unique<LocalFileSystem>();
}
