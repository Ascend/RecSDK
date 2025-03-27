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

#ifndef MXREC_RMA_SWAP_TEST_H
#define MXREC_RMA_SWAP_TEST_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <unordered_map>

#include <sys/stat.h>
#include <sys/shm.h>
#include <acl/acl.h>
#include "securec.h"

#include "hd_transfer/rma_shm_svm.h"

namespace TfTest {

template<class T>
void DumpVector(std::vector <T>& vec, std::string savePath)
{
    std::ofstream outputFile(savePath, std::ios::binary);
    if (!outputFile.is_open()) {
        std::cout << "Unable to open file " << savePath << std::endl;
        return;
    }

    // 写入向量的数据
    if (!vec.empty()) {
        outputFile.write(reinterpret_cast<const char*>(vec.data()), sizeof(T) * vec.size());
    }

    // 关闭文件
    outputFile.close();

    // 检查文件是否成功写入
    if (!outputFile.good()) {
        std::cerr << "写入文件时发生错误: " << savePath << std::endl;
    }
}

template<class T>
void DumpVector(T* arr, int64_t arrLen, std::string savePath)
{
    std::ofstream outputFile(savePath, std::ios::binary);
    if (!outputFile.is_open()) {
        std::cout << "Unable to open file " << savePath << std::endl;
        return;
    }

    if (arrLen > 0) {
        outputFile.write(reinterpret_cast<const char*>(arr), sizeof(T) * arrLen);
    }

    outputFile.close();

    if (!outputFile.good()) {
        std::cerr << "写入文件时发生错误: " << savePath << std::endl;
    }
}

int ReadShmData(std::string name, bool save = false, std::string savePath = "./");

int WriteShmData(std::string name, std::vector <int64_t>& shape, bool save = false, std::string savePath = "./");
}
#endif // MXREC_RMA_SWAP_TEST_H