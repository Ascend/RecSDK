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

#ifndef MX_REC_EMB_TABLE_H
#define MX_REC_EMB_TABLE_H

#include <iostream>
#include <list>
#include <stdexcept>
#include <bits/stdc++.h>

#include "utils/common.h"

namespace MxRec {

    using namespace std;

    class EmbTable {
    public:
        EmbTable() = default;

        void Init(const EmbInfo& eInfo, const RankInfo& rInfo, int initSeed = 0);

        ~EmbTable();

        // 从embeddingList获取获取一个可用的emb地址
        int64_t GetEmbAddress();

        // 打印emb表使用情况
        void PrintStatus() const;

        int64_t GetTableSize() const;

        int64_t GetTableCapacity() const;

        EmbTable(const EmbTable&) = delete;

        EmbTable(EmbTable&&) = delete;

        EmbTable& operator=(const EmbTable&) = delete;

        EmbTable& operator=(EmbTable&&) = delete;

        void ExecuteAclMemcpy(void* newBlock, vector<float> devEmb) const;

    GTEST_PRIVATE:
        constexpr static int BLOCK_EMB_COUNT = 100000;
        constexpr static int INIT_BLOCK_COUNT = 5;
        constexpr static int TEST_EMB_SIZE = 12;
        EmbInfo embInfo;
        RankInfo rankInfo;
        size_t blockSize = 1;
        int embSize = 1;
        size_t totalCapacity = 1;
        size_t usedCapacity = 0;
        int seed = 0;
        // embedding地址的列表
        list<float*> embeddingList;
        // 内存块列表
        vector<void*> memoryList;

        void RandomInit(void* newBlock);

        // embSize由embInfo得出
        void SplitMemoryBlock(void* newBlock);

        // 内部类，抛出内存不足异常
        class OutOfMemoryError : public runtime_error {
        public:
            OutOfMemoryError() : runtime_error("Out of memory!") {}
        };

        // 内部类，抛出acl异常
        class AclError : public runtime_error {
        public:
            AclError() : runtime_error("Acl failed!") {}
        };
    };
}

#endif // MX_REC_EMB_TABLE_MANAGER_H