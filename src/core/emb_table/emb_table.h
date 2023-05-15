/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: emb table
 * Author: MindX SDK
 * Date: 2023/5/6
 */

#ifndef MX_REC_EMB_TABLE_H
#define MX_REC_EMB_TABLE_H

#include <iostream>
#include <list>
#include <stdexcept>
#include <bits/stdc++.h>
#include "utils/common.h"
#include <acl/acl_rt.h>
#include <acl/acl.h>

namespace MxRec {

    using namespace std;

    class EmbTable {
    public:
        EmbTable() = default;

        void Init(const EmbInfo& embInfo, const RankInfo& rInfo, int seed = 0);

        ~EmbTable();

        // 从embeddingList获取获取一个可用的emb地址
        int64_t GetEmbAddress();

        // 将一个emb地址放入embeddingList中
        void PutEmbAddress(int64_t curAddress);

        // 打印emb表使用情况
        void PrintStatus();

        int GetTotalCap();

        int GetUsedCap();

        EmbTable(const EmbTable&) = delete;

        EmbTable(EmbTable&&) = delete;

        EmbTable& operator=(const EmbTable&) = delete;

        EmbTable& operator=(EmbTable&&) = delete;

        // 用于保存
        map<int64, vector<float>> SaveEmb();

        // 用于加载 输入一个vector，创建一个embeddingtable类，申请内存，存储输入信息 , list<flaot*>返回全部地址
        list<float*> LoadEmb(const vector<vector<float>> &savedEmb);

    GTEST_PRIVATE:
        constexpr static int BLOCK_EMB_COUNT = 1000;
        constexpr static int INIT_BLOCK_COUNT = 5;
        constexpr static int TEST_EMB_SIZE = 12;
        EmbInfo embInfo;
        RankInfo rankInfo;
        int blockSize = 1;
        int embSize = 1;
        int totalCapacity = 1;
        int usedCapacity = 0;
        int seed = 0;
        float mean = 0;
        float stddev = 1;
        // embedding地址的列表
        list<float*> embeddingList;
        // 内存块列表
        vector<void*> memoryList;

        void RandomInit(void* newBlock, const vector<InitializeInfo> &initializeInfos, int seed);

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