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

#include <list>
#include <stdexcept>
#include <random>
#include <acl/acl_rt.h>
#include "acl/acl_base.h"
#include "utils/common.h"
#include "initializer/initializer.h"
#include "emb_table/emb_table.h"


using namespace std;
using namespace MxRec;
using namespace tensorflow;

void EmbTable::Init(const EmbInfo& eInfo, const RankInfo& rInfo, int initSeed)
{
#ifndef GTEST
    this->rankInfo = rInfo;
    this->seed = initSeed;
    this->embInfo = eInfo;
    LOG_INFO("EmbTable init, deviceID {}, embSize {} running", rInfo.deviceId, embInfo.extEmbeddingSize);
    // 计算embedding table需要分配的内存块数
    auto ret = aclrtSetDevice(static_cast<int32_t>(rInfo.deviceId));
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("Set device failed, device_id:{}, ret={}", rInfo.deviceId, ret);
        throw AclError();
    }
    embSize = embInfo.extEmbeddingSize;
    blockSize = BLOCK_EMB_COUNT * embSize;
    for (int i = 0; i < INIT_BLOCK_COUNT; ++i) {
        // 申请新的内存块
        void *newBlock = nullptr;
        aclError ec = aclrtMalloc(&newBlock, blockSize * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        if (ec != ACL_SUCCESS) {
            LOG_ERROR("aclrtMalloc failed, ret={}", ec);
            throw AclError();
        }
        // 申请内存初始化
        RandomInit(newBlock);
        // 将新的内存块加入内存链表
        memoryList.push_back(newBlock);
        SplitMemoryBlock(newBlock);
    }
    totalCapacity = static_cast<int>(memoryList.size()) * BLOCK_EMB_COUNT;
    LOG_INFO("aclrtMalloc success, emb name:{}, total capacity:{}", embInfo.name, totalCapacity);
#endif
}

EmbTable::~EmbTable()
{
#ifndef GTEST
    for (void *block : memoryList) {
        // 释放内存块
        aclError ret = aclrtFree(block);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtFree failed, ret={}", ret);
        }
        block = nullptr;
    }
#endif
}

// 从embeddingList获取一个可用的emb地址
int64_t EmbTable::GetEmbAddress()
{
    int64_t ret = -1;
#ifndef GTEST
    if (embeddingList.empty()) {
        PrintStatus();
        LOG_DEBUG("GetEmbAddress, embedding_list size: empty! Add block!");
        void *addBlock = nullptr;
        aclError ret = aclrtMalloc(&addBlock, blockSize * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtMalloc failed, ret={}", ret);
            throw AclError();
        }
        RandomInit(addBlock);
        // 将新的内存块加入内存list
        memoryList.push_back(addBlock);
        SplitMemoryBlock(addBlock);
        totalCapacity += BLOCK_EMB_COUNT;
    }
    float *embAddr = embeddingList.front();
    embeddingList.pop_front();
    usedCapacity++;
    ret = reinterpret_cast<int64_t>(embAddr);
#endif
    return ret;
}

void EmbTable::RandomInit(void* newBlock)
{
#ifndef GTEST
    LOG_INFO("Device GenerateEmbData Start, seed:{}, initializer num: {}", seed, embInfo.initializeInfos.size());
    vector<float> devEmb(blockSize);
    for (const auto& initializeInfo: as_const(embInfo.initializeInfos)) {
        LOG_INFO("Device GenerateEmbData ing. name {}", initializeInfo.name.c_str());
        for (int i = 0; i < BLOCK_EMB_COUNT; i++) {
            initializeInfo.initializer->GenerateData(&devEmb[i * embSize], embSize);
        }
    }
    LOG_INFO("Device GenerateEmbData End, seed:{}", seed);
    ExecuteAclMemcpy(newBlock, devEmb);
#endif
}

void EmbTable::ExecuteAclMemcpy(void* newBlock, vector<float> devEmb) const
{
#ifndef GTEST
    aclError ret = aclrtMemcpy(
        newBlock, blockSize * sizeof(float), devEmb.data(), blockSize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMemcpy failed, ret={}", ret);
        throw AclError();
    }
#endif
}


void EmbTable::SplitMemoryBlock(void *newBlock)
{
#ifndef GTEST
    if (embSize == 0) {
        throw std::runtime_error("SplitMemoryBlock by embSize=0!");
    }
    for (int i = 0; i < BLOCK_EMB_COUNT; i++) {
        float *embPtr = static_cast<float*>(newBlock) + i * embSize;
        embeddingList.push_back(embPtr);
    }
#endif
}

void EmbTable::PrintStatus() const
{
    // 输出embedding table的总容量和未使用的使用容量
    LOG_INFO("Total capacity:{}, Unused capacity:{}",
        totalCapacity * embSize, totalCapacity * embSize - usedCapacity * embSize);
}

int64_t EmbTable::GetTableSize() const
{
    return static_cast<int64_t>(usedCapacity);
}

int64_t EmbTable::GetTableCapacity() const
{
    return static_cast<int64_t>(totalCapacity);
}
