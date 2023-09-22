/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: emb table
 * Author: MindX SDK
 * Date: 2023/5/6
 */

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
    totalCapacity = static_cast<int>(memoryList.size());
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
    }
#endif
}

// 从embeddingList获取一个可用的emb地址
int64_t EmbTable::GetEmbAddress()
{
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
        totalCapacity++;
    }
    float *embAddr = embeddingList.front();
    embeddingList.pop_front();
    usedCapacity++;
    return reinterpret_cast<int64_t>(embAddr);
#endif
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
        totalCapacity * blockSize, totalCapacity * blockSize - usedCapacity * embSize);
}
