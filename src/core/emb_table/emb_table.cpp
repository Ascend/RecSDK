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

void EmbTable::Init(const EmbInfo& embInfo, const RankInfo& rInfo, int seed)
{
#ifndef GTEST
    this->rankInfo = rInfo;
    this->seed = seed;
    LOG(INFO) << StringFormat(
        "EmbTable init, deviceID %d, embSize %d running", rInfo.deviceId, embInfo.extEmbeddingSize);
    // 计算embedding table需要分配的内存块数
    auto ret = aclrtSetDevice(static_cast<int32_t>(rInfo.deviceId));
    if (ret != ACL_ERROR_NONE) {
        LOG(ERROR) << StringFormat("Set device failed, device_id:%d, ret=%d", rInfo.deviceId, ret);
        throw AclError();
    }
    embSize = embInfo.extEmbeddingSize;
    blockSize = BLOCK_EMB_COUNT * embSize;
    for (int i = 0; i < INIT_BLOCK_COUNT; ++i) {
        // 申请新的内存块
        void *newBlock = nullptr;
        aclError ret = aclrtMalloc(&newBlock, blockSize * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG(ERROR) << StringFormat("aclrtMalloc failed, ret=%d", ret);
            throw AclError();
        }
        if (newBlock == nullptr) {
            // 内存不足，抛出异常
            throw OutOfMemoryError();
        } else {
            // 申请内存初始化
            RandomInit(newBlock, embInfo.initializeInfos, seed);
            // 将新的内存块加入内存链表
            memoryList.push_back(newBlock);
            SplitMemoryBlock(newBlock);
        }
    }
    totalCapacity = static_cast<int>(memoryList.size());
    LOG(INFO) << StringFormat(
        "aclrtMalloc success, emb name:%s, total capacity:%d", embInfo.name.c_str(), totalCapacity
    );
#endif
}

EmbTable::~EmbTable()
{
#ifndef GTEST
    for (void *block : memoryList) {
        // 释放内存块
        aclError ret = aclrtFree(block);
        if (ret != ACL_SUCCESS) {
            LOG(ERROR) << StringFormat("aclrtFree failed, ret=%d", ret);
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
        VLOG(GLOG_DEBUG) << "GetEmbAddress, embedding_list size: empty! Add block!";
        void *addBlock = nullptr;
        aclError ret = aclrtMalloc(&addBlock, blockSize * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG(ERROR) << StringFormat("aclrtMalloc failed, ret=%d", ret);
            throw AclError();
        }
        if (addBlock == nullptr) {
            // 内存不足，抛出异常
            throw OutOfMemoryError();
        } else {
            RandomInit(addBlock, embInfo.initializeInfos, seed);
            // 将新的内存块加入内存list
            memoryList.push_back(addBlock);
            SplitMemoryBlock(addBlock);
            totalCapacity++;
        }
    }
    float *embAddr = embeddingList.front();
    embeddingList.pop_front();
    usedCapacity++;
    return reinterpret_cast<int64_t>(embAddr);
#endif
}

// 将一个emb地址放入embeddingList中
void EmbTable::PutEmbAddress(int64_t curAddress)
{
#ifndef GTEST
    embeddingList.push_back(reinterpret_cast<float*>(curAddress));
    usedCapacity--;
#endif
}

void EmbTable::RandomInit(void* newBlock, const vector<InitializeInfo>& initializeInfos, int seed)
{
#ifndef GTEST
    LOG(INFO) << StringFormat(
        "Device GenerateEmbData Start, seed:%d, initializer num: %d", seed, initializeInfos.size());
    vector<float> devEmb(blockSize);
    for (auto initializeInfo: initializeInfos) {
        Initializer* initializer;
        switch (initializeInfo.initializerType) {
            case InitializerType::CONSTANT: {
                LOG(INFO) << StringFormat(
                    "Device GenerateEmbData ing using Constant Initializer by value %f. name %s, start %d, len %d.",
                    initializeInfo.constantInitializerInfo.constantValue,
                    initializeInfo.name.c_str(), initializeInfo.start, initializeInfo.len);
                initializer = &initializeInfo.constantInitializer;
                break;
            }
            case InitializerType::TRUNCATED_NORMAL: {
                LOG(INFO) << StringFormat(
                    "Device GenerateEmbData ing using Truncated Normal Initializer by mean: %f stddev: %f. "
                    "name %s, start %d, len %d.", initializeInfo.normalInitializerInfo.mean,
                    initializeInfo.normalInitializerInfo.stddev, initializeInfo.name.c_str(),
                    initializeInfo.start, initializeInfo.len);
                initializer = &initializeInfo.truncatedNormalInitializer;
                break;
            }
            case InitializerType::RANDOM_NORMAL: {
                LOG(INFO) << StringFormat(
                    "Device GenerateEmbData ing using Random Normal Initializer by mean: %f stddev: %f. "
                    "name %s, start %d, len %d.", initializeInfo.normalInitializerInfo.mean,
                    initializeInfo.normalInitializerInfo.stddev, initializeInfo.name.c_str(),
                    initializeInfo.start, initializeInfo.len);
                initializer = &initializeInfo.randomNormalInitializer;
                break;
            }
            default: {
                LOG(WARNING) << "Device Invalid Initializer Type. Using default Constant Initializer with value 0.";
                ConstantInitializer defaultInitializer(initializeInfo.start, initializeInfo.len, 0, 1);
                initializer = &defaultInitializer;
            }
        }
        for (int i = 0; i < BLOCK_EMB_COUNT; i++) {
            initializer->GenerateData(&devEmb[i * embSize], embSize);
        }
    }
    LOG(INFO) << StringFormat("Device GenerateEmbData End, seed:%d", seed);
    ExecuteAclMemcpy(newBlock, devEmb);
#endif
}

void EmbTable::ExecuteAclMemcpy(void* newBlock, vector<float> devEmb)
{
#ifndef GTEST
    aclError ret = aclrtMemcpy(
        newBlock, blockSize * sizeof(float), devEmb.data(), blockSize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << StringFormat("aclrtMemcpy failed, ret=%d", ret);
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

void EmbTable::PrintStatus()
{
    // 输出embedding table的总容量
    LOG(INFO) << StringFormat("Total capacity:%d", totalCapacity * blockSize);
    // 输出embedding table的未使用的使用容量
    LOG(INFO) << StringFormat("Unused capacity:%d", totalCapacity * blockSize - usedCapacity * embSize);
}

// 用于保存
map<int64, vector<float>> EmbTable::SaveEmb()
{
#ifndef GTEST
    if (embSize == 0) {
        throw std::runtime_error("SaveEmb Divided by Zero!");
    }
    map<int64, vector<float>> savedEmb;
    for (auto ptr : memoryList) {
        float* floatPtr = static_cast<float*>(ptr);
        for (int i = 0; i < BLOCK_EMB_COUNT; ++i) {
            // 访问 aclmemcpy
            vector<float> row(embSize);
            aclError ret = aclrtMemcpy(row.data(), embSize * sizeof(float),
                floatPtr + i * embSize, embSize * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                LOG(ERROR) << StringFormat("aclrtMemcpy failed, ret=%d", ret);
                throw AclError();
            }
            savedEmb[reinterpret_cast<int64>(floatPtr + i * embSize)] = move(row);
        }
    }
    return savedEmb;
#endif
}

// 用于加载 输入一个vector，申请内存，存储输入信息 , list<float*>返回全部地址
list<float*> EmbTable::LoadEmb(const vector<vector<float>> &savedEmb)
{
#ifndef GTEST
    list<float *> addressList;
    int embCapacity = static_cast<int>(savedEmb.size());
    if (savedEmb.size() == 0 || savedEmb[0].size() == 0) {
        LOG(ERROR) << "Load invalid savedEmb";
        return addressList;
    }
    embSize = static_cast<int>(savedEmb[0].size());
    void *newBlock = nullptr;
    aclError ret = aclrtMalloc(&newBlock, embCapacity * embSize * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << StringFormat("aclrtMalloc failed, ret=%d", ret);
        throw AclError();
    }
    if (newBlock == nullptr) {
        // 内存不足，抛出异常
        throw OutOfMemoryError();
    }
    float *floatPtr = static_cast<float*>(newBlock);
    for (int i = 0; i < embCapacity; i++) {
        aclError ret = aclrtMemcpy(floatPtr + i * embSize, embSize * sizeof(float),
            savedEmb[i].data(), embSize * sizeof(float),
            ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            LOG(ERROR) << StringFormat("aclrtMemcpy failed, ret=%d", ret);
            throw AclError();
        }
        addressList.push_back(floatPtr + i * embSize);
    }
    memoryList.push_back(newBlock);
    return addressList;
#endif
}
