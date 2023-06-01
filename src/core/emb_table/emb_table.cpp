/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: emb table
 * Author: MindX SDK
 * Date: 2023/5/6
 */

#include <list>
#include <stdexcept>
#include <random>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/ranges.h>
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
    spdlog::info("EmbTable init, deviceID {}, embSize {} running", rInfo.deviceId, embInfo.extEmbeddingSize);
    // 计算embedding table需要分配的内存块数
    auto ret = aclrtSetDevice(static_cast<int32_t>(rInfo.deviceId));
    if (ret != ACL_ERROR_NONE) {
        spdlog::error("Set device failed, device_id:{}, ret={}", rInfo.deviceId, ret);
        throw AclError();
    }
    embSize = embInfo.extEmbeddingSize;
    blockSize = BLOCK_EMB_COUNT * embSize;
    for (int i = 0; i < INIT_BLOCK_COUNT; ++i) {
        // 申请新的内存块
        void *newBlock = nullptr;
        aclError ret = aclrtMalloc(&newBlock, blockSize * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            spdlog::error("aclrtMalloc failed, ret={}", ret);
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
    totalCapacity = memoryList.size();
    spdlog::info("aclrtMalloc success, emb name:{}, total capacity:{}", embInfo.name, totalCapacity);
#endif
}

EmbTable::~EmbTable()
{
#ifndef GTEST
    for (void *block : memoryList) {
        // 释放内存块
        aclError ret = aclrtFree(block);
        if (ret != ACL_SUCCESS) {
            spdlog::error("aclrtFree failed, ret={}", ret);
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
        spdlog::debug("GetEmbAddress, embedding_list size: empty! Add block!");
        void *addBlock = nullptr;
        aclError ret = aclrtMalloc(&addBlock, blockSize * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            spdlog::error("aclrtMalloc failed, ret={}", ret);
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
    embeddingList.push_back(reinterpret_cast<float*>(curAddress));
    usedCapacity--;
}

void EmbTable::RandomInit(void* newBlock, const vector<InitializeInfo>& initializeInfos, int seed)
{
#ifndef GTEST
    spdlog::info("Device GenerateEmbData Start, seed:{}", seed);
    vector<float> devEmb(blockSize);
    for (auto initializeInfo: initializeInfos) {
        Initializer* initializer;
        switch (initializeInfo.initializerType) {
            case InitializerType::CONSTANT: {
                spdlog::info("Device GenerateEmbData ing using Constant Initializer by value {}.",
                             initializeInfo.constantInitializerInfo.constantValue);
                initializer = &initializeInfo.constantInitializer;
                break;
            }
            case InitializerType::TRUNCATED_NORMAL: {
                spdlog::info("Device GenerateEmbData ing using Truncated Normal Initializer by mean: {} stddev: {}.",
                             initializeInfo.normalInitializerInfo.mean, initializeInfo.normalInitializerInfo.stddev);
                initializer = &initializeInfo.truncatedNormalInitializer;
                break;
            }
            case InitializerType::RANDOM_NORMAL: {
                spdlog::info("Device GenerateEmbData ing using Random Normal Initializer by mean: {} stddev: {}.",
                             initializeInfo.normalInitializerInfo.mean, initializeInfo.normalInitializerInfo.stddev);
                initializer = &initializeInfo.randomNormalInitializer;
                break;
            }
            default: {
                spdlog::warn("Device Invalid Initializer Type. Using default Constant Initializer with value 0.");
                ConstantInitializer defaultInitializer(initializeInfo.start, initializeInfo.len, 0);
                initializer = &defaultInitializer;
            }
        }
        for (int i = 0; i < BLOCK_EMB_COUNT; i++) {
            initializer->GenerateData(&devEmb[i * embSize], embSize);
        }
    }
    spdlog::info("Device GenerateEmbData End, seed:{}", seed);
    aclError ret = aclrtMemcpy(newBlock, blockSize * sizeof(float),
        devEmb.data(), blockSize * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        spdlog::error("aclrtMemcpy failed, ret={}", ret);
        throw AclError();
    }
#endif
}


void EmbTable::SplitMemoryBlock(void *newBlock)
{
    if (embSize == 0) {
        throw std::runtime_error("SplitMemoryBlock by embSize=0!");
    }
    for (int i = 0; i < BLOCK_EMB_COUNT; i++) {
        float *embPtr = static_cast<float*>(newBlock) + i * embSize;
        embeddingList.push_back(embPtr);
    }
}

void EmbTable::PrintStatus()
{
    // 输出embedding table的总容量
    spdlog::info("Total capacity:{}", totalCapacity * blockSize);
    // 输出embedding table的未使用的使用容量
    spdlog::info("Unused capacity:{}", totalCapacity * blockSize - usedCapacity * embSize);
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
                spdlog::error("aclrtMemcpy failed, ret={}", ret);
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
    int embCapacity = savedEmb.size();
    if (savedEmb.size() == 0 || savedEmb[0].size() == 0) {
        spdlog::error("Load invalid savedEmb");
        return addressList;
    }
    embSize = savedEmb[0].size();
    void *newBlock = nullptr;
    aclError ret = aclrtMalloc(&newBlock, embCapacity * embSize * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        spdlog::error("aclrtMalloc failed, ret={}", ret);
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
            spdlog::error("aclrtMemcpy failed, ret={}", ret);
            throw AclError();
        }
        addressList.push_back(floatPtr + i * embSize);
    }
    memoryList.push_back(newBlock);
    return addressList;
#endif
}
