/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: EmbeddingDynamic HBM动态扩容embedding表实现
 * Author: MindX SDK
 * Date: 2023/12/11
 */

#include "emb_table/embedding_dynamic.h"

#include <acl/acl_rt.h>
#include <acl/acl_base.h>

#include "utils/logger.h"
#include "utils/singleton.h"

using namespace MxRec;

EmbeddingDynamic::EmbeddingDynamic()
{
}

EmbeddingDynamic::EmbeddingDynamic(const EmbInfo& info, const RankInfo& rankInfo, int inSeed)
    : EmbeddingTable(info, rankInfo, inSeed)
{
    if (isDynamic_) {
        auto ret = aclrtSetDevice(static_cast<int32_t>(rankInfo.deviceId));
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("Set device failed, device_id:{}, ret={}", rankInfo.deviceId, ret);
            throw runtime_error("Acl set device failed!");
        }
        MallocEmbeddingBlock(BLOCK_EMB_NUM);
    }
}

EmbeddingDynamic::~EmbeddingDynamic()
{
    for (auto& it: memoryList_) {
        aclError ret = aclrtFree(it);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtFree failed, ret={}", ret);
        }
    }
}

void EmbeddingDynamic::Key2Offset(std::vector<emb_key_t>& keys, int channel)
{
    constexpr emb_key_t INVALID_DYNAMIC_EXPANSION_ADDR = 0; // 动态扩容算子中的无效地址是0
    std::lock_guard<std::mutex> lk(mut_); // lock for PROCESS_THREAD
    for (emb_key_t& key : keys) {
        if (key == INVALID_KEY_VALUE) {
            key = INVALID_DYNAMIC_EXPANSION_ADDR;
            continue;
        }
        const auto& iter = keyOffsetMap.find(key);
        if (iter != keyOffsetMap.end()) {
            key = iter->second;
            continue;
        }
        // 新值
        if (channel == TRAIN_CHANNEL_ID) {
            int64_t addr = GetEmptyEmbeddingAddress();
            keyOffsetMap[key] = addr;
            key = addr;
            maxOffset++;
            continue;
        }
        key = INVALID_DYNAMIC_EXPANSION_ADDR;
    }
    LOG_DEBUG("current expansion emb:{}, usage:{}/{})", name, maxOffset, devVocabSize);
}

int64_t EmbeddingDynamic::capacity() const
{
    return capacity_;
}

int64_t EmbeddingDynamic::GetEmptyEmbeddingAddress()
{
    if (embeddingList_.empty()) {
        MallocEmbeddingBlock(BLOCK_EMB_NUM);
    }
    float *addr = embeddingList_.front();
    embeddingList_.pop_front();
    return reinterpret_cast<int64_t>(addr);
}

void EmbeddingDynamic::MallocEmbeddingBlock(int embNum)
{
    void *block = nullptr;
    aclError ec = aclrtMalloc(&block, embNum * embSize_ * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ec != 0) {
        throw std::bad_alloc();
    }
    RandomInit(block, embNum);
    memoryList_.push_back(block);
    for (int i = 0; i < embNum; i++) {
        float *embAddr = static_cast<float*>(block) + (i * embSize_);
        embeddingList_.push_back(embAddr);
    }
    capacity_ += embNum;
}

void EmbeddingDynamic::RandomInit(void* addr, size_t embNum)
{
    LOG_INFO("Device GenerateEmbData Start, seed:{}, initializer num: {}", seed_, embInfo_.initializeInfos.size());
    vector<float> hostmem(embNum * embSize_);
    for (const auto& initializeInfo: as_const(embInfo_.initializeInfos)) {
        for (size_t i = 0; i < embNum; ++i) {
            initializeInfo.initializer->GenerateData(&hostmem[i * embSize_], embSize_);
        }
    }
    LOG_INFO("Device GenerateEmbData End, seed:{}", seed_);

    aclError ret = aclrtMemcpy(addr, embNum * embSize_ * sizeof(float),
                               hostmem.data(), embNum * embSize_ * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMemcpy failed, ret={}", ret);
    }
}
