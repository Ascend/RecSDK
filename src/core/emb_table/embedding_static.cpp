/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: EmbeddingStatic HBM模式embedding表实现
 * Author: MindX SDK
 * Date: 2023/12/11
 */

#include "emb_table/embedding_static.h"
#include "utils/logger.h"

using namespace MxRec;

EmbeddingStatic::EmbeddingStatic()
{
}

EmbeddingStatic::EmbeddingStatic(const EmbInfo& info, const RankInfo& rankInfo, int inSeed)
    : EmbeddingTable(info, rankInfo, inSeed)
{
}

EmbeddingStatic::~EmbeddingStatic()
{
}

void EmbeddingStatic::Key2Offset(std::vector<emb_key_t>& keys, int channel)
{
    std::lock_guard<std::mutex> lk(mut_); // lock for PROCESS_THREAD
    for (emb_key_t& key : keys) {
        if (key == INVALID_KEY_VALUE) {
            continue;
        }
        const auto& iter = keyOffsetMap_.find(key);
        if (iter != keyOffsetMap_.end()) {
            key = iter->second;
            continue;
        }
        if (evictPos_.size() != 0 && channel == TRAIN_CHANNEL_ID) {
            // 新值, emb有pos可复用
            size_t offset = evictPos_.back();
            keyOffsetMap_[key] = offset;
            key = offset;
            evictPos_.pop_back();
            continue;
        }
        // 新值
        if (channel != TRAIN_CHANNEL_ID) {
            key = INVALID_KEY_VALUE;
            continue;
        }
        keyOffsetMap_[key] = maxOffset_;
        key = maxOffset_++;
    }
    if (maxOffset_ > devVocabSize_) {
        LOG_ERROR("dev cache overflow {} > {}", maxOffset_, devVocabSize_);
        throw std::runtime_error("dev cache overflow!");
    }
}

int64_t EmbeddingStatic::capacity() const
{
    return this->devVocabSize_;
}
