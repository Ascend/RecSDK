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
        const auto& iter = keyOffsetMap.find(key);
        if (iter != keyOffsetMap.end()) {
            key = iter->second;
            continue;
        }
        if (evictDevPos.size() != 0 && channel == TRAIN_CHANNEL_ID) {
            // 新值, emb有pos可复用
            size_t offset = evictDevPos.back();
            keyOffsetMap[key] = offset;
            key = offset;
            evictDevPos.pop_back();
            continue;
        }
        // 新值
        if (channel != TRAIN_CHANNEL_ID) {
            key = INVALID_KEY_VALUE;
            continue;
        }
        keyOffsetMap[key] = maxOffset;
        key = maxOffset++;
    }
    if (maxOffset > devVocabSize) {
        LOG_ERROR("dev cache overflow {} > {}", maxOffset, devVocabSize);
        throw std::runtime_error("dev cache overflow!");
    }
}

int64_t EmbeddingStatic::capacity() const
{
    return this->devVocabSize;
}
