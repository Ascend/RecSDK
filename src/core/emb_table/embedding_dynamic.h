/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: embeddingtable with dynamic expansion
 * Author: MindX SDK
 * Date: 2023/12/11
 */

#ifndef MX_REC_EMBEDDING_DYNAMIC_H
#define MX_REC_EMBEDDING_DYNAMIC_H

#include "emb_table/embedding_table.h"

namespace MxRec {

/**
 * 支持动态扩容的embedding表
 */
class EmbeddingDynamic : public EmbeddingTable {
public:
    EmbeddingDynamic();

    EmbeddingDynamic(const EmbInfo& info, const RankInfo& rankInfo, int inSeed);

    ~EmbeddingDynamic();

    virtual void Key2Offset(std::vector<emb_key_t>& keys, int channel);

    virtual int64_t capacity() const;

private:
    constexpr static int BLOCK_EMB_NUM = 100000; // 每次扩容分配10w条

    void RandomInit(void* addr, size_t embNum);

    int64_t GetEmptyEmbeddingAddress();

    void MallocEmbeddingBlock(int embNum);

    // embedding地址的列表
    list<float*> embeddingList_;
    // 内存块列表
    vector<void*> memoryList_;
};
}

#endif // MX_REC_EMBEDDING_DYNAMIC_H
