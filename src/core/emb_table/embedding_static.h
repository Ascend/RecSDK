/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: emb table
 * Author: MindX SDK
 * Date: 2023/12/11
 */

#ifndef MX_REC_EMBEDDING_STATIC_H
#define MX_REC_EMBEDDING_STATIC_H

#include "emb_table/embedding_table.h"

namespace MxRec {

/**
 * 静态大小的Embedding表。在HBM中分配好后大小无法改变
 */
class EmbeddingStatic : public EmbeddingTable {
public:
    EmbeddingStatic();

    EmbeddingStatic(const EmbInfo& info, const RankInfo& rankInfo, int inSeed);

    ~EmbeddingStatic();

    virtual void Key2Offset(std::vector<emb_key_t>& keys, int channel);

    virtual int64_t capacity() const;
};

}

#endif // MX_REC_EMBEDDING_STATIC_H
