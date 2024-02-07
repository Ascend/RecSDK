/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: emb table
 * Author: MindX SDK
 * Date: 2023/12/11
 */

#ifndef MX_REC_EMBEDDING_DDR_H
#define MX_REC_EMBEDDING_DDR_H

#include "emb_table/embedding_table.h"

namespace MxRec {

class EmbeddingDDR : public EmbeddingTable {
public:
    EmbeddingDDR();

    EmbeddingDDR(const EmbInfo& info, const RankInfo& rankInfo, int inSeed);

    EmbeddingDDR& operator=(const EmbeddingDDR& table);

    ~EmbeddingDDR();

    virtual void Key2Offset(std::vector<emb_key_t>& splitKey, int channel);

    virtual int64_t capacity() const;

    virtual std::vector<int32_t> FindOffset(const vector<emb_key_t>& keys,
                                            size_t batchId, int channelId,
                                            std::vector<size_t>& swapPos);

    emb_key_t FindOffsetHelper(const emb_key_t& key, int channelId);

    void UpdateBatchId(const vector<emb_key_t>& keys, size_t currentBatchId);

    emb_key_t FindSwapPosOld(emb_key_t key, size_t hostOffset, size_t batchId, std::vector<size_t>& swapPos);

    virtual void EvictKeys(const vector<emb_key_t>& keys);

//    std::vector<int32_t> lookUpVec; // 查询结果

    virtual void ClearLookupAndSwapOffset();

    void SetStartCount();

    int Load(const string& savePath);

    int Save(const string& savePath);

GTEST_PRIVATE:

    int LoadHashMap(const string& savePath);
    int LoadDevOffset(const string& savePath);
    int LoadCurrStat(const string& savePath);
    int LoadEvictPos(const string& savePath);
    int LoadEmbInfo(const string& savePath);
    int LoadEmbData(const string& savePath);

    int SaveHashMap(const string& savePath);
    int SaveDevOffset(const string& savePath);
    int SaveCurrStat(const string& savePath);
    int SaveEvictPos(const string& savePath);
    int SaveEmbInfo(const string& savePath);
    int SaveEmbData(const string& savePath);

    void EvictDeleteEmb(const vector<emb_key_t>& keys);

    std::vector<emb_key_t> devOffset2Key;

    size_t maxOffsetOld { 0 };
    std::vector<size_t> evictPosChange;
    std::vector<size_t> evictDevPosChange;
    std::vector<std::pair<int, emb_key_t>> devOffset2KeyOld;
    std::vector<std::pair<emb_key_t, emb_key_t>> oldSwap; // (old on dev, old on host)

    /*
     * HBM与DDR换入换出时,已存在于DDR且要转移到HBM的key(不包含新key); 用于SSD模式
     * (区别于oldSwap: pair.second为已存在于DDR key + 换入换出前映射到DDR的新key)
     */
    std::vector<emb_key_t> ddr2HbmKeys;
    bool isSSDEnabled;
    std::vector<int> devOffset2Batch; // has -1

    /**
     * 记录HBM上查找空位的当前位置
     * 值域为[0, devVocabSize_]
    **/
    size_t currentUpdatePos;
    size_t currentUpdatePosStart; // 记录HBM上查找空位的起始位置
};

}

#endif // MX_REC_EMBEDDING_DDR_H
