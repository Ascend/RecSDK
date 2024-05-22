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

    void Load(const string& savePath);

    void Save(const string& savePath);

    vector<int64_t> GetDeviceOffset();

    void SetOptimizerInfo(OptimizerInfo& optimizerInfo);

    void RefreshFreqInfoWithSwap();

    void AddKeyFreqInfo(const emb_key_t& key, RecordType type);

    void SetCacheManager(CacheManager *cm);

    void AddCacheManagerTraceLog() const;

    TableInfo GetTableInfo();

    void RefreshFreqInfoAfterLoad();

GTEST_PRIVATE:

    void LoadKey(const string& savePath);
    void LoadEmbAndOptim(const string& savePath);

    void SaveKey(const string& savePath);
    void SaveEmbData(const string &savePath);
    void SaveOptimData(const string& savePath);
    void SaveEmbAndOptim(const string& savePath);

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
    std::vector<int> devOffset2Batch; // has -1

    /**
     * 记录HBM上查找空位的当前位置
     * 值域为[0, devVocabSize]
    **/
    size_t currentUpdatePos;
    size_t currentUpdatePosStart; // 记录HBM上查找空位的起始位置

    vector<int64_t> hostKey;
    vector<int64_t> hostOffset;
    vector<int64_t> deviceKey;
    vector<int64_t> deviceOffset;

    vector<float *> embContent;

    std::string optimName;
    std::vector<std::string> optimParams;
    std::map<std::string, vector<float*>> optimContentMap;

    vector<int64_t> hostLoadOffset;
};

}

#endif // MX_REC_EMBEDDING_DDR_H
