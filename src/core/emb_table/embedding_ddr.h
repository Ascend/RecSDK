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

    virtual void Key2OffsetForDp(std::vector<emb_key_t>& keys, int channel);

    virtual int64_t capacity() const;

    virtual void EvictKeys(const vector<emb_key_t>& keys);

    void Load(const string& savePath, map<string, unordered_set<emb_cache_key_t>>& trainKeySet);

    void LoadKey(const string& savePath, vector<emb_cache_key_t>& keys);

    void LoadEmbedding(const string& savePath, vector<vector<float>>& embeddings);

    void LoadOptimizerSlot(const string& savePath, vector<vector<float>>& optimizerSlots);

    void Save(const string& savePath, const int pythonBatchId);

    void SyncLatestEmbedding(const int pythonBatchId);

    void SaveKey(const string& savePath, vector<emb_cache_key_t>& keys);

    void SaveEmbedding(const string& savePath, vector<vector<float>>& embeddings);

    void SaveOptimizerSlot(const string& savePath, vector<vector<float>>& optimizerSlots, size_t keySize);

    vector<int64_t> GetDeviceOffset();

    void SetOptimizerInfo(OptimizerInfo& optimizerInfo);

    void SetCacheManager(CacheManager *cm);

    TableInfo GetTableInfo();

    void SetHDTransfer(HDTransfer* hdTransfer);

    void LoadKey(const string& savePath);
    void LoadEmbAndOptim(const string& savePath);

    void SaveKey(const string& savePath);
    void SaveEmbData(const string &savePath);
    void SaveOptimData(const string& savePath);
    void SaveEmbAndOptim(const string& savePath);
    void SetEmbCache(ock::ctr::EmbCacheManagerPtr embCache);

    void BackUpTrainStatus();
    void RecoverTrainStatus();

GTEST_PRIVATE:

    void EvictDeleteEmb(const vector<emb_key_t>& keys);

    void EmbeddingUpdateWithSSD(const vector<uint64_t>& swapOutKeys, float* deviceDataPtr);

    size_t maxOffsetOld { 0 };
    std::vector<size_t> evictPosChange;
    std::vector<size_t> evictDevPosChange;
    std::vector<std::pair<int, emb_key_t>> devOffset2KeyOld;
    std::vector<std::pair<emb_key_t, emb_key_t>> oldSwap; // (old on dev, old on host)

    vector<float *> embContent;

    std::string optimName;
    std::vector<std::string> optimParams;

    vector<int64_t> hostLoadOffset;

    HDTransfer *hdTransfer = nullptr;
    ock::ctr::EmbCacheManagerPtr embCache = nullptr;
    int deviceId = -1;
};

}

#endif // MX_REC_EMBEDDING_DDR_H
