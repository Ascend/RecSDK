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

    virtual void Key2OffsetForDp(std::vector<emb_key_t>& keys, int channel);

    virtual int64_t capacity() const;

    void Load(const string& savePath, map<string, unordered_set<emb_cache_key_t>>& trainKeySet);

    void Save(const string& savePath, const int pythonBatchId, bool saveDelta, const map<emb_key_t, KeyInfo>& keyInfo);

private:
    constexpr static int BLOCK_EMB_NUM = 100000; // 每次扩容分配10w条

    void RandomInit(void* addr, size_t embNum);

    int64_t GetEmptyEmbeddingAddress();

    void MallocEmbeddingBlock(int embNum);

    void SaveKey(const string& savePath, bool saveDelta, const map<emb_key_t, KeyInfo>& keyInfo);

    void SaveEmbAndOptim(const string& savePath);

    void SetOptimizerInfo(OptimizerInfo& optimizerInfo);

    void LoadKey(const string& savePath);

    void LoadEmbAndOptim(const string& savePath);

    void SaveEmbData(const string& savePath);

    void SaveOptimData(const string& savePath);

    // embedding地址的列表
    list<float*> embeddingList_;
    // 内存块列表
    vector<void*> memoryList_;

    vector<int64_t> deviceKey;
    vector<int64_t> embAddress;
    vector<int64_t> optimAddress;

    std::string optimName;
    std::vector<std::string> optimParams;
    std::map<std::string, vector<int64_t>> optimAddressMap;
    int deviceId = -1;

    int64_t firstAddress;
};
}

#endif // MX_REC_EMBEDDING_DYNAMIC_H
