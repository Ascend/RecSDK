/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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

#ifndef MX_REC_EMBEDDING_STATIC_H
#define MX_REC_EMBEDDING_STATIC_H

#include "emb_table/embedding_table.h"
#include "hybrid_mgmt/hybrid_mgmt.h"

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

    virtual void Key2OffsetForDp(std::vector<emb_key_t>& keys, int channel);

    virtual int64_t capacity() const;

    void Load(const string& savePath, map<string, unordered_set<emb_cache_key_t>>& trainKeySet,
              const vector<string>& warmStartTables);

    void Save(const string& savePath, const int pythonBatchId, bool saveDelta, const map<emb_key_t, KeyInfo>& keyInfo);

    void BackUpTrainStatus();

    void RecoverTrainStatus();

    vector<int64_t> GetDeviceOffset();

    std::vector<size_t> FindKeyOffset(std::vector<emb_key_t>& keys);

    void EmplaceKeyOffset(std::vector<emb_key_t>& keys, const std::vector<size_t>& newKeysIdx, int channel);

GTEST_PRIVATE:
    void SaveKey(const string& savePath, bool saveDelta, const map<emb_key_t, KeyInfo>& keyInfo);

    void LoadKey(const string& savePath);

    vector<int64_t> deviceKey;
    vector<int64_t> deviceOffset;
};

}

#endif // MX_REC_EMBEDDING_STATIC_H
