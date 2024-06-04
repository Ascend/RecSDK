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

#ifndef MX_REC_L3_STORAGE_H
#define MX_REC_L3_STORAGE_H

#include <string>
#include <vector>

#include "utils/common.h"

using MxRec::emb_cache_key_t;
using std::string;
using std::vector;

namespace MxRec {

class L3Storage {
public:
    L3Storage();
    virtual ~L3Storage();

    virtual bool IsTableExist(const string& tableName);

    virtual bool IsKeyExist(const string& tableName, emb_cache_key_t key);

    virtual void CreateTable(const string& tableName, vector<string> savePaths, uint64_t maxTableSize);

    virtual int64_t GetTableAvailableSpace(const string& tableName);

    virtual void InsertEmbeddingsByAddr(const string& tableName, vector<emb_cache_key_t>& keys,
                                        vector<float*>& embeddingsAddr, uint64_t extEmbeddingSize);

    virtual void DeleteEmbeddings(const string& tableName, vector<emb_cache_key_t>& keys);

    virtual vector<vector<float>> FetchEmbeddings(const string& tableName, vector<emb_cache_key_t>& keys);

    virtual void Save(int step);

    virtual void Load(const string& tableName, vector<string> savePaths, uint64_t maxTableSize, int step);

    virtual void Start();

    virtual void Stop();

    virtual int64_t GetTableUsage(const string& tableName);

    virtual vector<std::pair<string, vector<emb_cache_key_t>>> ExportTableKey();
};
}  // namespace MxRec
#endif  // MX_REC_L3_STORAGE_H