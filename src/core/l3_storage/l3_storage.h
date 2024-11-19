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

namespace MxRec {

class L3Storage {
public:
    L3Storage();
    virtual ~L3Storage();

    virtual bool IsTableExist(const std::string& tableName);

    virtual bool IsKeyExist(const std::string& tableName, emb_cache_key_t key);

    virtual void CreateTable(const std::string& tableName, std::vector<std::string> savePaths, uint64_t maxTableSize);

    virtual int64_t GetTableAvailableSpace(const std::string& tableName);

    virtual void InsertEmbeddingsByAddr(const std::string& tableName, std::vector<emb_cache_key_t>& keys,
                                        std::vector<float*>& embeddingsAddr, uint64_t extEmbeddingSize);

    virtual void DeleteEmbeddings(const std::string& tableName, std::vector<emb_cache_key_t>& keys);

    virtual std::vector<std::vector<float>> FetchEmbeddings(const std::string& tableName,
                                                            std::vector<emb_cache_key_t>& keys);

    virtual void Save(int step, const map<string, map<emb_key_t, KeyInfo>>& keyInfoMap);

    virtual void Save(int step);

    virtual void Load(const std::string& tableName, std::vector<std::string> savePaths, uint64_t maxTableSize,
                      int step);

    virtual void Start();

    virtual void Stop();

    virtual int64_t GetTableUsage(const std::string& tableName);

    virtual std::vector<std::pair<std::string, std::vector<emb_cache_key_t>>> ExportTableKey();
};
}  // namespace MxRec
#endif  // MX_REC_L3_STORAGE_H