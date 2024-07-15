/* Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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

#ifndef EMBEDDING_CACHE_MANAGER_H
#define EMBEDDING_CACHE_MANAGER_H

#include <cstring>
#include <map>
#include <set>
#include <utility>

#include "embedding_cache.h"
#include "embedding_local_table/emb_local_table.h"
#include "error_code.h"
#include "offset_mapper/offset_mapper.h"

namespace EmbCache {
class EmbCacheManagerImpl : public EmbCacheManager {
public:
    EmbCacheManagerImpl() = default;

    ~EmbCacheManagerImpl() override = default;

    int CreateCacheForTable(const EmbCacheInfo& embCacheInfo, const std::vector<InitializerInfo>& initializerInfos,
                            int64_t invalidKey, uint64_t prefillBufferSize, uint32_t refillThreadNum) override;

    int GetSwapPairsAndKey2Offset(const std::string& tableName, std::vector<uint64_t>& keys,
                                  KeyOffsetPair& swapInKoPair, KeyOffsetPair& swapOutKoPair) override;

    int EmbeddingLookup(const std::string& tableName, const std::vector<uint64_t>& keys, float* embAddr,
                        uint32_t threadNum) override;

    int EmbeddingLookupAddrs(const std::string& tableName, const std::vector<uint64_t>& keys,
                             std::vector<float*>& addrs, uint32_t threadNum) override;

    int EmbeddingUpdate(const std::string& tableName, const std::vector<uint64_t>& keys, float* embAddr,
                        uint32_t threadNum) override;

    int EmbeddingRemove(const std::string& tableName, const std::vector<uint64_t>& keys, uint32_t threadNum) override;

    int EmbeddingLookupAndRemove(const std::string& tableName, const std::vector<uint64_t>& keys, float* embAddr,
                                 uint32_t threadNum) override;

    int RemoveEmbsByKeys(const std::string& tableName, const std::vector<uint64_t>& keys) override;

    int GetEmbTableNames(std::vector<std::string>& allTableNames) override;

    int ExportDeviceKeyOffsetPairs(const std::string& tableName,
                                   std::vector<std::pair<uint64_t, uint64_t>>& koVec) override;

    int Serialize(const std::string& tableName, std::vector<char>& buffer) override;

    int Deserialize(const std::string& tableName, const std::vector<char>& buffer) override;

    void Destroy() override;

    int GetEmbTableInfos(std::string tableName, std::vector<uint64_t>& keys,
                         std::vector<std::vector<float>>& embeddings,
                         std::vector<std::vector<float>>& optimizerSlots) override;

    int LoadEmbTableInfos(std::string tableName, const std::vector<uint64_t>& keys,
                          const std::vector<std::vector<float>>& embeddings,
                          const std::vector<std::vector<float>>& optimizerSlots) override;

    int BackUpTrainStatus(const std::string& tableName) override;

    int RecoverTrainStatus(const std::string& tableName) override;

    int ResetOffsetMappers() override;

    uint32_t GetUsage(const std::string& tableName) override;

private:
    std::map<std::string, EmbCacheInfo> embCacheInfos;
    std::map<std::string, OffsetMapper> offsetMappers;
    std::map<std::string, OffsetMapper> offsetMappersBackUp;
    std::map<std::string, EmbLocalTable> embTables;
    std::map<std::string, std::vector<std::pair<uint64_t, uint64_t>>> kvVecsBackUp;

    int CheckValidTableName(const std::string& tableName);

    bool CheckInitializer(uint32_t extEmbSize, std::vector<InitializerInfo> initializerInfos);

    bool CheckValidThreadNum(uint32_t threadNum);

    int CheckGetSwapPairsAndKey2Offset(const std::string& tableName, const KeyOffsetPair& swapInKoPair,
                                       const KeyOffsetPair& swapOutKoPair);

    int CheckCreateTableName(const std::string& tableName);
};
}  // namespace EmbCache
#endif  // EMBEDDING_CACHE_MANAGER_H
