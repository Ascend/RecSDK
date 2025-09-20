/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.s
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#pragma once

#include <gmock/gmock.h>

#include "l3_storage/cache_manager.h"

namespace EmbCache {
class EmbCacheManagerMock : public EmbCacheManager {
public:
    MOCK_METHOD5(CreateCacheForTable, int(const EmbCacheInfo& embCacheInfo,
                                          const std::vector<InitializerInfo>& initializerInfos,
                                          int64_t invalidKey,
                                          uint64_t prefillBufferSize,
                                          uint32_t refillThreadNum));

    MOCK_METHOD4(GetSwapPairsAndKey2Offset, int(const EmbBaseInfo& info,
                                                std::vector<uint64_t>& keys,
                                                KeyOffsetPair& swapInKoPair,
                                                KeyOffsetPair& swapOutKoPair));

    MOCK_METHOD4(EmbeddingLookupAddrs, int(const std::string& tableName,
                                           const std::vector<uint64_t>& keys,
                                           std::vector<float*>& addrs,
                                           uint32_t threadNum));

    MOCK_METHOD4(EmbeddingUpdate, int(const std::string& tableName,
                                      const std::vector<uint64_t>& keys,
                                      float* embAddr,
                                      uint32_t threadNum));

    MOCK_METHOD3(EmbeddingRemove, int(const std::string& tableName,
                                      const std::vector<uint64_t>& keys,
                                      uint32_t threadNum));

    MOCK_METHOD2(RemoveEmbsByKeys, int(const std::string& tableName,
                                       const std::vector<uint64_t>& keys));

    MOCK_METHOD1(GetEmbTableNames, int(std::vector<std::string>& allTableNames));

    MOCK_METHOD2(ExportDeviceKeyOffsetPairs, int(const std::string& tableName,
                                                 std::vector<std::pair<uint64_t, uint64_t>>& koVec));

    MOCK_METHOD2(Serialize, int(const std::string& tableName,
                                std::vector<char>& buffer));

    MOCK_METHOD2(Deserialize, int(const std::string& tableName,
                                  const std::vector<char>& buffer));

    MOCK_METHOD0(Destroy, void());

    MOCK_METHOD1(GetUsage, uint32_t(const std::string& tableName));

    MOCK_METHOD4(GetEmbTableInfos, int(std::string tableName,
                                       std::vector<uint64_t>& keys,
                                       std::vector<std::vector<float>>& embeddings,
                                       std::vector<std::vector<float>>& optimizerSlots));

    MOCK_METHOD4(LoadEmbTableInfos, int(std::string tableName,
                                        const std::vector<uint64_t>& keys,
                                        const std::vector<std::vector<float>>& embeddings,
                                        const std::vector<std::vector<float>>& optimizerSlots));

    MOCK_METHOD1(BackUpTrainStatus, int(const std::string& tableName));

    MOCK_METHOD1(RecoverTrainStatus, int(const std::string& tableName));

    MOCK_METHOD0(ResetOffsetMappers, int());

    MOCK_METHOD1(GetPaddingKeysOffset, std::unordered_set<uint64_t>(const std::string& tableName));
};
} // namespace EmbCache