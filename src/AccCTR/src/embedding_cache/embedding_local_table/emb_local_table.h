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

#ifndef EMB_LOCAL_TABLE_H
#define EMB_LOCAL_TABLE_H

#include <memory>
#include <string>
#include <vector>

#include "offset_mapper/address_mapper.h"

namespace EmbCache {
struct EmbPoolParam {
    uint64_t prefillBufferSize;
    uint32_t refillThreadNum;
};

class EmbLocalTable {
public:
    EmbLocalTable() = default;

    ~EmbLocalTable() = default;

    bool Initialize(const EmbCacheInfo& embCacheInfo, uint64_t reserve,
                    const std::vector<InitializerInfo>& initializerInfos, const EmbPoolParam& embPoolParam);

    void UnInitialize();

    int FindAndPutIfNotFound(uint64_t key, uint64_t& value);

    bool Remove(uint64_t key);

    int RemoveByKeys(const std::vector<uint64_t>& keys, uint32_t threadNum);

    int Gather(uint64_t startAddr, const std::vector<uint64_t>& keys, uint32_t threadNum);

    int GatherAddrs(const std::vector<uint64_t>& keys, std::vector<float*>& addrs, uint32_t threadNum);

    int Scatter(uint64_t startAddr, const std::vector<uint64_t>& keys, uint32_t threadNum);

    int OneThreadHandle(uint64_t startAddr, const std::vector<uint64_t>& keys, bool isGather);

    int GatherAndRemove(uint64_t startAddr, const std::vector<uint64_t>& keys, uint32_t threadNum);

    std::vector<std::pair<uint64_t, uint64_t>> ExportVec();

    std::vector<char> Serialize();

    bool Deserialize(const std::vector<char>& buffer);

    uint32_t GetUsage();

    void GetEmbTableInfos(std::vector<uint64_t>& keys, std::vector<std::vector<float>>& embeddings,
                          std::vector<std::vector<float>>& optimizerSlots);

    bool LoadEmbTableInfos(const std::vector<uint64_t>& keys, const std::vector<std::vector<float>>& embeddings,
                           const std::vector<std::vector<float>>& optimizerSlots);

private:
    std::shared_ptr<AutoRefillEmbeddingMemoryPool> emExpendMemInfo;
    AddressMapper embMap;
    uint32_t embeddingSize;
    uint32_t extEmbeddingSize;

    template <class T>
    void insertData(std::vector<char>& buffer, T& data);

    template <class T>
    bool getData(const std::vector<char>& buffer, T& data, uint64_t& i);
};
}  // namespace EmbCache
#endif  // EMB_LOCAL_TABLE_H
