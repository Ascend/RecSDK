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

#ifndef MX_REC_EMB_HASHMAP_H
#define MX_REC_EMB_HASHMAP_H

#include <vector>
#include <memory>
#include <array>
#include "absl/container/flat_hash_map.h"
#include "host_emb/host_emb.h"
#include "ssd_cache/cache_manager.h"
#include "utils/common.h"
#include "utils/time_cost.h"

namespace MxRec {
    using namespace std;

    class EmbHashMap {
    public:
        EmbHashMap() = default;

        void Init(const RankInfo& ri, const vector<EmbInfo>& embInfos, bool ifLoad = false);

        void Process(const string& embName, std::vector<emb_key_t>& keys, DDRParam& ddrParam, int channelId);

        auto GetHashMaps() -> absl::flat_hash_map<string, EmbHashMapInfo>;

        void LoadHashMap(absl::flat_hash_map<string, EmbHashMapInfo>& loadData);

        void EvictDeleteEmb(const string& embName, const vector<emb_key_t>& keys);

        absl::flat_hash_map<string, EmbHashMapInfo> embHashMaps;

        bool FindOffsetHelper(const emb_key_t& key, EmbHashMapInfo& embHashMap, int channelId, size_t& offset) const;

        void UpdateBatchId(const vector<emb_key_t>& keys, size_t currentBatchId, size_t keySize,
                           EmbHashMapInfo& embHashMap) const;

        bool FindSwapPosOld(const string& embName, emb_key_t key, size_t hostOffset, size_t currentBatchId,
                            size_t keepBatchId);

        std::vector<size_t>& GetEvictPos(const string& embName)
        {
            return embHashMaps.at(embName).evictPos;
        }

        bool isSSDEnabled { false };
        CacheManager* cacheManager;

    GTEST_PRIVATE:

        void FindOffset(const string& embName, const vector<emb_key_t>& keys,
                        size_t currentBatchId, size_t keepBatchId, int channelId);

        void AddCacheManagerTraceLog(const string& embTableName, const EmbHashMapInfo& embHashMap) const;

        void AddKeyFreqInfo(const string& embTableName, const emb_key_t& key, RecordType type) const;

        void ClearLookupAndSwapOffset(EmbHashMapInfo& embHashMap) const;

        void RefreshFreqInfoWithSwap(const string& embName, EmbHashMapInfo& embHashMap) const;

        RankInfo rankInfo;
        int swapId { 0 };
    };
}

#endif // MX_REC_EMB_HASHMAP_H
