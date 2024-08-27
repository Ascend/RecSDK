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

#ifndef MXREC_DDR_PREPROCESS_MAPPER_H
#define MXREC_DDR_PREPROCESS_MAPPER_H

#include <stdexcept>
#include "lfu_cache.h"
#include "utils/error.h"

namespace MxRec {
    /*
    * 专供keys处理的线程使用，每一个emb_local_table就有一个DDRPreProcessMapper
    * MapperBase中的桶存储k-v对，在这里value统一赋值为0
    */
    class PreProcessMapper {
    public:
        void Initialize(const string& embName, size_t ddrVocabSize, size_t l3StorageVocabSize)
        {
            tableName = embName;
            lfuCache = LFUCache(embName);
            ddrAvailableSize = ddrVocabSize;
            l3StorageAvailableSize = l3StorageVocabSize;
        }

        bool IsDDRKeyExist(uint64_t key)
        {
            return lfuCache.keyTable.find(key) != lfuCache.keyTable.end();
        }

        bool IsL3StorageKeyExist(uint64_t key)
        {
            return excludeDDRKeyCountMap.find(key) != excludeDDRKeyCountMap.end();
        }

        bool InsertDDRKey(uint64_t key)
        {
            if (IsDDRKeyExist(key)) {
                auto error = Error(ModuleName::M_L3_STORAGE, ErrorType::LOGIC_ERROR,
                                   "InsertDDRKey failed, key already exist.");
                LOG_ERROR(error.ToString());
                throw std::invalid_argument(error.ToString());
            }

            freq_num_t freq = excludeDDRKeyCountMap[key] + 1;
            lfuCache.PutWithInit(key, freq);
            return true;
        }

        bool InsertL3StorageKey(uint64_t key)
        {
            if (IsL3StorageKeyExist(key)) {
                auto error = Error(ModuleName::M_L3_STORAGE, ErrorType::LOGIC_ERROR,
                                   "InsertL3StorageKey failed! key already exist.");
                LOG_ERROR(error.ToString());
                throw std::invalid_argument(error.ToString());
            }

            excludeDDRKeyCountMap[key] = 1;
            return true;
        }

        bool RemoveL3StorageKey(uint64_t key)
        {
            if (!IsL3StorageKeyExist(key)) {
                auto error = Error(ModuleName::M_L3_STORAGE, ErrorType::LOGIC_ERROR,
                                   "RemoveKey failed, key not exist.");
                LOG_ERROR(error.ToString());
                throw std::invalid_argument(error.ToString());
            }
            excludeDDRKeyCountMap.erase(key);
            return true;
        }

        size_t DDRAvailableSize()
        {
            if (ddrAvailableSize < lfuCache.keyTable.size()) {
                auto error = Error(ModuleName::M_L3_STORAGE, ErrorType::LOGIC_ERROR,
                                   "ddrAvailableSize < existKeys.size().");
                LOG_ERROR(error.ToString());
                throw std::invalid_argument(error.ToString());
            }
            return ddrAvailableSize - lfuCache.keyTable.size();
        }

        size_t L3StorageAvailableSize()
        {
            if (l3StorageAvailableSize < excludeDDRKeyCountMap.size()) {
                auto error = Error(ModuleName::M_L3_STORAGE, ErrorType::LOGIC_ERROR,
                                   "l3StorageAvailableSize < existKeys.size().");
                LOG_ERROR(error.ToString());
                throw std::invalid_argument(error.ToString());
            }
            return l3StorageAvailableSize - excludeDDRKeyCountMap.size();
        }

        void GetAndDeleteLeastFreqDDRKey2L3Storage(uint64_t transNum, const std::vector<uint64_t>& keys,
                                                   std::vector<uint64_t>& DDRSwapOutKeys)
        {
            LOG_DEBUG("Start GetAndDeleteLeastFreqDDRKey2L3Storage, table:{}.", tableName);
            std::vector<freq_num_t> DDRSwapOutCounts;
            lfuCache.GetAndDeleteLeastFreqKeyInfo(transNum, keys, DDRSwapOutKeys, DDRSwapOutCounts);
            for (uint64_t i = 0; i < DDRSwapOutKeys.size(); i++) {
                excludeDDRKeyCountMap[DDRSwapOutKeys[i]] = DDRSwapOutCounts[i];
            }
            if (DDRSwapOutCounts.size() != transNum) {
                auto error = Error(ModuleName::M_L3_STORAGE, ErrorType::LOGIC_ERROR,
                                   "GetAndDeleteLeastFreqDDRKey2L3Storage failed! DDRSwapOutCounts.size()!=transNum.");
                LOG_ERROR(error.ToString());
                throw std::invalid_argument(error.ToString());
            }
        }

        string tableName;
        uint64_t ddrAvailableSize = 0;
        uint64_t l3StorageAvailableSize = 0;
        LFUCache lfuCache;
        std::unordered_map<uint64_t, freq_num_t> excludeDDRKeyCountMap;
    };
}

#endif // MXREC_DDR_PREPROCESS_MAPPER_H
