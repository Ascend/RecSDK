/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 * Description: ssd cache module
 * Author: MindX SDK
 * Date: 2024/2/18
 */

#ifndef MXREC_DDR_PREPROCESS_MAPPER_H
#define MXREC_DDR_PREPROCESS_MAPPER_H

#include <stdexcept>
#include "lfu_cache.h"

namespace MxRec {
    /*
    * 专供keys处理的线程使用，每一个emb_local_table就有一个DDRPreProcessMapper
    * MapperBase中的桶存储k-v对，在这里value统一赋值为0
    */
    class PreProcessMapper {
    public:
        void Initialize(const string& embName, uint32_t vocabSize, uint32_t ssdVocabSize)
        {
            tableName = embName;
            lfuCache = LFUCache(embName);
            ddrAvailableSize = vocabSize;
            ssdAvailableSize = ssdVocabSize;
        }

        bool IsDDRKeyExist(uint64_t key)
        {
            return lfuCache.keyTable.find(key) != lfuCache.keyTable.end();
        }

        bool IsSSDKeyExist(uint64_t key)
        {
            return excludeDDRKeyCountMap.find(key) != excludeDDRKeyCountMap.end();
        }

        bool InsertDDRKey(uint64_t key)
        {
            if (IsDDRKeyExist(key)) {
                throw std::invalid_argument("InsertDDRKey failed! key already exist");
            }

            freq_num_t freq = excludeDDRKeyCountMap[key] + 1;
            lfuCache.PutWithInit(key, freq);
            return true;
        }

        bool InsertSSDKey(uint64_t key)
        {
            if (IsSSDKeyExist(key)) {
                throw std::invalid_argument("InsertSSDKey failed! key already exist");
            }

            excludeDDRKeyCountMap[key] = 1;
            return true;
        }

        bool RemoveSSDKey(uint64_t key)
        {
            if (!IsSSDKeyExist(key)) {
                throw std::invalid_argument("RemoveKey failed! key not exist");
            }
            excludeDDRKeyCountMap.erase(key);
            return true;
        }

        size_t DDRAvailableSize()
        {
            if (ddrAvailableSize < lfuCache.keyTable.size()) {
                throw std::invalid_argument("ddrAvailableSize < existKeys.size()");
            }
            return ddrAvailableSize - lfuCache.keyTable.size();
        }

        size_t SSDAvailableSize()
        {
            if (ssdAvailableSize < excludeDDRKeyCountMap.size()) {
                throw std::invalid_argument("ssdAvailableSize < existKeys.size()");
            }
            return ssdAvailableSize - excludeDDRKeyCountMap.size();
        }

        void GetAndDeleteLeastFreqDDRKey2SSD(uint64_t transNum, const std::vector<uint64_t>& keys,
                                             std::vector<uint64_t>& DDRSwapOutKeys)
        {
            LOG_DEBUG("start GetAndDeleteLeastFreqDDRKey2SSD, table:{}", tableName);
            std::vector<freq_num_t> DDRSwapOutCounts;
            lfuCache.GetAndDeleteLeastFreqKeyInfo(transNum, keys, DDRSwapOutKeys, DDRSwapOutCounts);
            for (uint64_t i = 0; i < DDRSwapOutKeys.size(); i++) {
                excludeDDRKeyCountMap[DDRSwapOutKeys[i]] = DDRSwapOutCounts[i];
            }
            if (DDRSwapOutCounts.size() != transNum) {
                throw std::invalid_argument(
                    "GetAndDeleteLeastFreqDDRKey2SSD failed! DDRSwapOutCounts.size()!=transNum");
            }
        }

        string tableName;
        uint64_t ddrAvailableSize = 0;
        uint64_t ssdAvailableSize = 0;
        LFUCache lfuCache;
        std::unordered_map<uint64_t, freq_num_t> excludeDDRKeyCountMap;
    };
}

#endif // MXREC_DDR_PREPROCESS_MAPPER_H
