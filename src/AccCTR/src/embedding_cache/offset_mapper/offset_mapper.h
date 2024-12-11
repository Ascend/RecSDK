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

#ifndef MXREC_OFFSET_MAPPER_H
#define MXREC_OFFSET_MAPPER_H

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_set>
#include <vector>

#include "embedding_cache.h"
#include "mapper_base.h"

namespace EmbCache {
constexpr int TRAIN_CHANNEL_ID = 0;

class OffsetMapper : public MapperBase {
public:
    OffsetMapper() = default;

    ~OffsetMapper() = default;

    OffsetMapper(const OffsetMapper& other)
        : maxCacheSize(other.maxCacheSize),
          useLength(other.useLength),
          validPos(new LimitedSet(*other.validPos)),
          evictPos(new LimitedSet(*other.evictPos)),
          pos2Key(other.pos2Key),
          lastBatchPos(other.lastBatchPos),
          evictSize(other.evictSize)
    {
    }

    OffsetMapper& operator=(const OffsetMapper& other)
    {
        if (this != &other) {
            delete validPos;
            validPos = nullptr;
            delete evictPos;
            evictPos = nullptr;

            if (other.validPos != nullptr) {
                validPos = new LimitedSet(*other.validPos);
            }
            if (other.evictPos != nullptr) {
                evictPos = new LimitedSet(*other.evictPos);
            }

            maxCacheSize = other.maxCacheSize;
            useLength = other.useLength;
            pos2Key = other.pos2Key;
            lastBatchPos = other.lastBatchPos;
            evictSize = other.evictSize;
        }
        return *this;
    }

    bool Initialize(uint64_t reserve, uint64_t maxSize = 0)
    {
        maxCacheSize = maxSize;
        useLength = 0;
        pos2Key.resize(maxSize);
        std::fill(pos2Key.begin(), pos2Key.end(), INVALID_KEY);
        try {
            validPos = new LimitedSet(maxSize);
            evictPos = new LimitedSet(maxSize);
        } catch (const std::bad_alloc& e) {
            return false;
        }
        return MapperBase::Initialize(reserve);
    }

    void UnInitialize() override
    {
        if (validPos != nullptr) {
            delete validPos;
            validPos = nullptr;
        }
        if (evictPos != nullptr) {
            delete evictPos;
            evictPos = nullptr;
        }
        pos2Key.clear();
        lastBatchPos.clear();
        useLength = 0;
        MapperBase::UnInitialize();
    }

    FkvState Remove(uint64_t key)
    {
        return MapperBase::Remove(key, [&](uint64_t value) {
            validPos->remove(value);
            auto pos = std::find(lastBatchPos.begin(), lastBatchPos.end(), value);
            if (pos != lastBatchPos.end()) {
                lastBatchPos.erase(pos);
            }
            evictPos->insert(value);
            evictSize++;
            return BeforeRemoveFuncState::BEFORE_SUCCESS;
        });
    }

    std::vector<std::pair<uint64_t, uint64_t>> ExportSortedKVPairs()
    {
        auto koVec = ExportVec();
        std::sort(koVec.begin(), koVec.end(), [](const auto& u, const auto& v) { return u.second < v.second; });
        return koVec;
    }

    uint64_t GetFreeLength()
    {
        return maxCacheSize - useLength + evictSize;
    }

    int GetSwapPairsAndKey2Offset(const EmbBaseInfo& info, std::vector<uint64_t>& keys, KeyOffsetPair& swapInKoPair,
                                  KeyOffsetPair& swapOutKoPair)
    {
        std::vector<uint64_t> swapInKeysID = FilterKeys(keys, swapInKoPair);

        uint64_t swapInCnt = 0;
        auto ret = FindInUsedPos(info, keys, swapInCnt, swapInKeysID, swapInKoPair, swapOutKoPair);
        if (ret != ock::ctr::H_OK) {
            return ret;
        }

        // 剩下的Key从om中分配位置
        ret = FindInOffsetMapper(info, keys, swapInKoPair, swapInCnt, swapInKeysID);
        if (ret != ock::ctr::H_OK) {
            return ret;
        }

        // 上个batch中的pos可被换出，加入validPos中
        for (uint64_t pos : lastBatchPos) {
            if (HM_UNLIKELY(pos == static_cast<uint64_t>(INVALID_KEY))) {
                continue;
            }
            validPos->insert(pos);
        }

        // 这里keys都已被替换成offset，这个batch使用的pos在下个batch不能被换出，移出validPos
        for (uint64_t pos : keys) {
            if (HM_UNLIKELY(pos == static_cast<uint64_t>(INVALID_KEY))) {
                continue;
            }
            validPos->remove(pos);
            evictPos->remove(pos);
        }

        lastBatchPos = keys;
        return ock::ctr::H_OK;
    }

    uint64_t GetUsage()
    {
        return useLength - evictSize;
    }

    uint64_t FindInUsedPos(const EmbBaseInfo& info, std::vector<uint64_t>& keys, uint64_t& swapInCnt,
                           std::vector<uint64_t>& swapInKeysID, KeyOffsetPair& swapInKoPair,
                           KeyOffsetPair& swapOutKoPair)
    {
        std::vector<uint64_t>& swapInKeys = swapInKoPair.first;
        std::vector<uint64_t>& swapInPos = swapInKoPair.second;
        std::vector<uint64_t>& swapOutKeys = swapOutKoPair.first;
        std::vector<uint64_t>& swapOutPos = swapOutKoPair.second;

        // 换出量 = 换入量 - 剩余空间
        uint64_t swapOutNum = swapInKeys.size() <= GetFreeLength() ? 0 : swapInKeys.size() - GetFreeLength();
        swapOutKeys.resize(swapOutNum);
        swapOutPos.resize(swapOutNum);

        // 空间不足，前swapOutNum个Key从evictPos中拿可换出位置
        for (uint64_t pos : *evictPos) {
            if (swapInCnt == swapInKeys.size()) {
                break;
            }
            // 记录swapInPos
            swapInPos[swapInCnt] = pos;
            // key->offset
            RecordPaddingKeysOffset(info, keys[swapInKeysID[swapInCnt]], pos);
            keys[swapInKeysID[swapInCnt]] = pos;
            // 放入新key-pos
            Put(swapInKeys[swapInCnt], pos);
            // 更新pos2Key
            pos2Key[pos] = swapInKeys[swapInCnt];
            swapInCnt++;
            evictSize--;
        }

        uint64_t swapOutCnt = 0;
        // 空间不足，前swapOutNum个Key从validPos中拿可换出位置
        for (uint64_t pos : *validPos) {
            if (swapOutCnt == swapOutNum) {
                break;
            }
            // 记录swapInPos
            swapInPos[swapInCnt] = pos;
            // key->offset
            RecordPaddingKeysOffset(info, keys[swapInKeysID[swapInCnt]], pos);
            keys[swapInKeysID[swapInCnt]] = pos;
            // 删除原key-pos，放入新key-pos
            uint64_t key = pos2Key[pos];
            MapperBase::Remove(key);
            Put(swapInKeys[swapInCnt], pos);
            // 记录swapOutKoPair
            swapOutKeys[swapOutCnt] = key;
            swapOutPos[swapOutCnt] = pos;
            // 更新pos2Key
            pos2Key[pos] = swapInKeys[swapInCnt];
            swapInCnt++;
            swapOutCnt++;
        }

        if (swapOutCnt < swapOutNum) {
            ock::ExternalLogger::PrintLog(ock::LogLevel::ERROR, "max cache size is too small");
            return ock::ctr::H_MAX_CACHESIZE_TOO_SMALL;
        }

        return ock::ctr::H_OK;
    }

    int FindInOffsetMapper(const EmbBaseInfo& info, std::vector<uint64_t>& keys, KeyOffsetPair& swapInKoPair,
                           uint64_t swapInCnt, std::vector<uint64_t>& swapInKeysID)
    {
        std::vector<uint64_t>& swapInKeys = swapInKoPair.first;
        std::vector<uint64_t>& swapInPos = swapInKoPair.second;

        for (uint64_t i = swapInCnt; i < swapInKeys.size(); i++) {
            swapInPos[i] = useLength++;
            if (HM_UNLIKELY(swapInPos[i] >= maxCacheSize)) {
                ock::ExternalLogger::PrintLog(ock::LogLevel::ERROR, "max cache size is too small");
                return ock::ctr::H_MAX_CACHESIZE_TOO_SMALL;
            }
            // 放入新key-pos
            Put(swapInKeys[i], swapInPos[i]);
            // 更新pos2Key
            pos2Key[swapInPos[i]] = swapInKeys[i];
            // key->offset
            RecordPaddingKeysOffset(info, keys[swapInKeysID[i]], swapInPos[i]);
            keys[swapInKeysID[i]] = swapInPos[i];
        }
        return ock::ctr::H_OK;
    }

    std::vector<uint64_t> FilterKeys(std::vector<uint64_t>& keys, KeyOffsetPair& swapInKoPair)
    {
        std::vector<uint64_t>& swapInKeys = swapInKoPair.first;
        std::vector<uint64_t>& swapInPos = swapInKoPair.second;

        std::vector<uint64_t> swapInKeysID;
        for (uint64_t i = 0; i < keys.size(); i++) {
            // Invalid key 不考虑
            if (HM_UNLIKELY(keys[i] == static_cast<uint64_t>(INVALID_KEY))) {
                continue;
            }
            // 在HBM中的key, 原地替换为pos后从validPos中移除
            // 不在HBM中的key，加入swapInKeys，并记录在keys中的下标，用于后续key->offset
            if (Find(keys[i], keys[i])) {
                validPos->remove(keys[i]);
            } else {
                swapInKeys.push_back(keys[i]);
                swapInKeysID.push_back(i);
            }
        }
        swapInPos.resize(swapInKeys.size());
        return swapInKeysID;
    }

    void RecordPaddingKeysOffset(const EmbBaseInfo& info, uint64_t key, uint64_t offset)
    {
        if (info.channelId != TRAIN_CHANNEL_ID || !info.paddingKeysMask) {
            return;
        }

        auto it = std::find(info.paddingKeys.begin(), info.paddingKeys.end(), key);
        if (it != info.paddingKeys.end()) {
            paddingKeysOffset.insert(offset);
        }
    }

    std::unordered_set<uint64_t> GetPaddingKeysOffset()
    {
        return paddingKeysOffset;
    }

private:
    uint64_t maxCacheSize{};            // HBM可容纳embedding条数
    uint64_t useLength{};               // HBM存储的embedding条数
    LimitedSet *validPos{};             // HBM中可被换出的位置
    LimitedSet *evictPos{};             // 淘汰出的位置
    std::vector<uint64_t> pos2Key;      // HBM中每个位置对应的key
    std::vector<uint64_t> lastBatchPos; // 上个batch的keys在HBM中占用的pos
    uint64_t evictSize;                 // evictPos的长度
    std::unordered_set<uint64_t> paddingKeysOffset;
};

}  // namespace EmbCache

#endif  // MXREC_OFFSET_MAPPER_H
