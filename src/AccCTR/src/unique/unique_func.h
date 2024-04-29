/* Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
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

#ifndef SRC_UTILS_UNIQUE_H
#define SRC_UTILS_UNIQUE_H

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <map>
#include <limits>

#include "securec.h"
#include "common_includes.h"
#include "factory.h"

namespace ock {
namespace ctr {
using UniqueOutSelf = struct UniqueSelf {
    void *uniqueId = nullptr;           // 去重分桶填充之后最终的的ids(需要用户申请)必选
    uint32_t *index = nullptr;          // 去重后id的索引位置(需要用户申请)必选
    void *uniqueIdInBucket = nullptr;   // 去重之后的分桶内的ids(需要用户申请) shard开启后必选
    int *uniqueIdCntInBucket = nullptr; // 每个桶去重后的id个数(需要用户申请) shard开启后必选
    int *idCnt = nullptr;               // 每个id的重复次数(需要用户申请) idCnt开启后必选
    int *idCntFill = nullptr; // 每个id的重复次数带了填充(需要用户申请) idCnt和padding开启后必选
    int uniqueIdCnt = 0;      // 每个桶去重后的id个数(需要用户申请)
};

constexpr int UNIQUE_MAX_BUCKET_WIDTH = 5;

template <DataType> struct Map {};
template <> struct Map<DataType::INT64> {
    using type = int64_t;
};

template <> struct Map<DataType::INT32> {
    using type = int32_t;
};

template <DataType A> typename Map<A>::type *TypeTrans(void *input)
{
    return reinterpret_cast<typename Map<A>::type *>(input);
}

using StrategyFun = int (*)(const uint64_t &val, const int &groupCount);

class BucketStrategies {
public:
    static int SimpleGroupFun(const uint64_t &val, const int &groupCount)
    {
        return val % groupCount;
    }
};


class GroupMethod {
public:
    inline int GroupCount()
    {
        return groupCount_;
    }

    inline int GroupId(uint64_t val)
    {
        return strategyFun_(val, groupCount_);
    }
    void SetGroupCount(int count)
    {
        groupCount_ = count;
    }

    void SetStrategyFun(StrategyFun strategyFun)
    {
        strategyFun_ = strategyFun;
    }

    void SetStrategyFunByConf(BucketStrategy bucketStrategy)
    {
        if (bucketStrategy == BucketStrategy::MODULO) {
            SetStrategyFun(BucketStrategies::SimpleGroupFun);
        }
    }

private:
    int groupCount_;
    StrategyFun strategyFun_;
};

class Dedup {
    static constexpr uint32_t K_MINIMAL_WORKLOAD_PER_WORKER = 1 << 12;
    static constexpr size_t K_ALIGNMENT = 64;
    static const int kDefaultBucketCount = 1 << 24;
    static const int8_t n = 4;

    template <int M> struct Meta {
        static_assert(M <= UNIQUE_MAX_BUCKET_WIDTH, "should be no larger than max bucket width");
        SpinLockG lock;
        volatile int8_t count {};
        uint32_t replaceBase {};
        volatile uint64_t data[M] {};
        std::atomic<uint16_t> idCount[M] {};
    } __attribute__((__aligned__(64)));

    struct Statistics {
        uint64_t totalUniques = 0;
        uint64_t totalOverflowUniques = 0;
    };

public:
    explicit Dedup(int bucketCountPower2, int groups, bool idCountEnable)
        : bucketCount_(bucketCountPower2),
          bucketCountMask_(bucketCountPower2 - 1),
          groupCount_(groups),
          idCountEnable_(idCountEnable)
    {
        bucketCount_ = static_cast<uint64_t>(bucketCountPower2);
        bucketCountMask_ = bucketCount_ - 1;
        groupCount_ = groups;
        InitTable();
        Clear(bucketCount_);
    }

    ~Dedup()
    {
        if (table_ != nullptr) {
            free(table_);
        }
    }

public:
    void Insert(uint64_t val);
    int32_t GetReplaceOffsetUnsafe(uint64_t val);
    void InitTable();
    void TryIncreaseIdCount(std::atomic<uint16_t> &val);
    void Clear(uint64_t newBucketCountPowerOf2);
    void NewParameter();

    template <DataType T> uint32_t UniqueRaw(void *output, uint32_t priorTotal, int32_t *idCount)
    {
        uint32_t total = priorTotal;
        uint32_t replaceOffset = priorTotal;
        auto out = TypeTrans<T>(output);
        for (uint64_t i = 0; i < bucketCount_; ++i) {
            Meta<n> *bucket = &table_[i];
            if (bucket->count == 0) {
                continue;
            }
            bucket->replaceBase = replaceOffset;
            for (int j = 0; j < bucket->count; ++j) {
                if (idCountEnable_) {
                    idCount[total] = bucket->idCount[j];
                }
                out[total++] = static_cast<typename Map<DataType::INT64>::type>(bucket->data[j]);
            }
            replaceOffset += bucket->count;
        }
        auto it = overflow_.begin();
        int32_t totalOverflow = 0;
        while (it != overflow_.end()) {
            if (idCountEnable_) {
                idCount[total] = static_cast<int32_t>(idCountOverflow_[it->first]);
            }
            out[total++] = it->first;
            it->second = replaceOffset++;
            ++it;
            ++totalOverflow;
        }

        // set total overflow count
        stats_.totalUniques = static_cast<uint64_t>(total - priorTotal);
        stats_.totalOverflowUniques = static_cast<uint64_t>(totalOverflow);
        return total - priorTotal;
    }

private:
    uint64_t bucketCount_ = static_cast<uint64_t>(kDefaultBucketCount);
    uint64_t bucketCountMask_;
    int groupCount_ = 1;
    int largeCount_ { 0 };
    Meta<n> *table_ {};
    std::unordered_map<uint64_t, uint32_t> overflow_;
    std::unordered_map<uint64_t, uint32_t> idCountOverflow_;
    SpinLockG overflowMutex_;
    Statistics stats_;
    bool idCountEnable_ { false };

    static inline uint64_t Hash(uint64_t val)
    {
        return val ^ (val >> HASH_L_L) ^ (val >> HASH_L_L) ^ (val >> HASH_H);
    }

    void InsertOverflow(uint64_t val)
    {
        std::lock_guard<SpinLockG> lg(overflowMutex_);
        auto it = overflow_.find(val);
        if (it == overflow_.end()) {
            overflow_[val] = 0;
        }

        if (idCountEnable_) {
            idCountOverflow_[val]++;
        }
    }

    int32_t GetReplaceOffsetFromOverflowUnsafe(uint64_t val)
    {
        auto it = overflow_.find(val);
        return (it != overflow_.end()) ? it->second : -1;
    }
}; // Dedup

class ShardedDedup {
    static constexpr uint32_t K_MINIMAL_WORKLOAD_PER_WORKER = 1 << 13;
    static constexpr int K_DEFAULT_DUPLICATE_RATIO = 4;
    static constexpr int K_BUCKET_WIDTH = 4;

public:
    using DedupT = Dedup;

    ShardedDedup(const GroupMethod &groupMethod, const UniqueConf &uniqueConf,
        int estimatedDuplicateRatio = K_DEFAULT_DUPLICATE_RATIO)
        : groupMethod_(groupMethod), bucketCountPower2_(DEFAULT_NUM), conf(uniqueConf), partSize(0)
    {
        const int numOfGroupsInShard = groupMethod_.GroupCount();
        uint32_t totalSize = conf.desiredSize + (conf.desiredSize >> 1);
        while (bucketCountPower2_ * static_cast<uint32_t>(K_BUCKET_WIDTH) *
        static_cast<uint32_t>(numOfGroupsInShard) * static_cast<uint32_t>(estimatedDuplicateRatio) < totalSize) {
            bucketCountPower2_ <<= 1;
        }

        idCountEnable_ = (conf.outputType == OutputType::ENHANCED) && conf.useIdCount;
        try {
            for (int32_t i = 0; i < numOfGroupsInShard; ++i) {
                auto obj = new DedupT(bucketCountPower2_, numOfGroupsInShard, idCountEnable_);
                dedupShards_.emplace_back(obj);
            }
        } catch (const std::bad_alloc& e) {
            ExternalLogger::PrintLog(LogLevel::ERROR, "Memory allocation failed during loop: " + std::string(e.what()));
            throw;
        }
    }

    ~ShardedDedup() = default;

    void StartNewRound()
    {
        for (auto &s : dedupShards_) {
            s->NewParameter();
        }
    }

public:
    template <DataType T> int Compute(UniqueIn &uniqueIn, UniqueOutSelf &uniqueOut)
    {
        try {
            if (!firstEnterFlag_) {
                StartNewRound();
            }
        } catch (AllocError &) {
            ExternalLogger::PrintLog(LogLevel::ERROR, "memory alloc error");
            return H_MEMORY_ALLOC_ERROR;
        }
        firstEnterFlag_ = false;
        size_t threadNum = CalThreadNum();
        partSize = (uniqueIn.inputIdCnt + threadNum - 1) / threadNum;

        int ret = InsertVal<T>(uniqueIn, threadNum);
        if (ret != H_OK) {
            return ret;
        }

        DoUniqueRaw<T>(uniqueOut);

        partSize = CacheLineAlign(partSize);

        if (!IsPaddingValid(uniqueOut)) {
            return H_PADDING_SMALL;
        }

        std::vector<size_t> totalUniqueSize;
        totalUniqueSize.resize(conf.shardingNum);

        if (conf.outputType == OutputType::ENHANCED) {
            int totalNumber = 0;
            for (int i = 0; i < conf.shardingNum; i++) {
                totalUniqueSize[i] = static_cast<size_t>(totalNumber);
                if (conf.useSharding) {
                    totalNumber += uniqueOut.uniqueIdCntInBucket[i];
                }
            }
        }

        ret = CalUniqueOut<T>(uniqueIn, uniqueOut, totalUniqueSize);
        if (ret != H_OK) {
            ExternalLogger::PrintLog(LogLevel::ERROR, "CalUniqueOut ERROR");
            return ret;
        }

        if (conf.outputType == OutputType::ENHANCED) {
            HandleTileAndFill<T>(uniqueIn, uniqueOut);
        }

        return H_OK;
    }

private:
    template <typename T> T CacheLineAlign(T size)
    {
        return (((size) + 63uL) & ~63uL);
    }

    bool IsPaddingValid(UniqueOutSelf &uniqueOut);

    size_t CalThreadNum() const;

    int32_t GetFillOffset(const std::vector<size_t> &totalUniqueSize, int64_t val, int32_t group);

    template <DataType T> int HandleTileAndFill(UniqueIn &uniqueIn, UniqueOutSelf &uniqueOut)
    {
        int ret = H_OK;
        if (conf.useSharding) { // 使能shard
            ret = TileAndFill<T>(uniqueOut.uniqueIdInBucket, uniqueOut.uniqueIdCntInBucket, uniqueOut.uniqueId,
                uniqueOut.idCnt, uniqueOut.idCntFill);
        } else if (!conf.useSharding && conf.useIdCount) { // 不使能shard和使能特征计数
            std::vector<int32_t> count;
            count.emplace_back(uniqueOut.uniqueIdCnt); // 记录去重后id个数
            ret = TileAndFill<T>(uniqueOut.uniqueId, count.data(), uniqueOut.uniqueId, uniqueOut.idCnt,
                uniqueOut.idCntFill);
        }

        if (ret != H_OK) {
            ExternalLogger::PrintLog(LogLevel::ERROR, "TileAndFill ERROR");
            return ret;
        }

        return H_OK;
    }

    template <DataType T> void DoUniqueRaw(UniqueOutSelf &uniqueOut)
    {
        // Collect Unique and base vectors
        int32_t total = 0;
        for (int j = 0; j < groupMethod_.GroupCount(); ++j) {
            uint64_t inGroupTotal;
            if (conf.outputType == OutputType::ENHANCED) {
                if (conf.useSharding && conf.useIdCount) {
                    inGroupTotal = dedupShards_[j]->UniqueRaw<T>(uniqueOut.uniqueIdInBucket, total,
                        uniqueOut.idCnt); // 特征计数使能和shard同时使能
                    uniqueOut.uniqueIdCntInBucket[j] = static_cast<int32_t>(inGroupTotal);
                } else if (!conf.useSharding && conf.useIdCount) {
                    inGroupTotal = dedupShards_[j]->UniqueRaw<T>(uniqueOut.uniqueId, total,
                        uniqueOut.idCnt); // 特征计数使能和shard不使能
                } else if (conf.useSharding && !conf.useIdCount) {
                    inGroupTotal = dedupShards_[j]->UniqueRaw<T>(uniqueOut.uniqueIdInBucket, total,
                        nullptr); // 特征计数使能和shard不使能
                    uniqueOut.uniqueIdCntInBucket[j] = static_cast<int>(inGroupTotal);
                } else {
                    inGroupTotal = dedupShards_[j]->UniqueRaw<T>(uniqueOut.uniqueId, total,
                        nullptr); // 特征计数不使能和shard不使能，跟普通unique对等
                }
            } else {
                inGroupTotal = dedupShards_[j]->UniqueRaw<T>(uniqueOut.uniqueId, total, nullptr);
            }
            total += static_cast<int32_t>(inGroupTotal);
        }
        uniqueOut.uniqueIdCnt = total;
    }

    template <DataType T>
    int TileAndFill(void *uniqueIdInBucket, const int32_t *uniqueSizeInBucket, void *uniqueIds, const int32_t *idCnt,
        int32_t *idCntFill)
    {
        int start = 0;
        int index = 0;

        auto uIdInBucket = TypeTrans<T>(uniqueIdInBucket);
        auto uIds = TypeTrans<T>(uniqueIds);

        for (int i = 0; i < conf.shardingNum; i++) {
            GetIndexAndStart(uniqueSizeInBucket, conf.usePadding, i, start, index);

            uint32_t memSize = 0;
            if (T == DataType::INT64) {
                memSize = uniqueSizeInBucket[i] * sizeof(int64_t);
            } else if (T == DataType::INT32) {
                memSize = uniqueSizeInBucket[i] * sizeof(int32_t);
            }

            if (memSize == 0) {
                continue;
            }

            auto rc = memcpy_s(uIds + start, memSize, uIdInBucket + index, memSize);
            int ret = PrintMemCpyLog(rc, memSize, "[TileAndFill/uniqueIds]");
            if (ret != 0) {
                return ret;
            }

            if (conf.useIdCount && conf.usePadding) {
                memSize = uniqueSizeInBucket[i] * sizeof(int32_t);
                rc = memcpy_s(idCntFill + start, memSize, idCnt + index, memSize);
                ret = PrintMemCpyLog(rc, memSize, "[TileAndFill/idCntFill]");
            }
            if (ret != 0) {
                return ret;
            }
        }

        if (conf.usePadding) {
            HandleFill<T>(uIds, uniqueSizeInBucket, idCntFill);
        }

        return H_OK;
    }

    int PrintMemCpyLog(int rc, const uint32_t dstSize, const std::string &logMsg)
    {
        if (rc != 0) {
            std::stringstream ssm;
            ssm << "[" << logMsg << "] memcpy_s failed... dstSize: " << dstSize;
            ExternalLogger::PrintLog(LogLevel::ERROR, ssm.str());
            return H_COPY_ERROR;
        } else {
            return H_OK;
        }
    }

    template <DataType T>
    void HandleFill(typename Map<T>::type *uIds, const int32_t *uniqueSizeInBucket, int32_t *idCntFill)
    {
        int start = 0;
        int index = 0;

        for (int i = 0; i < conf.shardingNum; i++) {
            GetIndexAndStart(uniqueSizeInBucket, conf.usePadding, i, start, index);

            int fillLen = conf.paddingSize - uniqueSizeInBucket[i];
            for (int j = 0; j < fillLen; j++) {
                uIds[start + uniqueSizeInBucket[i] + j] = conf.paddingVal; // padding填充
            }

            if (idCntFill != nullptr) {
                for (int y = 0; y < fillLen; y++) {
                    idCntFill[start + uniqueSizeInBucket[i] + y] = 0; // 特征计数填充
                }
            }
        }
    }

    void GetIndexAndStart(const int32_t *uniqueSizeInBucket, bool usePadding, int shardingNumber, int &start,
        int &index)
    {
        if (shardingNumber > 0) {
            index += uniqueSizeInBucket[shardingNumber - 1];
        }

        if (usePadding) {
            start = shardingNumber * conf.paddingSize;
        } else {
            start = index;
        }
    }

    template <DataType T> int InsertVal(UniqueIn &uniqueIn, size_t threadNum)
    {
        auto val = TypeTrans<T>(uniqueIn.inputId);
        std::vector<std::function<void()>> tasks;
        int ret = H_OK;
        for (uint32_t i = 0; i < threadNum; ++i) {
            uint32_t start = i * partSize;
            uint32_t end = std::min(uniqueIn.inputIdCnt, (i + 1) * partSize);
            tasks.push_back([this, val, start, end, &ret]() {
                for (uint64_t j = start; j < end; ++j) {
                    auto value = val[j];
                    if (value > conf.maxIdVal) {
                        ExternalLogger::PrintLog(LogLevel::ERROR, "id val is larger than maxIdVal");
                        ret = H_ID_LARGE;
                        break;
                    }
                    auto group = groupMethod_.GroupId(value);
                    dedupShards_[group]->Insert(value);
                }
            });
        }

        try {
            if (!tasks.empty()) {
                auto threader = ExternalThreader::Instance();
                if (threader == nullptr) {
                    return H_ADDRESS_NULL;
                }
                threader->Run(tasks);
            }
        } catch (NullptrError &) {
            return H_ADDRESS_NULL;
        }

        return ret;
    }

    template <DataType T>
    int CalUniqueOut(UniqueIn &uniqueIn, UniqueOutSelf &uniqueOut, std::vector<size_t> &totalUniqueSize)
    {
        uint32_t *beginPtr = uniqueOut.index;
        uint32_t *finishPtr = beginPtr + uniqueIn.inputIdCnt;
        uint32_t *partBeginPtr = beginPtr;
        auto alignedAddress = CacheLineAlign(reinterpret_cast<uintptr_t>(partBeginPtr + partSize));
        auto *partEndPtr = reinterpret_cast<uint32_t *>(alignedAddress);
        std::vector<std::function<void()>> tasks;
        auto val = TypeTrans<T>(uniqueIn.inputId);
        while (partBeginPtr < finishPtr) {
            if (partEndPtr > finishPtr) {
                partEndPtr = finishPtr;
            }
            if (partBeginPtr < partEndPtr) {
                // Due to cacheline alignment computation, the actual number of
                // threads created here may not match threadNum exactly but
                // should be +/-1 off.
                tasks.push_back([this, val, beginPtr, partBeginPtr, partEndPtr, totalUniqueSize]() {
                    for (uint32_t *ptr = partBeginPtr; ptr < partEndPtr; ++ptr) {
                        auto group = groupMethod_.GroupId(val[ptr - beginPtr]);
                        int32_t fillOffset = GetFillOffset(totalUniqueSize, val[ptr - beginPtr], group);
                        *ptr = fillOffset;
                    }
                });
            }
            partBeginPtr = partEndPtr;
            partEndPtr += partSize;
        }

        try {
            if (!tasks.empty()) {
                auto threader = ExternalThreader::Instance();
                if (threader == nullptr) {
                    return H_ADDRESS_NULL;
                }
                threader->Run(tasks);
            }
        } catch (NullptrError &) {
            return H_ADDRESS_NULL;
        }
        return H_OK;
    }

private:
    GroupMethod groupMethod_;
    uint32_t bucketCountPower2_;
    UniqueConf conf;
    std::vector<std::unique_ptr<DedupT>> dedupShards_ {};
    uint32_t partSize;
    bool firstEnterFlag_ = false;
    bool idCountEnable_ { false };
};
}
}
#endif