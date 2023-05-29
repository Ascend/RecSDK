/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: unique keys module
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */
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

#include "securec.h"

#include <spdlog/fmt/chrono.h>
#include <spdlog/fmt/bundled/ranges.h>

#include "absl/container/flat_hash_map.h"

#include "common.h"
#include "spinlock.h"
#include "time_cost.h"

using namespace MxRec;
using namespace std;

struct UniqueData {
    void *inputData;
    size_t dataSize;
    int32_t *restore;
    int64_t *uniqueVector;
    int32_t *splitSize;
    int64_t *keySend;
    int32_t *idCount;
    int32_t *idCountFill;
};

struct UniqueFlag {
    bool isInt64;
    bool useStatic;
    bool useHot;
};

struct UniqueForHot {
    int hotOffset;
    int *hotPos;
    map<int64_t, int> &hotMap;
    absl::flat_hash_map<int64_t, int> &keyCountMap;
};

struct UniqueThreadNum {
    int minThread;
    int maxThread;
};

class SendCntTooSmallError : public std::exception {
};

class GroupMethod {
public:
    inline int GroupCount()
    {
        return groupCount_;
    }
    inline int GroupId(uint64_t val)
    {
        return val & (groupCount_ - 1);
    }
    void SetGroupCount(int count)
    {
        groupCount_ = count;
    }

private:
    int groupCount_;
};

class SimpleThreadPool {
public:
    void SyncRun(const std::vector<std::function<int()>> &tasks)
    {
        std::vector<std::future<int>> futs;
        for (auto &task : tasks) {
            futs.push_back(std::async(task));
        }
        for (auto &fut : futs) {
            fut.wait();
        }
    }
};
template <int N = 4> class Dedup {
    static constexpr uint32_t kMinimalWorkloadPerWorker = 1 << 12;
    static const int kDefaultBucketCount = 1 << 24;
    static const int kDefaultBucketCountMask = kDefaultBucketCount - 1;

    template <int M> struct Meta {
        static_assert(M <= UNIQUE_MAX_BUCKET_WIDTH, "should be no larger than max bucket width");
        SpinLock lock;
        volatile int8_t count;
        int8_t pad[3];
        int32_t replace_base;
        volatile uint64_t data[M];
        std::atomic<uint16_t> idCount[M];
    } __attribute__((__aligned__(64)));

    struct Statistics {
        uint64_t totalUniques = 0;
        uint64_t totalOverflowUniques = 0;
    };

public:
    Dedup(int bucketCountPower2 = kDefaultBucketCount, int groups = 1)
            : bucketCount_(bucketCountPower2), bucketCountMask_(bucketCount_ - 1), groupCount_(groups)
    {
        void *area = aligned_alloc(64, sizeof(Meta<N>) * bucketCount_);
        table_ = reinterpret_cast<Meta<N> *>(area);
        Clear(bucketCount_);
    }

    ~Dedup()
    {
        free(table_);
    }

    static size_t BucketSize()
    {
        return sizeof(Meta<N>);
    }

    void Insert(uint64_t val)
    {
        int32_t h = static_cast<int32_t>(hash(val) & bucketCountMask_);
        Meta<N> *bucket = &table_[h];

        int8_t count = bucket->count;

        int totalCount = 0;

        for (int i = 0; i < count; ++i) {
            if (bucket->data[totalCount] == val) {
                bucket->idCount[totalCount]++;
                // found one
                return;
            }
            totalCount++;
        }
        // try again, this time with lock acquired
        if (count < N) {
            std::lock_guard<SpinLock> lg(bucket->lock);
            for (int i = totalCount; i < bucket->count; ++i) {
                if (bucket->data[totalCount] == val) {
                    bucket->idCount[totalCount]++;
                    // found one
                    return;
                }
                totalCount++;
            }
            if (totalCount < N) {
                bucket->data[totalCount] = val;
                bucket->count++;
                bucket->idCount[totalCount]++;
                return;
            }
        }
        // shift to the overflow reservior
        insertOverflow(val);
    }

    int32_t GetReplaceOffset(uint64_t val)
    {
        int32_t h = static_cast<int32_t>(hash(val) & bucketCountMask_);
        Meta<N> *bucket = &table_[h];

        int8_t count = bucket->count;
        int totalCount = 0;
        for (int i = 0; i < count; ++i) {
            if (bucket->data[totalCount] == val) {
                // found one
                return bucket->replace_base + totalCount;
            }
            totalCount++;
        }
        // try again, this time with lock acquired
        if (count < N) {
            std::lock_guard<SpinLock> lg(bucket->lock);
            for (int i = totalCount; i < bucket->count; ++i) {
                if (bucket->data[totalCount] == val) {
                    return bucket->replace_base + totalCount;
                }
                totalCount++;
            }
            if (totalCount < N) {
                return -1;
            }
        }
        return getReplaceOffsetFromOverflow(val);
    }

    int32_t GetReplaceOffsetUnsafe(uint64_t val)
    {
        int32_t h = static_cast<int32_t>(hash(val) & bucketCountMask_);
        Meta<N> *bucket = &table_[h];

        int totalCount = 0;
        for (int i = 0; i < bucket->count; ++i) {
            if (bucket->data[totalCount] == val) {
                // found one
                return bucket->replace_base + totalCount;
            }
            totalCount++;
        }
        if (totalCount < N) {
            return -1;
        }
        return getReplaceOffsetFromOverflowUnsafe(val);
    }

    bool Contains(uint64_t val)
    {
        int32_t h = static_cast<int32_t>(hash(val) & bucketCountMask_);
        Meta<N> *bucket = &table_[h];
        {
            std::lock_guard<SpinLock> lg(bucket->lock);
            int totalCount = 0;
            for (int i = 0; i < bucket->count; ++i) {
                if (bucket->data[totalCount] == val) {
                    return true;
                }
                totalCount++;
            }
            if (totalCount < N) {
                // bucket isn't filled, no hit for sure
                return false;
            }
        }
        return checkOverflow(val);
    }

    void Clear(uint64_t newBucketCountPowerOf2 = 0)
    {
        std::lock_guard<SpinLock> lg(overflowMutex_);
        if (newBucketCountPowerOf2 > 0 && newBucketCountPowerOf2 != (uint64_t)bucketCount_) {
            free(table_);
            bucketCount_ = newBucketCountPowerOf2;
            bucketCountMask_ = bucketCount_ - 1;
            table_ = reinterpret_cast<Meta<N> *>(aligned_alloc(64, sizeof(Meta<N>) * bucketCount_));
        }
        bzero(table_, sizeof(Meta<N>) * bucketCount_);
        overflow_.clear();
        idCountOverflow_.clear();
    }

    void NewParameter()
    {
        int32_t newBucketCountPowerOf2 = bucketCount_;

        if (stats_.totalUniques > 0 && stats_.totalOverflowUniques > kMinimalWorkloadPerWorker) {
            // Time to check the proper size of sharded tables for performance
            // sake.
            uint64_t shardedTableSize = newBucketCountPowerOf2 * N * groupCount_;
            int largeCount = 0;
            while (shardedTableSize > stats_.totalUniques * 4 && largeCount_ != 1) {
                // too large
                newBucketCountPowerOf2 >>= 1;
                shardedTableSize >>= 1;
                largeCount++;
            }

            int count = ((largeCount == 1) && (largeCount != largeCount_)) ? 2 : 1;
            for (int i = 0; i < count; i++) {
                if (stats_.totalOverflowUniques > kMinimalWorkloadPerWorker) {
                    newBucketCountPowerOf2 <<= 1;
                    shardedTableSize <<= 1;
                }
            }

            while (shardedTableSize < stats_.totalUniques + (stats_.totalUniques >> 2)) {
                newBucketCountPowerOf2 <<= 1;
                shardedTableSize <<= 1;
            }

            if (largeCount_ != 1) {
                largeCount_ = largeCount;
            }
        }

        Clear(newBucketCountPowerOf2);
        bucketCount_ = newBucketCountPowerOf2;
        stats_.totalUniques = 0;
        stats_.totalOverflowUniques = 0;
    }

    // Warning: functions below are not thread safe!
    //
    // Return the unique values
    // Also update the hash-order base of each bucket
    std::vector<uint64_t> Unique()
    {
        int32_t replace_offset = 0;
        std::vector<uint64_t> output;

        for (int i = 0; i < bucketCount_; ++i) {
            Meta<N> *bucket = &table_[i];
            if (bucket->count == 0) { // 如果桶为0，则跳过
                continue;
            }
            bucket->replace_base = replace_offset; // 取桶的偏移量
            for (int j = 0; j < bucket->count; ++j) {
                auto data = bucket->data[j];
                output.push_back(data); // 挨个桶取数据，然后填到output中去
            }
            replace_offset += bucket->count;
        }
        auto it = overflow_.begin(); // 取overflow里面的，也添加到output中去
        while (it != overflow_.end()) {
            output.push_back(it->first);
            it->second = replace_offset++; // 记录偏移量++
            ++it;
        }
        return output;
    }

    // Used by ShardedDedup Only!
    uint32_t UniqueRaw(int64_t *output, uint32_t priorTotal, int32_t *idCount)
    {
        uint32_t total = priorTotal;
        int32_t replace_offset = priorTotal;

        for (int i = 0; i < bucketCount_; ++i) {
            Meta<N> *bucket = &table_[i];
            if (bucket->count == 0) {
                continue;
            }
            bucket->replace_base = replace_offset;
            for (int j = 0; j < bucket->count; ++j) {
                idCount[total] = bucket->idCount[j];
                output[total++] = bucket->data[j];
            }
            replace_offset += bucket->count;
        }
        auto it = overflow_.begin();
        int32_t totalOverflow = 0;
        while (it != overflow_.end()) {
            idCount[total] = idCountOverflow_[it->first];
            output[total++] = it->first;
            it->second = replace_offset++;
            ++it;
            ++totalOverflow;
        }

        // set total overflow count
        stats_.totalUniques = total - priorTotal;
        stats_.totalOverflowUniques = totalOverflow;
        return total - priorTotal;
    }

    void handleHotKey(int key, map<int64_t, int> &hotMap, map<int64_t, int> &hotPosMap, int &hotCount) {
        auto hot = hotMap.find(key);
        if (hot != hotMap.end()) {
            if (hot->second == -1) {
                int pos = hotCount;
                hotMap[key] = pos;
                hotPosMap[key] = pos;
                hotCount++;
            } else {
                hotPosMap[key] = -1;
            }
        }
    }

    uint32_t UniqueRawForHot(int64_t *output, uint32_t priorTotal, int32_t* idCount,
                             map<int64_t, int> &hotMap, map<int64_t, int> &hotPosMap, int &hotCount,
                             absl::flat_hash_map<emb_key_t, int> &keyCountMap)
    {
        uint32_t total = priorTotal;
        int32_t replace_offset = priorTotal;

        for (int i = 0; i < bucketCount_; ++i) {
            Meta<N> *bucket = &table_[i];
            if (bucket->count == 0) {
                continue;
            }
            bucket->replace_base = replace_offset;
            for (int j = 0; j < bucket->count; ++j) {
                idCount[total] = bucket->idCount[j];
                output[total++] = bucket->data[j];
                handleHotKey(bucket->data[j], hotMap, hotPosMap, hotCount);
                keyCountMap[bucket->data[j]] = bucket->idCount[j];
            }
            replace_offset += bucket->count;
        }
        auto it = overflow_.begin();
        int32_t totalOverflow = 0;
        while (it != overflow_.end()) {
            idCount[total] = idCountOverflow_[it->first];
            keyCountMap[it->first] = idCountOverflow_[it->first];
            output[total++] = it->first;
            handleHotKey(it->first, hotMap, hotPosMap, hotCount);
            it->second = replace_offset++;
            ++it;
            ++totalOverflow;
        }

        // set total overflow count
        stats_.totalUniques = total - priorTotal;
        stats_.totalOverflowUniques = totalOverflow;
        return total - priorTotal;
    }

    std::vector<uint32_t> Replacement(const std::vector<uint64_t> &input, std::vector<uint64_t> *unique = nullptr,
                                      int32_t base = 0)
    {
        std::vector<uint32_t> output;
        if (unique) {
            *unique = std::move(Unique());
        }
        for (auto &val : input) {
            output.push_back(GetReplaceOffsetUnsafe(val) + base);
        }
        return output;
    }

private:
    int bucketCount_;
    int bucketCountMask_;
    int upperRangeIndex_;
    int groupCount_;
    int largeCount_ { 0 };
    Meta<N> *table_;
    std::unordered_map<uint64_t, int32_t> overflow_;
    std::unordered_map<uint64_t, int32_t> idCountOverflow_;
    SpinLock overflowMutex_;
    Statistics stats_;

    static inline uint64_t hash(uint64_t val)
    {
        return val ^ (val >> 16) ^ (val >> 32) ^ (val >> 48);
    }

    void insertOverflow(uint64_t val)
    {
        std::lock_guard<SpinLock> lg(overflowMutex_);
        auto it = overflow_.find(val);
        if (it == overflow_.end()) {
            overflow_[val] = 0;
        }
        idCountOverflow_[val]++;
    }

    bool checkOverflow(uint64_t val)
    {
        std::lock_guard<SpinLock> lg(overflowMutex_);
        return overflow_.find(val) != overflow_.end();
    }

    int32_t getReplaceOffsetFromOverflow(uint64_t val)
    {
        std::lock_guard<SpinLock> lg(overflowMutex_);
        auto it = overflow_.find(val);
        return (it != overflow_.end()) ? it->second : -1;
    }

    int32_t getReplaceOffsetFromOverflowUnsafe(uint64_t val)
    {
        auto it = overflow_.find(val);
        return (it != overflow_.end()) ? it->second : -1;
    }
}; // Dedup

#define CACHE_LINE_ALIGN(size) (((size) + 63ul) & ~63ul)

class OneSimpleGroupMethod {
public:
    inline int GroupCount()
    {
        return 1;
    }
    inline int GroupId(uint64_t val)
    {
        return 0;
    }
};

template <class GroupMethod = OneSimpleGroupMethod, int BucketWidth = 4> class ShardedDedup {
    static constexpr uint32_t kMinimalWorkloadPerWorker = 1 << 13;
    static constexpr int kDefaultDuplicateRatio = 4;
    static constexpr int kMinimalWorkerCount = 2;
    static constexpr int kMaximalWorkerCount = 32;

public:
    using DedupT = Dedup<BucketWidth>;

    ShardedDedup(const GroupMethod &groupMethod, int desiredSize, int send_cnt,
                 int estimatedDuplicateRatio = kDefaultDuplicateRatio)
            : groupMethod_(groupMethod), bucketCountPower2_(256), send_cnt_(send_cnt)
    {
        const int numOfGroupsInShard = groupMethod_.GroupCount();

        desiredSize += (desiredSize >> 1);
        while (bucketCountPower2_ * BucketWidth * numOfGroupsInShard * estimatedDuplicateRatio < desiredSize) {
            bucketCountPower2_ <<= 1;
        }
        for (int32_t i = 0; i < numOfGroupsInShard; ++i) {
            dedupShards_.emplace_back(new DedupT(bucketCountPower2_, numOfGroupsInShard));
        }
    }

    ~ShardedDedup() {}

    const int NumOfGroupsInEachShard()
    {
        return groupMethod_.GroupCount();
    }

    /* *
     * @brief given the input vector, compute unique values and partition
     * them into regions delimited by the partition boundaries passed
     * as ctor input (see above)
     *
     *
     * @param pool thread pool which is used by unique task
     * @param input the data input
     * @param size the size of the data input
     * @param uniqueVector unique values
     * @param uniqueSize unique of sizes
     * @param output the output vector of index values
     * @param uniqueIds unique ids final
     * @param idCount key count
     * @param idCountFill key count and filled zero by send count
     * @param isStatic output and idCount Fill isFilled
     * @param isInt64 input data is int64 or int32
     * @param useHot hot embedding
     * @param offset add hot map size
     * @param hotMap hot key map
     * @param keyCountMap record key count
     * @param minThreadCount min thread number
     * @param maxThreadCount max thread number
     */
    template <typename TaskReturnType, class ThreadPool>
    int Compute(ThreadPool *pool, UniqueData &uniqueData, UniqueFlag &uniqueFlag,
                UniqueForHot &uniqueForHot, UniqueThreadNum &uniqueThreadNum)
    {
        // Now kick off the computation

        void *input = uniqueData.inputData;
        const size_t size = uniqueData.dataSize;
        int64_t *uniqueVector = uniqueData.uniqueVector;
        int32_t *uniqueSize = uniqueData.splitSize;
        int32_t *output = uniqueData.restore;
        int64_t *uniqueIds = uniqueData.keySend;
        int32_t *idCount = uniqueData.idCount;
        int32_t *idCountFill = uniqueData.idCountFill;

        map<int64_t, int> &hotMap = uniqueForHot.hotMap;
        absl::flat_hash_map<int64_t, int> &keyCountMap = uniqueForHot.keyCountMap;
        int offset = uniqueForHot.hotOffset;
        int *hotPos = uniqueForHot.hotPos;

        bool useStatic = uniqueFlag.useStatic;
        bool useHot = uniqueFlag.useHot;
        bool isInt64 = uniqueFlag.isInt64;

        uint32_t minThreadCount = uniqueThreadNum.minThread;
        uint32_t maxThreadCount = uniqueThreadNum.maxThread;

        std::vector<int64_t> uniqueSizeVector;
        uniqueSizeVector.resize(groupMethod_.GroupCount());

        size_t inputSize = size;

        uint32_t threadNum = (inputSize + kMinimalWorkloadPerWorker - 1) / kMinimalWorkloadPerWorker;
        threadNum = std::min(maxThreadCount, std::max(threadNum, minThreadCount));

        size_t partSize = (inputSize + threadNum - 1) / threadNum;

        std::vector<std::function<TaskReturnType()>> tasks;

        for (uint32_t i = 0; i < threadNum; ++i) {
            const int numOfGroupsInShard = groupMethod_.GroupCount();
            tasks.push_back([this, i, input, inputSize, partSize, numOfGroupsInShard, isInt64]() -> TaskReturnType {
                for (uint64_t j = i * partSize; j < std::min(inputSize, (i + 1) * partSize); ++j) {
                    auto val = isInt64 ? ((int64_t *)input)[j] : ((int32_t *)input)[j];
                    auto group = groupMethod_.GroupId(val);
                    dedupShards_[group]->Insert(val);
                }
                return TaskReturnType {};
            });
        }
        spdlog::debug("unique finish insert");

        if (!tasks.empty()) {
            pool->SyncRun(tasks);
        }

        std::vector<uint32_t> baseVector;
        // Collect Unique and base vectors
        uint64_t base = 0;
        uint64_t total = 0;

        int hotCount = 0;
        map<int64_t, int> hotPosMap;

        for (int j = 0; j < groupMethod_.GroupCount(); ++j) {
            uint64_t inGroupTotal = 0;
            if (useHot) {
                inGroupTotal = dedupShards_[j]->UniqueRawForHot(uniqueVector, total, idCount,
                                                                hotMap, hotPosMap, hotCount,
                                                                keyCountMap);
            } else {
                inGroupTotal = dedupShards_[j]->UniqueRaw(uniqueVector, total, idCount);
            }
            uniqueSizeVector[j] = inGroupTotal;
            total += inGroupTotal;
        }

        spdlog::debug("unique finish uniqueRaw");

        baseVector.push_back(base);
        base += total;

        partSize = CACHE_LINE_ALIGN(partSize);

        int32_t *beginPtr = output;
        int32_t *finishPtr = beginPtr + inputSize;

        int32_t *partBeginPtr = beginPtr;
        int32_t *partEndPtr =
                reinterpret_cast<int32_t *>(CACHE_LINE_ALIGN(reinterpret_cast<uintptr_t>(partBeginPtr + partSize)));

        if(uniqueFlag.useStatic){
            for (int i = 0; i < groupMethod_.GroupCount(); i++) {
                if (send_cnt_ < uniqueSizeVector[i]){
                    spdlog::error("sendCnt should not be smaller than uniqueSize, sendCnt {}, uniqueSize {}", send_cnt_, uniqueSizeVector[i]);
                }
            }
        }

        std::vector<size_t> totalUniqueSize;
        totalUniqueSize.resize(groupMethod_.GroupCount());

        size_t totalNumber = 0;
        for (int i = 0; i < groupMethod_.GroupCount(); i++) {
            totalUniqueSize[i] = totalNumber;
            totalNumber += uniqueSizeVector[i];
        }
        spdlog::debug("uniqueSize: {}", totalNumber);

        tasks.clear();
        while (partBeginPtr < finishPtr) {
            if (partEndPtr > finishPtr) {
                partEndPtr = finishPtr;
            }
            if (partBeginPtr < partEndPtr) {
                // Due to cacheline alignment computation, the actual number of
                // threads created here may not match threadNum exactly but
                // should be +/-1 off.
                const int numOfGroupsInShard = groupMethod_.GroupCount();
                tasks.push_back([this, input, &baseVector, beginPtr, partBeginPtr, partEndPtr, numOfGroupsInShard,
                                        totalUniqueSize, useStatic, isInt64, useHot, offset, hotMap, hotPos, hotPosMap]() -> TaskReturnType {
                    for (int32_t *ptr = partBeginPtr; ptr < partEndPtr; ++ptr) {
                        auto val = isInt64 ? ((int64_t *)input)[ptr - beginPtr] : ((int32_t *)input)[ptr - beginPtr];
                        auto group = groupMethod_.GroupId(val);
                        uint32_t fillOffset = GetFillOffset(useStatic, baseVector, totalUniqueSize, val, group);
                        ComputeRestore(useHot, offset, hotMap, hotPos, hotPosMap, ptr, val, fillOffset);
                    }
                    return TaskReturnType {};
                });
            }
            partBeginPtr = partEndPtr;
            partEndPtr += partSize;
        }

        if (!tasks.empty()) {
            pool->SyncRun(tasks);
        }


        TileAndFill(groupMethod_.GroupCount(), uniqueVector, uniqueSize, uniqueIds, idCount, idCountFill, useStatic, uniqueSizeVector);

        return 0;
    }

    void ComputeRestore(bool useHot, int offset,const map<int64_t, int> &hotMap, int *hotPos,const map<int64_t, int> &hotPosMap,
                        int32_t *ptr, int64_t val, uint32_t fillOffset) const {
        auto hot = hotPosMap.find(val);
        if (!useHot) {
            *ptr = fillOffset;
        } else if (hot == hotPosMap.end()) {
            *ptr = offset + fillOffset;
        } else if (hot->second == -1) {
            *ptr = hotMap.find(val)->second;
        } else {
            hotPos[hot->second] = fillOffset;
            *ptr = hot->second;
        }
    }

    uint32_t GetFillOffset(bool useStatic, const vector<uint32_t> &baseVector, const vector<size_t> &totalUniqueSize,
                           int64_t val, int32_t group) {
        if (!useStatic) {
            return dedupShards_[group]->GetReplaceOffsetUnsafe(val) + baseVector[0];
        } else {
            return dedupShards_[group]->GetReplaceOffsetUnsafe(val) + baseVector[0] + send_cnt_ * group - totalUniqueSize[group];
        }
    }

    void TileAndFill(int groupCount, const int64_t *uniqueVector, int32_t *uniqueSize, int64_t *uniqueIds, const int32_t *idCount,
                     int32_t *idCountFill, bool useStatic, const std::vector<int64_t> &uniqueSizeVector) const {
        int start = 0;
        int index = 0;

        for (int i = 0; i < groupCount; i++) {
            if (i > 0) {
                index += uniqueSizeVector[i - 1];
            }

            if (useStatic) {
                start = i * send_cnt_;
            } else {
                start = index;
            }

            if (uniqueSizeVector[i] > 0) {
                size_t mem_size = uniqueSizeVector[i] * sizeof(int64_t);
                auto rc = memcpy_s(uniqueIds + start, mem_size, uniqueVector + index, mem_size);
                if (rc != 0) {
                    spdlog::error("[TileAndFill/uniqueIds] memcpy_s failded... mem_size: {}",mem_size);
                    throw std::runtime_error(fmt::format("[TileAndFill/uniqueIds] memcpy_s failded... mem_size: {}",mem_size).c_str());
                }
                mem_size = uniqueSizeVector[i] * sizeof(int32_t);
                rc = memcpy_s(idCountFill + start, mem_size, idCount + index, mem_size);
                if (rc != 0) {
                    spdlog::error("[TileAndFill/idCountFill] memcpy_s failded... mem_size: {}", mem_size);
                    throw std::runtime_error(fmt::format("[TileAndFill/idCountFill] memcpy_s failded... mem_size: {}", mem_size).c_str());
                }
            }

            int fillLen = send_cnt_ - uniqueSizeVector[i];
            if (useStatic) {
                for (int j = 0; j < fillLen; j++) {
                    uniqueIds[start + uniqueSizeVector[i] + j] = -1;
                    idCountFill[start + uniqueSizeVector[i] + j] = 0;
                }
            }

            uniqueSize[i] = uniqueSizeVector[i];
        }
    }

    void StartNewRound()
    {
        for (auto &s : dedupShards_) {
            s->NewParameter();
        }
    }

private:
    GroupMethod groupMethod_;
    int32_t bucketCountPower2_;
    std::vector<std::unique_ptr<DedupT>> dedupShards_;
    int32_t send_cnt_;
};
#endif
