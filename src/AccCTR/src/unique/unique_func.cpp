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

#include "unique_func.h"

namespace ock {
namespace ctr {
void Dedup::Insert(uint64_t val)
{
    auto h = static_cast<int32_t>(Hash(val) & bucketCountMask_);
    Meta<n> *bucket = &table_[h];

    int8_t count = bucket->count;

    int8_t totalCount = 0;

    for (int8_t i = 0; i < count; ++i) {
        if (bucket->data[totalCount] == val) {
            TryIncreaseIdCount(bucket->idCount[totalCount]);
            // found one
            return;
        }
        totalCount++;
    }
    // try again, this time with lock acquired
    if (count < n) {
        std::lock_guard<SpinLockG> lg(bucket->lock);
        for (int8_t j = totalCount; j < bucket->count; ++j) {
            if (bucket->data[totalCount] == val) {
                TryIncreaseIdCount(bucket->idCount[totalCount]);
                // found one
                return;
            }
            totalCount++;
        }
        if (totalCount < n) {
            bucket->data[totalCount] = val;
            bucket->count++;
            TryIncreaseIdCount(bucket->idCount[totalCount]);
            return;
        }
    }
    // shift to the overflow reservior
    InsertOverflow(val);
}

inline void Dedup::TryIncreaseIdCount(std::atomic<uint16_t> &val)
{
    if (idCountEnable_) {
        val++;
    }
}

int32_t Dedup::GetReplaceOffsetUnsafe(uint64_t val)
{
    auto h = static_cast<int32_t>(Hash(val) & bucketCountMask_);
    Meta<n> *bucket = &table_[h];

    int8_t totalCount = 0;
    for (int8_t i = 0; i < bucket->count; ++i) {
        if (bucket->data[totalCount] == val) {
            // found one
            return bucket->replaceBase + totalCount;
        }
        totalCount++;
    }
    if (totalCount < n) {
        return -1;
    }
    return GetReplaceOffsetFromOverflowUnsafe(val);
}

void Dedup::InitTable()
{
    void *area = aligned_alloc(64, sizeof(Meta<n>) * bucketCount_);
    if (area == nullptr) {
        throw AllocError();
    } else {
        table_ = reinterpret_cast<Meta<n> *>(area);
    }
}

void Dedup::Clear(uint64_t newBucketCountPowerOf2)
{
    std::lock_guard<SpinLockG> lg(overflowMutex_);
    if (newBucketCountPowerOf2 > 0 && newBucketCountPowerOf2 != bucketCount_) {
        if (table_ != nullptr) {
            free(table_);
            table_ = nullptr;
        }
        bucketCount_ = newBucketCountPowerOf2;
        bucketCountMask_ = bucketCount_ - 1;
        table_ = reinterpret_cast<Meta<n> *>(aligned_alloc(K_ALIGNMENT, sizeof(Meta<n>) * bucketCount_));
        if (table_ == nullptr) {
            throw AllocError();
        }
    }
    bzero(table_, sizeof(Meta<n>) * bucketCount_);
    overflow_.clear();
    idCountOverflow_.clear();
}

void Dedup::NewParameter()
{
    uint64_t newBucketCountPowerOf2 = bucketCount_;

    if (stats_.totalUniques > 0 && stats_.totalOverflowUniques > K_MINIMAL_WORKLOAD_PER_WORKER) {
        // Time to check the proper size of sharded tables for performance
        // sake.
        uint64_t shardedTableSize = 0;
        if (std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(n) / static_cast<uint64_t>(groupCount_)
        < newBucketCountPowerOf2) {
            shardedTableSize = static_cast<uint64_t>(std::numeric_limits<int>::max());
        } else {
            shardedTableSize = newBucketCountPowerOf2 * n * static_cast<uint64_t>(groupCount_);
        }

        int largeCount = 0;
        while (shardedTableSize > stats_.totalUniques * FACTOR && largeCount_ != 1) {
            // too large
            newBucketCountPowerOf2 >>= 1;
            shardedTableSize >>= 1;
            largeCount++;
        }

        int count = ((largeCount == 1) && (largeCount != largeCount_)) ? 2 : 1;
        for (int i = 0; i < count; i++) {
            if (stats_.totalOverflowUniques > K_MINIMAL_WORKLOAD_PER_WORKER) {
                newBucketCountPowerOf2 <<= 1;
                shardedTableSize <<= 1;
            }
        }

        while (shardedTableSize < stats_.totalUniques + (stats_.totalUniques >> FACTOR_BIT)) {
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

int32_t ShardedDedup::GetFillOffset(const std::vector<size_t> &totalUniqueSize, int64_t val, int32_t group)
{
    if (!conf.usePadding) {
        return dedupShards_[group]->GetReplaceOffsetUnsafe(val);
    } else {
        return dedupShards_[group]->GetReplaceOffsetUnsafe(val) + conf.paddingSize * group - totalUniqueSize[group];
    }
}


size_t ShardedDedup::CalThreadNum() const
{
    uint32_t threadNum = (conf.desiredSize + K_MINIMAL_WORKLOAD_PER_WORKER - 1) / K_MINIMAL_WORKLOAD_PER_WORKER;
    threadNum = std::min(conf.maxThreadNum, std::max(threadNum, conf.minThreadNum));
    return threadNum;
}

bool ShardedDedup::IsPaddingValid(UniqueOutSelf &uniqueOut)
{
    if (conf.outputType == OutputType::ENHANCED && conf.usePadding) {
        for (int i = 0; i < conf.shardingNum; i++) {
            if (conf.paddingSize < uniqueOut.uniqueIdCntInBucket[i]) {
                std::stringstream ssm;
                ssm << "paddingSize should not be smaller than uniqueSize, paddingSize " << conf.paddingSize <<
                    " , uniqueSize " << uniqueOut.uniqueIdCntInBucket[i];
                ExternalLogger::PrintLog(LogLevel::ERROR, ssm.str());
                return false;
            }
        }
    }
    return true;
}
}
}