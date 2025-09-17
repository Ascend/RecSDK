/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fast_hashmap.h"

#include "securec.h"

#include "common/constants.h"
#include "utils/logger.h"

namespace Embcache {

bool FastHashMap::Init(uint64_t reserve)
{
    // already initialized
    if (mOverflowEntryAlloc != nullptr) {
        return true;
    }

    // get proper bucket count
    uint64_t bucketCount = std::max(reserve, uint64_t(128));
    if (bucketCount > gPrimes[gPrimesCount - 1]) {
        bucketCount = gPrimes[gPrimesCount - 1];
    } else {
        uint64_t i = 0;
        while (i < gPrimesCount && gPrimes[i] < bucketCount) {
            i++;
        }
        if (i == gPrimesCount) {
            i--;
        }

        bucketCount = gPrimes[i];
    }

    // allocate buckets for sub-maps
    for (auto& mSubMap : mSubMaps) {
        NetHashBucket* tmp;
        if (!NewAndSetBuckets(bucketCount, 0, tmp)) {
            return false;
        }
        mSubMap = tmp;
    }

    // create overflow entry allocator
    mOverflowEntryAlloc = new (std::nothrow) NetHeapAllocator();
    if (HM_UNLIKELY(mOverflowEntryAlloc == nullptr)) {
        FreeSubMaps();
        LOG_ERROR("Failed to new overflow entry allocator, probably out of memory");
        return false;
    }

    // set bucket count
    mBucketCount = bucketCount;
    LOG_INFO("Hashmap inited, mBucketCount: {}", mBucketCount);

    return true;
}

void FastHashMap::Destroy()
{
    if (mOverflowEntryAlloc == nullptr) {
        return;
    }

    // free overflowed entries firstly
    FreeOverFlowedEntries();

    // Free sub map secondly
    FreeSubMaps();

    // free overflow entry at last
    delete mOverflowEntryAlloc;
    mOverflowEntryAlloc = nullptr;
    mBucketCount = 0;
}

FkvState FastHashMap::FindOrInsert(uint64_t key, uint64_t& value,
                                   const std::function<BeforePutFuncState()>& beforePutFunc)
{
    if (HM_UNLIKELY(key == 0)) {
        if (zeroInside) {
            value = zeroValue;
            return FkvState::FKV_EXIST;
        }

        if (__sync_bool_compare_and_swap(&zeroInside, false, true)) {
            BeforePutFuncState ret = beforePutFunc();
            if (HM_UNLIKELY(ret == BeforePutFuncState::BEFORE_FAIL)) {
                return FkvState::FKV_BEFORE_PUT_FUNC_FAIL;
            }

            if (HM_UNLIKELY(ret == BeforePutFuncState::BEFORE_NO_SPACE)) {
                return FkvState::FKV_NO_SPACE;
            }
            zeroValue = value;
            currentSize_++;
            return FkvState::FKV_NOT_EXIST;
        }
        return FkvState::FKV_KEY_CONFLICT;
    }

    // get bucket
    auto buck = &(mSubMaps[key % gSubMapCount][key % mBucketCount]);

    // loop all buckets linked
    while (buck != nullptr) {
        buck->spinLock.Lock();
        if (buck->Find(key, value)) {
            buck->spinLock.UnLock();
            return FkvState::FKV_EXIST;
        }
        buck->spinLock.UnLock();

        if (buck->next != nullptr) {
            buck = buck->next;
        } else {
            break;
        }
    }

    // did not found, now do put. continue from the last bucket in find
    return PutKeyValue(key, value, buck, beforePutFunc);
}

bool FastHashMap::Find(const uint64_t key, uint64_t& value)
{
    if (HM_UNLIKELY(key == 0)) {
        if (zeroInside) {
            value = zeroValue;
            return true;
        }
        return false;
    }

    // get bucket
    auto buck = &(mSubMaps[key % gSubMapCount][key % mBucketCount]);

    // loop all buckets linked
    while (buck != nullptr) {
        if (buck->Find(key, value)) {
            return true;
        }
        buck = buck->next;
    }
    return false;
}

FkvState FastHashMap::Remove(uint64_t key)
{
    if (HM_UNLIKELY(key == 0)) {
        if (zeroInside) {
            if (__sync_bool_compare_and_swap(&zeroInside, true, false)) {
                zeroValue = 0;
                currentSize_--;
            }
            return FkvState::FKV_EXIST;
        }
        return FkvState::FKV_NOT_EXIST;
    }

    /* get bucket */
    auto buck = &(mSubMaps[key % gSubMapCount][key % mBucketCount]);

    /* loop all buckets linked */
    uint64_t value;
    while (buck != nullptr) {
        if (buck->Find(key, value)) {
            buck->Remove(key);
            currentSize_--;
            return FkvState::FKV_EXIST;
        }

        buck = buck->next;
    }

    return FkvState::FKV_NOT_EXIST;
}

FkvState FastHashMap::Remove(uint64_t key, const std::function<BeforeRemoveFuncState(uint64_t)>& beforeRemoveFunc)
{
    if (HM_UNLIKELY(key == 0)) {
        if (!zeroInside) {
            return FkvState::FKV_NOT_EXIST;
        }
        if (__sync_bool_compare_and_swap(&zeroInside, true, false)) {
            auto ret = beforeRemoveFunc(zeroValue);
            if (HM_UNLIKELY(ret == BeforeRemoveFuncState::BEFORE_FAIL)) {
                return FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL;
            }
            zeroValue = 0;
            currentSize_--;
        }
        return FkvState::FKV_EXIST;
    }

    /* get bucket */
    auto buck = &(mSubMaps[key % gSubMapCount][key % mBucketCount]);

    /* loop all buckets linked */
    uint64_t value;
    while (buck != nullptr) {
        if (buck->Find(key, value)) {
            auto ret = buck->Remove(key, beforeRemoveFunc);
            if (HM_UNLIKELY(ret == FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL)) {
                return FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL;
            }

            currentSize_--;
            return FkvState::FKV_EXIST;
        }
        buck = buck->next;
    }

    return FkvState::FKV_NOT_EXIST;
}

std::vector<std::pair<uint64_t, uint64_t>> FastHashMap::Export()
{
    std::vector<std::pair<uint64_t, uint64_t>> results;
    if (zeroInside) {
        results.emplace_back(0, zeroValue);
    }

    for (auto& mSubMap : mSubMaps) {
        for (uint64_t i = 0; i < mBucketCount; i++) {
            auto buck = &mSubMap[i];
            ExtractKeyValueInBuck(buck, results);
        }
    }
    return results;
}

void FastHashMap::FreeSubMaps()
{
    for (auto& mSubMap : mSubMaps) {
        if (mSubMap != nullptr) {
            delete[] mSubMap;
            mSubMap = nullptr;
        }
    }
}

void FastHashMap::FreeOverFlowedEntries()
{
    for (auto& mSubMap : mSubMaps) {
        if (mSubMap == nullptr) {
            continue;
        }

        // free overflow entries in one sub map
        for (uint64_t buckIdx = 0; buckIdx < mBucketCount; ++buckIdx) {
            auto curBuck = mSubMap[buckIdx].next;
            NetHashBucket* nextOverflowEntryBuck = nullptr;

            // exit loop when curBuck is null
            while (curBuck != nullptr) {
                // assign next overflow buck to tmp variable
                nextOverflowEntryBuck = curBuck->next;

                // free this overflow bucket
                mOverflowEntryAlloc->Free(curBuck);

                // assign next to current
                curBuck = nextOverflowEntryBuck;
            }
        }
    }
}

bool FastHashMap::NewAndSetBuckets(const uint64_t& bucketCount, const int& value, NetHashBucket*& bucketsPtr)
{
    bucketsPtr = new (std::nothrow) NetHashBucket[bucketCount];
    if (HM_UNLIKELY(bucketsPtr == nullptr)) {
        FreeSubMaps();
        LOG_ERROR("Failed to new NetHashBucket, probably out of memory");
        return false;
    }

    // make physical page and set to zero
    size_t bucketsBytes = sizeof(NetHashBucket) * bucketCount;
    char* destBytePtr = reinterpret_cast<char*>(bucketsPtr);
    for (size_t i = 0; i < bucketsBytes; i += MEMSET_S_MAX_SIZE) {
        size_t bytesOnceSet = (i + MEMSET_S_MAX_SIZE <= bucketsBytes) ? MEMSET_S_MAX_SIZE : (bucketsBytes - i);
        auto ret = memset_s(destBytePtr + i, bytesOnceSet, value, bytesOnceSet);
        if (ret != 0) {
            delete[] bucketsPtr;
            bucketsPtr = nullptr;
            FreeSubMaps();
            LOG_ERROR("memset_s failed... size: {}, error code: {}", bucketsBytes, ret);
            return false;
        }
    }
    return true;
}

FkvState FastHashMap::PutKeyValue(uint64_t key, uint64_t& value, NetHashBucket* buck,
                                  const std::function<BeforePutFuncState()>& beforePutFunc)
{
    // try 8192 times
    for (uint16_t i = 0; i < 8192; i++) {
        // loop all buckets linked
        while (buck != nullptr) {
            // if there is an entry to put, just break
            buck->spinLock.Lock();
            FkvState putRet = buck->Put(key, value, beforePutFunc);
            buck->spinLock.UnLock();
            if (putRet == FkvState::FKV_NOT_EXIST) {
                currentSize_++;
                return FkvState::FKV_NOT_EXIST;
            }

            if (HM_UNLIKELY(putRet == FkvState::FKV_KEY_CONFLICT)) {
                return FkvState::FKV_KEY_CONFLICT;
            }

            if (HM_UNLIKELY(putRet == FkvState::FKV_BEFORE_PUT_FUNC_FAIL)) {
                return FkvState::FKV_BEFORE_PUT_FUNC_FAIL;
            }

            if (HM_UNLIKELY(putRet == FkvState::FKV_NO_SPACE)) {
                return FkvState::FKV_NO_SPACE;
            }

            // if no next bucket exist, just for break,
            // else move to the next bucket linked
            if (buck->next == nullptr) {
                break;
            } else {
                buck = buck->next;
            }
        }

        /*
         * if not put successfully in existing buckets, allocate a new one
         *
         * NOTES: just allocate memory, don't access new bucket in the spin lock scope,
         * if access new bucket, which could trigger physical memory allocation which
         * could trigger page fault, that is quite slow. In this case, spin lock
         * could occupy too much CPU
         */
        if (buck == nullptr) {
            LOG_ERROR("buck is nullptr");
            return FkvState::FKV_FAIL;
        }

        auto& lock = buck->spinLock;
        lock.Lock();
        // if other thread allocated new buck already, unlock and continue
        if (buck->next != nullptr) {
            buck = buck->next;
            lock.UnLock();
            continue;
        }

        // firstly entered thread allocate new bucket
        auto newBuck = static_cast<NetHashBucket*>(mOverflowEntryAlloc->Allocate(sizeof(NetHashBucket)));
        if (HM_UNLIKELY(newBuck == nullptr)) {
            lock.UnLock();
            LOG_ERROR("Failed to allocate new bucket");
            return FkvState::FKV_FAIL;
        }
        // link to current buck, set buck to new buck
        buck->next = newBuck;
        buck = newBuck;

        // unlock
        lock.UnLock();
    }
    return FkvState::FKV_FAIL;
}

void FastHashMap::ExtractKeyValueInBuck(NetHashBucket* buck, std::vector<std::pair<uint64_t, uint64_t>>& results)
{
    while (buck) {
        for (size_t k = 0; k < K_KV_NUM_IN_BUCKET; k++) {
            if (buck->keys[k] == 0) {
                continue;
            }
            results.emplace_back(buck->keys[k].load(), buck->values[k]);
        }
        buck = buck->next;
    }
}

}  // namespace Embcache
