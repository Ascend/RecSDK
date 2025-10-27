/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mapper_base.h"

namespace {
    constexpr size_t BUCKET_IDX_FIRST = 0;
    constexpr size_t BUCKET_IDX_SECOND = 1;
    constexpr size_t BUCKET_IDX_THIRD = 2;
}

namespace EmbCache {

FkvState NetHashBucket::PutTrySlot(uint64_t key, uint64_t &value, std::atomic<uint64_t> &keySlot,
    uint64_t &valueSlot, const std::function<BeforePutFuncState()> &beforePutFunc) const
{
    uint64_t oldKey = 0;
    if (keySlot.load(std::memory_order_relaxed) == 0 && keySlot.compare_exchange_strong(oldKey, key)) {
        BeforePutFuncState ret = beforePutFunc();
        if (HM_UNLIKELY(ret == BeforePutFuncState::BEFORE_FAIL)) {
            keySlot = 0;
            return FkvState::FKV_BEFORE_PUT_FUNC_FAIL;
        }
        if (HM_UNLIKELY(ret == BeforePutFuncState::BEFORE_NO_SPACE)) {
            keySlot = 0;
            return FkvState::FKV_NO_SPACE;
        }
        valueSlot = value;
        return FkvState::FKV_NOT_EXIST;
    }

    if (HM_UNLIKELY(oldKey == key)) {
        return FkvState::FKV_KEY_CONFLICT;
    }

    return FkvState::FKV_FAIL;
}

FkvState NetHashBucket::Put(uint64_t key, uint64_t &value,
    const std::function<BeforePutFuncState()> &beforePutFunc)
{
    /* don't put them into loop, flat code is faster than loop */
    FkvState result;

    result = PutTrySlot(key, value, keys[BUCKET_IDX_FIRST],
                        values[BUCKET_IDX_FIRST], beforePutFunc);
    if (result != FkvState::FKV_FAIL) {
        return result;
    }

    result = PutTrySlot(key, value, keys[BUCKET_IDX_SECOND],
                        values[BUCKET_IDX_SECOND], beforePutFunc);
    if (result != FkvState::FKV_FAIL) {
        return result;
    }

    result = PutTrySlot(key, value, keys[BUCKET_IDX_THIRD],
                        values[BUCKET_IDX_THIRD], beforePutFunc);
    return result;
}

bool NetHashBucket::Find(const uint64_t key, uint64_t &value)
{
    /*
     * expand the loop, instead of put them into a for/while loop for performance
     */
    if (key == keys[BUCKET_IDX_FIRST].load(std::memory_order_relaxed)) {
        value = values[BUCKET_IDX_FIRST];
        return true;
    }

    if (key == keys[BUCKET_IDX_SECOND].load(std::memory_order_relaxed)) {
        value = values[BUCKET_IDX_SECOND];
        return true;
    }

    if (key == keys[BUCKET_IDX_THIRD].load(std::memory_order_relaxed)) {
        value = values[BUCKET_IDX_THIRD];
        return true;
    }

    return false;
}

FkvState NetHashBucket::Remove(uint64_t key)
{
    /* don't put them into loop, flat code is faster than loop */
    uint64_t oldValue = key;
    if (keys[BUCKET_IDX_FIRST].load(std::memory_order_relaxed) == key &&
        keys[BUCKET_IDX_FIRST].compare_exchange_strong(oldValue, 0)) {
        values[BUCKET_IDX_FIRST] = 0;
        return FkvState::FKV_EXIST;
    }
    if (HM_UNLIKELY(oldValue == 0)) {
        return FkvState::FKV_EXIST;
    }
    oldValue = key;

    if (keys[BUCKET_IDX_SECOND].load(std::memory_order_relaxed) == key &&
        keys[BUCKET_IDX_SECOND].compare_exchange_strong(oldValue, 0)) {
        values[BUCKET_IDX_SECOND] = 0;
        return FkvState::FKV_EXIST;
    }
    if (HM_UNLIKELY(oldValue == 0)) {
        return FkvState::FKV_EXIST;
    }
    oldValue = key;

    if (keys[BUCKET_IDX_THIRD].load(std::memory_order_relaxed) == key &&
        keys[BUCKET_IDX_THIRD].compare_exchange_strong(oldValue, 0)) {
        values[BUCKET_IDX_THIRD] = 0;
        return FkvState::FKV_EXIST;
    }
    if (HM_UNLIKELY(oldValue == 0)) {
        return FkvState::FKV_EXIST;
    }

    return FkvState::FKV_NOT_EXIST;
}

FkvState NetHashBucket::Remove(uint64_t key, const std::function<BeforeRemoveFuncState(uint64_t)> &beforeRemoveFunc)
{
    /* don't put them into loop, flat code is faster than loop */
    uint64_t oldValue = key;
    if (keys[BUCKET_IDX_FIRST].load(std::memory_order_relaxed) == key &&
        keys[BUCKET_IDX_FIRST].compare_exchange_strong(oldValue, 0)) {
        if (HM_UNLIKELY(beforeRemoveFunc(values[BUCKET_IDX_FIRST])
            == BeforeRemoveFuncState::BEFORE_FAIL)) {
            return FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL;
        }

        values[BUCKET_IDX_FIRST] = 0;
        return FkvState::FKV_EXIST;
    }
    if (HM_UNLIKELY(oldValue == 0)) {
        return FkvState::FKV_EXIST;
    }
    oldValue = key;

    if (keys[BUCKET_IDX_SECOND].load(std::memory_order_relaxed) == key &&
        keys[BUCKET_IDX_SECOND].compare_exchange_strong(oldValue, 0)) {
        if (HM_UNLIKELY(beforeRemoveFunc(values[BUCKET_IDX_SECOND])
            == BeforeRemoveFuncState::BEFORE_FAIL)) {
            return FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL;
        }

        values[BUCKET_IDX_SECOND] = 0;
        return FkvState::FKV_EXIST;
    }
    if (HM_UNLIKELY(oldValue == 0)) {
        return FkvState::FKV_EXIST;
    }
    oldValue = key;

    if (keys[BUCKET_IDX_THIRD].load(std::memory_order_relaxed) == key &&
        keys[BUCKET_IDX_THIRD].compare_exchange_strong(oldValue, 0)) {
        if (HM_UNLIKELY(beforeRemoveFunc(values[BUCKET_IDX_THIRD])
            == BeforeRemoveFuncState::BEFORE_FAIL)) {
            return FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL;
        }

        values[BUCKET_IDX_THIRD] = 0;
        return FkvState::FKV_EXIST;
    }
    if (HM_UNLIKELY(oldValue == 0)) {
        return FkvState::FKV_EXIST;
    }

    return FkvState::FKV_NOT_EXIST;
}

bool MapperBase::Initialize(uint64_t reserve)
{
    /* already initialized */
    if (mOverflowEntryAlloc != nullptr) {
        return true;
    }

    /* get proper bucket count */
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

    /* allocate buckets for sub-maps */
    for (auto &mSubMap : mSubMaps) {
        NetHashBucket* tmp;
        if (!NewAndSetBucket(bucketCount, 0, tmp)) { return false;}
        mSubMap = tmp;
    }

    /* create overflow entry allocator */
    mOverflowEntryAlloc = new (std::nothrow) NetHeapAllocator();
    if (HM_UNLIKELY(mOverflowEntryAlloc == nullptr)) {
        FreeSubMaps();
        ock::ExternalLogger::PrintLog(ock::LogLevel::ERROR,
            "Failed to new overflow entry allocator, probably out of memory");
        return false;
    }

    /* set bucket count */
    mBucketCount = bucketCount;
    ock::ExternalLogger::PrintLog(ock::LogLevel::INFO,
        "fastKV inited, mBucketCount: " + std::to_string(mBucketCount));
    return true;
}

void MapperBase::UnInitialize()
{
    if (mOverflowEntryAlloc == nullptr) {
        return;
    }

    /* free overflowed entries firstly */
    FreeOverFlowedEntries();

    /* free sub map secondly */
    FreeSubMaps();

    /* free overflow entry at last */
    delete mOverflowEntryAlloc;
    mOverflowEntryAlloc = nullptr;
    mBucketCount = 0;
    currentSize = 0;
    zeroInside = false;
}

FkvState MapperBase::FindAndPutIfNotFound(uint64_t key, uint64_t &value,
    const std::function<BeforePutFuncState()> &beforePutFunc)
{
    if (HM_UNLIKELY(key == 0)) {
        std::lock_guard<std::mutex> lock(zeroKeyMutex_);
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
            currentSize++;
            return FkvState::FKV_NOT_EXIST;
        }
        return FkvState::FKV_KEY_CONFLICT;
    }

    /* get bucket */
    auto buck = &(mSubMaps[key % gSubMapCount][key % mBucketCount]);

    /* loop all buckets linked */
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

    // did not find, now do put. continue from the last bucket in find
    return PutKeyValue(key, value, buck, beforePutFunc);
}

FkvState MapperBase::Remove(uint64_t key)
{
    if (HM_UNLIKELY(key == 0)) {
        std::lock_guard<std::mutex> lock(zeroKeyMutex_);
        if (zeroInside) {
            if (__sync_bool_compare_and_swap(&zeroInside, true, false)) {
                zeroValue = 0;
                currentSize--;
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
            currentSize--;
            return FkvState::FKV_EXIST;
        }

        buck = buck->next;
    }

    return FkvState::FKV_NOT_EXIST;
}

FkvState MapperBase::Remove(uint64_t key, const std::function<BeforeRemoveFuncState(uint64_t)> &beforeRemoveFunc)
{
    if (HM_UNLIKELY(key == 0)) {
        std::lock_guard<std::mutex> lock(zeroKeyMutex_);
        if (!zeroInside) {
            return FkvState::FKV_NOT_EXIST;
        }
        if (__sync_bool_compare_and_swap(&zeroInside, true, false)) {
            auto ret = beforeRemoveFunc(zeroValue);
            if (HM_UNLIKELY(ret == BeforeRemoveFuncState::BEFORE_FAIL)) {
                return FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL;
            }
            zeroValue = 0;
            currentSize--;
        }
        return FkvState::FKV_EXIST;
    }

    /* get bucket */
    auto buck = &(mSubMaps[key % gSubMapCount][key % mBucketCount]);

    /* loop all buckets linked */
    uint64_t value;
    while (buck != nullptr) {
        buck->spinLock.Lock();
        if (buck->Find(key, value)) {
            auto ret = buck->Remove(key, beforeRemoveFunc);
            buck->spinLock.UnLock();
            if (HM_UNLIKELY(ret == FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL)) {
                return FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL;
            }

            currentSize--;
            return FkvState::FKV_EXIST;
        }
        buck->spinLock.UnLock();
        buck = buck->next;
    }

    return FkvState::FKV_NOT_EXIST;
}

FkvState MapperBase::Put(uint64_t key, uint64_t value)
{
    if (HM_UNLIKELY(key == 0)) {
        std::lock_guard<std::mutex> lock(zeroKeyMutex_);
        if (__sync_bool_compare_and_swap(&zeroInside, false, true)) {
            zeroValue = value;
            currentSize++;
            return FkvState::FKV_NOT_EXIST;
        }
        return FkvState::FKV_KEY_CONFLICT;
    }

    /* get bucket */
    auto buck = &(mSubMaps[key % gSubMapCount][key % mBucketCount]);
    // Execute `Put` operation on the first bucket when not find.
    // `Put` from the first bucket to reuse the previously removed location.
    /* try 8192 times */
    for (uint16_t i = 0; i < 8192; i++) {
        /* loop all buckets linked */
        while (buck != nullptr) {
            /* if there is an entry to put, just break */
            FkvState putRet = buck->Put(key, value, []() -> BeforePutFuncState { return {}; });
            if (putRet == FkvState::FKV_NOT_EXIST) {
                currentSize++;
                return FkvState::FKV_NOT_EXIST;
            }

            if (HM_UNLIKELY(putRet == FkvState::FKV_KEY_CONFLICT)) {
                return FkvState::FKV_KEY_CONFLICT;
            }
            /*
                * if no next bucket exist, just for break,
                * else move to next bucket linked
                */
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
        auto &lock = buck->spinLock;
        lock.Lock();
        /* if other thread allocated new buck already, unlock and continue */
        if (buck->next != nullptr) {
            buck = buck->next;
            lock.UnLock();
            continue;
        }

        /* firstly entered thread allocate new bucket */
        auto newBuck = static_cast<NetHashBucket *>(mOverflowEntryAlloc->Allocate(sizeof(NetHashBucket)));
        if (HM_UNLIKELY(newBuck == nullptr)) {
            lock.UnLock();
            ock::ExternalLogger::PrintLog(ock::LogLevel::ERROR, "Failed to allocate new bucket");
            return FkvState::FKV_FAIL;
        }
        /* link to current buck, set buck to new buck */
        buck->next = newBuck;
        buck = newBuck;

        /* unlock */
        lock.UnLock();
    }
    return FkvState::FKV_FAIL;
}

FkvState MapperBase::FindAndDeleteIfFound(const uint64_t key, uint64_t &value,
    const std::function<BeforeRemoveFuncState(uint64_t)> &beforeRemoveFunc)
{
    if (HM_UNLIKELY(key == 0)) {
        std::lock_guard<std::mutex> lock(zeroKeyMutex_);
        if (!zeroInside) {
            return FkvState::FKV_NOT_EXIST;
        }
        value = zeroValue;
        if (__sync_bool_compare_and_swap(&zeroInside, true, false)) {
            auto ret = beforeRemoveFunc(zeroValue);
            if (HM_UNLIKELY(ret == BeforeRemoveFuncState::BEFORE_FAIL)) {
                return FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL;
            }
            zeroValue = 0;
            currentSize--;
        }

        return FkvState::FKV_EXIST;
    }
    /* get bucket */
    auto buck = &(mSubMaps[key % gSubMapCount][key % mBucketCount]);

    while (buck != nullptr) {
        if (buck->Find(key, value)) {
            auto ret = buck->Remove(key, beforeRemoveFunc);
            if (HM_UNLIKELY(ret == FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL)) {
                return FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL;
            }
            currentSize--;
            return FkvState::FKV_EXIST;
        }

        buck = buck->next;
    }

    return FkvState::FKV_NOT_EXIST;
}

std::vector<std::pair<uint64_t, uint64_t>> MapperBase::ExportVec()
{
    std::vector<std::pair<uint64_t, uint64_t>> kvVec;
    if (zeroInside) {
        kvVec.emplace_back(0, zeroValue);
    }
    for (auto &mSubMap : mSubMaps) {
        for (uint64_t j = 0; j < mBucketCount; j++) {
            auto buck = &mSubMap[j];
            ExtractKeyValInBuck(buck, kvVec);
        }
    }
    return kvVec;
}

void MapperBase::FreeSubMaps()
{
    /* free all sub maps */
    for (auto &mSubMap : mSubMaps) {
        if (mSubMap != nullptr) {
            delete[] mSubMap;
            mSubMap = nullptr;
        }
    }
}

bool MapperBase::NewAndSetBucket(const uint64_t& bucketCount, const int& c, NetHashBucket* &bucketPtr)
{
    bucketPtr = new (std::nothrow) NetHashBucket[bucketCount];
    if (HM_UNLIKELY(bucketPtr == nullptr)) {
        FreeSubMaps();
        ock::ExternalLogger::PrintLog(ock::LogLevel::ERROR,
                                      "Failed to new hash bucket, probably out of memory");
        return false;
    }

    /* make physical page and set to zero */
    size_t bucketsBytes = sizeof(NetHashBucket) * bucketCount;
    char* destBytePtr = reinterpret_cast<char*>(bucketPtr);
    for (size_t i = 0; i < bucketsBytes; i += MEMSET_S_MAX_SIZE) {
        size_t bytesOnceSet = (i + MEMSET_S_MAX_SIZE <= bucketsBytes) ? MEMSET_S_MAX_SIZE : (bucketsBytes - i);
        auto ret = memset_s(destBytePtr + i, bytesOnceSet, c, bytesOnceSet);
        if (ret != 0) {
            delete[] bucketPtr;
            bucketPtr = nullptr;
            FreeSubMaps();
            ock::ExternalLogger::PrintLog(ock::LogLevel::ERROR,
                "memset_s failed... size: " + std::to_string(bucketsBytes) + ", error code:" + std::to_string(ret));
            return false;
        }
    }
    return true;
}

void MapperBase::FreeOverFlowedEntries()
{
    for (auto &mSubMap : mSubMaps) {
        if (mSubMap == nullptr) {
            continue;
        }

        /* free overflow entries in one sub map */
        for (uint64_t buckIndex = 0; buckIndex < mBucketCount; ++buckIndex) {
            auto curBuck = mSubMap[buckIndex].next;
            NetHashBucket *nextOverflowEntryBuck = nullptr;

            /* exit loop when curBuck is null */
            while (curBuck != nullptr) {
                /* assign next overflow buck to tmp variable */
                nextOverflowEntryBuck = curBuck->next;

                /* free this overflow bucket */
                mOverflowEntryAlloc->Free(curBuck);

                /* assign next to current */
                curBuck = nextOverflowEntryBuck;
            }
        }
    }
}

FkvState MapperBase::PutKeyValue(uint64_t key, uint64_t& value, EmbCache::NetHashBucket *buck,
    const std::function<BeforePutFuncState()>& beforePutFunc)
{
        /* try 8192 times */
    for (uint16_t i = 0; i < 8192; i++) {
        /* loop all buckets linked */
        while (buck != nullptr) {
            /* if there is an entry to put, just break */
            buck->spinLock.Lock();
            FkvState putRet = buck->Put(key, value, beforePutFunc);
            buck->spinLock.UnLock();
            if (putRet == FkvState::FKV_NOT_EXIST) {
                currentSize++;
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

            /*
                * if no next bucket exist, just for break,
                * else move to next bucket linked
                */
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
        auto &lock = buck->spinLock;
        lock.Lock();
        /* if other thread allocated new buck already, unlock and continue */
        if (buck->next != nullptr) {
            buck = buck->next;
            lock.UnLock();
            continue;
        }

        /* firstly entered thread allocate new bucket */
        auto newBuck = static_cast<NetHashBucket *>(mOverflowEntryAlloc->Allocate(sizeof(NetHashBucket)));
        if (HM_UNLIKELY(newBuck == nullptr)) {
            lock.UnLock();
            ock::ExternalLogger::PrintLog(ock::LogLevel::ERROR, "Failed to allocate new bucket");
            return FkvState::FKV_FAIL;
        }
        /* link to current buck, set buck to new buck */
        buck->next = newBuck;
        buck = newBuck;

        /* unlock */
        lock.UnLock();
    }
    return FkvState::FKV_FAIL;
}

void MapperBase::ExtractKeyValInBuck(EmbCache::NetHashBucket *buck,
                                     std::vector<std::pair<uint64_t, uint64_t>>& kvVec) const
{
    while (buck) {
        for (size_t k = 0; k < K_KVNUMINBUCKET; k++) {
            if (buck->keys[k] == 0) {
                continue;
            }
            buck->spinLock.Lock();
            kvVec.emplace_back(buck->keys[k].load(), buck->values[k]);
            buck->spinLock.UnLock();
        }
        buck = buck->next;
    }
}
}