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

#ifndef MXREC_MAPPER_BASE_H
#define MXREC_MAPPER_BASE_H

#include <iostream>
#include <atomic>
#include <cstring>
#include <vector>
#include <mutex>
#include <bitset>
#include <future>
#include <cstdlib>
#include <thread>
#include <algorithm>
#include "securec.h"
#include "embedding_cache/common.h"
#include "external_logger.h"

namespace EmbCache {
/*
 * @brief Allocator template, for extend memory allocation for overflowed buckets
 */

static constexpr size_t K_ALIGNMENT = 64;
static constexpr size_t K_KVNUMINBUCKET = 3;

enum BucketIdx {
    FIRST,
    SECOND,
    THIRD
};

class NetHeapAllocator {
public:
    void *Allocate(uint64_t size)
    {
        return calloc(1, size);
    }

    void Free(void *p)
    {
        if (HM_LIKELY(p != nullptr)) {
            free(p);
            p = nullptr;
        }
    }
};

/*
 * @brief Spin lock entry in bucket
 * used for alloc overflowed buckets
 */

struct NetHashLockEntry {
    uint64_t lock = 0;

    /*
     * @brief Spin lock
     */
    void Lock()
    {
        while (!__sync_bool_compare_and_swap(&lock, 0, 1)) {
        }
    }

    /*
     * @brief Unlock
     */
    void UnLock()
    {
        __atomic_store_n(&lock, 0, __ATOMIC_SEQ_CST);
    }
} __attribute__((packed));

/*
 * @brief Store the key/value into a linked array with 6 items,
 * because 64bytes is one cache line
 */

struct alignas(K_ALIGNMENT)NetHashBucket {
    std::atomic<uint64_t> keys[K_KVNUMINBUCKET]{};
    uint64_t values[K_KVNUMINBUCKET]{};
    NetHashBucket *next = nullptr;
    NetHashLockEntry spinLock{};

    FkvState Put(uint64_t key, uint64_t &value, const std::function<BeforePutFuncState()> &beforePutFunc)
    {
        /* don't put them into loop, flat code is faster than loop */
        uint64_t oldKey = 0;
        if (keys[BucketIdx::FIRST].load(std::memory_order_relaxed) == 0 &&
            keys[BucketIdx::FIRST].compare_exchange_strong(oldKey, key)) {
            BeforePutFuncState ret = beforePutFunc();
            if (HM_UNLIKELY(ret == BeforePutFuncState::BEFORE_FAIL)) {
                keys[BucketIdx::FIRST] = 0;
                return FkvState::FKV_BEFORE_PUT_FUNC_FAIL;
            }
            if (HM_UNLIKELY(ret == BeforePutFuncState::BEFORE_NO_SPACE)) {
                keys[BucketIdx::FIRST] = 0;
                return FkvState::FKV_NO_SPACE;
            }
            values[BucketIdx::FIRST] = value;
            return FkvState::FKV_NOT_EXIST;
        }

        if (HM_UNLIKELY(oldKey == key)) {
            return FkvState::FKV_KEY_CONFLICT;
        }

        oldKey = 0;
        if (keys[BucketIdx::SECOND].load(std::memory_order_relaxed) == 0 &&
            keys[BucketIdx::SECOND].compare_exchange_strong(oldKey, key)) {
            BeforePutFuncState ret = beforePutFunc();
            if (HM_UNLIKELY(ret == BeforePutFuncState::BEFORE_FAIL)) {
                keys[BucketIdx::SECOND] = 0;
                return FkvState::FKV_BEFORE_PUT_FUNC_FAIL;
            }
            if (HM_UNLIKELY(ret == BeforePutFuncState::BEFORE_NO_SPACE)) {
                keys[BucketIdx::SECOND] = 0;
                return FkvState::FKV_NO_SPACE;
            }
            values[BucketIdx::SECOND] = value;
            return FkvState::FKV_NOT_EXIST;
        }

        if (HM_UNLIKELY(oldKey == key)) {
            return FkvState::FKV_KEY_CONFLICT;
        }

        oldKey = 0;
        if (keys[BucketIdx::THIRD].load(std::memory_order_relaxed) == 0 &&
            keys[BucketIdx::THIRD].compare_exchange_strong(oldKey, key)) {
            BeforePutFuncState ret = beforePutFunc();
            if (HM_UNLIKELY(ret == BeforePutFuncState::BEFORE_FAIL)) {
                keys[BucketIdx::THIRD] = 0;
                return FkvState::FKV_BEFORE_PUT_FUNC_FAIL;
            }
            if (HM_UNLIKELY(ret == BeforePutFuncState::BEFORE_NO_SPACE)) {
                keys[BucketIdx::THIRD] = 0;
                return FkvState::FKV_NO_SPACE;
            }
            values[BucketIdx::THIRD] = value;
            return FkvState::FKV_NOT_EXIST;
        }

        if (HM_UNLIKELY(oldKey == key)) {
            return FkvState::FKV_KEY_CONFLICT;
        }

        return FkvState::FKV_FAIL;
    }

    /*
     * @brief Remove the address from the bucket and get size
     */
    bool Find(const uint64_t key, uint64_t &value)
    {
        /*
         * expand the loop, instead of put them into a for/while loop for performance
         */
        if (key == keys[BucketIdx::FIRST].load(std::memory_order_relaxed)) {
            value = values[BucketIdx::FIRST];
            return true;
        }

        if (key == keys[BucketIdx::SECOND].load(std::memory_order_relaxed)) {
            value = values[BucketIdx::SECOND];
            return true;
        }

        if (key == keys[BucketIdx::THIRD].load(std::memory_order_relaxed)) {
            value = values[BucketIdx::THIRD];
            return true;
        }

        return false;
    }

    FkvState Remove(uint64_t key)
    {
        /* don't put them into loop, flat code is faster than loop */
        uint64_t oldValue = key;
        if (keys[BucketIdx::FIRST].load(std::memory_order_relaxed) == key &&
            keys[BucketIdx::FIRST].compare_exchange_strong(oldValue, 0)) {
            values[BucketIdx::FIRST] = 0;
            return FkvState::FKV_EXIST;
        }
        if (HM_UNLIKELY(oldValue == 0)) {
            return FkvState::FKV_EXIST;
        }
        oldValue = key;

        if (keys[BucketIdx::SECOND].load(std::memory_order_relaxed) == key &&
            keys[BucketIdx::SECOND].compare_exchange_strong(oldValue, 0)) {
            values[BucketIdx::SECOND] = 0;
            return FkvState::FKV_EXIST;
        }
        if (HM_UNLIKELY(oldValue == 0)) {
            return FkvState::FKV_EXIST;
        }
        oldValue = key;

        if (keys[BucketIdx::THIRD].load(std::memory_order_relaxed) == key &&
            keys[BucketIdx::THIRD].compare_exchange_strong(oldValue, 0)) {
            values[BucketIdx::THIRD] = 0;
            return FkvState::FKV_EXIST;
        }
        if (HM_UNLIKELY(oldValue == 0)) {
            return FkvState::FKV_EXIST;
        }

        return FkvState::FKV_NOT_EXIST;
    }

    FkvState Remove(uint64_t key, const std::function<BeforeRemoveFuncState(uint64_t)> &beforeRemoveFunc)
    {
        /* don't put them into loop, flat code is faster than loop */
        uint64_t oldValue = key;
        if (keys[BucketIdx::FIRST].load(std::memory_order_relaxed) == key &&
            keys[BucketIdx::FIRST].compare_exchange_strong(oldValue, 0)) {
            if (HM_UNLIKELY(beforeRemoveFunc(values[BucketIdx::FIRST]) == BeforeRemoveFuncState::BEFORE_FAIL)) {
                return FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL;
            }

            values[BucketIdx::FIRST] = 0;
            return FkvState::FKV_EXIST;
        }
        if (HM_UNLIKELY(oldValue == 0)) {
            return FkvState::FKV_EXIST;
        }
        oldValue = key;

        if (keys[BucketIdx::SECOND].load(std::memory_order_relaxed) == key &&
            keys[BucketIdx::SECOND].compare_exchange_strong(oldValue, 0)) {
            if (HM_UNLIKELY(beforeRemoveFunc(values[BucketIdx::SECOND]) == BeforeRemoveFuncState::BEFORE_FAIL)) {
                return FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL;
            }

            values[BucketIdx::SECOND] = 0;
            return FkvState::FKV_EXIST;
        }
        if (HM_UNLIKELY(oldValue == 0)) {
            return FkvState::FKV_EXIST;
        }
        oldValue = key;

        if (keys[BucketIdx::THIRD].load(std::memory_order_relaxed) == key &&
            keys[BucketIdx::THIRD].compare_exchange_strong(oldValue, 0)) {
            if (HM_UNLIKELY(beforeRemoveFunc(values[BucketIdx::THIRD]) == BeforeRemoveFuncState::BEFORE_FAIL)) {
                return FkvState::FKV_BEFORE_REMOVE_FUNC_FAIL;
            }

            values[BucketIdx::THIRD] = 0;
            return FkvState::FKV_EXIST;
        }
        if (HM_UNLIKELY(oldValue == 0)) {
            return FkvState::FKV_EXIST;
        }

        return FkvState::FKV_NOT_EXIST;
    }
};


class MapperBase {
public:
    // DEFINE_RDMA_REF_COUNT_FUNCTIONS
    std::atomic<uint64_t> currentSize{ 0 };

    MapperBase() = default;

    virtual ~MapperBase() = default;

    bool Initialize(uint64_t reserve)
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

    virtual void UnInitialize()
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

    FkvState FindAndPutIfNotFound(uint64_t key, uint64_t &value,
        const std::function<BeforePutFuncState()> &beforePutFunc)
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

    FkvState Remove(uint64_t key)
    {
        if (HM_UNLIKELY(key == 0)) {
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

    FkvState Remove(uint64_t key, const std::function<BeforeRemoveFuncState(uint64_t)> &beforeRemoveFunc)
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
                currentSize--;
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

                currentSize--;
                return FkvState::FKV_EXIST;
            }

            buck = buck->next;
        }

        return FkvState::FKV_NOT_EXIST;
    }

    FkvState Put(uint64_t key, uint64_t value)
    {
        if (HM_UNLIKELY(key == 0)) {
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

    bool Find(const uint64_t key, uint64_t &value)
    {
        if (HM_UNLIKELY(key == 0)) {
            if (zeroInside) {
                value = zeroValue;
                return true;
            }
            return false;
        }
        /* get bucket */
        auto buck = &(mSubMaps[key % gSubMapCount][key % mBucketCount]);

        /* loop all buckets linked */
        while (buck != nullptr) {
            if (buck->Find(key, value)) {
                return true;
            }

            buck = buck->next;
        }

        return false;
    }

    /* When used in muti thread, this function can only be used when keys are uniqued */
    FkvState FindAndDeleteIfFound(const uint64_t key, uint64_t &value,
        const std::function<BeforeRemoveFuncState(uint64_t)> &beforeRemoveFunc)
    {
        if (HM_UNLIKELY(key == 0)) {
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

    std::vector<std::pair<uint64_t, uint64_t>> ExportVec()
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

protected:
    static constexpr uint16_t gSubMapCount = 5; /* count of sub map */
    static constexpr uint64_t gPrimesCount = 256;

    /* make sure the size of this class is 64 bytes, fit into one cache line */
    NetHeapAllocator *mOverflowEntryAlloc = nullptr; /* allocate overflowed entry in one bucket */
    NetHashBucket *mSubMaps[gSubMapCount]{};         /* sub map */
    uint64_t mBucketCount = 0;                       /* bucket count of each sub map */
    uint64_t mBaseSize = 4096;                       /* base size */
    bool zeroInside = false;
    uint64_t zeroValue = 0;

    const uint64_t gPrimes[gPrimesCount] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                                            41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
                                            97, 103, 109, 113, 127, 137, 139, 149, 157, 167,
                                            179, 193, 199, 211, 227, 241, 257, 277, 293, 313,
                                            337, 359, 383, 409, 439, 467, 503, 541, 577, 619,
                                            661, 709, 761, 823, 887, 953, 1031, 1109, 1193, 1289,
                                            1381, 1493, 1613, 1741, 1879, 2029, 2179, 2357, 2549,
                                            2753, 2971, 3209, 3469, 3739, 4027, 4349, 4703, 5087,
                                            5503, 5953, 6427, 6949, 7517, 8123, 8783, 9497, 10273,
                                            11113, 12011, 12983, 14033, 15173, 16411, 17749, 19183,
                                            20753, 22447, 24281, 26267, 28411, 30727, 33223, 35933,
                                            38873, 42043, 45481, 49201, 53201, 57557, 62233, 67307,
                                            72817, 78779, 85229, 92203, 99733, 107897, 116731, 126271,
                                            136607, 147793, 159871, 172933, 187091, 202409, 218971, 236897,
                                            256279, 277261, 299951, 324503, 351061, 379787, 410857, 444487,
                                            480881, 520241, 562841, 608903, 658753, 712697, 771049, 834181,
                                            902483, 976369, 1056323, 1142821, 1236397, 1337629, 1447153,
                                            1565659, 1693859, 1832561, 1982627, 2144977, 2320627, 2510653,
                                            2716249, 2938679, 3179303, 3439651, 3721303, 4026031, 4355707,
                                            4712381, 5098259, 5515729, 5967347, 6456007, 6984629, 7556579,
                                            8175383, 8844859, 9569143, 10352717, 11200489, 12117689,
                                            13109983, 14183539, 15345007, 16601593, 17961079, 19431899,
                                            21023161, 22744717, 24607243, 26622317, 28802401, 31160981,
                                            33712729, 36473443, 39460231, 42691603, 46187573, 49969847,
                                            54061849, 58488943, 63278561, 68460391, 74066549, 80131819,
                                            86693767, 93793069, 101473717, 109783337, 118773397, 128499677,
                                            139022417, 150406843, 162723577, 176048909, 190465427,
                                            206062531, 222936881, 241193053, 260944219, 282312799,
                                            305431229, 330442829, 357502601, 386778277, 418451333,
                                            452718089, 489790921, 529899637, 573292817, 620239453,
                                            671030513, 725980837, 785430967, 849749479, 919334987,
                                            994618837, 1076067617, 1164186217, 1259520799, 1362662261,
                                            1474249943, 1594975441, 1725587117, 1866894511, 2019773507,
                                            2185171673, 2364114217, 2557710269, 2767159799, 2993761039,
                                            3238918481, 3504151727, 3791104843, 4101556399, 4294967291};

private:
    void FreeSubMaps()
    {
        /* free all sub maps */
        for (auto &mSubMap : mSubMaps) {
            if (mSubMap != nullptr) {
                delete[] mSubMap;
                mSubMap = nullptr;
            }
        }
    }

    /*
     * Description: allocate buckets and init it
     * Parameter: bucketCount - the bucket counts
     * Parameter: c - the value to be copied
     * Parameter: bucketPtr - pointing at the bucket array which is allocated
     * NOTES: SECUREC_MEM_MAX_LEN of memset_s function is 2GB
     */
    bool NewAndSetBucket(const uint64_t& bucketCount, const int& c, NetHashBucket* &bucketPtr)
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

    void FreeOverFlowedEntries()
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

    FkvState PutKeyValue(uint64_t key, uint64_t& value, EmbCache::NetHashBucket *buck,
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

    void ExtractKeyValInBuck(EmbCache::NetHashBucket *buck, std::vector<std::pair<uint64_t, uint64_t>>& kvVec)
    {
        while (buck) {
            for (size_t k = 0; k < K_KVNUMINBUCKET; k++) {
                if (buck->keys[k] == 0) {
                    continue;
                }
                kvVec.emplace_back(buck->keys[k].load(), buck->values[k]);
            }
            buck = buck->next;
        }
    }
};
}
#endif // MXREC_MAPPER_BASE_H
