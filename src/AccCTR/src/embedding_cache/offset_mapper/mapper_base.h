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

    FkvState Put(uint64_t key, uint64_t &value,
        const std::function<BeforePutFuncState()> &beforePutFunc);

    /*
     * @brief Remove the address from the bucket and get size
     */
    bool Find(const uint64_t key, uint64_t &value);

    FkvState Remove(uint64_t key);

    FkvState Remove(uint64_t key, const std::function<BeforeRemoveFuncState(uint64_t)> &beforeRemoveFunc);
};


class MapperBase {
public:
    // DEFINE_RDMA_REF_COUNT_FUNCTIONS
    std::atomic<uint64_t> currentSize{ 0 };

    MapperBase() = default;

    virtual ~MapperBase() = default;

    bool Initialize(uint64_t reserve);

    virtual void UnInitialize();

    FkvState FindAndPutIfNotFound(uint64_t key, uint64_t &value,
        const std::function<BeforePutFuncState()> &beforePutFunc);

    FkvState Remove(uint64_t key);

    FkvState Remove(uint64_t key, const std::function<BeforeRemoveFuncState(uint64_t)> &beforeRemoveFunc);

    FkvState Put(uint64_t key, uint64_t value);

    bool Find(const uint64_t key, uint64_t &value);

    /* When used in muti thread, this function can only be used when keys are uniqued */
    FkvState FindAndDeleteIfFound(const uint64_t key, uint64_t &value,
        const std::function<BeforeRemoveFuncState(uint64_t)> &beforeRemoveFunc);

    std::vector<std::pair<uint64_t, uint64_t>> ExportVec();

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
    std::mutex zeroKeyMutex_;

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
    void FreeSubMaps();

    /*
     * Description: allocate buckets and init it
     * Parameter: bucketCount - the bucket counts
     * Parameter: c - the value to be copied
     * Parameter: bucketPtr - pointing at the bucket array which is allocated
     * NOTES: SECUREC_MEM_MAX_LEN of memset_s function is 2GB
     */
    bool NewAndSetBucket(const uint64_t& bucketCount, const int& c, NetHashBucket* &bucketPtr);

    void FreeOverFlowedEntries();

    FkvState PutKeyValue(uint64_t key, uint64_t& value, EmbCache::NetHashBucket *buck,
        const std::function<BeforePutFuncState()>& beforePutFunc);

    void ExtractKeyValInBuck(EmbCache::NetHashBucket *buck, std::vector<std::pair<uint64_t, uint64_t>>& kvVec);
};
}
#endif // MXREC_MAPPER_BASE_H
