/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef HASH_TABLE_HASH_BUCKET_H
#define HASH_TABLE_HASH_BUCKET_H

#include <atomic>
#include <functional>

#include "common/common.h"

namespace Embcache {

constexpr size_t BUCKET_IDX_FIRST = 0;
constexpr size_t BUCKET_IDX_SECOND = 1;
constexpr size_t BUCKET_IDX_THIRD = 2;
constexpr size_t K_ALIGNMENT = 64;
constexpr size_t K_KV_NUM_IN_BUCKET = 3;

/*
 * @brief Spin lock entry in bucket
 * used for alloc overflowed buckets
 */
struct NetHashLockEntry {
    uint64_t lock = 0;

    void Lock()
    {
        while (!__sync_bool_compare_and_swap(&lock, 0, 1)) {}
    }

    void UnLock()
    {
        __atomic_store_n(&lock, 0, __ATOMIC_SEQ_CST);
    }
} __attribute__((packed));

struct alignas(K_ALIGNMENT) NetHashBucket {
    std::atomic<uint64_t> keys[K_KV_NUM_IN_BUCKET]{};
    uint64_t values[K_KV_NUM_IN_BUCKET]{};
    NetHashBucket* next = nullptr;
    NetHashLockEntry spinLock{};

    FkvState Put(uint64_t key, uint64_t& value, const std::function<BeforePutFuncState()>& beforePutFunc);

    bool Find(const uint64_t key, uint64_t& value);

    FkvState Remove(uint64_t key);

    FkvState Remove(uint64_t key, const std::function<BeforeRemoveFuncState(uint64_t)>& beforeRemoveFunc);
};

}  // namespace Embcache
#endif  // HASH_TABLE_HASH_BUCKET_H
