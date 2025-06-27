/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "hash_bucket.h"

namespace Embcache {

FkvState NetHashBucket::Put(uint64_t key, uint64_t& value, const std::function<BeforePutFuncState()>& beforePutFunc)
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

bool NetHashBucket::Find(const uint64_t key, uint64_t& value)
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

FkvState NetHashBucket::Remove(uint64_t key)
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

FkvState NetHashBucket::Remove(uint64_t key, const std::function<BeforeRemoveFuncState(uint64_t)>& beforeRemoveFunc)
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

}  // namespace Embcache