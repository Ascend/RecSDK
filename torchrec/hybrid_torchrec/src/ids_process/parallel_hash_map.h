/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
* Copyright (c) huawei Platforms, Inc. and affiliates.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree.
 */
#ifndef HYBRID_HASHMAP_H
#define HYBRID_HASHMAP_H

#include <memory.h>

#include <cstdint>
#include <cstdio>
#include <utility>

namespace hybrid {

class ParallelHashMap {
public:
    ParallelHashMap() {}
    explicit ParallelHashMap(int64_t capacity){};
    virtual std::pair<bool, int64_t> Find(int64_t key) = 0;
    virtual bool Insert(int64_t k, int64_t v) = 0;
    virtual int64_t Size() = 0;
    virtual bool CheckOrExpansion(int64_t insertNum) = 0;
    virtual int64_t Capacity() = 0;
    virtual void Clear() = 0;
    virtual ~ParallelHashMap(){};
};

// Create
using CreateMap = ParallelHashMap*(int64_t n);
using DestroyMap = void(ParallelHashMap*);

}  // namespace hybrid
#endif