/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_CONSTANTS_H
#define EMBEDDING_CACHE_CONSTANTS_H

namespace Embcache {

const int INVALID_EMB_SIZE = -1;

constexpr float CONSTANT_VALUE_MAX = 1e9;
constexpr float CONSTANT_VALUE_MIN = -1e9;

constexpr float NORMAL_MEAN_MAX = 1e9;
constexpr float NORMAL_MEAN_MIN = -1e9;

constexpr float NORMAL_STDDEV_MAX = 100;
constexpr float NORMAL_STDDEV_MIN = 0;

constexpr float INIT_K_MAX = 10000;
constexpr float INIT_K_MIN = -10000;

constexpr int64_t INVALID_KEY = -1;

const size_t MEMSET_S_MAX_SIZE = 2LL * 1024 * 1024 * 1024 - 1;

struct EmbMemPoolConfigConstants {
    static constexpr uint64_t bufferSize = 1024;
    // 512 * 1024 * 1024 * 128 (emb_dim) * 4 (fp32) = 256 GB
    static constexpr uint64_t hostVocabSize = 512 * 1024 * 1024;
    static constexpr uint32_t refillThreadNum = 4;
};

constexpr uint64_t FAST_HASHMAP_RESERVE_BUCKET_NUM = 2 * 1024 * 1024;

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_CONSTANTS_H