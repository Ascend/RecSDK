/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_RET_CODE_H
#define EMBEDDING_CACHE_RET_CODE_H

namespace Embcache {

using RetCode = enum : int {
    H_OK = 0,
    H_ERROR = 1,
    H_NEW_OBJECT_FAILED = 2,
    H_ADDRESS_NULL = 3,
    H_NUM_SMALL = 4,
    H_COPY_ERROR = 5,
    H_ID_LARGE = 6,
    H_PADDING_SMALL = 7,
    H_OUTPUT_TYPE_ERROR = 8,
    H_SCENE_ERROR = 9,
    H_MEMORY_ALLOC_ERROR = 10,
    H_UNIQUE_UNINITIALIZED_ERROR = 11,
    H_TABLE_NOT_EXIST = 12,
    H_LOAD_ERROR = 13,
    H_INITIALIZER_INVALID = 14,
    H_EXT_EMBEDDING_SIZE_INVALID = 15,
    H_MAX_CACHESIZE_TOO_SMALL = 16,
    H_HOST_VOCAB_SIZE_TOO_SMALL = 17,
    H_THREAD_NUM_ERROR = 18,
    H_TABLE_CREATE_DUPLICATE = 19,
    H_ARG_NOT_EMPTY = 20,
    H_SIZE_ZERO = 21,
    H_TABLE_NAME_EMPTY = 22,
    H_PREFILL_BUFFER_SIZE_INVALID = 23,
    H_TABLE_NAME_TOO_LONG = 24,
    H_EMB_CACHE_INFO_LOST = 25
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_RET_CODE_H
