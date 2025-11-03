/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#ifndef LCCL_COMM_ARGS_H
#define LCCL_COMM_ARGS_H
#include <cstdint>
#include <stdexcept>
#include <string>

#if !defined(__DAV_C220_VEC__) && !defined(__DAV_C310__)
#define GM_ADDR int8_t*
#else
#define FORCE_INLINE_AICORE __attribute__((always_inline)) inline __aicore__
#include "kernel_operator.h"
#endif
namespace Lcal {

constexpr int LCAL_MAX_RANK_SIZE = 32;  // max tested rank size
constexpr int RANK_SIZE_TWO = 2;  // for 910_93
constexpr int64_t FLAG_NUM = 256 * 1024;  // calculate from IPC_DATA_OFFSET, each card get 512K int64 for flag
constexpr int64_t IPC_DATA_OFFSET = 2 * 1024 * 1024;  // 2MB for flag
constexpr int64_t SYNC_FLAG_BIT_NUM = 10;
constexpr int64_t MEM_DMA_UNIT_INT_NUM = 4;
constexpr int64_t EVENT_ID_MASK = 0xFFFFFFFF;
constexpr int64_t PING_PONG_SIZE = 2;
constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = 190 * 1024;
constexpr int64_t SMALL_DATA_SIZE = 1 * 1024 * 1024;  // distinguish different op according to data volumn
constexpr int64_t UB_SINGLE_PING_PONG_ADD_SIZE_MAX = UB_SINGLE_DMA_SIZE_MAX / 2;
constexpr int UB_ALIGN_SIZE = 32;

struct CommArgs {
    void SetBuff(int8_t* b[LCAL_MAX_RANK_SIZE])
    {
        if (rankSize > LCAL_MAX_RANK_SIZE) {
            throw std::invalid_argument(
                "Max support rank size is " + std::to_string(LCAL_MAX_RANK_SIZE) +
                ", get:" + std::to_string(rankSize));
        }
        for (int i = 0; i < rankSize; ++i) {
            peerMems[i] = b[i];
        }
    }

    int rank = 0;  // attr rank_id, global rank
    int localRank = 0;
    int rankSize = 0;  // global rank size
    int localRankSize = 0;
    uint32_t extraFlag = 0;  // 32 bit map，
    GM_ADDR peerMems[LCAL_MAX_RANK_SIZE] = {};
    int64_t sendCountMatrix[LCAL_MAX_RANK_SIZE * LCAL_MAX_RANK_SIZE] = {};  // for all2allv
};
}
#endif // LCCL_COMM_ARGS_H
